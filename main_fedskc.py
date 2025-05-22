#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
from torch import eq
from torch.optim import SGD
from tqdm import tqdm
import copy, sys
import torch
import random
from pathlib import Path
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from utils.options import args_parser
from model.resnet_base import ResNet_cifar
from utils.long_tailed_cifar10 import train_long_tail
from utils.dataset import classify_label, show_clients_data_distribution, Indices2Dataset
from utils.sample_dirichlet import clients_indices
from utils.util_func import average_kns
from utils.infonce_lcl import InfoNCE_LCL

model_dir = (Path(__file__).parent / "model").resolve()
if str(model_dir) not in sys.path: sys.path.insert(0, str(model_dir))
utils_dir = (Path(__file__).parent / "utils").resolve()
if str(utils_dir) not in sys.path: sys.path.insert(0, str(utils_dir))


class Global(object):
    def __init__(self,num_classes: int,device: str,args):
        self.device = device
        self.num_classes = num_classes
        self.args = args
        if (args.data_name=='cifar10') or (args.data_name=='cifar100'):
            self.global_model = ResNet_cifar(
                resnet_size=8, scaling=4, save_activations=False, group_norm_num_groups=None,
                freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes, in_channels=args.in_channels
            ).to(args.device)
        else:
            exit('Load model error: unknown model!')

    def average_parameters(self, list_dicts_local_params: list, list_nums_local_data: list):
        global_params = copy.deepcopy(list_dicts_local_params[0])
        for name_param in list_dicts_local_params[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            global_params[name_param] = value_global_param
        return global_params
    
    def get_dis_weights(self, list_nums_local_data, list_local_kns, global_kn):
        list_local_dis = []
        for each_local_kn in list_local_kns:
            temp_dis = 0.0
            for key,value in each_local_kn.items():
                temp_dis += torch.norm(value - global_kn[key], p=2).item()
            list_local_dis.append(temp_dis)
        list_discrepancy_weights = []
        for local_dis,local_num in zip(list_local_dis,list_nums_local_data):
            temp_weight = local_num - local_dis/sum(list_local_dis) * local_dis + local_num/sum(list_nums_local_data)
            temp_weight = 1.0 / (1.0 + np.exp(-temp_weight))
            list_discrepancy_weights.append(temp_weight)
        return list_discrepancy_weights / sum(list_discrepancy_weights)
    
    def discrepancy_aggregation(self, list_dicts_local_params, list_discrepancy_weights):
        global_params = copy.deepcopy(list_dicts_local_params[0])
        for name_param in list_dicts_local_params[0]:
            list_values_param = []
            for dict_local_params, local_dis_weight in zip(list_dicts_local_params, list_discrepancy_weights):
                list_values_param.append(dict_local_params[name_param] * local_dis_weight)
            value_global_param = sum(list_values_param)
            global_params[name_param] = value_global_param
        return global_params
    
    def period_review(self, global_params, previous_global_params, global_kn, previous_global_kn):
        list_cur_var = []
        list_pre_var = []
        for key,value in global_kn.items():
            variance = torch.var(global_kn[key]).item()
            list_cur_var.append(variance)
        for key,value in previous_global_kn.items():
            variance = torch.var(previous_global_kn[key]).item()
            list_pre_var.append(variance)
        for name_param in global_params:
            cur_value = global_params[name_param]
            pre_value = previous_global_params[name_param]
            beta = 0.99
            temp_v = beta*cur_value+(1.0-beta)*((sum(list_cur_var)-sum(list_pre_var))/sum(list_pre_var))*(pre_value-cur_value)
            global_params[name_param] = temp_v
        return global_params
    
    def get_global_kn(self, list_local_kns):
        global_kn = {}
        for each_local_kn in list_local_kns:
            for key,value in each_local_kn.items():
                if key in global_kn:
                    global_kn[key].append(value)
                else:
                    global_kn[key] = [value]
        for key,value in global_kn.items():
            temp = torch.stack(value)
            distances = torch.cdist(temp, temp, p=2)
            distances.fill_diagonal_(float("inf"))
            min_distances, min_indexs = torch.min(distances, dim=1)
            for id,index in enumerate(min_indexs):
                temp[id] = (value[id] + value[index]) * 0.5
            global_kn[key] = torch.mean(temp, dim=0)
        return global_kn

    def global_eval(self, average_params, data_test, batch_size_test):
        self.global_model.load_state_dict(average_params)
        self.global_model.eval()
        with torch.no_grad():
            num_corrects = 0
            test_loader = DataLoader(data_test, batch_size_test, shuffle=False)
            for data_batch in test_loader:
                images, labels = data_batch
                _, outputs = self.global_model(images.to(self.device))
                _, predicts = torch.max(outputs, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            accuracy = num_corrects / len(data_test)
        return accuracy

    def download_params(self):
        return self.global_model.state_dict()


class Local(object):
    def __init__(self, data_client, class_list, global_kn):
        args = args_parser()
        self.data_client = data_client
        self.device = args.device
        self.class_compose = class_list
        if (args.data_name=='cifar10') or (args.data_name=='cifar100'):
            self.local_model = ResNet_cifar(
                resnet_size=8, scaling=4, save_activations=False, group_norm_num_groups=None,
                freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes, in_channels=args.in_channels
            ).to(args.device)
        else:
            exit('Load model error: unknown model!')
        self.criterion = CrossEntropyLoss().to(args.device)
        self.optimizer = SGD(self.local_model.parameters(), lr=args.lr_local_training)
        self.global_kn = global_kn
        self.lcl_func = InfoNCE_LCL(tau=0.08)

    def local_train(self, args, global_params, round_id):
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4), 
                transforms.RandomHorizontalFlip()])
        self.local_model.load_state_dict(global_params)
        temp_parameters = self.local_model.parameters()
        self.local_model.train()
        local_kn_labels = {}
        for tr_idx in range(args.num_epochs_local_training):
            data_loader = DataLoader(dataset=self.data_client, batch_size=args.batch_size_local_training, shuffle=True)
            for data_batch in data_loader:
                images, labels_g = data_batch
                images = images.to(self.device)
                labels = labels_g.to(self.device)
                if (args.data_name=='cifar10') or (args.data_name=='cifar100'):
                    images = transform_train(images)
                _, outputs = self.local_model(images)
                proximal_term = 0.0
                for w, w_t in zip(self.local_model.parameters(), temp_parameters):
                    proximal_term += (w - w_t).norm(2)
                # Cross-Entropy loss
                ce_loss = self.criterion(outputs, labels) + (0.01/2)*proximal_term
                '''Local Contrastive Learning (LCL)'''
                pos_key, neg_keys = [], []
                for i in range(len(labels)):
                    label_id = labels_g[i].item()
                    try:
                        pos_key.append(self.global_kn[label_id])
                    except:
                        pos_key.append(outputs[i].detach())
                    neg_keys.append(torch.stack(list(self.global_kn.values())))
                # LCL loss
                lcl_loss = self.lcl_func.infonce_lcl_loss(outputs, torch.stack(pos_key), torch.stack(neg_keys))
                lcl_rate = round_id/100 if round_id<100 else 1.0
                loss = ce_loss + lcl_loss * lcl_rate
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                # structured knowledge of all labels
                if tr_idx == args.num_epochs_local_training - 1:  # last local epoch
                    for i in range(len(labels)):
                        if labels_g[i].item() in local_kn_labels:
                            local_kn_labels[labels_g[i].item()].append(outputs[i,:])
                        else:
                            local_kn_labels[labels_g[i].item()] = [outputs[i,:]]
        # get local structured knowledge
        local_kn = average_kns(local_kn_labels)
        return self.local_model.state_dict(), local_kn


def FedSKC_main():
    args = args_parser()
    print(
        '=====> long-tail rate (imb_factor): {ib}\n'
        '=====> non-iid rate (non_iid): {non_iid}\n'
        '=====> activate clients (num_online_clients): {num_online_clients}\n'
        '=====> dataset classes (num_classes): {num_classes}\n'.format(
            ib=args.imb_factor,  # long-tail imbalance factor
            non_iid=args.non_iid_alpha,  # non-iid alpha based on Dirichlet-distribution
            num_online_clients=args.num_online_clients,  # activate clients in FL
            num_classes=args.num_classes,  # dataset classes
        )
    )
    random_state = np.random.RandomState(args.seed)
    # load dataset
    transform_all = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.data_name=='cifar10': 
        data_local_training = datasets.CIFAR10('./data/cifar10/', train=True, download=True, transform=transform_all)
        data_global_test = datasets.CIFAR10('./data/cifar10/', train=False, transform=transform_all)
    elif args.data_name=='cifar100':
        data_local_training = datasets.CIFAR100('./data/cifar100/', train=True, download=True, transform=transform_all)
        data_global_test = datasets.CIFAR100('./data/cifar100/', train=False, transform=transform_all)
    else:
        exit('Load dataset error: unknown dataset!')
    # distribute dataset
    list_label2indices = classify_label(data_local_training, args.num_classes)
    # heterogeneous and long_tailed setting
    _, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices), args.num_classes,
                                                      args.imb_factor, args.imb_type)
    list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), args.num_classes,
                                          args.num_clients, args.non_iid_alpha, args.seed)
    original_dict_per_client = show_clients_data_distribution(data_local_training, list_client2indices,
                                                              args.num_classes)

    server_model = Global(num_classes=args.num_classes,device=args.device,args=args)
    total_clients = list(range(args.num_clients))
    indices2data = Indices2Dataset(data_local_training)
    fedskc_trained_acc = []
    global_kn = {}
    for key in range(args.num_classes): # init global structured knowledge
        global_kn[key] = torch.empty(args.num_classes, device=args.device, requires_grad=False)

    # federated learning training
    for round_id in tqdm(range(1, args.num_rounds+1), desc='server-training'):
        global_params = server_model.download_params()
        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        list_dicts_local_params = []
        list_nums_local_data = []
        list_local_kns = []
        # local model training
        for client in online_clients:
            indices2data.load(list_client2indices[client])
            data_client = indices2data
            list_nums_local_data.append(len(data_client))
            local_model = Local(data_client=data_client, class_list=original_dict_per_client[client], global_kn=global_kn)
            # local model update
            local_params, local_kn = local_model.local_train(args, copy.deepcopy(global_params), round_id)
            list_dicts_local_params.append(copy.deepcopy(local_params))
            list_local_kns.append(copy.deepcopy(local_kn))
        
        pre_global_params = global_params
        pre_global_kn = global_kn
        # send all local structured knowledge to server, get global structured knowledge
        global_kn = server_model.get_global_kn(list_local_kns)
        '''Global Discrepancy Aggregation (GDA)'''
        # list_discrepancy_weights = server_model.get_dis_weights(list_nums_local_data, list_local_kns, global_kn)
        # global_params = server_model.discrepancy_aggregation(list_dicts_local_params, list_discrepancy_weights)
        global_params = server_model.average_parameters(list_dicts_local_params, list_nums_local_data)
        '''Global Period Review (GPR)'''
        # if round_id!=1:
        #     global_params = server_model.period_review(global_params, pre_global_params, global_kn, pre_global_kn)
        # global model eval
        one_re_train_acc = server_model.global_eval(global_params, data_global_test, args.batch_size_test)
        fedskc_trained_acc.append(one_re_train_acc)
        server_model.global_model.load_state_dict(copy.deepcopy(global_params))
        print("\nRound {} FedSKC Accuracy: {}".format(round_id, fedskc_trained_acc))
    print("\n FedSKC: ", fedskc_trained_acc)
    print("\n FedSKC Top-1   Acc: ", max(fedskc_trained_acc))
    print("\n FedSKC Average Acc: ", np.mean(fedskc_trained_acc))

if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)  # cpu
    torch.cuda.manual_seed(args.seed)  # gpu
    np.random.seed(args.seed)  # numpy
    random.seed(args.seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    FedSKC_main()
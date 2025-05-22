import argparse
import os
from utils.param_aug import ParamDiffAug


def args_parser():
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)
    parser.add_argument('--data_name', type=str, default='cifar10')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--num_online_clients', type=int, default=8)
    parser.add_argument('--num_rounds', type=int, default=200)
    parser.add_argument('--server_lr', type=float, default=1.0)
    parser.add_argument('--num_epochs_local_training', type=int, default=10)
    parser.add_argument('--batch_size_local_training', type=int, default=64)
    parser.add_argument('--batch_size_test', type=int, default=100)
    parser.add_argument('--lr_local_training', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--non_iid_alpha', type=float, default=0.05) # non-iid alpha
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor') # long-tailed distribution factor
    parser.add_argument('--save_path', type=str, default=os.path.join(path_dir, 'result/'))
    # FedProx
    parser.add_argument('--mu', type=float, default=0.01)
    # FedAvgM
    parser.add_argument('--init_belta', type=float, default=0.97)
    
    args = parser.parse_args()
    args.dsa_param = ParamDiffAug()
    args.dsa = True

    return args

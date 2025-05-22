# FedSKC: Federated Learning with Non-IID Data via Structural Knowledge Collaboration

This is an official implementation of the following paper:
> Huan Wang, Haoran Li, Huaming Chen, Jun Yan, Lijuan Wang, Jiahua Shi, Shiping Chen, Jun Shen. *"FedSKC: Federated Learning with Non-IID Data via Structural Knowledge Collaboration"*. IEEE International Conference on Web Services (ICWS), ICWS 2025.
---

**Abstract:** With the advancement of edge computing, federated learning (FL) displays a bright promise as a privacy-preserving collaborative learning paradigm. However, one major challenge for FL is the data heterogeneity issue, which refers to the biased labeling preferences among multiple clients, negatively impacting convergence and model performance. Most previous FL methods attempt to tackle the data heterogeneity issue locally or globally, neglecting underlying class-wise structure information contained in each client. In this paper, we first study how data heterogeneity affects the divergence of the model and decompose it into local, global, and sampling drift sub-problems. To explore the potential of using intra-client class-wise structural knowledge in handling these drifts, we thus propose Federated Learning with Structural Knowledge Collaboration (FedSKC). The key idea of FedSKC is to extract and transfer domain preferences from inter-client data distributions, offering diverse class-relevant knowledge and a fair convergent signal. FedSKC comprises three components: i) local contrastive learning, to prevent weight divergence resulting from local training; ii) global discrepancy aggregation, which addresses the parameter deviation between the server and clients; iii) global period review, correcting for the sampling drift introduced by the server randomly selecting devices. We have theoretically analyzed FedSKC under non-convex objectives and empirically validated its superiority through extensive experimental results.

---

Here is an example to run FedSKC on CIFAR-10 with noniid_factor=0.05 & imb_factor=0.1:


```python
python3 main_fedskc.py --data_name cifar10 --num_classes 10 --non_iid_alpha 0.05 --imb_factor 0.1
```
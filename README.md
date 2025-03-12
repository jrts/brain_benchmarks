## Brain Network Analysis Benchmarks

### Implementations ([`./nets`](./nets))
|  Name              | Source |
| ------------------ | ------- |
| `bnt`              | [Brain Network Transformer](https://arxiv.org/abs/2210.06681) |
| `brain_gnn_net`    | [BrainGNN](https://www.sciencedirect.com/science/article/pii/S1361841521002784) |
| `brain_net_cnn`    | [BrainNetCNN](https://www.sciencedirect.com/science/article/abs/pii/S1053811916305237) |
| `contrastpool_net` | [ContrastPool](https://arxiv.org/abs/2307.11133) |
| `gat_net`          | [Graph Attention Networks](https://arxiv.org/abs/1710.10903) |
| `gated_gcn_net`    | [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493) |
| `gcn_net`          | [Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) |
| `gin_net`          | [Graph Isomorphism Networks](https://arxiv.org/abs/1810.00826) |
| `graphsage_net`    | [GraphSAGE](https://arxiv.org/abs/1706.02216) |
| `gcn_net`          | [Pooling Regularized Graph Neural Network for fMRI Biomarker Analysis](https://arxiv.org/abs/2007.14589) |
| `prgnn_net`        | [Proximity Relational Graph Neural Network](https://www.sciencedirect.com/science/article/abs/pii/S0925231224006283) |

### Environment
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install pyg_lib torch_scatter==2.1.1 torch_sparse==0.6.16 torch_cluster==1.6.1 torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu116.html
pip install torch-geometric tensorboard scikit-learn
pip install dgl-cu116 -f https://data.dgl.ai/wheels/repo.html
```

### Credit
Benchmarking Graph Neural Networks [[paper](https://arxiv.org/abs/2003.00982)] [[Github](https://github.com/graphdeeplearning/benchmarking-gnns)]

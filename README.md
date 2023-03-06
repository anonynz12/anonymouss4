## Enhancing Graph Neural Networks via Memorized Global Information

This repository contains the PyTorch implementation for our MGIGNN. Further details about MGIGNN can be found in our paper.

## Abstract

Most Graph neural networks (GNNs) merely leverage information in a limited range of local neighbors, which is not able to capture long-range dependencies or global information in the graph. However, extending information to the whole graph provides GNNs with more information from similar nodes in large scale neighbors, which enables GNNs to learn more informative node representations. To this end, we propose MGIGNN (Memorized Global Information Graph Neural Network), an approach that leverages global information from global similar nodes to enhance GNNs. Unfortunately, finding global similar nodes requires calculating representations of all nodes in the graph using GNNs, which is computationally expensive. To circumvent the heavy burden of computing all node representations, MGIGNN uses an external memory module to store node representations and utilizes those representations from memory to efficiently find global similar nodes. Moreover, to efficiently make predictions at test time, MGIGNN retrieves global similar nodes from a set of candidate nodes, which are selected from a sparse node selection distribution with Dirichlet prior. Experimental results on seven real-world datasets show that our MGIGNN can improve the effectiveness of existing GNNs on node classification task under both inductive and transductive settings.

## Model Framework Overview

<p align="center">
  <img src="./pic/Model_Framework.png" alt="Model_Framework" width="500"/>
</p>


## Installation

```shell
pip install -r requirements.txt
```

## Implement Details

MGIGNN can be equipped with various GNN models, now MGIGNN supports the following GNN models:

- GCN
- GraphSAGE
- GAT
- SGC
- DNA
- GCNII
- FILM
- SSGC
- FAGCN
- GATv2Conv

In the future, we aim at enabling MGIGNN to support more GNN models.

## Data Download (Update: 2023.3.6)

Please first download the dataset and unzip it into `data` directory.

Google Drive Link: https://drive.google.com/file/d/1BkIIoQsowSaFOY0TLFbVYkIlgVgnstQP/view?usp=share_link

## MGIGNN Training 

- For PPI dataset, please use the following command:

```bash
python train_ppi.py --cuda_id 0 --model [gcn/graphsage/gat/sgc/dna/gcnii/film/ssgc/fagcn/gatv2] --criterion sigmoid --hidden_dim 16 --log_dir ./your_log --k 3 --eta 1 --val_test_batch_size 2 --epochs 1 --run_times 5 --normalize True --gnn_lr 0.01 --vae_lr 0.00001 --sim_function feature_base
```

- For DBLP dataset, please use the following command:

```bash
python train_dblp.py --cuda_id 0 --model [gcn/graphsage/gat/sgc/dna/gcnii/film/ssgc/fagcn/gatv2] --criterion softmax --hidden_dim 16 --log_dir ./your_log --k 3 --eta 1 --val_test_batch_size 2 --epochs 1 --run_times 5 --normalize True --gnn_lr 0.01 --vae_lr 0.00001 --sim_function feature_base
```

- For Photo dataset, please use the following command:

```bash
python train_photo.py --cuda_id 0 --model [gcn/graphsage/gat/sgc/dna/gcnii/film/ssgc/fagcn/gatv2] --criterion softmax --hidden_dim 16 --log_dir ./your_log --k 3 --eta 1 --val_test_batch_size 2 --epochs 1 --run_times 5 --normalize True --gnn_lr 0.01 --vae_lr 0.00001 --sim_function feature_base
```

- For Computer dataset, please use the following command:

```bash
python train_computer.py --cuda_id 4 --model [gcn/graphsage/gat/sgc/dna/gcnii/film/ssgc/fagcn/gatv2] --criterion softmax --hidden_dim 16 --log_dir ./your_log --k 3 --eta 1 --val_test_batch_size 2 --epochs 1 --run_times 5 --normalize True --gnn_lr 0.01 --vae_lr 0.00001 --sim_function feature_base
```


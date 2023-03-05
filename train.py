#!/usr/bin/env python
# -*- coding:utf8 -*-

import argparse
import copy
import os.path
import sys

import torch
import time
from utils import convert_to_one_hot_label, CitationDataset
from sklearn import metrics
from torch.optim import Adam
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from model import MGIGNN
import pickle
from torch.nn import functional as F
import torch_geometric.transforms as T
from tqdm import tqdm
import random
import numpy as np

random_seed = 1024
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)


def train(run_time, args):

    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    device = torch.device("cuda:{}".format(str(args.cuda_id)) if torch.cuda.is_available()
                          and args.cuda else 'cpu')
    print(device)



    if args.dataset == 'ppi':

        if args.model == 'gcnii':
            path = "./data/GCN2_PPI"
            pre_transform = T.Compose([T.GCNNorm(), T.ToSparseTensor()])
            train_dataset = PPI(path, split='train',
                                pre_transform=pre_transform)
        else:
            train_dataset = PPI("./data/PPI", split='train')
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        non_hop_adjacency_matrix_dict_path = os.path.join("./data/PPI/non_hop_2_adjacency_matrix_dict_val_batch_size_2.pkl")
        train_node_id_dict_path = os.path.join("./data/PPI/train_node_id_dict.pkl")

    with open(non_hop_adjacency_matrix_dict_path, 'rb') as f:
        non_hop_adjacency_matrix_dict = pickle.load(f)

    with open(train_node_id_dict_path, 'rb') as f:
        id_dict = pickle.load(f)

    num_nodes = sum(id_dict.keys())

    if args.criterion == 'sigmoid':
        criterion = torch.nn.BCELoss()
    elif args.criterion == 'softmax':
        criterion = torch.nn.CrossEntropyLoss()

    # Define the model
    model = MGIGNN(
        input_dim=train_dataset.num_features,
        hidden_dim=args.hidden_dim,
        num_classes=train_dataset.num_classes,
        nodes_numbers=num_nodes,
        normalize=True,
        k=args.k,
        eta=args.eta,
        device=device,
        model_name=args.model,
        criterion=args.criterion).to(device)

    embedding_optimizer = Adam(model.embedding_encoder.parameters(), lr=args.gnn_lr)
    nodevae_optimizer = Adam(list(model.node_vae.parameters()), lr=args.vae_lr)

    # Pretrain
    for epoch in range(50):
        model.train()
        pbar = tqdm(total=int(len(train_loader.dataset)))
        pbar.set_description(f'Epoch {epoch:04d}')
        for data in train_loader:
            embedding_optimizer.zero_grad()
            data = data.to(device)
            output = model.pretrain_forward(data)
            loss = criterion(output, data.y)
            loss.backward()
            embedding_optimizer.step()
            pbar.update()

    for epoch in range(args.epochs):
        model.train()

        pbar = tqdm(total=int(len(train_loader.dataset)))
        pbar.set_description(f'Epoch {epoch:04d}')
        total_loss = 0.0

        for batch in train_loader:
            batch_size = batch.x.shape[0]
            # E Step
            model.embedding_encoder.requires_grad = False
            batch = batch.to(device)
            nodevae_optimizer.zero_grad()
            y = batch.y
            train_info = model.get_training_embedding_memory(
                batch,
                id_dict=id_dict,
                non_hop_2_adjacency_matrix_dict=non_hop_adjacency_matrix_dict,
                neighbors_info=None
            )
            sim_info = model.get_sim_info(
                train_info['train_embedding'],
                train_info['train_label'],
                train_info['train_embedding'],
                train_info['train_label'],
                train_info['train_batch_id'].to(device),
                train_info['mask'])
            output = model.reconstruct_z_q(train_info, sim_info)
            output = output.float()
            bce_loss = criterion(output, y)
            loss_e = bce_loss + 0.01 * model.node_vae.compute_KL() + 0.01 * model.compute_kl_theta()
            total_loss += loss_e.item()
            loss_e.backward(retain_graph=True)
            for param in nodevae_optimizer.param_groups[0]['params']:
                """
                torch.nn.utils.clip_grad_value_(parameters, clip_value)
                """
                torch.nn.utils.clip_grad_value_(param, 1)
            nodevae_optimizer.step()

            model.update_lambda_stats_sum(sim_info)
            model.distributed_update_lambda(
                lambda_stats_sum=model.lambda_stats_sum,
                num_of_nodes=model.batch_size,
                update_num=model._num_updates)

            model._num_updates += 1

            model.embedding_encoder.requires_grad = True
            model.node_vae.requires_grad = False
            embedding_optimizer.zero_grad()
            output, _ = model.m_step_forward(batch,
                                             non_hop_adjacency_matrix_dict[batch_size])
            output = output.float()
            loss_m = criterion(output, y)
            total_loss += loss_m.item()
            loss_m.backward()
            for param in embedding_optimizer.param_groups[0]['params']:
                """
                torch.nn.utils.clip_grad_value_(parameters, clip_value)
                """
                torch.nn.utils.clip_grad_value_(param, 1)
            embedding_optimizer.step()

            pbar.update()

    print("Optimization Finished!")

    torch.save(model.state_dict(), os.path.join(log_dir, "{}.pt".format(run_time)))


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true',
                        default=True, help='Use CUDA or not')
    parser.add_argument('--cuda_id', type=int, required=True,
                        help='Use CUDA or not')
    parser.add_argument('--epochs', type=int,
                        required=True, help='Number of epochs to train')
    parser.add_argument('--gnn_lr', type=float, default=0.005, help='GNN learning rate')
    parser.add_argument('--vae_lr', type=float, default=1e-5, help='VAE learning rate')
    parser.add_argument('--hidden_dim', type=int,
                        required=True,
                        help='Number of hidden units.')
    parser.add_argument('--patience', type=int, default=100,
                        help='Patience')
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--model', required=True,
                        help='training_model')
    parser.add_argument('--eta', type=float, required=True,
                        help='trade-off parameters')
    parser.add_argument('--k', type=int, required=True,
                        help='Number of global similar nodes')
    parser.add_argument('--criterion', type=str, default='sigmoid',
                        help='softmax or sigmoid')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--run_times', type=int, default=1)
    parser.add_argument('--log_dir', type=str, required=True)

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    print(args)
    run_times = args.run_times

    for run_time in range(run_times):
        print("|> Run time is ", run_time)
        train(run_time, args)
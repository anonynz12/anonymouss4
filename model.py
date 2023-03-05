#!/usr/bin/env python
# -*- coding:utf8 -*-

import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, SGConv, GCN2Conv, GATv2Conv
from torch_geometric.nn import FiLMConv
from torch_geometric.nn import DNAConv
from torch_geometric.nn import SSGConv
from torch_geometric.nn import FAConv
from torch_geometric.nn import MLP, GINConv, global_add_pool, global_mean_pool
from utils import convert_to_one_hot_label, sim_matrix
from torch.nn import BatchNorm1d
import math
from tqdm import tqdm

class GraphSAGE_Encoder_SubGraph(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_classes,
                 criterion,
                 k,
                 eta,
                 device):
        super().__init__()
        self.layer_1 = SAGEConv(input_dim, hidden_dim)
        self.layer_2 = SAGEConv(hidden_dim, num_classes)
        self.criterion = criterion
        self.k = k
        self.eta = eta
        if criterion == 'sigmoid':
            self.m = nn.Sigmoid()
        elif criterion == 'softmax':
            self.m = nn.LogSoftmax(dim=1)
        self.num_classes = num_classes
        self.device = device

    def pretrain_with_batch(self, data):

        x, edge_index = data.x, data.edge_index
        x = F.elu(self.layer_1(x, edge_index))
        # x = x.relu_()
        # x = F.dropout(x, p=0.5, training=self.training)
        lc_x = F.elu(self.layer_2(x, edge_index))
        p_lc = self.m(lc_x)

        return p_lc

    @torch.no_grad()
    def get_embedding(self, data):
        dic = {}
        dic['full_data'] = data
        x, edge_index = data.x, data.edge_index
        y = data.y

        x = F.elu(self.layer_1(x, edge_index))
        # x = x.relu_()
        embedding = x

        dic['train_embedding'] = embedding
        dic['train_label'] = y

        return dic

    @torch.no_grad()
    def decode_z_q(self, train_info):

        # print(train_info)

        data = train_info['full_data']
        x, edge_index = data.x, data.edge_index

        y = data.y

        x = F.elu(self.layer_1(x, edge_index))
        # x = x.relu_()
        # x = F.dropout(x, p=0.5, training=self.training)
        lc_x = F.elu(self.layer_2(x, edge_index))
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(train_info['z_q'],
                                                              train_info['z_q'],
                                                              y,
                                                              train_info['mask'],
                                                              train_info['neighbors_info']))
        # print(p_sim)

        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution

    def forward(self,
                data,
                mask,
                neighbors_info=None):
        x, edge_index = data.x, data.edge_index

        y = data.y

        x = F.elu(self.layer_1(x, edge_index))
        # x = x.relu_()
        # x = F.dropout(x, p=0.5, training=self.training)
        embedding = x

        lc_x = F.elu(self.layer_2(x, edge_index))
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                              embedding,
                                                              y,
                                                              mask,
                                                              neighbors_info))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution, embedding

    def generate_label_with_similar_nodes(self,
                                          embedding_a,
                                          embedding_b,
                                          b_y,
                                          mask,
                                          neighbors_info=None):

        with torch.no_grad():
            if torch.is_tensor(neighbors_info):
                # print(neighbors_info)
                # print("|> [generate_label_with_similar_nodes] Using neighbors information")
                feature_similarity_matrix = neighbors_info
            else:
                # print("|> [generate_label_with_similar_nodes] Not Using neighbors information")
                feature_similarity_matrix = sim_matrix(embedding_a, embedding_b)
            feature_similarity_matrix = feature_similarity_matrix.cpu()
            if self.training:
                feature_similarity_matrix = feature_similarity_matrix * mask
            topk_similar_nodes_information = torch.topk(feature_similarity_matrix, self.k)
        fetch_similarity = torch.exp(topk_similar_nodes_information.values)
        fetch_similarity = fetch_similarity.to(self.device)
        fetch_label = []
        if self.criterion == 'sigmoid':
            one_hot_label = b_y
        elif self.criterion == 'softmax':
            one_hot_label = convert_to_one_hot_label(
                labels=b_y,
                num_classes=self.num_classes,
                batch_size=b_y.shape[0],
                device=self.device
            )
            one_hot_label = one_hot_label.to(self.device)

        for i in range(topk_similar_nodes_information.indices.shape[0]):
            fetch_label.append(
                torch.index_select(one_hot_label, 0, topk_similar_nodes_information.indices[i].to(self.device)))
        fetch_label = torch.stack(fetch_label)
        node_label_distribution = fetch_similarity.unsqueeze(-1) * fetch_label.to(self.device)
        total_fuse_distribution = node_label_distribution.sum(dim=1)

        return total_fuse_distribution

    def inference(self,
                  test_data,
                  training_embedding,
                  training_y):

        x, edge_index = test_data.x, test_data.edge_index

        x = F.elu(self.layer_1(x, edge_index))
        # x = x.relu_()
        _data_embedding = x
        lc_x = F.elu(self.layer_2(x, edge_index))
        p_lc = self.m(lc_x)

        p_sim = self.m(
            self.generate_label_with_similar_nodes(_data_embedding,
                                                   training_embedding,
                                                   training_y,
                                                   mask=None,
                                                   neighbors_info=None))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim
        return final_label_distribution


class GAT_Encoder_SubGraph(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_classes,
                 criterion,
                 k,
                 eta,
                 device):
        super().__init__()
        self.layer_1 = GATConv(input_dim, out_channels=hidden_dim, heads=4)
        self.layer_2 = GATConv(4 * hidden_dim, hidden_dim, heads=4)
        self.layer_3 = GATConv(4 * hidden_dim, num_classes, heads=6, concat=False)

        self.criterion = criterion
        self.k = k
        self.eta = eta
        if criterion == 'sigmoid':
            self.m = nn.Sigmoid()
        elif criterion == 'softmax':
            self.m = nn.LogSoftmax(dim=1)
        self.num_classes = num_classes
        self.device = device

    def pretrain_with_batch(self, data):
        """
        预训练环节
        :param data:
        :return:
        """
        x, edge_index = data.x, data.edge_index
        # print(self.layer_1(x, edge_index))

        x = F.elu(self.layer_1(x, edge_index))
        x = F.elu(self.layer_2(x, edge_index))
        lc_x = self.layer_3(x, edge_index)

        p_lc = self.m(lc_x)

        return p_lc

    @torch.no_grad()
    def get_embedding(self, data):
        dic = {}
        dic['full_data'] = data
        x, edge_index = data.x, data.edge_index
        y = data.y

        x = F.elu(self.layer_1(x, edge_index))
        x = F.elu(self.layer_2(x, edge_index))
        embedding = x

        dic['train_embedding'] = embedding
        dic['train_label'] = y

        return dic

    @torch.no_grad()
    def decode_z_q(self, train_info):

        data = train_info['full_data']
        x, edge_index = data.x, data.edge_index

        y = data.y

        x = F.elu(self.layer_1(x, edge_index))
        x = F.elu(self.layer_2(x, edge_index))

        lc_x = self.layer_3(x, edge_index)
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(train_info['z_q'],
                                                              train_info['z_q'],
                                                              y,
                                                              train_info['mask'],
                                                              train_info['neighbors_info']))
        # print(p_sim)

        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution

    def forward(self,
                data,
                mask,
                neighbors_info=None):
        x, edge_index = data.x, data.edge_index

        y = data.y

        x = F.elu(self.layer_1(x, edge_index))
        x = F.elu(self.layer_2(x, edge_index))
        embedding = x

        lc_x = self.layer_3(x, edge_index)
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                              embedding,
                                                              y,
                                                              mask,
                                                              neighbors_info))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution, embedding

    def generate_label_with_similar_nodes(self,
                                          embedding_a,
                                          embedding_b,
                                          b_y,
                                          mask,
                                          neighbors_info=None):

        with torch.no_grad():
            if torch.is_tensor(neighbors_info):
                feature_similarity_matrix = neighbors_info
            else:
                feature_similarity_matrix = sim_matrix(embedding_a, embedding_b)
            feature_similarity_matrix = feature_similarity_matrix.cpu()
            if self.training:
                feature_similarity_matrix = feature_similarity_matrix * mask
            topk_similar_nodes_information = torch.topk(feature_similarity_matrix, self.k)
        fetch_similarity = torch.exp(topk_similar_nodes_information.values)
        fetch_similarity = fetch_similarity.to(self.device)
        fetch_label = []
        if self.criterion == 'sigmoid':
            one_hot_label = b_y
        elif self.criterion == 'softmax':
            one_hot_label = convert_to_one_hot_label(
                labels=b_y,
                num_classes=self.num_classes,
                batch_size=b_y.shape[0],
                device=self.device
            )
            one_hot_label = one_hot_label.to(self.device)

        for i in range(topk_similar_nodes_information.indices.shape[0]):
            fetch_label.append(
                torch.index_select(one_hot_label, 0, topk_similar_nodes_information.indices[i].to(self.device)))
        fetch_label = torch.stack(fetch_label)
        node_label_distribution = fetch_similarity.unsqueeze(-1) * fetch_label.to(self.device)
        total_fuse_distribution = node_label_distribution.sum(dim=1)

        return total_fuse_distribution

    def inference(self,
                  test_data,
                  training_embedding,
                  training_y):

        x, edge_index = test_data.x, test_data.edge_index

        x = F.elu(self.layer_1(x, edge_index))
        x = F.elu(self.layer_2(x, edge_index))
        _data_embedding = x
        lc_x = self.layer_3(x, edge_index)
        p_lc = self.m(lc_x)


        p_sim = self.m(
            self.generate_label_with_similar_nodes(_data_embedding,
                                                   training_embedding,
                                                   training_y,
                                                   mask=None,
                                                   neighbors_info=None))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim
        return final_label_distribution


class GATv2Conv_Encoder_SubGraph(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_classes,
                 criterion,
                 k,
                 eta,
                 device):
        super().__init__()
        self.layer_1 = GATv2Conv(input_dim, out_channels=hidden_dim, heads=4)
        self.layer_2 = GATv2Conv(4 * hidden_dim, hidden_dim, heads=4)
        self.layer_3 = GATv2Conv(4 * hidden_dim, num_classes, heads=6, concat=False)

        self.criterion = criterion
        self.k = k
        self.eta = eta
        if criterion == 'sigmoid':
            self.m = nn.Sigmoid()
        elif criterion == 'softmax':
            self.m = nn.LogSoftmax(dim=1)
        self.num_classes = num_classes
        self.device = device

    def pretrain_with_batch(self, data):
        """
        预训练环节
        :param data:
        :return:
        """
        x, edge_index = data.x, data.edge_index
        # print(self.layer_1(x, edge_index))

        x = F.elu(self.layer_1(x, edge_index))
        x = F.elu(self.layer_2(x, edge_index))
        lc_x = self.layer_3(x, edge_index)

        p_lc = self.m(lc_x)

        return p_lc

    @torch.no_grad()
    def get_embedding(self, data):
        dic = {}
        dic['full_data'] = data
        x, edge_index = data.x, data.edge_index
        y = data.y

        x = F.elu(self.layer_1(x, edge_index))
        x = F.elu(self.layer_2(x, edge_index))
        embedding = x

        dic['train_embedding'] = embedding
        dic['train_label'] = y

        return dic

    @torch.no_grad()
    def decode_z_q(self, train_info):

        data = train_info['full_data']
        x, edge_index = data.x, data.edge_index

        y = data.y

        x = F.elu(self.layer_1(x, edge_index))
        x = F.elu(self.layer_2(x, edge_index))

        lc_x = self.layer_3(x, edge_index)
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(train_info['z_q'],
                                                              train_info['z_q'],
                                                              y,
                                                              train_info['mask'],
                                                              train_info['neighbors_info']))
        # print(p_sim)

        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution

    def forward(self,
                data,
                mask,
                neighbors_info=None):
        x, edge_index = data.x, data.edge_index

        y = data.y

        x = F.elu(self.layer_1(x, edge_index))
        x = F.elu(self.layer_2(x, edge_index))
        embedding = x

        lc_x = self.layer_3(x, edge_index)
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                              embedding,
                                                              y,
                                                              mask,
                                                              neighbors_info))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution, embedding

    def generate_label_with_similar_nodes(self,
                                          embedding_a,
                                          embedding_b,
                                          b_y,
                                          mask,
                                          neighbors_info=None):

        with torch.no_grad():
            if torch.is_tensor(neighbors_info):
                feature_similarity_matrix = neighbors_info
            else:
                feature_similarity_matrix = sim_matrix(embedding_a, embedding_b)
            feature_similarity_matrix = feature_similarity_matrix.cpu()
            if self.training:
                feature_similarity_matrix = feature_similarity_matrix * mask
            topk_similar_nodes_information = torch.topk(feature_similarity_matrix, self.k)
        fetch_similarity = torch.exp(topk_similar_nodes_information.values)
        fetch_similarity = fetch_similarity.to(self.device)
        fetch_label = []
        if self.criterion == 'sigmoid':
            one_hot_label = b_y
        elif self.criterion == 'softmax':
            one_hot_label = convert_to_one_hot_label(
                labels=b_y,
                num_classes=self.num_classes,
                batch_size=b_y.shape[0],
                device=self.device
            )
            one_hot_label = one_hot_label.to(self.device)

        for i in range(topk_similar_nodes_information.indices.shape[0]):
            fetch_label.append(
                torch.index_select(one_hot_label, 0, topk_similar_nodes_information.indices[i].to(self.device)))
        fetch_label = torch.stack(fetch_label)
        node_label_distribution = fetch_similarity.unsqueeze(-1) * fetch_label.to(self.device)
        total_fuse_distribution = node_label_distribution.sum(dim=1)

        return total_fuse_distribution

    def inference(self,
                  test_data,
                  training_embedding,
                  training_y):

        x, edge_index = test_data.x, test_data.edge_index

        x = F.elu(self.layer_1(x, edge_index))
        x = F.elu(self.layer_2(x, edge_index))
        _data_embedding = x
        lc_x = self.layer_3(x, edge_index)
        p_lc = self.m(lc_x)


        p_sim = self.m(
            self.generate_label_with_similar_nodes(_data_embedding,
                                                   training_embedding,
                                                   training_y,
                                                   mask=None,
                                                   neighbors_info=None))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim
        return final_label_distribution


class GCN_Encoder_SubGraph(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_classes,
                 criterion,
                 k,
                 eta,
                 device):
        super().__init__()
        self.layer_1 = GCNConv(input_dim, hidden_dim)
        self.layer_2 = GCNConv(hidden_dim, num_classes)

        self.criterion = criterion
        self.k = k
        self.eta = eta
        if criterion == 'sigmoid':
            self.m = nn.Sigmoid()
        elif criterion == 'softmax':
            self.m = nn.LogSoftmax(dim=1)
        self.num_classes = num_classes
        self.device = device

    def pretrain_with_batch(self, data):
        """
        预训练环节
        :param data:
        :return:
        """
        x, edge_index = data.x, data.edge_index

        x = self.layer_1(x, edge_index)
        x = F.relu(x)
        lc_x = self.layer_2(x, edge_index)

        p_lc = self.m(lc_x)

        return p_lc

    @torch.no_grad()
    def get_embedding(self, data):
        dic = {}
        dic['full_data'] = data
        x, edge_index = data.x, data.edge_index
        y = data.y

        x = self.layer_1(x, edge_index)
        x = F.relu(x)
        embedding = x

        dic['train_embedding'] = embedding
        dic['train_label'] = y

        return dic

    @torch.no_grad()
    def decode_z_q(self, train_info):

        data = train_info['full_data']
        x, edge_index = data.x, data.edge_index

        y = data.y

        x = self.layer_1(x, edge_index)
        x = F.relu(x)

        lc_x = self.layer_2(x, edge_index)
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(train_info['z_q'],
                                                              train_info['z_q'],
                                                              y,
                                                              train_info['mask'],
                                                              train_info['neighbors_info']))
        # print(p_sim)

        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution

    def forward(self,
                data,
                mask,
                neighbors_info=None):
        x, edge_index = data.x, data.edge_index

        y = data.y

        x = self.layer_1(x, edge_index)
        x = F.relu(x)
        embedding = x

        lc_x = self.layer_2(x, edge_index)
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                              embedding,
                                                              y,
                                                              mask,
                                                              neighbors_info))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution, embedding

    def generate_label_with_similar_nodes(self,
                                          embedding_a,
                                          embedding_b,
                                          b_y,
                                          mask,
                                          neighbors_info=None):

        with torch.no_grad():
            if torch.is_tensor(neighbors_info):
                feature_similarity_matrix = neighbors_info
            else:
                feature_similarity_matrix = sim_matrix(embedding_a, embedding_b)
            feature_similarity_matrix = feature_similarity_matrix.cpu()
            if self.training:
                feature_similarity_matrix = feature_similarity_matrix * mask
            topk_similar_nodes_information = torch.topk(feature_similarity_matrix, self.k)
        fetch_similarity = torch.exp(topk_similar_nodes_information.values)
        fetch_similarity = fetch_similarity.to(self.device)
        fetch_label = []
        if self.criterion == 'sigmoid':
            one_hot_label = b_y
        elif self.criterion == 'softmax':
            one_hot_label = convert_to_one_hot_label(
                labels=b_y,
                num_classes=self.num_classes,
                batch_size=b_y.shape[0],
                device=self.device
            )
            one_hot_label = one_hot_label.to(self.device)

        for i in range(topk_similar_nodes_information.indices.shape[0]):
            fetch_label.append(
                torch.index_select(one_hot_label, 0, topk_similar_nodes_information.indices[i].to(self.device)))
        fetch_label = torch.stack(fetch_label)
        node_label_distribution = fetch_similarity.unsqueeze(-1) * fetch_label.to(self.device)
        total_fuse_distribution = node_label_distribution.sum(dim=1)

        return total_fuse_distribution

    def inference(self,
                  test_data,
                  training_embedding,
                  training_y):

        x, edge_index = test_data.x, test_data.edge_index

        x = self.layer_1(x, edge_index)
        x = F.relu(x)
        _data_embedding = x
        lc_x = self.layer_2(x, edge_index)
        p_lc = self.m(lc_x)


        p_sim = self.m(
            self.generate_label_with_similar_nodes(_data_embedding,
                                                   training_embedding,
                                                   training_y,
                                                   mask=None,
                                                   neighbors_info=None))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim
        return final_label_distribution


class SGC_Encoder_SubGraph(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_classes,
                 criterion,
                 k,
                 eta,
                 device):
        super().__init__()
        self.layer_1 = SGConv(input_dim, hidden_dim, K=2)
        self.layer_2 = SGConv(hidden_dim, num_classes, K=2)

        self.criterion = criterion
        self.k = k
        self.eta = eta
        if criterion == 'sigmoid':
            self.m = nn.Sigmoid()
        elif criterion == 'softmax':
            self.m = nn.LogSoftmax(dim=1)
        self.num_classes = num_classes
        self.device = device

    def pretrain_with_batch(self, data):
        """
        预训练环节
        :param data:
        :return:
        """
        x, edge_index = data.x, data.edge_index

        x = self.layer_1(x, edge_index)
        lc_x = self.layer_2(x, edge_index)

        p_lc = self.m(lc_x)

        return p_lc

    @torch.no_grad()
    def get_embedding(self, data):
        dic = {}
        dic['full_data'] = data
        x, edge_index = data.x, data.edge_index
        y = data.y

        x = self.layer_1(x, edge_index)
        embedding = x

        dic['train_embedding'] = embedding
        dic['train_label'] = y

        return dic

    @torch.no_grad()
    def decode_z_q(self, train_info):

        data = train_info['full_data']
        x, edge_index = data.x, data.edge_index

        y = data.y

        x = self.layer_1(x, edge_index)

        lc_x = self.layer_2(x, edge_index)
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(train_info['z_q'],
                                                              train_info['z_q'],
                                                              y,
                                                              train_info['mask'],
                                                              train_info['neighbors_info']))
        # print(p_sim)

        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution

    def forward(self,
                data,
                mask,
                neighbors_info=None):
        x, edge_index = data.x, data.edge_index

        y = data.y

        x = self.layer_1(x, edge_index)
        embedding = x

        lc_x = self.layer_2(x, edge_index)
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                              embedding,
                                                              y,
                                                              mask,
                                                              neighbors_info))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution, embedding

    def generate_label_with_similar_nodes(self,
                                          embedding_a,
                                          embedding_b,
                                          b_y,
                                          mask,
                                          neighbors_info=None):

        with torch.no_grad():
            if torch.is_tensor(neighbors_info):
                feature_similarity_matrix = neighbors_info
            else:
                feature_similarity_matrix = sim_matrix(embedding_a, embedding_b)
            feature_similarity_matrix = feature_similarity_matrix.cpu()
            if self.training:
                feature_similarity_matrix = feature_similarity_matrix * mask
            topk_similar_nodes_information = torch.topk(feature_similarity_matrix, self.k)
        fetch_similarity = torch.exp(topk_similar_nodes_information.values)
        fetch_similarity = fetch_similarity.to(self.device)
        fetch_label = []
        if self.criterion == 'sigmoid':
            one_hot_label = b_y
        elif self.criterion == 'softmax':
            one_hot_label = convert_to_one_hot_label(
                labels=b_y,
                num_classes=self.num_classes,
                batch_size=b_y.shape[0],
                device=self.device
            )
            one_hot_label = one_hot_label.to(self.device)

        for i in range(topk_similar_nodes_information.indices.shape[0]):
            fetch_label.append(
                torch.index_select(one_hot_label, 0, topk_similar_nodes_information.indices[i].to(self.device)))
        fetch_label = torch.stack(fetch_label)
        node_label_distribution = fetch_similarity.unsqueeze(-1) * fetch_label.to(self.device)
        total_fuse_distribution = node_label_distribution.sum(dim=1)

        return total_fuse_distribution

    def inference(self,
                  test_data,
                  training_embedding,
                  training_y):

        x, edge_index = test_data.x, test_data.edge_index

        x = self.layer_1(x, edge_index)
        _data_embedding = x
        lc_x = self.layer_2(x, edge_index)
        p_lc = self.m(lc_x)


        p_sim = self.m(
            self.generate_label_with_similar_nodes(_data_embedding,
                                                   training_embedding,
                                                   training_y,
                                                   mask=None,
                                                   neighbors_info=None))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim
        return final_label_distribution


class GCNII_Encoder_SubGraph(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_classes,
                 criterion,
                 k,
                 eta,
                 device,
                 n_layers=9,
                 alpha=0.5,
                 theta=1.0,
                 shared_weights=False,
                 dropout=0.5):
        super().__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(input_dim, hidden_dim))
        self.lins.append(nn.Linear(hidden_dim, num_classes))
        self.convs = nn.ModuleList()
        for layer in range(n_layers):
            self.convs.append(
                GCN2Conv(hidden_dim, alpha, theta,
                         layer + 1, shared_weights, normalize=False))
        self.dropout = dropout

        self.criterion = criterion
        self.k = k
        self.eta = eta
        if criterion == 'sigmoid':
            self.m = nn.Sigmoid()
        elif criterion == 'softmax':
            self.m = nn.LogSoftmax(dim=1)
        self.num_classes = num_classes
        self.device = device

    def pretrain_with_batch(self, data):
        """
        预训练环节
        :param data:
        :return:
        """
        x, adj_t = data.x, data.adj_t
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            h = F.dropout(x, self.dropout, training=self.training)
            h = conv(h, x_0, adj_t)
            x = h + x
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)  # Embedding
        lc_x = self.lins[1](x)

        p_lc = self.m(lc_x)

        return p_lc

    @torch.no_grad()
    def get_embedding(self, data):
        dic = {}
        dic['full_data'] = data

        x, adj_t = data.x, data.adj_t
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        y = data.y

        for conv in self.convs:
            h = F.dropout(x, self.dropout, training=False)
            h = conv(h, x_0, adj_t)
            x = h + x
            x = x.relu()

        embedding = x

        dic['train_embedding'] = embedding
        dic['train_label'] = y

        return dic

    @torch.no_grad()
    def decode_z_q(self, train_info):

        data = train_info['full_data']
        x, adj_t = data.x, data.adj_t
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        y = data.y

        for conv in self.convs:
            h = F.dropout(x, self.dropout, training=self.training)
            h = conv(h, x_0, adj_t)
            x = h + x
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)  # Embedding
        lc_x = self.lins[1](x)

        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(train_info['z_q'],
                                                              train_info['z_q'],
                                                              y,
                                                              train_info['mask'],
                                                              train_info['neighbors_info']))
        # print(p_sim)

        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution

    def forward(self,
                data,
                mask,
                neighbors_info=None):

        y = data.y

        x, adj_t = data.x, data.adj_t
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        for conv in self.convs:
            h = F.dropout(x, self.dropout, training=self.training)
            h = conv(h, x_0, adj_t)
            x = h + x
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)  # Embedding
        embedding = x
        lc_x = self.lins[1](x)

        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                              embedding,
                                                              y,
                                                              mask,
                                                              neighbors_info))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution, embedding

    def generate_label_with_similar_nodes(self,
                                          embedding_a,
                                          embedding_b,
                                          b_y,
                                          mask,
                                          neighbors_info=None):

        with torch.no_grad():
            if torch.is_tensor(neighbors_info):
                feature_similarity_matrix = neighbors_info
            else:
                feature_similarity_matrix = sim_matrix(embedding_a, embedding_b)
            feature_similarity_matrix = feature_similarity_matrix.cpu()
            if self.training:
                feature_similarity_matrix = feature_similarity_matrix * mask
            topk_similar_nodes_information = torch.topk(feature_similarity_matrix, self.k)
        fetch_similarity = torch.exp(topk_similar_nodes_information.values)
        fetch_similarity = fetch_similarity.to(self.device)
        fetch_label = []
        if self.criterion == 'sigmoid':
            one_hot_label = b_y
        elif self.criterion == 'softmax':
            one_hot_label = convert_to_one_hot_label(
                labels=b_y,
                num_classes=self.num_classes,
                batch_size=b_y.shape[0],
                device=self.device
            )
            one_hot_label = one_hot_label.to(self.device)

        for i in range(topk_similar_nodes_information.indices.shape[0]):
            fetch_label.append(
                torch.index_select(one_hot_label, 0, topk_similar_nodes_information.indices[i].to(self.device)))
        fetch_label = torch.stack(fetch_label)
        node_label_distribution = fetch_similarity.unsqueeze(-1) * fetch_label.to(self.device)
        total_fuse_distribution = node_label_distribution.sum(dim=1)

        return total_fuse_distribution

    def inference(self,
                  test_data,
                  training_embedding,
                  training_y):

        x, adj_t = test_data.x, test_data.adj_t
        x = F.dropout(x, self.dropout, training=False)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            h = F.dropout(x, self.dropout, training=False)
            h = conv(h, x_0, adj_t)
            x = h + x
            x = x.relu()

        _data_embedding = F.dropout(x, self.dropout, training=False)

        lc_x = self.lins[1](_data_embedding)
        p_lc = self.m(lc_x)

        p_sim = self.m(
            self.generate_label_with_similar_nodes(_data_embedding,
                                                   training_embedding,
                                                   training_y,
                                                   mask=None,
                                                   neighbors_info=None))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution

class FILM_Encoder_SubGraph(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes,
                 criterion,
                 k,
                 eta,
                 device,
                 hidden_dim=320,
                 dropout=0.1,
                 n_layers=4,):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(FiLMConv(input_dim, hidden_dim))
        for _ in range(n_layers - 2):
            self.convs.append(FiLMConv(hidden_dim, hidden_dim))
        self.convs.append(FiLMConv(hidden_dim, num_classes, act=None))

        self.norms = torch.nn.ModuleList()
        for _ in range(n_layers - 1):
            self.norms.append(BatchNorm1d(hidden_dim))

        self.criterion = criterion
        self.k = k
        self.eta = eta
        if criterion == 'sigmoid':
            self.m = nn.Sigmoid()
        elif criterion == 'softmax':
            self.m = nn.LogSoftmax(dim=1)

        self.num_classes = num_classes
        self.device = device
        self.dropout = dropout

    def pretrain_with_batch(self, data):
        """
        Pretrain
        :param data:
        :return:
        """
        x, edge_index = data.x, data.edge_index

        for conv, norm in zip(self.convs[:-1], self.norms):
            x = norm(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        lc_x = self.convs[-1](x, edge_index)

        p_lc = self.m(lc_x)
        return p_lc

    @torch.no_grad()
    def get_embedding(self, data):
        dic = {}
        dic['full_data'] = data

        x, edge_index = data.x, data.edge_index
        y = data.y

        for conv, norm in zip(self.convs[:-1], self.norms):
            x = norm(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=False)

        embedding = x

        dic['train_embedding'] = embedding
        dic['train_label'] = y

        return dic

    @torch.no_grad()
    def decode_z_q(self, train_info):

        data = train_info['full_data']

        x, edge_index = data.x, data.edge_index

        y = data.y

        for conv, norm in zip(self.convs[:-1], self.norms):
            x = norm(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=False)

        lc_x = self.convs[-1](x, edge_index)
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(train_info['z_q'],
                                                              train_info['z_q'],
                                                              y,
                                                              train_info[
                                                                  'mask'],
                                                              train_info[
                                                                  'neighbors_info']))

        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution

    def forward(self,
                data,
                mask,
                neighbors_info=None):

        x, edge_index = data.x, data.edge_index

        y = data.y

        for conv, norm in zip(self.convs[:-1], self.norms):
            x = norm(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=False)

        embedding = x

        lc_x = self.convs[-1](x, edge_index)
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                              embedding,
                                                              y,
                                                              mask,
                                                              neighbors_info))

        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution, embedding

    def generate_label_with_similar_nodes(self,
                                          embedding_a,
                                          embedding_b,
                                          b_y,
                                          mask,
                                          neighbors_info=None):

        with torch.no_grad():
            if torch.is_tensor(neighbors_info):
                feature_similarity_matrix = neighbors_info
            else:
                feature_similarity_matrix = sim_matrix(embedding_a, embedding_b)
            feature_similarity_matrix = feature_similarity_matrix.cpu()
            if self.training:
                feature_similarity_matrix = feature_similarity_matrix * mask
            topk_similar_nodes_information = torch.topk(feature_similarity_matrix, self.k)
        fetch_similarity = torch.exp(topk_similar_nodes_information.values)
        fetch_similarity = fetch_similarity.to(self.device)
        fetch_label = []
        if self.criterion == 'sigmoid':
            one_hot_label = b_y
        elif self.criterion == 'softmax':
            one_hot_label = convert_to_one_hot_label(
                labels=b_y,
                num_classes=self.num_classes,
                batch_size=b_y.shape[0],
                device=self.device
            )
            one_hot_label = one_hot_label.to(self.device)

        for i in range(topk_similar_nodes_information.indices.shape[0]):
            fetch_label.append(
                torch.index_select(one_hot_label, 0, topk_similar_nodes_information.indices[i].to(self.device)))
        fetch_label = torch.stack(fetch_label)
        node_label_distribution = fetch_similarity.unsqueeze(-1) * fetch_label.to(self.device)
        total_fuse_distribution = node_label_distribution.sum(dim=1)

        return total_fuse_distribution

    def inference(self,
                  test_data,
                  training_embedding,
                  training_y):

        x, edge_index = test_data.x, test_data.edge_index

        for conv, norm in zip(self.convs[:-1], self.norms):
            x = norm(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=False)

        _data_embedding = x
        lc_x = self.convs[-1](x, edge_index)
        p_lc = self.m(lc_x)

        p_sim = self.m(
            self.generate_label_with_similar_nodes(_data_embedding,
                                                   training_embedding,
                                                   training_y,
                                                   mask=None,
                                                   neighbors_info=None))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim
        return final_label_distribution

class DNA_Encoder_SubGraph(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes,
                 criterion,
                 k,
                 eta,
                 device,
                 hidden_dim=128,
                 n_layers=4,
                 heads=8,
                 groups=16
                 ):
        super().__init__()

        self.hidden_channels = hidden_dim
        self.lin1 = torch.nn.Linear(input_dim, hidden_dim)
        self.convs = torch.nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(
                DNAConv(hidden_dim, heads, groups, dropout=0.8,
                        cached=False)
            )
        self.lin2 = torch.nn.Linear(hidden_dim, num_classes)

        self.criterion = criterion
        self.k = k
        self.eta = eta
        if criterion == 'sigmoid':
            self.m = nn.Sigmoid()
        elif criterion == 'softmax':
            self.m = nn.LogSoftmax(dim=1)

        self.num_classes = num_classes
        self.device = device

    def pretrain_with_batch(self, data):
        """
        :param data:
        :return:
        """
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x_all = x.view(-1, 1, self.hidden_channels)
        for conv in self.convs:
            x = F.relu(conv(x_all, edge_index))
            x = x.view(-1, 1, self.hidden_channels)
            x_all = torch.cat([x_all, x], dim=1)
        x = x_all[:, -1]
        x = F.dropout(x, p=0.5, training=self.training)
        lc_x = self.lin2(x)

        p_lc = self.m(lc_x)
        return p_lc

    @torch.no_grad()
    def get_embedding(self, data):
        dic = {}
        dic['full_data'] = data

        x, edge_index = data.x, data.edge_index
        y = data.y

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x_all = x.view(-1, 1, self.hidden_channels)
        for conv in self.convs:
            x = F.relu(conv(x_all, edge_index))
            x = x.view(-1, 1, self.hidden_channels)
            x_all = torch.cat([x_all, x], dim=1)
        x = x_all[:, -1]
        x = F.dropout(x, p=0.5, training=False)

        embedding = x

        dic['train_embedding'] = embedding
        dic['train_label'] = y

        return dic

    @torch.no_grad()
    def decode_z_q(self, train_info):

        data = train_info['full_data']

        x, edge_index = data.x, data.edge_index

        y = data.y

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x_all = x.view(-1, 1, self.hidden_channels)
        for conv in self.convs:
            x = F.relu(conv(x_all, edge_index))
            x = x.view(-1, 1, self.hidden_channels)
            x_all = torch.cat([x_all, x], dim=1)
        x = x_all[:, -1]
        x = F.dropout(x, p=0.5, training=self.training)
        lc_x = self.lin2(x)

        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(train_info['z_q'],
                                                              train_info['z_q'],
                                                              y,
                                                              train_info[
                                                                  'mask'],
                                                              train_info[
                                                                  'neighbors_info']))

        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution

    def forward(self,
                data,
                mask,
                neighbors_info=None):

        x, edge_index = data.x, data.edge_index

        y = data.y

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x_all = x.view(-1, 1, self.hidden_channels)
        for conv in self.convs:
            x = F.relu(conv(x_all, edge_index))
            x = x.view(-1, 1, self.hidden_channels)
            x_all = torch.cat([x_all, x], dim=1)
        x = x_all[:, -1]
        x = F.dropout(x, p=0.5, training=self.training)

        embedding = x

        lc_x = self.lin2(x)
        p_lc = self.m(lc_x)
        p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                              embedding,
                                                              y,
                                                              mask,
                                                              neighbors_info))

        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution, embedding

    def generate_label_with_similar_nodes(self,
                                          embedding_a,
                                          embedding_b,
                                          b_y,
                                          mask,
                                          neighbors_info=None):
        with torch.no_grad():
            if torch.is_tensor(neighbors_info):
                feature_similarity_matrix = neighbors_info
            else:
                feature_similarity_matrix = sim_matrix(embedding_a, embedding_b)
            feature_similarity_matrix = feature_similarity_matrix.cpu()
            if self.training:
                feature_similarity_matrix = feature_similarity_matrix * mask
            topk_similar_nodes_information = torch.topk(feature_similarity_matrix, self.k)
        fetch_similarity = torch.exp(topk_similar_nodes_information.values)
        fetch_similarity = fetch_similarity.to(self.device)
        fetch_label = []
        if self.criterion == 'sigmoid':
            one_hot_label = b_y
        elif self.criterion == 'softmax':
            one_hot_label = convert_to_one_hot_label(
                labels=b_y,
                num_classes=self.num_classes,
                batch_size=b_y.shape[0],
                device=self.device
            )
            one_hot_label = one_hot_label.to(self.device)

        for i in range(topk_similar_nodes_information.indices.shape[0]):
            fetch_label.append(
                torch.index_select(one_hot_label, 0, topk_similar_nodes_information.indices[i].to(self.device)))
        fetch_label = torch.stack(fetch_label)
        node_label_distribution = fetch_similarity.unsqueeze(-1) * fetch_label.to(self.device)
        total_fuse_distribution = node_label_distribution.sum(dim=1)

        return total_fuse_distribution

    def inference(self,
                  test_data,
                  training_embedding,
                  training_y):

        x, edge_index = test_data.x, test_data.edge_index

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x_all = x.view(-1, 1, self.hidden_channels)
        for conv in self.convs:
            x = F.relu(conv(x_all, edge_index))
            x = x.view(-1, 1, self.hidden_channels)
            x_all = torch.cat([x_all, x], dim=1)
        x = x_all[:, -1]
        x = F.dropout(x, p=0.5, training=False)

        _data_embedding = x

        lc_x = self.lin2(x)
        p_lc = self.m(lc_x)

        p_sim = self.m(
            self.generate_label_with_similar_nodes(_data_embedding,
                                                   training_embedding,
                                                   training_y,
                                                   mask=None,
                                                   neighbors_info=None))

        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim
        return final_label_distribution

class SSGC_Encoder_SubGraph(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_classes,
                 criterion,
                 k,
                 eta,
                 device):
        super().__init__()
        self.layer_1 = SSGConv(input_dim, hidden_dim, alpha=0.1, K=2)
        self.layer_2 = SSGConv(hidden_dim, num_classes, alpha=0.1, K=2)

        self.criterion = criterion
        self.k = k
        self.eta = eta
        if criterion == 'sigmoid':
            self.m = nn.Sigmoid()
        elif criterion == 'softmax':
            self.m = nn.LogSoftmax(dim=1)
        self.num_classes = num_classes
        self.device = device

    def pretrain_with_batch(self, data):
        """
        预训练环节
        :param data:
        :return:
        """
        x, edge_index = data.x, data.edge_index

        x = self.layer_1(x, edge_index)
        lc_x = self.layer_2(x, edge_index)

        p_lc = self.m(lc_x)

        return p_lc

    @torch.no_grad()
    def get_embedding(self, data):
        dic = {}
        dic['full_data'] = data
        x, edge_index = data.x, data.edge_index
        y = data.y

        x = self.layer_1(x, edge_index)
        embedding = x

        dic['train_embedding'] = embedding
        dic['train_label'] = y

        return dic

    @torch.no_grad()
    def decode_z_q(self, train_info):

        data = train_info['full_data']
        x, edge_index = data.x, data.edge_index

        y = data.y

        x = self.layer_1(x, edge_index)

        lc_x = self.layer_2(x, edge_index)
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(train_info['z_q'],
                                                              train_info['z_q'],
                                                              y,
                                                              train_info['mask'],
                                                              train_info['neighbors_info']))
        # print(p_sim)

        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution

    def forward(self,
                data,
                mask,
                neighbors_info=None):
        x, edge_index = data.x, data.edge_index

        y = data.y

        x = self.layer_1(x, edge_index)
        embedding = x

        lc_x = self.layer_2(x, edge_index)
        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                              embedding,
                                                              y,
                                                              mask,
                                                              neighbors_info))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution, embedding

    def generate_label_with_similar_nodes(self,
                                          embedding_a,
                                          embedding_b,
                                          b_y,
                                          mask,
                                          neighbors_info=None):

        with torch.no_grad():
            if torch.is_tensor(neighbors_info):
                feature_similarity_matrix = neighbors_info
            else:
                feature_similarity_matrix = sim_matrix(embedding_a, embedding_b)
            feature_similarity_matrix = feature_similarity_matrix.cpu()
            if self.training:
                feature_similarity_matrix = feature_similarity_matrix * mask
            topk_similar_nodes_information = torch.topk(feature_similarity_matrix, self.k)
        fetch_similarity = torch.exp(topk_similar_nodes_information.values)
        fetch_similarity = fetch_similarity.to(self.device)
        fetch_label = []
        if self.criterion == 'sigmoid':
            one_hot_label = b_y
        elif self.criterion == 'softmax':
            one_hot_label = convert_to_one_hot_label(
                labels=b_y,
                num_classes=self.num_classes,
                batch_size=b_y.shape[0],
                device=self.device
            )
            one_hot_label = one_hot_label.to(self.device)

        for i in range(topk_similar_nodes_information.indices.shape[0]):
            fetch_label.append(
                torch.index_select(one_hot_label, 0, topk_similar_nodes_information.indices[i].to(self.device)))
        fetch_label = torch.stack(fetch_label)
        node_label_distribution = fetch_similarity.unsqueeze(-1) * fetch_label.to(self.device)
        total_fuse_distribution = node_label_distribution.sum(dim=1)

        return total_fuse_distribution

    def inference(self,
                  test_data,
                  training_embedding,
                  training_y):

        x, edge_index = test_data.x, test_data.edge_index

        x = self.layer_1(x, edge_index)
        _data_embedding = x
        lc_x = self.layer_2(x, edge_index)
        p_lc = self.m(lc_x)


        p_sim = self.m(
            self.generate_label_with_similar_nodes(_data_embedding,
                                                   training_embedding,
                                                   training_y,
                                                   mask=None,
                                                   neighbors_info=None))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim
        return final_label_distribution

class FAGCN_Encoder_SubGraph(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_classes,
                 criterion,
                 k,
                 eta,
                 device,
                 n_layers=4,
                 eps=0.2,
                 dropout=0.5):
        super().__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(input_dim, hidden_dim))
        self.lins.append(nn.Linear(hidden_dim, num_classes))
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                FAConv(hidden_dim, dropout=dropout))
        self.dropout = dropout

        self.criterion = criterion
        self.k = k
        self.eta = eta
        if criterion == 'sigmoid':
            self.m = nn.Sigmoid()
        elif criterion == 'softmax':
            self.m = nn.LogSoftmax(dim=1)
        self.num_classes = num_classes
        self.device = device
        self.eps = eps

    def pretrain_with_batch(self, data):
        """
        预训练环节
        :param data:
        :return:
        """
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        x_0 = F.dropout(x_0, self.dropout, training=self.training)

        for conv in self.convs:
            h = conv(x, x_0, edge_index)
            x = self.eps * x_0 + h

        lc_x = self.lins[1](x)

        p_lc = self.m(lc_x)

        return p_lc

    @torch.no_grad()
    def get_embedding(self, data):
        dic = {}
        dic['full_data'] = data
        y = data.y

        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        x_0 = F.dropout(x_0, self.dropout, training=self.training)

        for conv in self.convs:
            h = conv(x, x_0, edge_index)
            x = self.eps * x_0 + h
            x = x.relu()

        embedding = x

        dic['train_embedding'] = embedding
        dic['train_label'] = y

        return dic

    @torch.no_grad()
    def decode_z_q(self, train_info):

        data = train_info['full_data']
        y = data.y

        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        x_0 = F.dropout(x_0, self.dropout, training=self.training)

        for conv in self.convs:
            h = conv(x, x_0, edge_index)
            x = self.eps * x_0 + h
            x = x.relu()

        lc_x = self.lins[1](x)

        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(train_info['z_q'],
                                                              train_info['z_q'],
                                                              y,
                                                              train_info['mask'],
                                                              train_info['neighbors_info']))
        # print(p_sim)

        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution

    def forward(self,
                data,
                mask,
                neighbors_info=None):

        y = data.y

        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        x_0 = F.dropout(x_0, self.dropout, training=self.training)

        for conv in self.convs:
            h = conv(x, x_0, edge_index)
            x = self.eps * x_0 + h
            x = x.relu()

        embedding = x
        lc_x = self.lins[1](x)

        p_lc = self.m(lc_x)

        p_sim = self.m(self.generate_label_with_similar_nodes(embedding,
                                                              embedding,
                                                              y,
                                                              mask,
                                                              neighbors_info))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution, embedding

    def generate_label_with_similar_nodes(self,
                                          embedding_a,
                                          embedding_b,
                                          b_y,
                                          mask,
                                          neighbors_info=None):

        with torch.no_grad():
            if torch.is_tensor(neighbors_info):
                feature_similarity_matrix = neighbors_info
            else:
                feature_similarity_matrix = sim_matrix(embedding_a, embedding_b)
            feature_similarity_matrix = feature_similarity_matrix.cpu()
            if self.training:
                feature_similarity_matrix = feature_similarity_matrix * mask
            topk_similar_nodes_information = torch.topk(feature_similarity_matrix, self.k)
        fetch_similarity = torch.exp(topk_similar_nodes_information.values)
        fetch_similarity = fetch_similarity.to(self.device)
        fetch_label = []
        if self.criterion == 'sigmoid':
            one_hot_label = b_y
        elif self.criterion == 'softmax':
            one_hot_label = convert_to_one_hot_label(
                labels=b_y,
                num_classes=self.num_classes,
                batch_size=b_y.shape[0],
                device=self.device
            )
            one_hot_label = one_hot_label.to(self.device)

        for i in range(topk_similar_nodes_information.indices.shape[0]):
            fetch_label.append(
                torch.index_select(one_hot_label, 0, topk_similar_nodes_information.indices[i].to(self.device)))
        fetch_label = torch.stack(fetch_label)
        node_label_distribution = fetch_similarity.unsqueeze(-1) * fetch_label.to(self.device)
        total_fuse_distribution = node_label_distribution.sum(dim=1)

        return total_fuse_distribution

    def inference(self,
                  test_data,
                  training_embedding,
                  training_y):

        x, edge_index = test_data.x, test_data.edge_index
        x = F.dropout(x, self.dropout, training=False)
        x = x_0 = self.lins[0](x).relu()
        x_0 = F.dropout(x_0, self.dropout, training=self.training)

        for conv in self.convs:
            h = conv(x, x_0, edge_index)
            x = self.eps * x_0 + h
            x = x.relu()

        _data_embedding = F.dropout(x, self.dropout, training=False)

        lc_x = self.lins[1](_data_embedding)
        p_lc = self.m(lc_x)

        p_sim = self.m(
            self.generate_label_with_similar_nodes(_data_embedding,
                                                   training_embedding,
                                                   training_y,
                                                   mask=None,
                                                   neighbors_info=None))
        final_label_distribution = self.eta * p_lc + (1 - self.eta) * p_sim

        return final_label_distribution


class NodeReprVAE(nn.Module):

    def __init__(self,
                 model_name,
                 hidden_dim1,
                 hidden_dim2,
                 num_classes,
                 delta,
                 k,
                 sigma,
                 device):
        super(NodeReprVAE, self).__init__()
        self.logstd_p_z = None
        self.mu_p_z = None
        self.logstd_q_z = None
        self.mu_q_z = None
        if 'gat' in model_name:
            n = 4
        else:
            n = 1
        self.gnn_mean_q_z = nn.Linear(hidden_dim1*n, hidden_dim2)
        self.gnn_logstd_q_z = nn.Linear(hidden_dim1*n, hidden_dim2)
        self.gnn_mean_p_z = nn.Linear(hidden_dim1*n, hidden_dim2)
        self.gnn_logstd_p_z = nn.Linear(hidden_dim1*n, hidden_dim2)
        self.linear_layer = nn.Linear(hidden_dim1*n, hidden_dim1*n)
        self.delta = delta
        self.delta = delta
        self.sigma = sigma
        self.device = device
        self.k = k
        self.num_classes = num_classes

    def encode(self, gnn_embedding, sim_info):

        self.mu_q_z = self.gnn_mean_q_z(
            gnn_embedding + self.delta * self.linear_layer(sim_info['most_similar_embedding']))
        self.logstd_q_z = self.gnn_logstd_q_z(
            gnn_embedding + self.delta * self.linear_layer(sim_info['most_similar_embedding']))

        self.mu_p_z = self.gnn_mean_p_z(gnn_embedding)
        self.logstd_p_z = self.gnn_logstd_p_z(gnn_embedding)

        return self.mu_q_z, self.logstd_q_z, self.mu_p_z, self.logstd_p_z

    def reparameterize(self, mu, logstd):

        if self.training:
            std = torch.exp(self.sigma * logstd)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, train_info, sim_info):
        mu_q_z, logstd_q_z, mu_p_z, logstd_p_z = self.encode(train_info['train_embedding'], sim_info)
        z_q = self.reparameterize(mu_q_z, logstd_q_z)
        train_info['z_q'] = z_q
        return [train_info, z_q, mu_q_z, logstd_q_z]

    def compute_KL(self):
        kl = (self.logstd_p_z - self.logstd_q_z) + (self.logstd_q_z.exp() ** 2 + (self.mu_p_z - self.mu_q_z) ** 2) / (
                2 * self.logstd_p_z.exp() ** 2) - 0.5
        return -torch.mean(torch.sum(kl, 1))


class SimilarNodeDistribution_SubGraph(object):

    def __init__(self, k, device, criterion, num_classes, training_status):
        self.k = k
        self.device = device
        self.criterion = criterion
        self.status = criterion
        self.num_classes = num_classes
        self.training = training_status

    def transform_label(self,
                        label):

        if self.criterion == 'sigmoid':
            one_hot_label = label
            return one_hot_label

        elif self.criterion == 'softmax':
            one_hot_label = convert_to_one_hot_label(
                labels=label,
                num_classes=self.num_classes,
                batch_size=label.shape[0],
                device=self.device
            )
            return one_hot_label

    def fetch_sim_info(self,
                       embedding_a,
                       a_y,
                       embedding_b,
                       b_y,
                       train_batch_id,
                       mask,
                       neighbors_info=None):
        one_hot_label_a = self.transform_label(a_y)
        one_hot_label_b = self.transform_label(b_y)

        embedding_combine_a = torch.cat((embedding_a, one_hot_label_a), dim=1)
        embedding_combine_b = torch.cat((embedding_b, one_hot_label_b), dim=1)

        with torch.no_grad():
            if torch.is_tensor(neighbors_info):
                similarity_matrix = neighbors_info
            else:
                similarity_matrix = sim_matrix(embedding_combine_a, embedding_combine_b)
            similarity_matrix = similarity_matrix.cpu()
            if self.training:
                similarity_matrix = similarity_matrix * mask

            gamma_weights = F.softmax(similarity_matrix, dim=1)
            topk_similar_nodes_information = torch.topk(gamma_weights, self.k)

        dic = {}

        batch_id_top_k = []
        for i in range(topk_similar_nodes_information.indices.shape[0]):
            batch_id_top_k.append(
                torch.index_select(train_batch_id, 0,
                                   topk_similar_nodes_information.indices[i].to(self.device))
            )
        batch_id_top_k = torch.stack(batch_id_top_k)

        similarity_values = []
        for i in range(topk_similar_nodes_information.indices.shape[0]):
            similarity_values.append(
                torch.index_select(similarity_matrix[i], 0,
                                   topk_similar_nodes_information.indices[i]))
        similarity_values = torch.stack(similarity_values)

        similarity_labels = []
        for i in range(topk_similar_nodes_information.indices.shape[0]):
            similarity_labels.append(
                torch.index_select(one_hot_label_b, 0,
                                   topk_similar_nodes_information.indices[i].to(self.device)))

        similarity_labels = torch.stack(similarity_labels)

        similarity_embedding = []
        for i in range(topk_similar_nodes_information.indices.shape[0]):
            similarity_embedding.append(
                torch.index_select(embedding_b, 0,
                                   topk_similar_nodes_information.indices[i].to(self.device)))

        similarity_embedding = torch.stack(similarity_embedding)

        scalar_output = torch.sum(one_hot_label_a.repeat(self.k, 1, 1).permute(1, 0, 2) == similarity_labels,
                                  dim=-1)
        scalar_output = scalar_output / one_hot_label_a.shape[1]

        b_uv = F.softmax(scalar_output * topk_similar_nodes_information.values.to(self.device), dim=1)

        most_similar_embedding = b_uv.unsqueeze(-1) * similarity_embedding
        most_similar_embedding = torch.sum(most_similar_embedding, dim=1)

        dic['tu_indexes'] = topk_similar_nodes_information.indices
        dic['tu_indexes_in_global'] = batch_id_top_k
        dic['gamma_weights'] = topk_similar_nodes_information.values
        dic['similarity_values'] = similarity_values
        dic['similarity_labels'] = similarity_labels
        dic['similarity_embedding'] = similarity_embedding
        dic['most_similar_embedding'] = most_similar_embedding.float()
        return dic


class MGIGNN(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_classes,
                 normalize,
                 nodes_numbers,
                 model_name,
                 criterion,
                 k,
                 eta,
                 device,
                 alpha=0.1):
        super(MGIGNN, self).__init__()

        self.k = k
        self.eta = eta
        self.device = device
        self.criterion = criterion
        self.num_classes = num_classes

        self.nodes_numbers = nodes_numbers
        self.alpha = torch.tensor([alpha] * self.nodes_numbers).to(self.device)
        self.lambda_ = nn.Parameter(torch.ones(self.nodes_numbers).fill_(alpha))
        self.lambda_.requires_grad = False
        self.alpha_stats = self.digamma_stats1(self.alpha)
        self.forget_rate = 0.8
        self.decay_rate = 1.0
        self._num_updates = 0
        self.lambda_stats_sum = nn.Parameter(torch.ones(self.nodes_numbers).fill_(alpha))

        self.embedding_encoder = self.construct_embedding_model(
            input_dim,
            hidden_dim,
            num_classes,
            normalize,
            model_name,
            criterion,
            k,
            eta,
            device)

        self.similar_node_handler = SimilarNodeDistribution_SubGraph(
            k=self.k,
            device=self.device,
            criterion=self.criterion,
            num_classes=self.num_classes,
            training_status=self.training)

        self.node_vae = NodeReprVAE(
            model_name=model_name,
            hidden_dim1=hidden_dim,
            hidden_dim2=64,
            num_classes=num_classes,
            delta=0.2,
            k=k,
            sigma=0.1,
            device=device
        )

        self.training_embedding_memory = None
        self.train_label_memory = None
        self.batch_size = None

    def construct_embedding_model(self,
                                  input_dim,
                                  hidden_dim,
                                  num_classes,
                                  normalize,
                                  model_name,
                                  criterion,
                                  k,
                                  eta,
                                  device):

        if model_name == 'graphsage':

            embedding_encoder = GraphSAGE_Encoder_SubGraph(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                criterion=criterion,
                k=k,
                eta=eta,
                device=device
            )

        elif model_name == 'gat':
            embedding_encoder = GAT_Encoder_SubGraph(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                criterion=criterion,
                k=k,
                eta=eta,
                device=device
            )

        elif model_name == 'gatv2':
            embedding_encoder = GATv2Conv_Encoder_SubGraph(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                criterion=criterion,
                k=k,
                eta=eta,
                device=device
            )

        elif model_name == 'gcn':
            embedding_encoder = GCN_Encoder_SubGraph(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                criterion=criterion,
                k=k,
                eta=eta,
                device=device
            )

        elif model_name == 'sgc':
            embedding_encoder = SGC_Encoder_SubGraph(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                criterion=criterion,
                k=k,
                eta=eta,
                device=device
            )

        elif model_name == 'gcnii':
            embedding_encoder = GCNII_Encoder_SubGraph(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                criterion=criterion,
                k=k,
                eta=eta,
                device=device
            )

        elif model_name == 'film':
            embedding_encoder = FILM_Encoder_SubGraph(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                criterion=criterion,
                k=k,
                eta=eta,
                device=device
            )

        elif model_name == 'dna':
            embedding_encoder = DNA_Encoder_SubGraph(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                criterion=criterion,
                k=k,
                eta=eta,
                device=device
            )

        elif model_name == 'gin':
            embedding_encoder = GIN0_Encoder_SubGraph(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                criterion=criterion,
                k=k,
                eta=eta,
                device=device
            )

        elif model_name == 'ssgc':
            embedding_encoder = SSGC_Encoder_SubGraph(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                criterion=criterion,
                k=k,
                eta=eta,
                device=device
            )

        elif model_name == 'fagcn':
            embedding_encoder = FAGCN_Encoder_SubGraph(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                criterion=criterion,
                k=k,
                eta=eta,
                device=device
            )

        return embedding_encoder

    def pretrain_forward(self, data):

        return self.embedding_encoder.pretrain_with_batch(data)

    def m_step_forward(self,
                       data,
                       mask,
                       neighbors_info=None):

        return self.embedding_encoder(data, mask, neighbors_info)

    def get_training_embedding_memory(self,
                                     data,
                                     non_hop_2_adjacency_matrix_dict,
                                     id_dict,
                                     neighbors_info
                                     ):

        self.embedding_encoder.eval()
        train_info = self.embedding_encoder.get_embedding(data)
        self.training_embedding_memory = train_info['train_embedding']
        self.train_label_memory = train_info['train_label']

        tag = data.x.shape[0]
        train_info['train_batch_id'] = id_dict[tag]
        train_info['mask'] = non_hop_2_adjacency_matrix_dict[tag]
        train_info['neighbors_info'] = neighbors_info
        self.embedding_encoder.train()

        return train_info

    def get_sim_info(self,
                     embedding_a,
                     a_y,
                     embedding_b,
                     b_y,
                     train_batch_id,
                     mask,
                     neighbors_info=None):

        return self.similar_node_handler.fetch_sim_info(
            embedding_a=embedding_a,
            embedding_b=embedding_b,
            a_y=a_y,
            b_y=b_y,
            train_batch_id=train_batch_id,
            mask=mask,
            neighbors_info=neighbors_info
        )

    def reconstruct_z_q(self, train_info, sim_info):

        _train_info, _z_q, _mu_q_z, _logstd_q_z = self.node_vae(train_info=train_info, sim_info=sim_info)

        final_label_distribution = self.embedding_encoder.decode_z_q(_train_info)

        return final_label_distribution

    def update_lambda(self, value):
        self.lambda_.copy_(value)

    def distributed_update_lambda(self,
                                  lambda_stats_sum,
                                  num_of_nodes,
                                  update_num):

        forget = math.pow(update_num + self.decay_rate, -self.forget_rate)
        new_lambda = self.alpha + lambda_stats_sum / num_of_nodes * self.nodes_numbers

        # Update the Dirichlet posterior
        new_lambda = (1 - forget) * self.lambda_ + forget * new_lambda
        self.update_lambda(new_lambda)

    def digamma_stats1(self, alpha):
        return torch.lgamma(alpha.sum()) - torch.lgamma(alpha).sum()

    def digamma_stats2(self, alpha1, alpha2):
        return ((alpha1 - 1.) * (torch.digamma(alpha2) - torch.digamma(alpha2.sum()))).sum()

    def update_lambda_stats_sum(self, sim_info):

        with torch.no_grad():
            lambda_stats_sum_i = nn.Parameter(torch.ones(self.nodes_numbers).fill_(0)).to(self.device)
            self.batch_size = sim_info['similarity_labels'].shape[0]
            lambda_stats_sum_i = lambda_stats_sum_i.repeat(self.batch_size, 1).scatter(-1,
                                                                                       sim_info['tu_indexes_in_global'],
                                                                                       self.k * sim_info[
                                                                                           'gamma_weights'].float().to(
                                                                                           self.device))
            lambda_stats_sum_i = sum(lambda_stats_sum_i)
            self.lambda_stats_sum += lambda_stats_sum_i

    def compute_kl_theta(self):
        if self.training:
            term1 = self.alpha_stats + self.digamma_stats2(self.alpha, self.lambda_) \
                    - (self.digamma_stats1(self.lambda_) +
                       self.digamma_stats2(self.lambda_, self.lambda_))
            term1 = term1 / self.nodes_numbers
        else:
            term1 = 0.

        return -term1
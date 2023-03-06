import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, SGConv, GCN2Conv, GATv2Conv
from torch_geometric.nn import FiLMConv
from torch_geometric.nn import DNAConv
from torch_geometric.nn import SSGConv
from torch_geometric.nn import FAConv
from torch_geometric.nn import MLP, GINConv, global_add_pool, global_mean_pool
from utils import convert_to_one_hot_label, sim_matrix
from torch.nn import BatchNorm1d


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


class GIN0_Encoder_SubGraph(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_classes,
                 criterion,
                 k,
                 eta,
                 device,
                 num_layers=5
                 ):
        super().__init__()

        self.hidden_channels = hidden_dim
        self.conv1 = GINConv(
            Sequential(
                Linear(input_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                BN(hidden_dim),
            ), train_eps=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden_dim, hidden_dim),
                        ReLU(),
                        Linear(hidden_dim, hidden_dim),
                        ReLU(),
                        BN(hidden_dim),
                    ), train_eps=False))
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, num_classes)

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

        x, edge_index, batch = data.x, data.edge_index, data.batch
        # print(data.batch)
        # print(x.shape)
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_mean_pool(x, batch, size=x.shape[0])
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        lc_x = self.lin2(x)
        # print(lc_x.shape)
        p_lc = self.m(lc_x)
        return p_lc

    @torch.no_grad()
    def get_embedding(self, data):
        dic = {}
        dic['full_data'] = data
        y = data.y
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_mean_pool(x, batch, size=x.shape[0])
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=False)
        embedding = x

        dic['train_embedding'] = embedding
        dic['train_label'] = y

        return dic

    @torch.no_grad()
    def decode_z_q(self, train_info):

        data = train_info['full_data']
        y = data.y

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_mean_pool(x, batch, size=x.shape[0])
        x = F.relu(self.lin1(x))
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

    def forward(self,
                data,
                mask,
                neighbors_info=None):
        y = data.y

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_mean_pool(x, batch, size=x.shape[0])
        x = F.relu(self.lin1(x))
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

    def inference(self,
                  test_data,
                  training_embedding,
                  training_y):

        x, edge_index, batch = test_data.x, test_data.edge_index, test_data.batch

        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_mean_pool(x, batch, size=x.shape[0])
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=False)

        _data_embedding = x
        lc_x = self.lin2(x)
        p_lc = self.m(lc_x)

        p_sim = self.m(
            self.generate_label_with_similar_nodes(_data_embedding,
                                                   training_embedding,
                                                   training_y,
                                                   mask=None,
                                                   neighbors_info=None)
        )

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

class template(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_classes,
                 criterion,
                 k,
                 eta,
                 device):
        pass

    def pretrain_with_batch(self, data):
        pass

    def get_embedding(self, data):
        pass

    def decode_z_q(self, train_info):
        pass

    def forward(self, data, mask, neighbors_info=None):
        pass

    def generate_label_with_similar_nodes(self,
                                          embedding_a,
                                          embedding_b,
                                          b_y,
                                          mask,
                                          neighbors_info=None):
        pass

    def inference(self, test_data, training_embedding, training_y):
        pass

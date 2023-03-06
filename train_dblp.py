import argparse
import copy
import os.path
import sys

sys.path.append("./")

import torch
import time
from utils import CitationDataset
from sklearn import metrics
from torch.optim import Adam
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from model_with_dirichlet_subgraph import MGIGNN_With_SubGraph
import pickle
from torch.nn import functional as F
import torch_geometric.transforms as T
from tqdm import tqdm


def train_test(run_time, args):

    sim_function = args.sim_function

    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    device = torch.device("cuda:{}".format(str(args.cuda_id)) if torch.cuda.is_available()
                          and args.cuda else 'cpu')
    print(device)

    # Load dataset

    if args.model == 'gcnii':
        pre_transform = T.Compose([T.GCNNorm(), T.ToSparseTensor()])
    else:
        pre_transform = None
    path = "./data/DBLP/dblp.pkl"

    train_dataset = CitationDataset(root=path, split='train', pre_transform=pre_transform)  # , transform=T.NormalizeFeatures())
    val_dataset = CitationDataset(root=path, split='val', pre_transform=pre_transform)  # , transform=T.NormalizeFeatures())
    test_dataset = CitationDataset(root=path, split='test', pre_transform=pre_transform)  # , transform=T.NormalizeFeatures())
    # train_dataset, val_dataset, test_dataset = map(precompute_edge_label_and_reverse, prepare_dblp(args))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    non_hop_2_adjacency_matrix_dict_path = os.path.join("./data/DBLP",
                                                        "non_hop_2_adjacency_matrix_dict_val_batch_size_{}.pkl".format(
                                                            1))
    train_node_id_dict_path = os.path.join("./data/DBLP",
                                           "train_node_id_dict.pkl")
    common_neighbors_dict_path = os.path.join(
        "./data/DBLP",
        "common_neighbors_dblp.pkl")
    degree_dict_path = os.path.join(
        "./data/DBLP",
        "degree_count_dblp.pkl"
    )


    print("|> Length of train_loader ", len(train_loader))
    print("|> Length of val_loader ", len(val_loader))
    print("|> Length of test_loader ", len(test_loader))

    with open(non_hop_2_adjacency_matrix_dict_path, 'rb') as f:
        non_hop_2_adjacency_matrix_dict = pickle.load(f)

    with open(train_node_id_dict_path, 'rb') as f:
        id_dict = pickle.load(f)

    if sim_function == 'common_neighbor':
        with open(common_neighbors_dict_path, 'rb') as f:
            common_neighbors_dict = pickle.load(f)

    if sim_function == 'degree':
        with open(degree_dict_path, 'rb') as f:
            degree_count_dict = pickle.load(f)

    num_nodes = sum(id_dict.keys())

    if args.criterion == 'sigmoid':
        criterion = torch.nn.BCELoss()
    elif args.criterion == 'softmax':
        criterion = torch.nn.CrossEntropyLoss()

    # Define the model
    model = MGIGNN_With_SubGraph(
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
    print(model)
    # print(model.device)

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
            # print(batch)
            batch_size = batch.x.shape[0]
            # E Step
            model.embedding_encoder.requires_grad = False
            batch = batch.to(device)
            nodevae_optimizer.zero_grad()
            y = batch.y
            # print(y.shape)
            if sim_function == 'feature_base':
                train_info = model.get_training_embedding_memory(
                    batch,
                    id_dict=id_dict,
                    non_hop_2_adjacency_matrix_dict=non_hop_2_adjacency_matrix_dict,
                    neighbors_info=None
                )

            elif sim_function == 'common_neighbor':
                train_info = model.get_training_embedding_memory(
                    batch,
                    id_dict=id_dict,
                    non_hop_2_adjacency_matrix_dict=non_hop_2_adjacency_matrix_dict,
                    neighbors_info=common_neighbors_dict[batch_size]
                )

            elif sim_function == 'degree':
                train_info = model.get_training_embedding_memory(
                    batch,
                    id_dict=id_dict,
                    non_hop_2_adjacency_matrix_dict=non_hop_2_adjacency_matrix_dict,
                    neighbors_info=degree_count_dict[batch_size]
                )

            if sim_function == 'feature_base':
                sim_info = model.get_sim_info(
                    train_info['train_embedding'],
                    train_info['train_label'],
                    train_info['train_embedding'],
                    train_info['train_label'],
                    train_info['train_batch_id'].to(device),
                    train_info['mask'],
                    None
                )

            elif sim_function == 'common_neighbor':
                sim_info = model.get_sim_info(
                    train_info['train_embedding'],
                    train_info['train_label'],
                    train_info['train_embedding'],
                    train_info['train_label'],
                    train_info['train_batch_id'].to(device),
                    train_info['mask'],
                    train_info['neighbors_info']
                )
            elif sim_function == 'degree':
                sim_info = model.get_sim_info(
                    train_info['train_embedding'],
                    train_info['train_label'],
                    train_info['train_embedding'],
                    train_info['train_label'],
                    train_info['train_batch_id'].to(device),
                    train_info['mask'],
                    train_info['neighbors_info']
                )

            # print(sim_info)
            output = model.reconstruct_z_q(train_info, sim_info)
            # print(output.shape)
            output = output.float()
            bce_loss = criterion(output, y)
            # print(bce_loss)
            # print(bce_loss.shape)
            # print(output.shape)
            # print(y.shape)
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

            # M Step
            model.embedding_encoder.requires_grad = True
            model.node_vae.requires_grad = False
            embedding_optimizer.zero_grad()
            if sim_function == 'feature_base':
                print("|> [sim_function] sim_function is feature_base")
                output, _ = model.m_step_forward(batch,
                                                 non_hop_2_adjacency_matrix_dict[batch_size],
                                                 None)
            elif sim_function == 'common_neighbor':
                print("|> [sim_function] sim_function is common_neighbor")
                output, _ = model.m_step_forward(batch,
                                                 non_hop_2_adjacency_matrix_dict[batch_size],
                                                 common_neighbors_dict[batch_size])
            elif sim_function == 'degree':
                print("|> [sim_function] sim_function is degree")
                output, _ = model.m_step_forward(batch,
                                                 non_hop_2_adjacency_matrix_dict[batch_size],
                                                 degree_count_dict[batch_size])
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

    test(model,
                    test_loader,
                    train_loader,
                    device,
                    run_time,
                    non_hop_2_adjacency_matrix_dict,
                    log_dir,
                    args)



def test(model,
                    test_loader,
                    train_loader,
                    device,
                    run_time,
                    non_hop_2_adjacency_matrix_dict,
                    log_dir,
                    args):

    model.eval()

    with torch.no_grad():
        training_embedding = []
        training_label = []

        for batch in train_loader:
            tag = batch.x.shape[0]
            batch = batch.to(device)
            mask = non_hop_2_adjacency_matrix_dict[tag]
            output, embedding = model.m_step_forward(batch,
                                                     mask)
            training_embedding.append(embedding)
            y = batch.y
            training_label.append(y)

        training_embedding = torch.concat(training_embedding)
        training_label = torch.cat(training_label)

    dirichlet_macro_p, dirichlet_macro_r, dirichlet_macro_f1, dirichlet_micro_f1, dirichlet_num_of_nodes = memory_evaluate(
        model.embedding_encoder,
        model.lambda_,
        test_loader,
        training_embedding,
        training_label,
        device)

    original_macro_p, original_macro_r, original_macro_f1, original_micro_f1, original_num_of_nodes = original_evaluate(
        model.embedding_encoder,
        model.lambda_,
        test_loader,
        training_embedding,
        training_label,
        device)

    with open("{}/{}_memory_with_dirichlet_result.txt".format(log_dir, str(run_time)), 'w') as f:
        f.write(str(dirichlet_macro_p))
        f.write("\n")
        f.write(str(dirichlet_macro_r))
        f.write("\n")
        f.write(str(dirichlet_macro_f1))
        f.write("\n")
        f.write(str(dirichlet_micro_f1))
        f.write("\n")
        f.write(str(dirichlet_num_of_nodes))
        f.write("\n")

    with open("{}/{}_memory_with_original_result.txt".format(log_dir, str(run_time)), 'w') as f:
        f.write(str(original_macro_p))
        f.write("\n")
        f.write(str(original_macro_r))
        f.write("\n")
        f.write(str(original_macro_f1))
        f.write("\n")
        f.write(str(original_micro_f1))
        f.write("\n")
        f.write(str(original_num_of_nodes))
        f.write("\n")

    print("Test set Dirichlet result:",
          "macro_p= {:.2f}".format(dirichlet_macro_p),
          "macro_r= {:.2f}".format(dirichlet_macro_r),
          "macro_f1= {:.2f}".format(dirichlet_macro_f1),
          "micro_f1= {:.2f}".format(dirichlet_micro_f1))


def memory_evaluate(_model,
                   _lambda_,
                   _test_loader,
                   _training_embedding,
                   _training_label,
                   _device):
    _train_lambda = _lambda_
    prob = _train_lambda / _train_lambda.sum()
    res = []
    sorted_prob, indices = torch.sort(prob, descending=True)
    sum_ = 0.

    for prob_i, id_i in zip(sorted_prob, indices):
        sum_ += prob_i.item()
        res.append(id_i.item())
        if sum_ >= 0.7:
            break

    _optimized_training_embedding = torch.index_select(_training_embedding, 0, torch.tensor(res).to(_device))
    _optimized_training_label = torch.index_select(_training_label, 0, torch.tensor(res).to(_device))

    _model.eval()
    ys, preds = [], []

    for test_data in _test_loader:
        test_data = test_data.to(_device)
        y = test_data.y
        ys.append(y)
        with torch.no_grad():
            output = _model.inference(test_data,
                                      _optimized_training_embedding,
                                      _optimized_training_label)
            memory_size = _optimized_training_embedding.shape[0]
            predicts = output.max(1)[1].cpu()
            preds.append(predicts)

    gold, pred = torch.cat(ys, dim=0).cpu(), torch.cat(preds, dim=0).numpy()
    acc = metrics.accuracy_score(gold, pred) * 100
    macro_p = metrics.precision_score(gold, pred, average='macro') * 100
    macro_r = metrics.recall_score(gold, pred, average='macro') * 100
    macro_f1 = metrics.f1_score(gold, pred, average='macro') * 100
    micro_f1 = metrics.f1_score(gold, pred, average='micro') * 100
    return macro_p, macro_r, macro_f1, micro_f1, memory_size


def original_evaluate(_model,
                      _lambda_,
                      _test_loader,
                      _training_embedding,
                      _training_label,
                      _device):
    _model.eval()

    ys, preds = [], []

    for test_data in _test_loader:
        test_data = test_data.to(_device)
        y = test_data.y
        ys.append(y)

        with torch.no_grad():
            output = _model.inference(test_data,
                                      _training_embedding,
                                      _training_label)
            memory_size = _training_embedding.shape[0]
            predicts = output.max(1)[1].cpu()
            preds.append(predicts)

    gold, pred = torch.cat(ys, dim=0).cpu(), torch.cat(preds, dim=0).numpy()
    acc = metrics.accuracy_score(gold, pred) * 100
    macro_p = metrics.precision_score(gold, pred, average='macro') * 100
    macro_r = metrics.recall_score(gold, pred, average='macro') * 100
    macro_f1 = metrics.f1_score(gold, pred, average='macro') * 100
    micro_f1 = metrics.f1_score(gold, pred, average='micro') * 100
    return macro_p, macro_r, macro_f1, micro_f1, memory_size


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
    parser.add_argument('--normalize', required=True)
    parser.add_argument('--model', required=True,
                        help='training_model')
    parser.add_argument('--eta', type=float, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--k', type=int, required=True,
                        help='Number of similar nodes')
    parser.add_argument('--val_test_batch_size', type=int, required=True,
                        help='validation and test dataloader batch size')
    parser.add_argument('--criterion', type=str, required=True,
                        help='softmax or sigmoid')
    parser.add_argument('--run_times', type=int, required=True)
    parser.add_argument('--sim_function',
                        type=str,
                        required=True,
                        help='feature_base, common_neighbor, degree')


    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()
    print(args)
    run_times = args.run_times

    for run_time in range(run_times):
        print("|> Run time is ", run_time)
        train_test(run_time, args)
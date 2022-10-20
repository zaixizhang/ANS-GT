import numpy as np
import torch
import os.path as osp
import pickle
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch.nn import functional as F
from torch.utils.data import DataLoader
from functools import partial
import scipy.sparse as sp
from numpy.linalg import inv
from torch_geometric.datasets import Planetoid, Amazon
from pygsp import graphs
from graph_coarsening.coarsening_utils import coarsen
from torch.nn.functional import normalize
import torch_geometric.transforms as T
from torch_geometric.utils.undirected import is_undirected, to_undirected
from torch_sparse import coalesce


def adj_normalize(mx):
    "A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2"
    mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def eigenvector(L):
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    return torch.tensor(EigVec[:, 1:11], dtype = torch.float32)


def column_normalize(mx):
    "A' = A * D^-1 "
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = mx.dot(r_mat_inv)
    return mx


def coarse_graph_adj(mx, p):
    p[p > 0] = 1.
    p = np.array(p)
    rowsum = p.sum(1)
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    p = np.matmul(p.T, r_mat_inv)
    mx = np.matmul(mx.toarray(), p)
    mx = np.matmul(p.T, mx)
    return mx


def coarse_adj_normalize(adj):
    adj += np.diag(np.ones(adj.shape[0]))
    r_inv = np.power(adj.sum(1), -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    adj = np.matmul(r_mat_inv, adj)
    adj = np.matmul(adj, r_mat_inv)
    return adj


def process_data(p=None):
    name = 'cora'
    dataset = Planetoid(root='./data/', name=name)
    data = dataset[0]
    adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
                                shape=(data.y.shape[0], data.y.shape[0]),
                                dtype=np.float32)
    normalized_adj = adj_normalize(adj)
    column_normalized_adj = column_normalize(adj)
    sp.save_npz('./dataset/'+name+'/normalized_adj.npz', normalized_adj)
    sp.save_npz('./dataset/' + name + '/column_normalized_adj.npz', column_normalized_adj)
    c = 0.15
    k1 = 14
    k2 = 0
    Samples = 8 # sampled subgraphs for each node
    power_adj_list = [normalized_adj]
    for m in range(5):
        power_adj_list.append(power_adj_list[0]*power_adj_list[m])
    torch.save(data.x, './dataset/' + name + '/x.pt')
    torch.save(data.y, './dataset/' + name + '/y.pt')
    torch.save(data.edge_index, './dataset/' + name + '/edge_index.pt')

    #Sampling heuristics: 0,1,2,3
    eigen_adj = c * inv((sp.eye(adj.shape[0]) - (1 - c) * normalized_adj).toarray())
    eigen_adj1 = power_adj_list[0].toarray()
    eigen_adj2 = power_adj_list[1].toarray()
    x = normalize(data.x, dim=1)
    eigen_adj3 = np.array(torch.matmul(x, x.transpose(1, 0)))

    G = graphs.Graph(adj)
    C, Gc, _, _ = coarsen(G, K=10, r=0.9, method='algebraic_JC')
    C = torch.tensor(C/C.sum(1), dtype=torch.float32)
    super_node_feature = torch.matmul(C, data.x)
    feature = torch.cat([data.x, super_node_feature])
    node_supernode_dict = {}
    for i in range(data.y.shape[0]):
        node_supernode_dict[i] = torch.where(C[:, i] > 0)[0].item()
    Coarse_adj = coarse_graph_adj(adj, C)
    Coarse_graph_dim = Coarse_adj.shape[0]
    normalized_coarse_graph = coarse_adj_normalize(Coarse_adj)
    coarse_power_adj_list = [normalized_coarse_graph]
    for m in range(5):
        coarse_power_adj_list.append(np.matmul(normalized_coarse_graph, coarse_power_adj_list[m]))

    #create subgraph samples
    data_list = []
    for id in range(data.y.shape[0]):
        sub_data_list = []
        s = eigen_adj[id]
        s[id] = -1000.0
        top_neighbor_index = s.argsort()[-(k1+k2):]
        super_node_id = node_supernode_dict[id]

        s = eigen_adj[id]
        s[id] = 0
        s = np.maximum(s, 0)
        if p is not None:
            s1 = eigen_adj1[id]
            s2 = eigen_adj2[id]
            s3 = eigen_adj3[id]
            s1[id] = 0
            s2[id] = 0
            s3[id] = 0
            s1 = np.maximum(s1, 0)
            s2 = np.maximum(s2, 0)
            s3 = np.maximum(s3, 0)
            s = p[0]*s/(s.sum()+1e-5) + p[1]*s1/(s1.sum()+1e-5) + p[2]*s2/(s2.sum()+1e-5) + p[3]*s3/(s3.sum()+1e-5)
        sample_num1 = np.minimum(k1, (s > 0).sum())
        sample_num2 = np.minimum(k2, (Coarse_adj[super_node_id] > 0).sum())
        #create subgraph samples for ensemble
        for _ in range(Samples):
            if sample_num1 > 0:
                sample_index1 = np.random.choice(a=np.arange(data.y.shape[0]), size=sample_num1, replace=False, p=s/s.sum())
            else:
                sample_index1 = np.array([], dtype=int)
            if sample_num2 > 0:
                sample_index2 = np.random.choice(a=np.arange(Coarse_graph_dim), size=sample_num2, replace=False, p=Coarse_adj[super_node_id]/Coarse_adj[super_node_id].sum())
            else:
                sample_index2 = np.array([], dtype=int)

            node_feature_id = torch.cat([torch.tensor([id, ]), torch.tensor(sample_index1, dtype=int),
                                    torch.tensor(top_neighbor_index[: k1+k2-sample_num2-sample_num1], dtype=int)])

            super_node_list = np.concatenate([[super_node_id], sample_index2])
            node2supernode_list = np.array([node_supernode_dict[i.item()] for i in node_feature_id])
            all_node_list = np.concatenate([node2supernode_list, super_node_list])

            attn_bias = torch.cat([torch.tensor(i[node_feature_id, :][:, node_feature_id].toarray(), dtype=torch.float32).unsqueeze(0) for i in power_adj_list])

            attn_bias_complem1 = torch.cat([torch.tensor(i[super_node_list, :][:, node2supernode_list], dtype=torch.float32).unsqueeze(0) for i in coarse_power_adj_list])
            attn_bias_complem2 = torch.cat([torch.tensor(i[all_node_list, :][:, super_node_list], dtype=torch.float32).unsqueeze(0) for i in coarse_power_adj_list])

            attn_bias = torch.cat([attn_bias, attn_bias_complem1], dim=1)
            attn_bias = torch.cat([attn_bias, attn_bias_complem2], dim=2)

            attn_bias = attn_bias.permute(1, 2, 0)

            label = torch.cat([data.y[node_feature_id], torch.zeros(len(super_node_list))]).long()
            feature_id = torch.cat([node_feature_id, torch.tensor(super_node_list + data.y.shape[0], dtype=int)])
            assert len(feature_id) == k1+k2+2
            sub_data_list.append([attn_bias, feature_id, label])
        data_list.append(sub_data_list)

    torch.save(data_list, './dataset/'+name+'/data.pt')
    torch.save(feature, './dataset/'+name+'/feature.pt')


def node_sampling(p=None):
    print('Sampling Nodes!')
    name = 'cora'
    edge_index = torch.load('./dataset/'+name+'/edge_index.pt')
    data_x = torch.load('./dataset/'+name+'/x.pt')
    data_y = torch.load('./dataset/'+name+'/y.pt')
    adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                                shape=(data_y.shape[0], data_y.shape[0]),
                                dtype=np.float32)
    normalized_adj = sp.load_npz('./dataset/'+name+'/normalized_adj.npz')
    column_normalized_adj = sp.load_npz('./dataset/' + name + '/column_normalized_adj.npz')
    c = 0.15
    k1 = 14
    k2 = 0
    Samples = 8 # sampled subgraphs for each node
    power_adj_list = [normalized_adj]
    for m in range(5):
        power_adj_list.append(power_adj_list[0]*power_adj_list[m])

    eigen_adj = c * inv((sp.eye(adj.shape[0]) - (1 - c) * column_normalized_adj).toarray())
    eigen_adj1 = power_adj_list[0].toarray()
    eigen_adj2 = power_adj_list[1].toarray()
    x = normalize(data_x, dim=1)
    eigen_adj3 = np.array(torch.matmul(x, x.transpose(1, 0)))

    G = graphs.Graph(adj)
    C, Gc, _, _ = coarsen(G, K=10, r=0.9, method='variation_neighborhood')
    C = torch.tensor(C/C.sum(1), dtype=torch.float32)
    super_node_feature = torch.matmul(C, data_x)
    feature = torch.cat([data_x, super_node_feature])
    node_supernode_dict = {}
    for i in range(data_y.shape[0]):
        node_supernode_dict[i] = torch.where(C[:, i] > 0)[0].item()
    Coarse_adj = coarse_graph_adj(adj, C)
    Coarse_graph_dim = Coarse_adj.shape[0]
    normalized_coarse_graph = coarse_adj_normalize(Coarse_adj)
    coarse_power_adj_list = [normalized_coarse_graph]
    for m in range(5):
        coarse_power_adj_list.append(np.matmul(normalized_coarse_graph, coarse_power_adj_list[m]))

    #create subgraph samples
    data_list = []
    for id in range(data_y.shape[0]):
        sub_data_list = []
        s = eigen_adj[id]
        s[id] = -1000.0
        top_neighbor_index = s.argsort()[-(k1+k2):]
        super_node_id = node_supernode_dict[id]

        s = eigen_adj[id]
        s[id] = 0
        s = np.maximum(s, 0)
        if p is not None:
            s1 = eigen_adj1[id]
            s2 = eigen_adj2[id]
            s3 = eigen_adj3[id]
            s1[id] = 0
            s2[id] = 0
            s3[id] = 0
            s1 = np.maximum(s1, 0)
            s2 = np.maximum(s2, 0)
            s3 = np.maximum(s3, 0)
            s = p[0]*s/(s.sum()+1e-5) + p[1]*s1/(s1.sum()+1e-5) + p[2]*s2/(s2.sum()+1e-5) + p[3]*s3/(s3.sum()+1e-5)
        sample_num1 = np.minimum(k1, (s > 0).sum())
        sample_num2 = np.minimum(k2, (Coarse_adj[super_node_id] > 0).sum())
        # sample_num2 = 0
        for _ in range(Samples):
            if sample_num1 > 0:
                sample_index1 = np.random.choice(a=np.arange(data_y.shape[0]), size=sample_num1, replace=False, p=s/s.sum())
            else:
                sample_index1 = np.array([], dtype=int)

            if sample_num2 > 0:
                sample_index2 = np.random.choice(a=np.arange(Coarse_graph_dim), size=sample_num2, replace=False,
                                                 p=Coarse_adj[super_node_id]/Coarse_adj[super_node_id].sum())
            else:
                sample_index2 = np.array([], dtype=int)

            node_feature_id = torch.cat([torch.tensor([id, ]), torch.tensor(sample_index1, dtype=int),
                                    torch.tensor(top_neighbor_index[: k1+k2-sample_num2-sample_num1], dtype=int)])

            super_node_list = np.concatenate([[super_node_id], sample_index2])
            node2supernode_list = np.array([node_supernode_dict[i.item()] for i in node_feature_id])
            all_node_list = np.concatenate([node2supernode_list, super_node_list])

            attn_bias = torch.cat([torch.tensor(i[node_feature_id, :][:, node_feature_id].toarray(), dtype=torch.float32).unsqueeze(0) for i in power_adj_list])

            attn_bias_complem1 = torch.cat([torch.tensor(i[super_node_list, :][:, node2supernode_list], dtype=torch.float32).unsqueeze(0) for i in coarse_power_adj_list])
            attn_bias_complem2 = torch.cat([torch.tensor(i[all_node_list, :][:, super_node_list], dtype=torch.float32).unsqueeze(0) for i in coarse_power_adj_list])

            attn_bias = torch.cat([attn_bias, attn_bias_complem1], dim=1)
            attn_bias = torch.cat([attn_bias, attn_bias_complem2], dim=2)

            attn_bias = attn_bias.permute(1, 2, 0)

            label = torch.cat([data_y[node_feature_id], torch.zeros(len(super_node_list))]).long()
            feature_id = torch.cat([node_feature_id, torch.tensor(super_node_list + data_y.shape[0], dtype=int)])

            sub_data_list.append([attn_bias, feature_id, label])
        data_list.append(sub_data_list)

    return data_list, feature


if __name__ == '__main__':
    process_data()




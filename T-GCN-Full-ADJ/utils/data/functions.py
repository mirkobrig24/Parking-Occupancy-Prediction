import numpy as np
import pandas as pd
import torch
import pickle
from scipy.sparse import csr_matrix
from utils.graph_conv import calculate_laplacian_with_self_loop
from torch_geometric.loader import DataLoader
#from torch_sparse import SparseTensor

def load_features(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path, header=None)
    feat = np.array(feat_df, dtype=dtype)
    feat = feat.T
    return feat

'''
def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    return adj
'''

def load_adjacency_matrix(adj_path, dtype=np.float32):
    with open(adj_path, 'rb') as f:
        adj_sparse = pickle.load(f)

    adj_dense = []
    for i in range(len(adj_sparse)):
        mat = adj_sparse[i].todense()
        mat = calculate_laplacian_with_self_loop(torch.FloatTensor(mat))
        mat = np.array(mat.cpu())
        mat = csr_matrix(mat)
        adj_dense.append(mat)
        
    print('---ADJ---', len(adj_dense))
    return adj_dense


def generate_dataset(feat, adj, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True):
    """
    :param data: feature matrix
    :param seq_len: length of the train data sequence
    :param pre_len: length of the prediction data sequence
    :param time_len: length of the time series in total
    :param split_ratio: proportion of the training set
    :param normalize: scale the data to (0, 1], divide by the maximum value in the data
    :return: train set (X, Y) and test set (X, Y)
    """
    if time_len is None:
        #time_len = feat.shape[1]    #2322
        time_len = feat.shape[0] 
        print('---TIME---', time_len)                                
    if normalize:
        #Feat matrix
        max_val = np.max(feat)
        feat = feat / max_val
        feat = np.nan_to_num(feat)

    train_size = int(time_len * split_ratio)   #same for feat and adj          
    print('---TRAIN SIZE---', train_size)

    # train FEAT matrix 
    train_data_feat = feat[:train_size]
    print('---LEN TRAIN FEAT---', train_data_feat.shape)

    # train ADJ matrix 
    train_data_adj = adj[:train_size]
    print('---LEN TRAIN ADJ---', len(train_data_adj))

    #test FEAT matrix
    test_data_feat = feat[train_size:time_len]
    print('---TEST LEN FEAT---', test_data_feat.shape)

    #test ADJ matrix
    test_data_adj = adj[train_size:time_len]
    print('---TEST LEN ADJ---', len(test_data_adj))

    train_tensor_adj = list()
    for i in range(0, len(train_data_adj)):
        tensor = torch.FloatTensor(train_data_adj[i].todense()).to_sparse()
        #mat = train_data_adj[i].tocoo()
        #tensor = torch.sparse.LongTensor(torch.LongTensor([mat.row.tolist(), mat.col.tolist()]),
                              #torch.LongTensor(mat.data.astype(np.float32)))

        #tensor = SparseTensor.from_scipy(train_data_adj[i]) NON VA
        #tensor = tensor.to_sparse_csr() NON VA
        train_tensor_adj.append(tensor)
    
    test_tensor_adj = list()
    for i in range(0, len(test_data_adj)):
        tensor = torch.FloatTensor(test_data_adj[i].todense()).to_sparse()
        
        #mat = test_data_adj[i].tocoo()
        #tensor = torch.sparse.LongTensor(torch.LongTensor([mat.row.tolist(), mat.col.tolist()]),
                              #torch.LongTensor(mat.data.astype(np.float32)))
        
        #tensor = SparseTensor.from_scipy(test_data_adj[i]) NON VA
        #tensor = tensor.to_sparse_csr() NON VA
        test_tensor_adj.append(tensor)

    #tensor_tot_adj = torch.stack((list_tot_adj))

    train_X_feat, train_Y_feat, test_X_feat, test_Y_feat = list(), list(), list(), list()
    train_X_adj, train_Y_adj, test_X_adj, test_Y_adj = list(), list(), list(), list() 
    train_X_tot, train_Y_tot, test_X_tot, test_Y_tot = list(), list(), list(), list() 

    for i in range(0, train_data_feat.shape[0] - seq_len - pre_len):
        train_X_feat.append(np.array(train_data_feat[i : i + seq_len]))
        train_Y_feat.append(np.array(train_data_feat[i + seq_len : i + seq_len + pre_len]))

        #train_X_adj.append(list_tot_adj[i : i + seq_len])
        #train_Y_adj.append(list_tot_adj[i + seq_len : i + seq_len + pre_len])
        train_X_adj.append(torch.stack(train_tensor_adj[i : i + seq_len]))
        train_Y_adj.append(torch.stack(train_tensor_adj[i + seq_len : i + seq_len + pre_len]))

    print('---TRAIN X ADJ---', len(train_X_adj))
    print('---TRAIN X FEAT---', len(train_X_feat))

    #tensor_train_X_adj = torch.stack((train_X_adj))
    #tensor_train_Y_adj = torch.stack((train_X_adj))
    
    #train_X_tot.append(torch.FloatTensor(train_X_feat))
    #train_X_tot.append(torch.stack(train_X_adj))
    #train_Y_tot.append(torch.FloatTensor(train_Y_feat))
    #train_Y_tot.append(torch.stack(train_Y_adj))
        
    for i in range(0, test_data_feat.shape[0] - seq_len - pre_len):
        test_X_feat.append(np.array(test_data_feat[i : i + seq_len]))
        test_Y_feat.append(np.array(test_data_feat[i + seq_len : i + seq_len + pre_len]))

        test_X_adj.append(torch.stack(test_tensor_adj[i : i + seq_len]))
        test_Y_adj.append(torch.stack(test_tensor_adj[i + seq_len : i + seq_len + pre_len]))

    
    #test_X_tot.append(torch.FloatTensor(test_X_feat).unsqueeze(-1))
    #test_X_tot.append(torch.stack(test_X_adj))
    #test_Y_tot.append(torch.FloatTensor(test_Y_feat).unsqueeze(-1))
    #test_Y_tot.append(torch.stack(test_Y_adj))

    #(800, 12, 2322)
    #(800, 12, 2322, 2322)
    
    #print('---NP TOT---', np.array(train_X_tot).shape)
    #exit()
    #return np.array(train_X_tot, dtype=object), np.array(train_Y_tot, dtype=object), np.array(test_X_tot, dtype=object), np.array(test_Y_tot, dtype=object)
    return np.array(train_X_feat), np.array(train_Y_feat), train_X_adj, np.array(test_X_feat), np.array(test_Y_feat), test_X_adj

def generate_torch_datasets(feat, adj, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True):
    train_X_f, train_Y_f, train_a, test_X_f, test_Y_f, test_a = generate_dataset(
        feat,
        adj,
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
    )
    #train_dataset = torch.utils.data.TensorDataset(
        #torch.FloatTensor(train_X) torch.FloatTensor(train_Y))

    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_X_f), torch.stack(train_a).coalesce(), torch.FloatTensor(train_Y_f))

    print('---TRAIN_X_FEAT---', torch.FloatTensor(train_X_f).size())
    print('---TRAIN_X_FEAT---', torch.FloatTensor(train_Y_f).size())
    print('---TRAIN_X_ADJ---', torch.stack(train_a).size())

    print('---TRAIN-DATASET-LEN---', type(train_dataset))
    print('---TRAIN-DATASET-LEN---', len(train_dataset))
    print('----------------')
    
    
    #test_dataset = torch.utils.data.TensorDataset(
        #torch.FloatTensor(test_X), torch.FloatTensor(test_Y))

    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(test_X_f), torch.stack(test_a).coalesce(), torch.FloatTensor(test_Y_f))

    print('---TEST_X_FEAT---', torch.FloatTensor(test_X_f).size())
    print('---TEST_X_FEAT---', torch.FloatTensor(test_Y_f).size())
    print('---TEST-DATASET-LEN---', len(test_dataset))
    print('----------------')
    
    #loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    #print('---LOADER---', loader)
    

    return train_dataset, test_dataset

'''
def generate_adj(adj, seq_len, pre_len):
    split_ratio = 0.8

    time_len = len(adj)
    train_size = int(time_len * split_ratio)
    train_data_adj = adj[:train_size]

    train_tensor_adj = list()
    for i in range(0, len(train_data_adj)):
        tensor = torch.FloatTensor(train_data_adj[i].todense()).to_sparse()
        train_tensor_adj.append(tensor)
        
    train_X_adj = list()
    for i in range(0, len(train_tensor_adj) - seq_len - pre_len):
        train_X_adj.append(torch.stack(train_tensor_adj[i : i + seq_len]))
    
    return train_X_adj
'''


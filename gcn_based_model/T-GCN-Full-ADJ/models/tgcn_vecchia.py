import argparse
import torch
import torch.nn as nn
from utils.graph_conv import calculate_laplacian_with_self_loop
import scipy
import pickle

class TGCNGraphConvolution(nn.Module):
    def __init__(self, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        print('---TGCN-3---')
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 1, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, input_adj, hidden_state):
        #print('---GRAPH-CONV---', inputs.size())
        #print('---GRAPH-CONV---', input_adj.size())
        
        laplacian = input_adj
        batch_size, num_nodes = inputs.shape
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        #print('---GRAPH-CONV---', inputs.size())
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [x, h] (batch_size, num_nodes, num_gru_units + 1)
        #print('---CONCAT-PRE--', hidden_state.size())
        concatenation = torch.cat((inputs, hidden_state), dim=2).unsqueeze(-1)
        #print('---CONCAT---', concatenation.size())
        # [x, h] (num_nodes, num_gru_units + 1, batch_size)
        
        #concatenation = concatenation.transpose(0, 1).transpose(1, 2).unsqueeze(-1)
        
        # [x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        #print('---CONCAT---', concatenation.size())       
        #concatenation = concatenation.reshape(
            #(num_nodes, (self._num_gru_units + 1) * batch_size)
        #)
        
        #print('---CONCAT---', concatenation.size())
        #print('---LAPL---', laplacian.size())
        #exit()
        # [bathc_size, num_nodes, num_nodes] -> [bathc_size, num_nodes, num_nodes, num_gru_units + 1]
        #laplacian_2 = torch.cat((laplacian, hidden_state), dim=3)
        '''
        laplacian_2 = laplacian.unsqueeze(0).repeat((self._num_gru_units + 1), 1, 1, 1)
        print('---LAP-2---', laplacian_2.size())
        
        laplacian_2 = laplacian_2.reshape(((self._num_gru_units + 1) * batch_size), num_nodes, num_nodes)
        print('---LAP-2---', laplacian_2.size())
        exit()
        '''
        #print('---TYPE---', type(laplacian))
        #for i in range(self._num_gru_units):
            #laplacian = torch.cat((laplacian, input_adj))
        #print('---TYPE---', type(laplacian))
        #exit()

        #concatenation = concatenation.reshape(num_nodes, self._num_gru_units, batch_size)
        res = [torch.bmm(laplacian, concatenation[:, :, i, :]) for i in range(self._num_gru_units + 1)]

        '''
        
        pos = 0
        for i in range(batch_size * (self._num_gru_units + 1)):
            #laplacian_3 = laplacian_2[pos, :, :]
            #concatenation_2 = concatenation[:, i]
            if pos < len(laplacian):
                a_times_concat = laplacian[pos] @ concatenation[:, i]
                print('---DENTRO-FOR---', a_times_concat.size())
                #a_times_concat = torch.sparse.mm(laplacian[pos], concatenation[:, i])
                res.append(a_times_concat)
                pos += 1
            else:
                pos = 0
                a_times_concat = laplacian[pos] @ concatenation[:, i]
                #a_times_concat = torch.sparse.mm(laplacian[pos], concatenation[:, i])
                res.append(a_times_concat)
                pos += 1
        '''
        a_times_concat = torch.vstack(res)
        #print('---DOPO-FOR---', a_times_concat.size())
        a_times_concat = a_times_concat.transpose(0,1).reshape(num_nodes, (batch_size * (self._num_gru_units + 1)), 1).transpose(0,1).squeeze()
        #print('---DOPO-FOR---', a_times_concat.size())
        

        #print('---A-CONCAT---', a_times_concat.size())
        #exit()
        # A[x, h] (num_nodes, num_gru_units + 1, batch_size)
        a_times_concat = a_times_concat.reshape(
            (num_nodes, self._num_gru_units + 1, batch_size)
        )
        # A[x, h] (batch_size, num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # A[x, h] (batch_size * num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_gru_units + 1)
        )
        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = a_times_concat @ self.weights + self.biases
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


class TGCNCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        print('---TGCN-2---')
        #self._adj = adj
        #self.register_buffer("adj", adj)
        self.graph_conv1 = TGCNGraphConvolution(
            self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )
        self.graph_conv2 = TGCNGraphConvolution(
            self._hidden_dim, self._hidden_dim
        )

    def forward(self, inputs, input_adj, hidden_state):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        
        #print('---TGCN-CELL---', inputs.shape)
        concatenation = torch.sigmoid(self.graph_conv1(inputs, input_adj, hidden_state))
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.graph_conv2(inputs, input_adj, r * hidden_state))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class TGCN(nn.Module):
    def __init__(self, hidden_dim: int):
        super(TGCN, self).__init__()
        self._input_dim = 2322
        self._hidden_dim = hidden_dim
        print('---TGCN-1---')
        #self._adj = adj
        #self.register_buffer("adj", adj)
        self.tgcn_cell = TGCNCell(input_dim=self._input_dim, hidden_dim=self._hidden_dim)

    def forward(self, inputs, input_adj):
        #print('---INPUT---', inputs[:, 1, :].shape)
        #print('---TGCN-INPUT---', inputs.size())
        #print('---TGCN-ADJ---', input_adj.size())
        #exit()
        #input(f, adj)
        
        #input_adj = input_adj.to_dense()
        #input_adj = torch.unbind(input_adj)
        batch_size, seq_len, num_nodes = inputs.shape
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
            inputs
        )
        output = None
        for i in range(seq_len):
            new_input_adj = list()
            for j in range(len(input_adj)):
                new_input_adj.append(input_adj[j][i])
            
            new_input_adj = torch.stack(new_input_adj)
            #print('---NEW---', new_input_adj.size())
            
            output, hidden_state = self.tgcn_cell(inputs[:, i, :], new_input_adj.to_dense(), hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
        return output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}

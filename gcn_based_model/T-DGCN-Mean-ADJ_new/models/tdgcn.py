import argparse
import torch
import torch.nn as nn
from utils.graph_conv import get_Sign_Magnetic_Laplacian, get_identity
import torch.nn.functional as F



class TDGCNGraphConvolution(nn.Module):
    def __init__(self, adj, num_gru_units: int, output_dim: int, features: int = 1, bias: float = 0.0):
        super(TDGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._features = features
        self._bias_init_value = bias
        self.register_buffer(
            "laplacian", get_Sign_Magnetic_Laplacian(torch.FloatTensor(adj))
        )
        self.register_buffer(
            "identity", get_identity(torch.FloatTensor(adj))
        )
        self.weights = nn.Parameter(
            torch.FloatTensor(2, self._num_gru_units + self._features, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)


     # Possiamo utilizzare  questa funzione per elaborare la parte Tx0=I
    def process(self, mul_L_real, mul_L_imag, weight, X_real, X_imag, num_nodes, features, batch_size):

       #data = torch.spmm(mul_L_real, X_real) sparse matrix
        Tx_0_real_real = torch.matmul(mul_L_real, X_real).reshape( (num_nodes, self._num_gru_units + features, batch_size)).transpose(0, 2).transpose(1, 2).reshape((batch_size * num_nodes, self._num_gru_units + features))
        real_real = torch.matmul(Tx_0_real_real, weight) 
        Tx_0_imag_imag = torch.matmul(mul_L_imag, X_imag).reshape( (num_nodes, self._num_gru_units + features, batch_size)).transpose(0, 2).transpose(1, 2).reshape((batch_size * num_nodes, self._num_gru_units + features))
        imag_imag = torch.matmul(Tx_0_imag_imag, weight) 

        Tx_0_real_imag = torch.matmul(mul_L_imag, X_real).reshape( (num_nodes, self._num_gru_units + features, batch_size)).transpose(0, 2).transpose(1, 2).reshape((batch_size * num_nodes, self._num_gru_units + features)) # L_imag e x_reale --> real_imag
        real_imag = torch.matmul(Tx_0_real_imag, weight)
        Tx_0_imag_real = torch.matmul(mul_L_real, X_imag).reshape( (num_nodes, self._num_gru_units + features, batch_size)).transpose(0, 2).transpose(1, 2).reshape((batch_size * num_nodes, self._num_gru_units + features)) # L_real e x_imag --> imag_real
        imag_real = torch.matmul(Tx_0_imag_real, weight)
        return real_real,imag_imag, imag_real, real_imag     
    
    # Possiamo utilizzare  questa funzione per elaborare la parte Tx0=I
    def propagate(self, mul_L_real, mul_L_imag, X_real, X_imag, weight, num_nodes, features, batch_size, real_real,imag_imag, imag_real, real_imag ):
        #data = torch.spmm(mul_L_real, X_real) sparse matrix
        Tx_0_real_real = torch.matmul(mul_L_real, X_real).reshape( (num_nodes, self._num_gru_units + features, batch_size)).transpose(0, 2).transpose(1, 2).reshape((batch_size * num_nodes, self._num_gru_units + features))
        real_real = real_real+ torch.matmul(Tx_0_real_real, weight)
        Tx_0_imag_imag = torch.matmul(mul_L_imag, X_imag).reshape( (num_nodes, self._num_gru_units + features, batch_size)).transpose(0, 2).transpose(1, 2).reshape((batch_size * num_nodes, self._num_gru_units + features))
        imag_imag = imag_imag + torch.matmul(Tx_0_imag_imag, weight)
        Tx_0_real_imag = torch.matmul(mul_L_imag, X_real).reshape( (num_nodes, self._num_gru_units + features, batch_size)).transpose(0, 2).transpose(1, 2).reshape((batch_size * num_nodes, self._num_gru_units + features))
        real_imag = real_imag + torch.matmul(Tx_0_real_imag, weight)
        Tx_0_imag_real = torch.matmul(mul_L_real, X_imag).reshape( (num_nodes, self._num_gru_units + features, batch_size)).transpose(0, 2).transpose(1, 2).reshape((batch_size * num_nodes, self._num_gru_units + features))
        imag_real = imag_real + torch.matmul(Tx_0_imag_real, weight)
        out_real = real_real - imag_imag
        out_imag = imag_real + real_imag
        return out_real, out_imag

    def forward(self, inputs, hidden_state):
        try:
            batch_size, num_nodes, features = inputs.shape
        except:
            batch_size, num_nodes = inputs.shape
            features = 1
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, features)) #1))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [x, h] (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (num_nodes, num_gru_units + 1, batch_size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        concatenation = concatenation.reshape(
            (num_nodes, (self._num_gru_units + features) * batch_size)
        )
        # creating two section operation between real and imag part
        concatenation_real = concatenation
        concatenation_imag = concatenation_real.clone()

        real_real,imag_imag, imag_real, real_imag  = self.process(self.identity, self.identity, self.weights[0],concatenation_real.float(), concatenation_imag.float(), \
        num_nodes, features, batch_size)


        # All operation in one
        out_real, out_imag = self.propagate(self.laplacian.real, self.laplacian.imag, concatenation_real.float(), concatenation_imag.float(), \
        self.weights[1], num_nodes, features, batch_size, real_real, imag_imag, imag_real, real_imag)
        out_real += self.biases
        out_imag += self.biases


        # Reshap 
        # 1) A[x, h]W + b (batch_size, num_nodes, output_dim)
        # 2) A[x, h]W + b (batch_size, num_nodes * output_dim)
        
        out_real = out_real.reshape((batch_size, num_nodes, self._output_dim)).reshape((batch_size, num_nodes * self._output_dim))
        out_imag = out_imag.reshape((batch_size, num_nodes, self._output_dim)).reshape((batch_size, num_nodes * self._output_dim))      

        # Unwind
        outputs = torch.cat((out_real, out_imag), dim = -1)
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


class TDGCNCell(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int):
        super(TDGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.graph_conv1 = TDGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim, #self._hidden_dim, 
            bias=1.0
        )
        self.graph_conv2 = TDGCNGraphConvolution(
            self.adj, self._hidden_dim, int(self._hidden_dim*0.5), #self._hidden_dim
        )
        
    def forward(self, inputs, hidden_state):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class TDGCN(nn.Module):
    def __init__(self, adj, hidden_dim: int, **kwargs):
        super(TDGCN, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.tdgcn_cell = TDGCNCell(self.adj, self._input_dim, self._hidden_dim)
        #self.graph_conv =  GCNGraphConvolution(self.adj, self._hidden_dim)

    def forward(self, inputs):
        #print('---INPUT---', inputs.shape)
        batch_size, seq_len, num_nodes = inputs.shape
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
            inputs
        )
        output = None
        for i in range(seq_len):
            #print(inputs[:, i, :].size())
            #input = self.graph_conv(inputs[:, i, :])
            #input = F.relu(input)
            output, hidden_state = self.tdgcn_cell(inputs[:, i, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))

            #print('---OUTPUT---', output.size())
            #exit()
        return output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}

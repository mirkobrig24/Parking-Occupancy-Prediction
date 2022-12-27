import numpy as np
import torch

def get_identity(matrix):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.eye(matrix.size(0))


def remove_self_loop(matrix):
    dg =  torch.diag(torch.diagonal(matrix, 0))
    matrix = matrix - dg
    return matrix

def get_Sign_Magnetic_Laplacian(matrix):
    r""" Computes our Sign Magnetic Laplacian of the graph given by :obj:`matrix`
    Arg types:
        * **matrix** (PyTorch Tensor) - matrix.
    Return types:
        * **edge_index** (PyTorch LongTensor) - The edge indices of the magnetic Laplacian.
        * **edge_weight.real, edge_weight.imag** (PyTorch Tensor) - Real and imaginary parts of the one-dimensional edge weights for the magnetic Laplacian.
        * **lambda_max** (float, optional) - The maximum eigenvalue of the magnetic Laplacian, only returns this when required by setting return_lambda_max as True.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    matrix = remove_self_loop(matrix)
    #matrix = matrix + torch.eye(matrix.size(0)) # add self-loop
    matrix_sym = 0.5*(matrix + matrix.transpose(0, 1))
    operation = torch.ones(matrix.size(0), matrix.size(0)) - torch.sign(torch.abs(matrix - matrix.transpose(0,1))) + (torch.sign(torch.abs(matrix) - torch.abs(matrix.transpose(0, 1))))*1j
    deg = torch.abs(matrix_sym).sum(1) # out degree
    deg[deg == 0]= 1
    d_inv_sqrt = torch.pow(deg, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix_sym.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    normalized_laplacian = normalized_laplacian.multiply(operation)
    return - normalized_laplacian


        

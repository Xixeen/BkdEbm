import numpy as np
import scipy.sparse as sp
import time
import torch
import scipy
import random
import pdb
import copy
import os
import argparse
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
# from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GINConv,SAGEConv,GATConv,PNAConv, GraphSAGE
from torch_geometric.utils import to_dense_adj
from pygod.utils import load_data
from pygod.metric import eval_roc_auc

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.linears[layer](h)
                if len(h.shape) > 2:
                    h = torch.transpose(h, 0, 1)
                    h = torch.transpose(h, 1, 2)
                # h = self.batch_norms[layer](h)
                if len(h.shape) > 2:
                    h = torch.transpose(h, 1, 2)
                    h = torch.transpose(h, 0, 1)
                h = F.relu(h)
                # h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)


class GNN(nn.Module):
    def __init__(self, in_dim, out_dim, GNN_name="GCN"):
        super(GNN, self).__init__()
        self.mlp0 = MLP(3, in_dim, out_dim, out_dim)
        if GNN_name == "GIN":
            self.linear1 = MLP(4, out_dim, out_dim, out_dim)
            self.graphconv1 = GINConv(self.linear1)
        elif GNN_name == "GCN":
            self.graphconv1 = GCNConv(out_dim, out_dim, aggr='mean')
        elif GNN_name == "GAT":
            self.graphconv1 = GATConv(out_dim, out_dim, aggr='mean')
        elif GNN_name == "SAGE":
            self.graphconv1 = SAGEConv(out_dim, out_dim, aggr='mean')
        self.mlp1 = nn.Linear(out_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self, x, edge_index):
        h0 = self.mlp0(x)
        h1 = self.graphconv1(h0, edge_index)
        h2 = self.mlp1(h1)
        h2 = self.relu(h2)
        p = torch.exp(h2)
        return p


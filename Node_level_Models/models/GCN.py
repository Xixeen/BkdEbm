#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Node_level_Models.helpers.func_utils import accuracy
from copy import deepcopy
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2,device=None,layer_norm_first=False,use_ln=False):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        #self.energy = energy(nhid, nclass)
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid))
        self.lns = nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(nfeat))
        for _ in range(layer-2):
            self.convs.append(GCNConv(nhid,nhid))
            self.lns.append(nn.LayerNorm(nhid))
        self.lns.append(nn.LayerNorm(nhid))
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = dropout
        self.lr = lr
        self.output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None 
        self.weight_decay = weight_decay

        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln

    def forward(self, x, edge_index, edge_weight=None):
        if(self.layer_norm_first):
            x = self.lns[0](x)
        i=0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index,edge_weight))
            if self.use_ln:
                x = self.lns[i+1](x)
            i+=1
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index,edge_weight)
        log_softmax_output = F.log_softmax(x, dim=1)
        return log_softmax_output

    def forward_energy(self, x, edge_index, edge_weight=None):
        if (self.layer_norm_first):
            x = self.lns[0](x)
        i = 0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))
        if self.use_ln:
            x = self.lns[i + 1](x)
        i += 1
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index, edge_weight)
        #p = torch.exp(x)
        p = x.logsumexp(dim=1).detach().cpu().numpy()
        #p=x
        return p

    def get_h(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        
        return x

    # def fit(self, global_model, features, edge_index, edge_weight, labels, idx_train, args, idx_val=None, train_iters=200, verbose=False):
    #     """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.
    #     Parameters
    #     ----------
    #     features :
    #         node features
    #     adj :
    #         the adjacency matrix. The format could be torch.tensor or scipy matrix
    #     labels :
    #         node labels
    #     idx_train :
    #         node training indices
    #     idx_val :
    #         node validation indices. If not given (None), GCN training process will not adpot early stopping
    #     train_iters : int
    #         number of training epochs
    #     initialize : bool
    #         whether to initialize parameters before training
    #     verbose : bool
    #         whether to show verbose logs
    #     """
    #     self.edge_index, self.edge_weight = edge_index, edge_weight
    #     self.features = features.to(self.device)
    #     self.labels = labels.to(self.device)
    #
    #     if idx_val is None:
    #         self._train_without_val(self.labels, idx_train, train_iters, verbose)
    #     else:
    #         loss_train, loss_val, acc_train, acc_val = self._train_with_val(global_model,self.labels, idx_train, idx_val, train_iters, verbose,args)
    #     return  loss_train, loss_val, acc_train, acc_val

    def fit(self, global_model, features, edge_index, edge_weight, aug_edge_index, aug_edge_weight, labels, idx_train,
            args, idx_val=None, train_iters=200, verbose=False):
        """Train the GCN model, when idx_val is not None, pick the best model according to the validation loss.
        Parameters
        ----------
        features : node features
        edge_index : the adjacency matrix of original graph.
        edge_weight : weights of the original graph edges.
        aug_edge_index : the adjacency matrix of augmented graph.
        aug_edge_weight : weights of the augmented graph edges.
        labels : node labels
        idx_train : node training indices
        idx_val : node validation indices. If not given (None), GCN training process will not adopt early stopping
        train_iters : number of training epochs
        verbose : whether to show verbose logs
        """
        self.edge_index, self.edge_weight = edge_index, edge_weight  # Original graph
        self.aug_edge_index, self.aug_edge_weight = aug_edge_index, aug_edge_weight  # Augmented graph
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)

        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            loss_train, loss_val, acc_train, acc_val = self._train_with_val(self, global_model, features, labels, idx_train, idx_val, edge_index, edge_weight,
                        aug_edge_index, aug_edge_weight, train_iters, verbose, args)
        return loss_train, loss_val, acc_train, acc_val

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        batch_count = 0

        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
            batch_count += 1

        self.eval()
        output = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output
        # torch.cuda.empty_cache()

    # def calculate_energies(self, x, edge_index, edge_weight=None):
    #     logits = self.get_logits(x, edge_index, edge_weight)
    #     energies = torch.logsumexp(logits, dim=1)  # 计算并返回能量
    #     return energies

    # def _train_with_val(self,global_model,labels, idx_train, idx_val, train_iters, verbose,args):
    #     if verbose:
    #         print('=== training gcn model ===')
    #     optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    #
    #     best_loss_val = 100
    #     best_acc_val = -10
    #
    #     for i in range(train_iters):
    #         self.train()
    #         optimizer.zero_grad()
    #         output = self.forward(self.features, self.edge_index, self.edge_weight)
    #         loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    #         # if args.agg_method == "FedProx":
    #         #     # compute proximal_term
    #         #     proximal_term = 0.0
    #         #     for w, w_t in zip(self.parameters(), global_model.parameters()):
    #         #         proximal_term += (w - w_t).norm(2)
    #         #     loss_train = loss_train + (args.mu / 2) * proximal_term
    #
    #         loss_train.backward()
    #         optimizer.step()
    #
    #         self.eval()
    #         with torch.no_grad():
    #             output = self.forward(self.features, self.edge_index, self.edge_weight)
    #             loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    #             acc_val = accuracy(output[idx_val], labels[idx_val])
    #             acc_train = accuracy(output[idx_train], labels[idx_train])
    #
    #         if verbose and i % 10 == 0:
    #             print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
    #             print("acc_val: {:.4f}".format(acc_val))
    #         if acc_val > best_acc_val:
    #             best_acc_val = acc_val
    #             self.output = output
    #             weights = deepcopy(self.state_dict())
    #
    #     if verbose:
    #         print('=== picking the best model according to the performance on validation ===')
    #     self.load_state_dict(weights)
    #
    #     return loss_train.item(), loss_val.item(), acc_train, acc_val

    def _train_with_val(self, global_model, features, labels, idx_train, idx_val, edge_index, edge_weight,
                        aug_edge_index, aug_edge_weight, train_iters, verbose):

        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = -10

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(features, edge_index, edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            p_data = self.forward_energy(features, edge_index, edge_weight)
            # Additional loss computation based on augmented graph
            #p_data = output  # original output used for comparison
            shuf_feats = features[:, torch.randperm(features.size(1))]  # shuffle features
            p_neigh = self.forward_energy(shuf_feats, aug_edge_index, aug_edge_weight)  # output from augmented graph
            c_theta_j1 = p_neigh / p_data
            c_theta_j2 = p_data / p_neigh

            j1 = (c_theta_j1 ** 2 + 2 * c_theta_j1).mean()
            j2 = (2 * c_theta_j2).mean()

            neigh_loss = j1 - j2
            total_loss = loss_train + neigh_loss  # combine the losses

            total_loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                output = self.forward(features, edge_index, edge_weight)
                loss_val = F.nll_loss(output[idx_val], labels[idx_val])
                acc_val = accuracy(output[idx_val], labels[idx_val])
                acc_train = accuracy(output[idx_train], labels[idx_train])

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {:.4f}, additional loss: {:.4f}'.format(i, loss_train.item(),
                                                                                        neigh_loss.item()))
                print("acc_val: {:.4f}".format(acc_val))
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

        return loss_train.item(), loss_val.item(), acc_train, acc_val

    def test(self, features, edge_index, edge_weight, labels,idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(features, edge_index, edge_weight)
            acc_test = accuracy(output[idx_test], labels[idx_test])
        # torch.cuda.empty_cache()
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return float(acc_test)
    
    def test_with_correct_nodes(self, features, edge_index, edge_weight, labels,idx_test):
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        correct_nids = (output.argmax(dim=1)[idx_test]==labels[idx_test]).nonzero().flatten()   # return a tensor
        acc_test =accuracy(output[idx_test], labels[idx_test])
        # torch.cuda.empty_cache()
        return acc_test,correct_nids

# %%

import pdb
import  random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Node_level_Models.helpers.func_utils import accuracy
from copy import deepcopy
import copy
from scipy.linalg import svd

def fed_avg(severe_model,local_models,args):
    #selected_models = random.sample(model_list, args.num_selected_models)
    for param_tensor in local_models[0].state_dict():
        avg = (sum(c.state_dict()[param_tensor] for c in local_models)) / len(local_models)
        # Update the global
        severe_model.state_dict()[param_tensor].copy_(avg)
        # Send global to the local
        # for cl in model_list:
        #     cl.state_dict()[param_tensor].copy_(avg)
    return severe_model
def _initialize_global_optimizer(model, args):
    # global optimizer
    if args.glo_optimizer == "SGD":
        # similar as FedAvgM
        global_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.glo_lr,
            momentum=0.9,
            weight_decay=0.0
        )
    elif args.glo_optimizer == "Adam":
        global_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.glo_lr,
            betas=(0.9, 0.999),
            weight_decay=0.0
        )
    else:
        raise ValueError("No such glo_optimizer: {}".format(
            args.glo_optimizer
        ))
    return global_optimizer
def fed_opt(global_model,local_models,args):
    #local_models = random.sample(model_list, args.num_selected_models)
    global_optimizer = _initialize_global_optimizer(
        model=global_model, args=args
    )
    mean_state_dict = {}

    for name, param in global_model.state_dict().items():
        vs = []
        for id,client in enumerate(local_models):
            vs.append(local_models[id].state_dict()[name])
        vs = torch.stack(vs, dim=0)

        try:
            mean_value = vs.mean(dim=0)
        except Exception:
            # for BN's cnt
            mean_value = (1.0 * vs).mean(dim=0).long()
        mean_state_dict[name] = mean_value

    # zero_grad
    global_optimizer.zero_grad()
    global_optimizer_state = global_optimizer.state_dict()

    # new_model
    new_model = copy.deepcopy(global_model)
    new_model.load_state_dict(mean_state_dict, strict=True)

    # set global_model gradient
    with torch.no_grad():
        for param, new_param in zip(
                global_model.parameters(), new_model.parameters()
        ):
            param.grad = param.data - new_param.data

    # replace some non-parameters's state dict
    state_dict = global_model.state_dict()
    for name in dict(global_model.named_parameters()).keys():
        mean_state_dict[name] = state_dict[name]
    global_model.load_state_dict(mean_state_dict, strict=True)

    # optimization
    global_optimizer = _initialize_global_optimizer(
        global_model, args
    )
    global_optimizer.load_state_dict(global_optimizer_state)
    global_optimizer.step()

    return global_model
########################################################################
def init_control(model,device):
    """ a dict type: {name: params}
    """
    control = {
        name: torch.zeros_like(
            p.data
        ).to(device) for name, p in model.state_dict().items()
    }
    return control
def get_delta_model(model0, model1):
    """ return a dict: {name: params}
    """
    state_dict = {}
    for name, param0 in model0.state_dict().items():
        param1 = model1.state_dict()[name]
        state_dict[name] = param0.detach() - param1.detach()
    return state_dict
class ScaffoldOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(
            lr=lr, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def step(self, server_control, client_control, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            for p, c, ci in zip(group['params'], server_control.values(), client_control.values()):
                if p.grad is None:
                    continue
                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp.data * group['lr']

        return loss

def init_random_graph(batch_size, feat_dim):
    """
    针对图数据生成随机特征。
    batch_size: 一批中节点的数量
    feat_dim: 每个节点的特征维度
    返回: 随机初始化的节点特征张量
    """
    return torch.FloatTensor(batch_size, feat_dim).uniform_(-1, 1)

# energy model
class EnergyModel(nn.Module):
    def __init__(self, model):
        super(EnergyModel, self).__init__()
        self.f = model  # 'model' 是一个图神经网络，比如GCN

    def classify(self, x, edge_index, edge_weight):
        """
        利用图神经网络处理图数据，这里x是节点特征，edge_index是图的邻接信息，edge_weight是边的权重
        """
        penult_z = self.f(x, edge_index, edge_weight)  # 在大多数图神经网络中，也需要边的权重
        return penult_z

    def forward(self, x, edge_index, edge_weight, y=None):
        """
        如果不提供标签y，则返回对数概率的最大值和logits；
        如果提供标签y，则返回对应标签的logits。
        """
        logits = self.classify(x, edge_index, edge_weight)
        if y is None:
            return logits.logsumexp(1), logits
        else:
            return torch.gather(logits, 1, y[:, None]), logits


def copy_model_and_optimizer(model, optimizer):
    """
    复制模型和优化器的状态。
    """
    model_state = deepcopy(model.state_dict())  # 复制模型状态
    optimizer_state = deepcopy(optimizer.state_dict())  # 复制优化器状态
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """
    加载模型和优化器的状态。
    """
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)


def update_local(model,server_control, client_control, global_model,
                 features, edge_index, edge_weight, labels, idx_train,
                 args, idx_val=None, train_iters=200):
    glo_model = copy.deepcopy(global_model)
    optimizer = ScaffoldOptimizer(
        model.parameters(),
        lr=args.scal_lr,
        weight_decay=args.weight_decay
    )
    best_loss_val = 100
    best_acc_val = 0

    for i in range(train_iters):
        model.train()

        output = model.forward(features, edge_index, edge_weight)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])

        optimizer.zero_grad()
        loss_train.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(), args.max_grad_norm
        )
        optimizer.step(
            server_control=server_control,
            client_control=client_control
        )

        model.eval()
        with torch.no_grad():
            output = model.forward(features, edge_index, edge_weight)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            acc_train = accuracy(output[idx_train], labels[idx_train])

        if acc_val > best_acc_val:
            best_acc_val = acc_val

            weights = deepcopy(model.state_dict())
            model.load_state_dict(weights)

    delta_model = get_delta_model(glo_model, model)

    local_steps = train_iters

    return delta_model, local_steps,loss_train.item(), loss_val.item(), acc_train, acc_val
def update_local_control(delta_model, server_control,
        client_control, steps, lr):

    new_control = copy.deepcopy(client_control)
    delta_control = copy.deepcopy(client_control)

    for name in delta_model.keys():
        c = server_control[name]
        ci = client_control[name]
        delta = delta_model[name]

        new_ci = ci.data - c.data + delta.data / (steps * lr)
        new_control[name].data = new_ci
        delta_control[name].data = ci.data - new_ci
    return new_control, delta_control
def scaffold(global_model,server_control,client_control,model,
                 features, edge_index, edge_weight, labels, idx_train,
                 args, idx_val=None, train_iters=200):
    # update local with control variates / ScaffoldOptimizer
    delta_model, local_steps,loss_train, loss_val, acc_train, acc_val = update_local(
        model, server_control, client_control, global_model,
        features, edge_index, edge_weight, labels, idx_train,
        args, idx_val=idx_val, train_iters=train_iters
    )

    client_control, delta_control = update_local_control(
        delta_model=delta_model,
        server_control=server_control,
        client_control=client_control,
        steps=local_steps,
        lr=args.lr,
    )
    return loss_train, loss_val, acc_train, acc_val,client_control, delta_control, delta_model
######################################defense ########################
def fed_median(global_model,client_models, args):
    """
    Implementation of median refers to `Byzantine-robust distributed
    learning: Towards optimal statistical rates`
    [Yin et al., 2018]
    (http://proceedings.mlr.press/v80/yin18a/yin18a.pdf)

    It computes the coordinate-wise median of recieved updates from clients

    The code is adapted from https://github.com/alibaba/FederatedScope/blob/master/federatedscope/core/aggregators/median_aggregator.py
    """
    client_parameters = [model.parameters() for model in client_models]
    for global_param, *client_params in zip(global_model.parameters(),
                                            *client_parameters):
        temp = torch.stack(client_params, dim=0)
        temp_pos, _ = torch.median(temp, dim=0)
        temp_neg, _ = torch.median(-temp, dim=0)
        new_temp = (temp_pos - temp_neg) / 2
        global_param.data = new_temp
    return global_model
###################################### fed_trimmedmean ##############################################################
def fed_trimmedmean(global_model,client_models, args):
    """
    Implementation of median refer to `Byzantine-robust distributed
    learning: Towards optimal statistical rates`
    [Yin et al., 2018]
    (http://proceedings.mlr.press/v80/yin18a/yin18a.pdf)

    The code is adapted from https://github.com/alibaba/FederatedScope/blob/master/federatedscope/core/aggregators/trimmedmean_aggregator.py
    """


    client_parameters = [model.parameters() for model in client_models]
    excluded_ratio = args.excluded_ratio
    excluded_num = int(len(client_models) * excluded_ratio)
    for global_param, *client_params in zip(global_model.parameters(),
                                            *client_parameters):
        temp = torch.stack(client_params, dim=0)
        pos_largest, _ = torch.topk(temp, excluded_num, dim=0)
        neg_smallest, _ = torch.topk(-temp, excluded_num, dim=0)
        new_stacked = torch.cat([temp, -pos_largest, neg_smallest], dim=0).sum(dim=0).float()
        new_stacked /= len(temp) - 2 * excluded_num
        global_param.data = new_stacked
    return global_model
###################################### fed_multi_krum ##############################################################
def _calculate_score( models,args):
    """
    Calculate Krum scores
    """
    byzantine_node_num = args.num_mali
    model_num = len(models)
    closest_num = model_num - byzantine_node_num - 2

    distance_matrix = torch.zeros(model_num, model_num)
    for index_a in range(model_num):
        for index_b in range(index_a, model_num):
            if index_a == index_b:
                distance_matrix[index_a, index_b] = float('inf')
            else:
                distance_matrix[index_a, index_b] = distance_matrix[
                    index_b, index_a] = _calculate_distance(
                    models[index_a], models[index_b])

    sorted_distance = torch.sort(distance_matrix)[0]
    krum_scores = torch.sum(sorted_distance[:, :closest_num], axis=-1)
    return krum_scores
def _calculate_distance(model_a, model_b):
    """
    Calculate the Euclidean distance between two given model parameter lists
    """
    distance = 0.0
    #model_a_params,model_b_params = model_a.parameters(), model_b.parameters()
    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        distance += torch.dist(param_a.data, param_b.data, p=2)

    return distance
def fed_multi_krum(global_model, client_models, args):
    federate_ignore_weight = False
    byzantine_node_num = args.num_mali
    client_num = len(client_models)
    agg_num = args.agg_num
    assert 2 * byzantine_node_num + 2 < client_num, \
        "it should be satisfied that 2*byzantine_node_num + 2 < client_num"
    # each_model: (sample_size, model_para)
    #models_para = [model.parameters() for model in client_models]
    krum_scores = _calculate_score(client_models, args)
    index_order = torch.sort(krum_scores)[1].numpy()
    reliable_models = list()
    reliable_client_train_loaders = []
    for number, index in enumerate(index_order):
        if number < agg_num:
            reliable_models.append(client_models[index])


    client_parameters = [model.parameters() for model in reliable_models]
    if  federate_ignore_weight:
        weights = torch.as_tensor([len(train_loader) for train_loader in reliable_client_train_loaders])
        weights = weights / weights.sum()
    else:
        weights = torch.as_tensor([1 for _ in range(len(reliable_models))])
        weights = weights / weights.sum()

    for model_parameter in zip(global_model.parameters(), *client_parameters):
        global_parameter = model_parameter[0]
        client_parameter = [client_parameter.data * weight for client_parameter, weight in
                            zip(model_parameter[1:], weights)]
        client_parameter = torch.stack(client_parameter, dim=0).sum(dim=0)
        global_parameter.data = client_parameter

    return global_model
###################################### fed_bulyan ##############################################################
def fed_bulyan(global_model, client_models, args):
    """
    Implementation of Bulyan refers to `The Hidden Vulnerability
    of Distributed Learning in Byzantium`
    [Mhamdi et al., 2018]
    (http://proceedings.mlr.press/v80/mhamdi18a/mhamdi18a.pdf)

    It combines the MultiKrum aggregator and the treamedmean aggregator
    """

    agg_num = args.agg_num
    byzantine_node_num = args.num_mali
    client_num = len(client_models)
    assert 2 * byzantine_node_num + 2 < client_num, \
        "it should be satisfied that 2*byzantine_node_num + 2 < client_num"
    # assert 4 * byzantine_node_num + 3 <= client_num, \
    #     "it should be satisfied that 4 * byzantine_node_num + 3 <= client_num"

    # models_para = [model.parameters() for model in client_models]


    krum_scores = _calculate_score(client_models, args)
    index_order = torch.sort(krum_scores)[1].numpy()
    reliable_models = []
    #reliable_client_train_loaders = []
    for number, index in enumerate(index_order):
        if number < agg_num:
            reliable_models.append(client_models[index])


    client_parameters = [model.parameters() for model in reliable_models]


    '''
    Sort parameter for each coordinate of the rest \theta reliable
    local models, and find \gamma (gamma<\theta-2*self.byzantine_num)
    parameters closest to the median to perform averaging
    '''
    excluded_num = args.excluded_num

    for global_param, *client_params in zip(global_model.parameters(),
                                            *client_parameters):
        temp = torch.stack(client_params, dim=0)
        pos_largest, _ = torch.topk(temp, excluded_num, dim=0)
        neg_smallest, _ = torch.topk(-temp, excluded_num, dim=0)
        new_stacked = torch.cat([temp, -pos_largest, neg_smallest], dim=0).sum(dim=0).float()
        new_stacked /= len(temp) - 2 * excluded_num
        global_param.data = new_stacked

    return global_model
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul

def pad_sequences(sequences, maxlen=None, dtype='float32', padding='post', truncating='post', value=0.0):
    """将序列填充到同一长度"""
    lengths = [len(seq) for seq in sequences]
    if maxlen is None:
        maxlen = max(lengths)
    sample_shape = np.asarray(sequences[0]).shape[1:]
    padded_sequences = np.full((len(sequences), maxlen) + sample_shape, value, dtype=dtype)
    for idx, seq in enumerate(sequences):
        if len(seq) > maxlen:
            if truncating == 'post':
                truncated = seq[:maxlen]
            elif truncating == 'pre':
                truncated = seq[-maxlen:]
            else:
                raise ValueError(f'Truncating type "{truncating}" not understood')
            padded_sequences[idx, :len(truncated)] = truncated
        else:
            if padding == 'post':
                padded_sequences[idx, :len(seq)] = seq
            elif padding == 'pre':
                padded_sequences[idx, -len(seq):] = seq
            else:
                raise ValueError(f'Padding type "{padding}" not understood')
    return padded_sequences

def compute_similarity_matrix(energies):
    """计算能量之间的相似度矩阵"""
    # 填充能量序列使其长度一致
    padded_energies = pad_sequences(energies)
#    print("Padded energies shape:", padded_energies.shape)
#    print("Padded energies dtype:", padded_energies.dtype)
    similarity_matrix = cosine_similarity(padded_energies)
    return similarity_matrix

def build_edge_index(similarity_matrix, threshold=0.85):
    """根据相似度矩阵和阈值构建边索引"""
    edge_index = []
    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            if i != j and similarity_matrix[i, j] > threshold:
                edge_index.append([i, j])
    if len(edge_index) == 0:
        return torch.empty(2, 0, dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

def energy_propagation(energies, edge_index, prop_layers=1, alpha=0.5):
    '''能量信念传播，返回传播后的能量'''
    energies = pad_sequences(energies)  # 确保能量序列被填充
    e = torch.tensor(energies).float()

    if edge_index.numel() == 0:
        print("Edge index is empty. Skipping propagation.")
        return energies  # 如果 edge_index 为空，直接返回原始能量

    N = e.shape[0]
    row, col = edge_index
    d = degree(col, N).float()
    d_norm = 1. / d[col]
    value = torch.ones_like(row) * d_norm
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    for _ in range(prop_layers):
        e = e * alpha + matmul(adj, e) * (1 - alpha)
    return e.cpu().numpy()


def fed_EnergyBelief(global_model, selected_models, client_energies, args):
    # 计算选定模型之间的相似度矩阵
    similarity_matrix = compute_similarity_matrix(client_energies)

    # 根据相似度矩阵和阈值构建边索引
    edge_index = build_edge_index(similarity_matrix, threshold=args.tau)

    # 计算每个客户端的传播权重（根据edge_index的个数）
    propagation_weights = np.zeros(len(client_energies))
    if edge_index.numel() > 0:
        for i in range(len(client_energies)):
            propagation_weights[i] = (edge_index[1] == i).sum().item()
        propagation_weights /= propagation_weights.sum()  # 归一化传播权重

    # 对选定模型的能量进行能量信念传播
    propagated_energies = energy_propagation(client_energies, edge_index, prop_layers=args.prop_layers,
                                             alpha=args.prop_alpha)

    # 确保传播后的能量是一维数组
    propagated_energies = np.mean(propagated_energies, axis=1)

    # 反转能量值
    inverted_energies = -propagated_energies

    # 根据反转后的能量和传播权重计算最终的聚合权重
    combined_weights = inverted_energies * propagation_weights
    combined_weights = torch.softmax(torch.tensor(combined_weights), dim=0).numpy()

    # 检查是否有客户端与其他所有客户端的相似度都低于阈值，并排除这些客户端
    excluded_clients = set()
    for i, similarities in enumerate(similarity_matrix):
        if np.all(similarities < args.tau):
            excluded_clients.add(i)

    # 确保权重是一维数组，排除恶意客户端
    selected_models_filtered = [model for i, model in enumerate(selected_models) if i not in excluded_clients]
    weights_filtered = [weight for i, weight in enumerate(combined_weights) if i not in excluded_clients]

    if len(selected_models_filtered) != len(weights_filtered):
        raise ValueError("Mismatch between filtered models and weights.")

    # 初始化全局模型参数
    global_params = global_model.state_dict()
    for param_tensor in global_params.keys():
        # 获取所有选定模型的当前参数
        model_params = [model.state_dict()[param_tensor].cpu() for model in selected_models_filtered]

        # 检查所有参数形状是否一致
        shapes = [param.shape for param in model_params]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError(f"Shape mismatch in parameter {param_tensor}: {shapes}")

        # 计算加权平均值
        avg = sum(param * weight for param, weight in zip(model_params, weights_filtered))
        global_params[param_tensor].copy_(avg)

    # 更新全局模型参数
    global_model.load_state_dict(global_params)

    return global_model


def compute_cosine_similarities(local_models):
    num_clients = len(local_models)
    similarities = torch.zeros((num_clients, num_clients))

    client_params = [torch.cat([param.view(-1) for param in model.parameters()]) for model in local_models]

    for i in range(num_clients):
        for j in range(num_clients):
            if i != j:
                similarities[i, j] = torch.nn.functional.cosine_similarity(client_params[i], client_params[j], dim=0)
            else:
                similarities[i, j] = 1.0

    return similarities


def compute_fools_gold_weights(similarities):
    if isinstance(similarities, torch.Tensor):
        similarities = similarities.detach().numpy()

    num_clients = similarities.shape[0]
    weights = np.ones(num_clients)

    max_similarities = np.max(similarities, axis=1)

    for i in range(num_clients):
        if max_similarities[i] == 1.0:
            weights[i] = 0.0
        else:
            weights[i] = 1.0 - max_similarities[i]

    sum_weights = np.sum(weights)
    if sum_weights == 0:
        weights = np.ones(num_clients) / num_clients
    else:
        weights = weights / sum_weights

    return weights


def fools_gold(global_model, local_models, args):
    similarities = compute_cosine_similarities(local_models)
    weights = compute_fools_gold_weights(similarities)

    global_state_dict = global_model.state_dict()
    for name, param in global_state_dict.items():
        aggregated_param = sum(
            weight * local_model.state_dict()[name] for weight, local_model in zip(weights, local_models))
        global_state_dict[name].copy_(aggregated_param)

    global_model.load_state_dict(global_state_dict)
    return global_model


def fed_dnc(global_model, local_models, args,fraction_to_remove=0.1):
    """
    Divide-and-Conquer (DnC) 算法用于联邦学习中检测和移除离群点。

    :param global_model: 全局模型
    :param local_models: 各客户端的本地模型列表
    :param fraction_to_remove: 要移除的最大投影向量的比例
    :return: 更新后的全局模型
    """

    # 收集所有客户端的模型更新
    model_updates = []
    for model in local_models:
        update = []
        for param in model.parameters():
            update.append(param.data.cpu().numpy().flatten())
        model_updates.append(np.concatenate(update))

    # 转换为矩阵形式
    update_matrix = np.stack(model_updates)

    # 对更新矩阵进行奇异值分解
    U, S, Vt = svd(update_matrix, full_matrices=False)

    # 计算每个模型更新在主成分方向上的投影
    projections = np.dot(update_matrix, Vt[0])

    # 计算要移除的更新数量
    num_to_remove = int(fraction_to_remove * len(projections))

    # 识别并移除最大投影的更新
    indices_to_keep = np.argsort(np.abs(projections))[:-num_to_remove]

    # 如果没有剩余的更新，则保留至少一个更新
    if len(indices_to_keep) == 0:
        indices_to_keep = np.argsort(np.abs(projections))[:1]

    filtered_updates = update_matrix[indices_to_keep]

    # 聚合剩余的模型更新
    aggregated_update = np.mean(filtered_updates, axis=0)

    # 更新全局模型
    index = 0
    for param in global_model.parameters():
        length = param.data.numel()
        param.data.copy_(torch.tensor(aggregated_update[index:index + length]).reshape(param.data.shape))
        index += length

    return global_model

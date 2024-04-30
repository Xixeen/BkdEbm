import  random
import torch
import torch.nn as nn
import torch.nn.functional as F
from Node_level_Models.helpers.func_utils import accuracy
from copy import deepcopy
import copy
import energy
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

class Energy(nn.Module):
    """Tent adapts a model by entropy minimization during testing.
        Once tented, a model adapits itself by updating on every forward.
        """

    def __init__(self, model, optimizer, steps=1, episodic=False,
                 buffer_size=10000, sgld_steps=20, sgld_lr=1, sgld_std=0.01, reinit_freq=0.05, feat_dim=50,
                 device='cpu', path=None, logger=None):
        super().__init__()
        self.energy_model = EnergyModel(model)
        self.replay_buffer = init_random_graph(buffer_size, feat_dim)  # Assuming buffer_size is the number of nodes
        self.replay_buffer_old = deepcopy(self.replay_buffer)
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # 储存模型和优化器的初始状态
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.energy_model, self.optimizer)

        self.sgld_steps = sgld_steps
        self.sgld_lr = sgld_lr
        self.sgld_std = sgld_std
        self.reinit_freq = reinit_freq

        self.feat_dim = feat_dim  # Number of features per node

        self.path = path
        self.logger = logger
        self.device = device

        # Note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.energy_model, self.optimizer)
        # 初始化计数器
        self.forward_calls = 0

    def compute_energy(self, x, edge_index, edge_weight=None):
        """计算给定图数据的能量值。

        参数:
        x - 节点特征矩阵。
        edge_index - 边索引。
        edge_weight - 边权重，如果模型需要的话。

        返回:
        energy - 计算得到的能量值。
        """
        self.energy_model.eval()  # 确保模型处于评估模式
        with torch.no_grad():  # 确保不会计算梯度
            logits = self.energy_model(x, edge_index, edge_weight)
            # 使用负的对数概率来表示能量
            energy = -torch.log_softmax(logits, dim=1).mean()
        return energy
    def forward(self, x, edge_index, if_adapt=True):
        if self.episodic:
            self.reset()

        # 更新forward方法的调用次数
        self.forward_calls += 1
        print(f'Forward method called {self.forward_calls} times')

        if if_adapt:
            for i in range(self.steps):
                outputs = forward_and_adapt_graph(x, edge_index, self.energy_model, self.optimizer,
                                                  self.replay_buffer, self.sgld_steps, self.sgld_lr, self.sgld_std,
                                                  self.reinit_freq, self.feat_dim, x.device)
        else:
            self.energy_model.eval()
            with torch.no_grad():
                outputs = self.energy_model.classify(x, edge_index)

        return outputs

    def reset(self):
        """Reset model and optimizer states for episodic adaptation."""
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.energy_model, self.optimizer,
                                 self.model_state, self.optimizer_state)

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

def init_random_graph(num_nodes, feat_dim):
    """
    初始化图数据的节点特征。
    num_nodes: 图中的节点数量
    feat_dim: 每个节点的特征维度
    返回: 随机初始化的节点特征张量
    """
    return torch.FloatTensor(num_nodes, feat_dim).uniform_(-1, 1)

def sample_p_0_graph(reinit_freq, replay_buffer, num_nodes, feat_dim, device):
    """
    生成或从重放缓冲中抽取图节点的特征。
    reinit_freq: 重新初始化（生成新随机样本）的频率
    replay_buffer: 存储旧样本的缓冲区
    batch_size: 一批中节点的数量
    feat_dim: 节点特征的维度
    device: 将数据移动到指定设备
    返回: 节点特征及其在缓冲区中的索引
    """
    if len(replay_buffer) == 0:
        return init_random_graph(num_nodes, feat_dim).to(device), []
    buffer_size = len(replay_buffer)
    inds = torch.randint(0, buffer_size, (num_nodes,))  # Randomly select indices from the replay buffer
    buffer_samples = replay_buffer[inds]  # Get samples from replay buffer based on indices
    random_samples = init_random_graph(num_nodes, feat_dim)  # Generate random samples
    choose_random = (torch.rand(num_nodes) < reinit_freq).float()[:, None]  # Decide whether to use random samples
    samples = choose_random * random_samples + (1 - choose_random) * buffer_samples  # Combine samples
    return samples.to(device), inds

def sample_q_graph(f, replay_buffer, n_steps, sgld_lr, sgld_std, reinit_freq, num_nodes, feat_dim, device, y=None):
    """
    这个函数现在处理图数据的特性，对节点特征进行SGLD采样。
    """
    n = 0
    f.eval()
    # 获取批量大小
    bs = num_nodes if y is None else y.size(0)
    # 生成初始样本和这些样本的缓冲区索引
    init_sample, buffer_inds = sample_p_0_graph(reinit_freq=reinit_freq, replay_buffer=replay_buffer, batch_size=bs,
                                                feat_dim=feat_dim, device=device)
    init_samples = deepcopy(init_sample)
    x_k = torch.autograd.Variable(init_sample, requires_grad=True)

    # 执行SGLD
    for k in range(n_steps):
        print(f'entering {n}')
        n += 1
        f_prime = torch.autograd.grad(f(x_k, y=y)[0].sum(), [x_k], retain_graph=True)[0]
        x_k.data += sgld_lr * f_prime + sgld_std * torch.randn_like(x_k)
    f.train()
    final_samples = x_k.detach()

    # 更新重放缓冲区
    if len(replay_buffer) > 0:
        replay_buffer[buffer_inds] = final_samples.cpu()

    return final_samples, init_samples.detach()

def forward_and_adapt_graph(x, energy_model, optimizer, replay_buffer, sgld_steps, sgld_lr, sgld_std, reinit_freq, feat_dim, device, if_cond=False, n_classes=10):
    batch_size = x.shape[0]

    if if_cond:
        # 假设条件采样依赖于节点类别，这里我们随机生成一些类别标签
        y = torch.randint(0, n_classes, (batch_size,)).to(device)
        x_fake, _ = sample_q_graph(energy_model, replay_buffer, n_steps=sgld_steps, sgld_lr=sgld_lr, sgld_std=sgld_std, reinit_freq=reinit_freq, num_nodes=batch_size, feat_dim=feat_dim, device=device, y=y)
    else:
        # 无条件采样，不依赖于任何外部条件
        x_fake, _ = sample_q_graph(energy_model, replay_buffer, n_steps=sgld_steps, sgld_lr=sgld_lr, sgld_std=sgld_std, reinit_freq=reinit_freq, num_nodes=batch_size, feat_dim=feat_dim, device=device)

    # 测量能量
    energy_real = energy_model(x)[0].mean()
    energy_fake = energy_model(x_fake)[0].mean()

    # 适应性调整
    loss = (- (energy_real - energy_fake))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    outputs = energy_model.classify(x)

    return outputs

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
import pdb

import numpy as np
import torch
from sklearn.cluster import KMeans
from finch import FINCH

from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from eval import EnergyModel, min_max_normalize


from random import sample
def print_model_shapes(model, model_name):
    print(f"Model: {model_name}")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
def select_models_based_on_energy(client_energies):
    """
    根据能量差异对模型进行聚类，筛选出恶意客户端。

    参数:
    - client_energies: 每个客户端的能量差异列表（假设每个元素是一个一维的能量分布数组）

    返回:
    - selected_models_index: 被选择的模型索引列表
    """

    # 找出最长的能量分布长度
    max_length = max(len(energy) for energy in client_energies)

    # 将能量分布填充到相同的长度
    padded_energies = np.array(
        [np.pad(energy, (0, max_length - len(energy)), 'constant') for energy in client_energies])

    # 使用FINCH进行聚类
    c, num_clust, req_c = FINCH(padded_energies)

    # 获取第一层聚类的标签
    labels = c[:, 0]

    # 获取聚类中心
    cluster_centers = [padded_energies[labels == i].mean(axis=0) for i in range(num_clust[0])]

    # 处理只有一个簇的情况
    if num_clust[0] == 1:
        # 随机选择七个客户端
        selected_models_index = sample(range(len(client_energies)), 7)
    else:
        # 找到能量差异较大的那个簇
        malicious_cluster = np.argmax(np.sum(cluster_centers, axis=1))

        # 选择非恶意客户端的索引
        selected_models_index = [i for i, label in enumerate(labels) if label != malicious_cluster]

    return selected_models_index



# 示例用法
client_energies = [np.random.rand(np.random.randint(5, 10)) for _ in range(10)]
selected_models_index = select_models_based_on_energy(client_energies)
print(f'Selected models index: {selected_models_index}')


def calculate_energy(model, data_x, data_edge_index, data_edge_weight):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    energy_model = EnergyModel(model).to(device)
    energy_model.eval()
    with torch.no_grad():
        _, energies = energy_model(data_x, data_edge_index, data_edge_weight)
        energies = energies.logsumexp(1).cpu().numpy()  # 将能量转换为NumPy数组
        normalized_energies = min_max_normalize(energies)  # 归一化处理
    return normalized_energies


def record_client_energies_js(model_list, client_data, client_poison_x, client_poison_edge_index,
                           client_poison_edge_weights, rs):
    client_energies_1 = []
    client_energies_2 = []
    js_divergences = []

    for client_id, model in enumerate(model_list):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Energy 1 - 使用可能被污染的数据
        if client_id in rs:
            data_x = client_poison_x[client_id].to(device)
            data_edge_index = client_poison_edge_index[client_id].to(device)
            data_edge_weight = client_poison_edge_weights[client_id].to(device) if client_poison_edge_weights[
                                                                                       client_id] is not None else None
        else:
            data_x = client_data[client_id].x.to(device)
            data_edge_index = client_data[client_id].edge_index.to(device)
            data_edge_weight = client_data[client_id].edge_weight.to(device) if 'edge_weight' in client_data[
                client_id] else None
        energies_1 = calculate_energy(model, data_x, data_edge_index, data_edge_weight)
        client_energies_1.append(energies_1)
        # Energy 2 - 使用干净的数据集
        data_x = client_data[client_id].x.to(device)
        data_edge_index = client_data[client_id].edge_index.to(device)
        data_edge_weight = client_data[client_id].edge_weight.to(device) if 'edge_weight' in client_data[
            client_id] else None
        energies_2 = calculate_energy(model, data_x, data_edge_index, data_edge_weight)
        client_energies_2.append(energies_2)
        # 确保能量数组是归一化的概率分布
        p = np.exp(energies_1 - np.max(energies_1))  # 防止数值问题
        p /= p.sum()
        q = np.exp(energies_2 - np.max(energies_2))
        q /= q.sum()
        # 计算JS散度
        js_div = calculate_js_divergence(p, q)
        js_divergences.append(js_div)
    return client_energies_1, client_energies_2, js_divergences

def record_client_energies_ks(model_list, client_data, client_poison_x, client_poison_edge_index,
                           client_poison_edge_weights, rs):
    client_energies_1 = []
    client_energies_2 = []
    energy_differences = []

    for client_id, model in enumerate(model_list):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Energy 1 - 使用可能被污染的数据
        if client_id in rs:
            data_x = client_poison_x[client_id].to(device)
            data_edge_index = client_poison_edge_index[client_id].to(device)
            data_edge_weight = client_poison_edge_weights[client_id].to(device) if client_poison_edge_weights[
                                                                                       client_id] is not None else None
        else:
            data_x = client_data[client_id].x.to(device)
            data_edge_index = client_data[client_id].edge_index.to(device)
            data_edge_weight = client_data[client_id].edge_weight.to(device) if 'edge_weight' in client_data[
                client_id] else None

        energies_1 = calculate_energy(model, data_x, data_edge_index, data_edge_weight)
        client_energies_1.append(energies_1)

        # Energy 2 - 使用干净的数据集
        data_x = client_data[client_id].x.to(device)
        data_edge_index = client_data[client_id].edge_index.to(device)
        data_edge_weight = client_data[client_id].edge_weight.to(device) if 'edge_weight' in client_data[
            client_id] else None

        energies_2 = calculate_energy(model, data_x, data_edge_index, data_edge_weight)
        client_energies_2.append(energies_2)

        # 计算能量分布差异
        # energy_difference = np.mean(np.abs(energies_1 - energies_2))
        # energy_differences.append(energy_difference)

        # 使用KS测试比较两个分布
        statistic, p_value = ks_2samp(energies_1, energies_2)
        energy_difference = 1 - p_value  # 用1减p值得到“差异性”的度量，p值越大，差异越小
        energy_differences.append(energy_difference)

    return client_energies_1, client_energies_2, energy_differences

def rfa(global_model, local_models, args):
    """
    Implementation of RFA (Robust Federated Aggregation)
    Reference: Robust aggregation for federated learning

    Parameters:
    - global_model: the global model to be updated
    - local_models: list of local models from clients
    - args: additional arguments

    Returns:
    - global_model: the updated global model
    """
    local_params = [model.state_dict() for model in local_models]
    global_params = global_model.state_dict()

    # Compute the geometric median for each parameter
    for param_tensor in global_params:
        stacked_params = torch.stack([local_params[i][param_tensor] for i in range(len(local_models))], dim=0)
        #print(f"Stacked parameters for {param_tensor}: {stacked_params.shape}")
        median_tensor = compute_geometric_median(stacked_params)
        #print(f"Median tensor for {param_tensor}: {median_tensor.shape}")
        global_params[param_tensor] = median_tensor
        #print(f"Updated global parameter {param_tensor}: {global_params[param_tensor].shape}")

    # Load the updated parameters into the global model
    global_model.load_state_dict(global_params)
    return global_model


def rtr(global_model, local_models, args):
    """
    Implementation of RTR (Robust Learning Rate)
    Reference: Defending against backdoors in federated learning with robust learning rate

    Parameters:
    - global_model: the global model to be updated
    - local_models: list of local models from clients
    - args: additional arguments

    Returns:
    - global_model: the updated global model
    """
    local_params = [model.state_dict() for model in local_models]
    global_params = global_model.state_dict()

    # Compute the robust learning rate for each parameter
    for param_tensor in global_params:
        stacked_params = torch.stack([local_params[i][param_tensor] for i in range(len(local_models))], dim=0)
        #print(f"Stacked parameters for {param_tensor}: {stacked_params.shape}")

        # Calculate robust learning rate
        median_tensor = compute_geometric_median(stacked_params)
        deviations = torch.norm(stacked_params - median_tensor, dim=tuple(range(1, stacked_params.dim())))
        robust_lr = 1.0 / (1.0 + deviations.mean())

        # Update global parameter with robust learning rate
        avg_update = (stacked_params.mean(dim=0) - global_params[param_tensor]) * robust_lr
        global_params[param_tensor] += avg_update

        #print(f"Updated global parameter {param_tensor}: {global_params[param_tensor].shape}")

    # Load the updated parameters into the global model
    global_model.load_state_dict(global_params)
    return global_model


def crfl(global_model, local_models, args):
    """
    Implementation of CRFL (Certifiably Robust Federated Learning)
    Reference: Crfl: Certifiably robust federated learning against backdoor attacks

    Parameters:
    - global_model: the global model to be updated
    - local_models: list of local models from clients
    - args: additional arguments

    Returns:
    - global_model: the updated global model
    """
    local_params = [model.state_dict() for model in local_models]
    global_params = global_model.state_dict()

    # Initialize the certifiably robust aggregation
    certified_updates = {}

    # Compute the certified update for each parameter
    for param_tensor in global_params:
        stacked_params = torch.stack([local_params[i][param_tensor] for i in range(len(local_models))], dim=0)
        # Calculate the certified robust update
        robust_update = compute_certified_update(stacked_params)
        # Store the certified update
        certified_updates[param_tensor] = robust_update
    # Apply the certified updates to the global model
    for param_tensor in global_params:
        global_params[param_tensor] += certified_updates[param_tensor]

    # Load the updated parameters into the global model
    global_model.load_state_dict(global_params)
    return global_model


def sageflow(global_model, local_models, args):
    """
    Implementation of Sageflow (Robust Federated Learning against Both Stragglers and Adversaries)
    Reference: Sageflow: Robust Federated Learning against Both Stragglers and Adversaries

    Parameters:
    - global_model: the global model to be updated
    - local_models: list of local models from clients
    - args: additional arguments

    Returns:
    - global_model: the updated global model
    """
    local_params = [model.state_dict() for model in local_models]
    global_params = global_model.state_dict()

    # Default values for thresholds
    staleness_threshold = getattr(args, 'staleness_threshold', 0.1)
    entropy_threshold = getattr(args, 'entropy_threshold', 0.5)

    # Initialize the robust aggregation
    aggregated_updates = {}

    # Compute the robust update for each parameter
    for param_tensor in global_params:
        stacked_params = torch.stack([local_params[i][param_tensor] for i in range(len(local_models))], dim=0)

        # Calculate the robust update using staleness-aware grouping, entropy-based filtering, and loss-weighted averaging
        robust_update = compute_robust_update(stacked_params, staleness_threshold, entropy_threshold)

        # Store the robust update
        aggregated_updates[param_tensor] = robust_update

    # Apply the robust updates to the global model
    for param_tensor in global_params:
        global_params[param_tensor] += aggregated_updates[param_tensor]

    # Load the updated parameters into the global model
    global_model.load_state_dict(global_params)
    return global_model


def compute_robust_update(tensor_list, staleness_threshold, entropy_threshold):
    """
    Compute the robust update of a list of tensors.
    This function implements staleness-aware grouping, entropy-based filtering, and loss-weighted averaging.

    Parameters:
    - tensor_list: a tensor of shape (num_tensors, ...)
    - staleness_threshold: the threshold for staleness
    - entropy_threshold: the threshold for entropy

    Returns:
    - robust_update: the robust update of the input tensors
    """
    # Implement staleness-aware grouping
    grouped_tensors = staleness_aware_grouping(tensor_list, staleness_threshold)

    # Apply entropy-based filtering
    filtered_tensors = entropy_based_filtering(grouped_tensors, entropy_threshold)

    # Calculate loss-weighted averaging
    robust_update = loss_weighted_averaging(filtered_tensors)

    return robust_update


def staleness_aware_grouping(tensor_list, staleness_threshold):
    """
    Group tensors based on staleness threshold.

    Parameters:
    - tensor_list: a tensor of shape (num_tensors, ...)
    - staleness_threshold: the threshold for staleness

    Returns:
    - grouped_tensors: the tensors grouped by staleness
    """
    staleness = compute_staleness(tensor_list)
    grouped_tensors = tensor_list[staleness < staleness_threshold]
    return grouped_tensors


def compute_staleness(tensor_list):
    """
    Compute staleness for each tensor.

    Parameters:
    - tensor_list: a tensor of shape (num_tensors, ...)

    Returns:
    - staleness: staleness for each tensor
    """
    # Placeholder: use actual staleness calculation logic
    return torch.randn(tensor_list.size(0))


def entropy_based_filtering(tensor_list, entropy_threshold):
    """
    Filter tensors based on entropy threshold.

    Parameters:
    - tensor_list: a tensor of shape (num_tensors, ...)
    - entropy_threshold: the threshold for entropy

    Returns:
    - filtered_tensors: the tensors filtered by entropy
    """
    entropies = compute_entropy(tensor_list)
    filtered_tensors = tensor_list[entropies < entropy_threshold]
    return filtered_tensors


def compute_entropy(tensor_list):
    """
    Compute entropy for each tensor.

    Parameters:
    - tensor_list: a tensor of shape (num_tensors, ...)

    Returns:
    - entropies: entropy for each tensor
    """
    entropies = -torch.sum(tensor_list * torch.log(tensor_list + 1e-5), dim=1)
    return entropies


def loss_weighted_averaging(tensor_list):
    """
    Perform loss-weighted averaging of the tensors.

    Parameters:
    - tensor_list: a tensor of shape (num_tensors, ...)

    Returns:
    - averaged_tensor: the loss-weighted average of the tensors
    """
    losses = compute_losses(tensor_list)
    weights = 1.0 / (losses + 1e-5)
    weights /= torch.sum(weights)
    weighted_average = torch.sum(tensor_list * weights.view(-1, 1, 1, 1), dim=0)
    return weighted_average


def compute_losses(tensor_list):
    """
    Compute losses for each tensor.

    Parameters:
    - tensor_list: a tensor of shape (num_tensors, ...)

    Returns:
    - losses: losses for each tensor
    """
    # Placeholder: use actual loss calculation logic
    return torch.randn(tensor_list.size(0))
def compute_certified_update(tensor_list):
    """
    Compute the certified update of a list of tensors.
    Reference: Crfl: Certifiably robust federated learning against backdoor attacks

    Parameters:
    - tensor_list: a tensor of shape (num_tensors, ...)

    Returns:
    - certified_update: the certified robust update of the input tensors
    """
    # Compute the geometric median
    median_tensor = tensor_list.median(dim=0)[0]

    # Calculate the deviation of each update from the median
    deviations = torch.norm(tensor_list - median_tensor, dim=tuple(range(1, tensor_list.dim())))

    # Set the deviation threshold (this is a hyperparameter that might need tuning)
    deviation_threshold = deviations.median() + 1.5 * torch.std(deviations)

    # Filter out the updates with large deviations
    filtered_updates = tensor_list[deviations < deviation_threshold]

    # Compute the certified robust update using the filtered updates
    certified_update = filtered_updates.mean(dim=0)

    return certified_update

def compute_geometric_median(tensor_list):
    """
    Compute the geometric median of a list of tensors.
    Reference: https://en.wikipedia.org/wiki/Geometric_median

    Parameters:
    - tensor_list: a tensor of shape (num_tensors, ...)

    Returns:
    - median_tensor: the geometric median of the input tensors
    """
    def aggregate_tensors(tensor_list, weights):
        """
        Aggregate the tensors using given weights.
        """
        return torch.sum(weights.view(-1, *([1] * (tensor_list.dim() - 1))) * tensor_list, dim=0)

    def compute_weights(distances):
        """
        Compute the weights for each tensor based on distances.
        """
        inverse_distances = 1.0 / (distances + 1e-5)
        return inverse_distances / torch.sum(inverse_distances)

    median_tensor = tensor_list.mean(dim=0)
    for _ in range(10):  # Perform a fixed number of iterations
        distances = torch.norm(tensor_list - median_tensor, dim=tuple(range(1, tensor_list.dim())))
        weights = compute_weights(distances)
        median_tensor = aggregate_tensors(tensor_list, weights)
    return median_tensor.squeeze()  # Ensure the output tensor has the correct shape


import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from eval import EnergyModel, min_max_normalize

def calculate_js_divergence(p, q):
    """
    计算两个分布p和q之间的Jensen-Shannon散度。
    p和q需要是同样长度的概率分布数组。
    """
    return jensenshannon(p, q)**2  # 返回JS散度的平方，即JS距离

def select_models_based_on_energy(energy_differences, threshold):
    # 识别低能量差异的模型（视为非恶意客户端）
    selected_models_index = [i for i, diff in enumerate(energy_differences) if diff < threshold]
    return selected_models_index

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

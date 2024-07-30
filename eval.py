import os

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
from seaborn.external.kde import gaussian_kde
from sklearn.manifold import TSNE

import plotly.graph_objs as go

from Node_level_Models.models.construct import model_construct



class EnergyModel(nn.Module):
    def __init__(self, model):
        super(EnergyModel, self).__init__()
        self.f = model

    def classify(self, x, edge_index, edge_weight=None):
        logits = self.f(x, edge_index, edge_weight)  # 获取 GCN 输出
        return logits

    def forward(self, x, edge_index, edge_weight=None, y=None):
        logits = self.classify(x, edge_index, edge_weight)
        if y is None:
            return logits.logsumexp(1), logits  # 如果需要 logsumexp 处理
        else:
            return torch.gather(logits, 1, y[:, None]), logits  # 使用类标签


def visualize_energies_kde(client_energies, is_malicious, save_path='./visualization/'):
    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, energies in enumerate(client_energies):
        # 计算 Kernel Density Estimate
        kde = gaussian_kde(energies)
        x = np.linspace(min(energies), max(energies), 1000)
        density = kde(x)

        # 创建图像并保存
        plt.figure(figsize=(8, 6))
        plt.plot(x, density, color='red' if is_malicious[i] else 'green')
        plt.fill_between(x, density, alpha=0.5, color='red' if is_malicious[i] else 'green')
        plt.title(f'Energy Distribution for Client {i} {"(Malicious)" if is_malicious[i] else "(Normal)"}')
        plt.xlabel('Energy')
        plt.ylabel('Density')
        plt.grid(True)
        plt.savefig(
            os.path.join(save_path, f'client_{i}_{"malicious" if is_malicious[i] else "normal"}_energy_kde.png'))
        plt.close()

def visualize_energies(energies, client_id, is_malicious=False):
    # 使用matplotlib进行能量分布的直方图
    plt.figure(figsize=(10, 6))
    plt.hist(energies, bins=30, alpha=0.75, color='blue' if not is_malicious else 'red')
    plt.title(f'Energy Distribution for Client {client_id} {"(Malicious)" if is_malicious else "(Normal)"}')
    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    plt.grid(True)

    # 根据客户端类型保存图像
    filename = f"./visualization/client_{client_id}_{'malicious' if is_malicious else 'normal'}_energy_distribution.png"
    plt.savefig(filename)
    plt.close()

def visualize_all_energies_kde_clean(client_energies, client_ids, is_malicious, save_path='./visualization/',
                               figsize=(20, 6)):
    num_clients = len(client_ids)
    plt.figure(figsize=figsize)

    for i in range(num_clients):
        ax = plt.subplot(1, num_clients, i + 1)  # 分配子图
        energies = client_energies[i]

        # 计算 Kernel Density Estimate
        kde = gaussian_kde(energies)
        x = np.linspace(min(energies), max(energies), 1000)
        density = kde(x)

        # 绘制 KDE 曲线
        plt.plot(x, density, color='red' if is_malicious[i] else 'green')
        plt.fill_between(x, density, alpha=0.5, color='red' if is_malicious[i] else 'green')
        plt.title(f'Client {client_ids[i]} {"(Malicious)" if is_malicious[i] else "(Normal)"}')
        plt.xlabel('Energy')
        if i == 0:  # 只在第一个子图显示y轴标签
            plt.ylabel('Density')
        plt.grid(True)

    plt.tight_layout()  # 调整布局
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig("./visualization/combinded_energy_kde_distribution(clean datasets).png")
    plt.show()
    plt.close()

def visualize_all_energies_kde(client_energies, client_ids, is_malicious, save_path='./visualization/',
                               figsize=(20, 6)):
    num_clients = len(client_ids)
    plt.figure(figsize=figsize)

    for i in range(num_clients):
        ax = plt.subplot(1, num_clients, i + 1)  # 分配子图
        energies = client_energies[i]

        # 计算 Kernel Density Estimate
        kde = gaussian_kde(energies)
        x = np.linspace(min(energies), max(energies), 1000)
        density = kde(x)

        # 绘制 KDE 曲线
        plt.plot(x, density, color='red' if is_malicious[i] else 'green')
        plt.fill_between(x, density, alpha=0.5, color='red' if is_malicious[i] else 'green')
        plt.title(f'Client {client_ids[i]} {"(Malicious)" if is_malicious[i] else "(Normal)"}')
        plt.xlabel('Energy')
        if i == 0:  # 只在第一个子图显示y轴标签
            plt.ylabel('Density')
        plt.grid(True)

    plt.tight_layout()  # 调整布局
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig("./visualization/combinded_energy_kde_distribution.png")
    plt.show()
    plt.close()

def visualize_combined_energies_kde_clean(args,client_energies, client_ids, is_malicious, agg_method, poisoning_intensity, save_path='./visualization/', figsize=(10, 6)):
    plt.figure(figsize=figsize)

    for i, energies in enumerate(client_energies):
        # 计算 Kernel Density Estimate
        kde = gaussian_kde(energies)
        x = np.linspace(min(energies), max(energies), 1000)
        density = kde(x)

        # 绘制 KDE 曲线
        plt.plot(x, density, label=f'Client {client_ids[i]} {"(Malicious)" if is_malicious[i] else "(Normal)"}',
                 color='red' if is_malicious[i] else 'green')

    # 更新标题以包含额外信息
    plt.title(f'Combined KDE of All Clients\n Aggregation Method: {agg_method}, Poisoning Intensity: {poisoning_intensity}')
    plt.xlabel('Energy')
    plt.ylabel('Density')
    plt.grid(True)
    plt.legend()  # 显示图例
    plt.tight_layout()  # 调整布局

    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    full_path = os.path.join(save_path, "all_clients_energy_kde_clean_dataset.png")
    plt.savefig(full_path)
    print(f"Saved combined KDE plot to {full_path}")
    plt.show()
    plt.close()


def visualize_combined_energies_kde(args, client_energies, client_ids, is_malicious, agg_method, poisoning_intensity,
                                    save_path='./visualization/', figsize=(10, 6)):
    plt.figure(figsize=figsize)

    # 定义颜色列表，用于恶意客户端
    malicious_colors = ['red', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'yellow']

    # 记录恶意客户端颜色分配
    malicious_color_map = {}

    for i, energies in enumerate(client_energies):
        # 计算 Kernel Density Estimate
        kde = gaussian_kde(energies)
        x = np.linspace(min(energies), max(energies), 1000)
        density = kde(x)

        if is_malicious[i]:
            # 为每个恶意客户端分配不同颜色
            color = malicious_colors[len(malicious_color_map) % len(malicious_colors)]
            malicious_color_map[client_ids[i]] = color
        else:
            color = 'green'

        # 绘制 KDE 曲线
        plt.plot(x, density, label=f'Client {client_ids[i]} {"(Malicious)" if is_malicious[i] else "(Normal)"}',
                 color=color)

    # 更新标题以包含额外信息
    plt.title(
        f'Combined KDE of All Clients\n Seed: {args.seed}, inner_epochs: {args.inner_epochs}, energy_epochs: {args.energy_epochs}\n Poisoning Intensity: {poisoning_intensity}  is_energy: {args.is_energy}')
    plt.xlabel('Energy')
    plt.ylabel('Density')
    plt.grid(True)
    plt.legend()  # 显示图例
    plt.tight_layout()  # 调整布局

    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    full_path = os.path.join(save_path, "all_clients_energy_kde.png")
    plt.savefig(full_path)
    print(f"Saved combined KDE plot to {full_path}")
    plt.show()
    plt.close()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def visualize_combined_energies_kde_exp(args, client_energies, client_ids, is_malicious, agg_method, poisoning_intensity, save_path='./visualization/', figsize=(10, 6)):
    plt.figure(figsize=figsize)

    for i, energies in enumerate(client_energies):
        # 检查并修正数据
        if np.std(energies) < 1e-10:
            energies += np.random.normal(0, 1e-8, size=len(energies))  # 增加噪声幅度

        # 尝试使用不同的带宽方法
        try:
            kde = gaussian_kde(energies, bw_method='silverman')
        except np.linalg.LinAlgError:
            # 如果默认方法失败，尝试增大带宽
            kde = gaussian_kde(energies, bw_method=0.5)  # 增大带宽

        min_energy = np.min(energies)
        max_energy = np.max(energies)
        x = np.linspace(min_energy, max_energy, 1000)
        density = kde(x)
        plt.plot(x, density, label=f'Client {client_ids[i]} {"(Malicious)" if is_malicious[i] else "(Normal)"}', color='red' if is_malicious[i] else 'green')

    plt.title(f'Combined KDE of All Clients\n Seed: {args.seed}, inner_epochs: {args.inner_epochs}, energy_epochs: {args.energy_epochs}, Poisoning Intensity: {poisoning_intensity}')
    plt.xlabel('Energy')
    plt.ylabel('Density')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    full_path = os.path.join(save_path, "all_clients_energy_kde.png")
    plt.savefig(full_path)
    print(f"Saved combined KDE plot to {full_path}")
    plt.show()
    plt.close()


def visualize_all_energies(client_energies, client_ids, is_malicious, figsize=(20, 6)):
    num_clients = len(client_ids)
    plt.figure(figsize=figsize)
    for i in range(num_clients):
        ax = plt.subplot(1, num_clients, i + 1)  # 分配子图
        plt.hist(client_energies[i], bins=30, alpha=0.75, color='red' if is_malicious[i] else 'blue')
        plt.title(f'Client {client_ids[i]} {"(Malicious)" if is_malicious[i] else "(Normal)"}')
        plt.xlabel('Energy')
        if i == 0:  # 只在第一个子图显示y轴标签
            plt.ylabel('Frequency')
        plt.grid(True)

    plt.tight_layout()  # 调整布局
    plt.savefig("./visualization/combined_energy_distribution.png")
    plt.show()
    plt.close()

def visualize_with_tsne(client_energies, is_malicious, figsize=(10, 6), save_path='./visualization_tsne/'):
    # 确保保存路径存在

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 将能量列表转换为一个矩阵
    all_energies = np.vstack(client_energies)

    # 使用TSNE进行降维
    tsne_model = TSNE(n_components=2, random_state=42)
    tsne_results = tsne_model.fit_transform(all_energies)

    # 绘制结果
    plt.figure(figsize=figsize)
    colors = ['green' if not mal else 'red' for mal in is_malicious]  # 正常绿色，恶意红色
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, alpha=0.5)
    plt.title('t-SNE Visualization of Client Energies')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.grid(True)

    # 保存图像
    filename = os.path.join(save_path, 'tsne_visualization.png')
    plt.savefig(filename)
    plt.close()
# def visualize_all_energies_kde(client_energies, client_ids, is_malicious, figsize=(20, 6)):
#     num_clients = len(client_ids)
#     plt.figure(figsize=figsize)
#
#     for i in range(num_clients):
#         ax = plt.subplot(1, num_clients, i + 1)  # 分配子图
#         sns.kdeplot(client_energies[i], fill=True, color='red' if is_malicious[i] else 'blue')
#         plt.title(f'Client {client_ids[i]} {"(Malicious)" if is_malicious[i] else "(Normal)"}')
#         plt.xlabel('Energy')
#         if i == 0:  # 只在第一个子图显示y轴标签
#             plt.ylabel('Density')
#         plt.grid(True)
#
#     plt.tight_layout()  # 调整布局
#     plt.savefig("./visualization_kde/combined_energy_kde_distribution.png")
#     plt.show()
#     plt.close()
def plot_energy_distribution(energies, labels, title,file_name):
    plt.figure(figsize=(10, 5))
    for label in set(labels):
        plt.hist([e for e, l in zip(energies, labels) if l == label], label=f'Client {label}', alpha=0.7, bins=50)
    plt.title(title)
    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(file_name)  # Save the figure to a file
    plt.close()  # Close the plot to free up memory

def min_max_normalize(energies):
    """将能量值归一化到[0, 1]范围."""
    min_energy = np.min(energies)
    max_energy = np.max(energies)
    if max_energy > min_energy:
        normalized_energies = (energies - min_energy) / (max_energy - min_energy)
    else:
        normalized_energies = energies
    return normalized_energies

def energy_distribution(model_saves, energy_instance, data, args, device):
    # 为可视化准备数据
    energy_results = pd.DataFrame()

    print('======================Start Preparing the Models========================================')
    # 从保存的模型状态加载每个客户端的模型，并计算能量
    for client_id, state_dict in model_saves.items():
        # 根据提供的args参数构建模型
        model = model_construct(args, args.model, data, device, args.nclass)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # 使用energy_instance计算能量值
        energy_values = energy_instance.compute_energy(data.x.to(device), data.edge_index.to(device)).detach().cpu().numpy()

        # 记录结果
        client_type = "Malicious" if client_id == "client_0_model" else "Normal"
        for energy in energy_values:
            energy_results = energy_results.append({
                "Energy": energy,
                "Client Type": client_type,
                "Client ID": client_id
            }, ignore_index=True)

    # 可视化结果
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Client Type", y="Energy", hue="Client ID", data=energy_results)
    plt.title("Energy Distribution per Client")
    plt.ylabel("Energy")
    plt.xlabel("Client Type")
    plt.legend(title="Client ID")
    plt.show()
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from Node_level_Models.models.construct import model_construct


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
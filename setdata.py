import torch.optim as optim
from Node_level_Models.models.GCN import GCN


def setup_energy(client_model_state, data, args, nclass, device):
    """
    根据提供的客户端模型状态设置并初始化能量模型。

    参数:
    client_model_state - 客户端的模型状态字典。
    data - 包含特征等信息的数据对象。
    args - 包含各种训练参数的命名空间。
    nclass - 输出类别数。
    device - 训练或测试将要执行的设备。
    use_ln - 是否使用层归一化。
    layer_norm_first - 层归一化的应用顺序。
    """
    # 初始化GCN模型
    if (args.dataset == 'Reddit2'):
        use_ln = True
        layer_norm_first = False
    else:
        use_ln = False
        layer_norm_first = False
    model = GCN(nfeat=data.x.shape[1],
                nhid=args.hidden,
                nclass=nclass,
                dropout=args.dropout,
                lr=args.train_lr,
                weight_decay=args.weight_decay,
                device=device,
                use_ln=use_ln,
                layer_norm_first=layer_norm_first)
    model.load_state_dict(client_model_state)
    model.to(device)

    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)

    # 创建Energy模型实例
    energy = Energy(model, optimizer, steps=1, episodic=False, buffer_size=10000,
                    sgld_steps=20, sgld_lr=1, sgld_std=0.01, reinit_freq=0.05,
                    feat_dim=data.x.shape[1], device=device)

    return energy
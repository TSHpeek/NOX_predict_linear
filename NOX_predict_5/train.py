from torch.utils.data import TensorDataset, DataLoader  # 数据集和数据加载器
from config import *
import torch.nn as nn

# 模型训练
def train(model, train_dataset, test_dataset, lr, epoch_num, batch_size, device):
    print("=" * 60)
    print("开始训练...")
    print("=" * 60)
    # 将模型加载到设备
    model = model.to(device)
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 定义训练误差每轮变化列表
    train_loss_list = []
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # epoch循环
    for epoch in range(epoch_num):
        model.train()
        train_loss_total = 0
        # batch循环
        for batch_idx, (X, y) in enumerate(train_loader):
            # 将数据加载到设备
            X, y = X.to(device), y.to(device)
            # 前向传播
            y_pred = model(X)
            # 计算损失
            mse_loss = nn.MSELoss()  # 先创建损失函数对象
            loss_value = mse_loss(y_pred.squeeze(), y)  # 再调用它计算损失
            train_loss_total += loss_value.item() * X.shape[0]  # 累加损失
            # 反向传播
            loss_value.backward()
            # 更新参数
            optimizer.step()
            # 梯度清零
            optimizer.zero_grad()
        # 计算每轮损失+添加到列表
        this_train_loss = train_loss_total / len(train_dataset)
        train_loss_list.append(this_train_loss)
        print(f"epoch: {epoch+1}/{epoch_num}, train loss: {this_train_loss}")
    # 返回记录每轮误差的列表，用于画损失曲线
    return train_loss_list
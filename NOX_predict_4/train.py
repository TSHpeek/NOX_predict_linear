from torch.utils.data import TensorDataset, DataLoader  # 数据集和数据加载器
from config import *
import torch.nn as nn

# 4. 模型训练
def train(model, train_dataset, test_dataset, lr, epoch_num, batch_size, device):
    #  将模型加载到设备
    model = model.to(device)
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 定义训练误差和测试误差变化列表
    train_loss_list = []

    # 2. 模型训练
    for epoch in range(epoch_num):
        model.train()
        # 2.1 创建DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loss_total = 0
        # 2.2 按批次迭代训练模型
        for batch_idx, (X, y) in enumerate(train_loader):
            # 将数据加载到设备
            X, y = X.to(device), y.to(device)
            # 2.3.1 前向传播
            y_pred = model(X)
            # 2.3.2 计算损失
            # loss_value = log_rmse(y_pred.squeeze(), y)
            mse_loss = nn.MSELoss()  # 先创建损失函数对象
            loss_value = mse_loss(y_pred.squeeze(), y)  # 再调用它计算损失

            # 2.3.3 反向传播
            loss_value.backward()
            # 2.3.4 更新参数
            optimizer.step()
            optimizer.zero_grad()   # 梯度清零

            # 累加损失
            train_loss_total += loss_value.item() * X.shape[0]
        this_train_loss = train_loss_total / len(train_dataset)
        train_loss_list.append(this_train_loss)

        print(f"epoch: {epoch+1}, train loss: {this_train_loss}")
    return train_loss_list
from torch.utils.data import DataLoader  # 数据集和数据加载器
from config import *
import torch.nn as nn
from data_process import create_dataset

def test(model,device):
    # 1. 加载数据
    train_dataset, test_dataset, feature_num = create_dataset()
    print(f"feature_num:{feature_num}")

    # 3. 测试
    test_loss_total = 0

    # ### 新增：创建空列表，用于收集预测值和真实值 ###
    all_preds = []
    all_targets = []

    # 3.1 定义DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    # 3.2 计算测试误差
    with torch.no_grad():  # 测试时关闭梯度计算
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            mse_loss = nn.MSELoss()  # 先创建损失函数对象
            loss_value = mse_loss(y_pred.squeeze(), y)  # 再调用它计算损失
            test_loss_total += loss_value.item() * X.shape[0]

            # ### 新增：将当前批次的预测值和真实值存入列表 ###
            all_preds.extend(y_pred.squeeze().cpu().numpy())
            all_targets.extend(y.cpu().numpy())

        this_test_loss = test_loss_total / len(test_dataset)
        print(f"test loss: {this_test_loss}")
        return all_preds,all_targets
from torch.utils.data import DataLoader  # 数据集和数据加载器
from config import *
import torch.nn as nn
from data_process import create_dataset

def test(test_dataset,model,device):
    print("\n" + "=" * 60)
    print("开始测试...")
    print("=" * 60)
    # 创建空列表，用于收集预测值和真实值
    all_preds = []
    all_targets = []
    # 创建DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    test_loss_total = 0
    # 计算测试误差
    with torch.no_grad():  # 测试时关闭梯度计算
        for X, y in test_loader:
            # 将数据加载到设备
            X, y = X.to(device), y.to(device)
            # 前向传播
            y_pred = model(X)
            # 计算损失
            mse_loss = nn.MSELoss()  # 先创建损失函数对象
            loss_value = mse_loss(y_pred.squeeze(), y) # 再调用它计算损失
            test_loss_total += loss_value.item() * X.shape[0]  # 累加损失

            # 将当前批次的"预测值"和"真实值"存入列表
            all_preds.extend(y_pred.squeeze().cpu().numpy())
            all_targets.extend(y.cpu().numpy())
        #计算该轮的损失
        this_test_loss = test_loss_total / len(test_dataset)
        print(f"test loss(MSE): {this_test_loss}")
        return all_preds,all_targets
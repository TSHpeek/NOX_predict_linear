"""
不用Sequential构建模型,训练集训练完所有轮，测试集才开始测试
1.一个文件写所有逻辑
2.特征工程方法
3.MSE损失函数
"""
import os
import matplotlib.pyplot as plt
import numpy as np

from config import *
from data_process import create_dataset
from model import Model
from test import test
from train import train

if __name__ == "__main__":
    # 1. 加载数据
    train_dataset, test_dataset, feature_num = create_dataset()
    print(f"feature_num:{feature_num}")

    # 2. 定义模型
    model = Model(device=device, feature_num=feature_num)

    # 3. 训练 + 画"损失函数"曲线
    train_loss_list = []
    if TRAIN:
        # 训练并保存模型
        train_loss_list = train(model, train_dataset, test_dataset, lr, epoch_num, batch_size, device)
        os.makedirs("model", exist_ok=True)
        torch.save(model.state_dict(), "model/model.pth")
        print("模型已保存到 model/model.pth")
        # 画"损失函数"曲线
        plt.plot(train_loss_list, 'r-', label='train loss', linewidth=3)
        plt.legend()
        plt.title("train loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.tight_layout()
        plt.savefig("outputs/train_loss.png", dpi=150)
        plt.show()

    # 4. 测试 + 画“预测值”与“真实值”对比图
    all_preds = []
    all_targets = []
    if TEST:
        if not TRAIN:  # 没有训练就导入模型，训练了就用训练的模型
            model_path = "model/model.pth"
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            print(f"已加载模型：{model_path}")
        # 测试
        all_preds, all_targets = test(test_dataset,model, device)

        # ====== 在这里加指标计算 ======
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        mae = np.mean(np.abs(all_preds - all_targets))
        mse = np.mean((all_preds - all_targets) ** 2)
        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
        mape = np.mean(np.abs((all_targets - all_preds) / (all_targets + 1e-8))) * 100

        print(f"MAE:  {mae:.4f}")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")
        # ============================

        # 画“预测值”与“真实值”对比图
        plt.figure(figsize=(12, 6))
        plt.plot(all_targets, 'g-', label='True Values', linewidth=0.7)
        plt.plot(all_preds, 'r-', label='Predicted Values', linewidth=0.7)
        plt.title('prediction vs true values')
        plt.xlabel('Sample Index')
        plt.ylabel('NOX')
        plt.legend()
        plt.tight_layout()
        plt.savefig("outputs/test_pred_vs_true_value.png", dpi=150)
        plt.show()



import torch
import torch.nn as nn
import os
from config import *
from predict_1 import create_dataset, Model  # 把原文件名改成你的文件名

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. 准备测试数据
train_dataset, test_dataset, feature_num = create_dataset()

# 2. 创建相同结构的模型
model = Model(device=device, feature_num=feature_num).to(device)

# 3. 加载训练好的权重
model_path = "model/model.pth"
state_dict = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)

# 4. 切换到评估模式
model.eval()

# 5. 测试测试集
test_loss_total = 0
all_preds = []
all_targets = []

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
mse_loss = nn.MSELoss()

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss_value = mse_loss(y_pred.squeeze(), y)
        test_loss_total += loss_value.item() * X.shape[0]

        all_preds.extend(y_pred.squeeze().cpu().numpy())
        all_targets.extend(y.cpu().numpy())

this_test_loss = test_loss_total / len(test_dataset)
print(f"test loss: {this_test_loss}")
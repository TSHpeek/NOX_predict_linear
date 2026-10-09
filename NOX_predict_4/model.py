import torch.nn as nn
from config import *

# 2. 创建模型
class Model(nn.Module):  #继承父类
    def __init__(self,device='cpu',feature_num=feature_num):
        # 初始化参数
        super().__init__()
        # 定义2个线性层
        self.linear1 = nn.Linear(feature_num,128,device=device)
        nn.init.xavier_uniform_(self.linear1.weight)  # 初始化
        self.bn1 = nn.BatchNorm1d(128,device=device)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(128,1,device=device)

    # 前向传播
    def forward(self,x):
        # Linear -> BatchNorm -> ReLU -> Dropout
        x = self.linear1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
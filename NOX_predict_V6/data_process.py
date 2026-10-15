import pandas as pd
from sklearn.model_selection import train_test_split    # 划分数据集
from sklearn.compose import ColumnTransformer   # 列转换器
from sklearn.pipeline import Pipeline   # 管道操作
from sklearn.impute import SimpleImputer    # 缺省值处理
from sklearn.preprocessing import StandardScaler, OneHotEncoder # 标准化和独热编码
from torch.utils.data import TensorDataset, DataLoader  # 数据集和数据加载器
from config import *

# 创建数据集dataset
def create_dataset():
    print("\n" + "=" * 60)
    print("加载数据并重采样为分钟级...")
    print("=" * 60)
    # 1、获取数据
    df_raw = pd.read_csv(CSV_PATH, encoding='gbk')
    print(f"df_raw.shape:{df_raw.shape}")
    # 把第一列重命名为 datetime 并转换为 Pandas 的时间格式，然后将其设为索引
    df_raw.rename(columns={df_raw.columns[0]: "datetime"}, inplace=True)
    df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
    df_raw.set_index("datetime", inplace=True)
    df_raw = df_raw[SENSOR_COLS].sort_index()
    # 按分钟聚合，计算每分钟内所有秒级数据的平均值;  dropna()直接删除当前 DataFrame（df_minute）中所有包含空值（NaN）的行
    data = df_raw.resample("1min").mean().dropna()
    print(f'df_min.shape:{data.shape}')

    # 2、划分特征和目标
    print("\n" + "=" * 60)
    print("划分特征和目标...")
    print("=" * 60)
    feature = data.drop("#5炉FGD出口NOX浓度(6%氧量)", axis=1)
    print(f"x.shape:{feature.shape}")
    target = data["#5炉FGD出口NOX浓度(6%氧量)"]
    print(f"y.shape:{target.shape}")
    print(f"x列名：{feature.columns.tolist()}")  # Dataframe用.columns.tolist()查看列名
    print(f"y列名：{target.name}")  # series要用.name查看列名

    # 3、划分训练集和测试集
    print("\n" + "=" * 60)
    print("划分训练集和测试集...")
    print("=" * 60)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)
    # 时间序列不能随机打乱，否则每个滑动窗口内的点不再是连续时间。
    split_index = int( len(feature) * 0.95)
    train_feature, test_feature = feature.iloc[:split_index], feature.iloc[split_index:]
    train_target, test_target = target.iloc[:split_index], target.iloc[split_index:]
    print(f"x_train.shape:{train_feature.shape}")
    print(f"y_train.shape:{train_target.shape}")
    print(f"x_test.shape:{test_feature.shape}")
    print(f"y_test.shape:{test_target.shape}")

    # 4、特征处理：填缺失 + 标准化
    imputer = SimpleImputer(strategy="mean")  # 均值填充
    scaler = StandardScaler()  # 标准化归一化器--适用波动大的数据
    # 用"训练集特征"训练归一化器
    train_feature = imputer.fit_transform(train_feature)
    train_feature_scaled = scaler.fit_transform(train_feature)
    # 对"测试集特征"使用归一化器
    test_feature = imputer.transform(test_feature)
    test_feature_scaled = scaler.transform(test_feature)

    # 5、构建Tensor数据集
    train_dataset = TensorDataset(torch.tensor(train_feature_scaled).float(), torch.tensor(train_target.values).float())
    # x_train 已是 ndarray，直接用;# y_train 是 Series，需要 .values
    test_dataset = TensorDataset(torch.tensor(test_feature_scaled).float(), torch.tensor(test_target.values).float())
    # 返回训练集和测试集，以及特征的数量
    return train_dataset, test_dataset, train_feature.shape[1]


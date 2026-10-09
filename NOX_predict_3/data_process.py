import pandas as pd
from sklearn.model_selection import train_test_split    # 划分数据集
from sklearn.compose import ColumnTransformer   # 列转换器
from sklearn.pipeline import Pipeline   # 管道操作
from sklearn.impute import SimpleImputer    # 缺省值处理
from sklearn.preprocessing import StandardScaler, OneHotEncoder # 标准化和独热编码
from torch.utils.data import TensorDataset, DataLoader  # 数据集和数据加载器
from config import *

# 创建数据集
def create_dataset():
    print("=" * 60)
    print("加载数据并重采样为分钟级...")
    print("=" * 60)
    # 获取原始数据
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
    # 划分特征和目标
    X = data.drop("#5炉FGD出口NOX浓度(6%氧量)", axis=1)
    print(f"X.shape:{X.shape}")
    y = data["#5炉FGD出口NOX浓度(6%氧量)"]
    print(f"y.shape:{y.shape}")
    print(f"X列名：{X.columns.tolist()}")  # Dataframe用..columns.tolist()查看列名
    print(f"y列名：{y.name}")  # series要用.name查看列名

    print("=" * 60)
    print("划分训练集和测试集...")
    print("=" * 60)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
    print(f"x_train.shape:{x_train.shape}")
    print(f"y_train.shape:{y_train.shape}")
    print(f"x_test.shape:{x_test.shape}")
    print(f"y_test.shape:{y_test.shape}")

    # 5. 特征工程（特征转换）
    # 5.1 按照特征数据类型划分成数值型和类别型
    numerical_features = X.select_dtypes(exclude=['object']).columns  #数值型列
    # categorical_features = X.select_dtypes(include=['object']).columns # 种类型列
    # 5.2 定义列转换器
    # 5.2.1 数值型特征：用平均值填充缺失项，再进行标准化
    numerical_transformer = Pipeline(
        steps=[
            ('fillna', SimpleImputer(strategy='mean')),
            ('std', StandardScaler())
        ]
    )

    # 5.2.3 组合列转换器
    transformer = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            # ('cat', categorical_transformer, categorical_features)
        ]
    )
    # 5.3 进行特征转换，构建新的列，组成最终的数据集
    x_train = transformer.fit_transform(x_train)
    x_test = transformer.transform(x_test)
    x_train = pd.DataFrame(x_train, columns=transformer.get_feature_names_out())
    x_test = pd.DataFrame(x_test, columns=transformer.get_feature_names_out())
    # 6. 构建Tensor数据集
    train_dataset = TensorDataset(torch.tensor(x_train.values).float(), torch.tensor(y_train.values).float())
    test_dataset = TensorDataset(torch.tensor(x_test.values).float(), torch.tensor(y_test.values).float())
    # 返回训练集和测试集，以及特征的数量
    return train_dataset, test_dataset, x_train.shape[1]
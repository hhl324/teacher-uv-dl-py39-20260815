import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder


def get_digital_data():
    # 1. 加载数据集
    data = pd.read_csv('../data/train.csv')

    # 2. 区分特征和标签
    x = data.drop("label", axis=1)
    y = data["label"]

    # 3. 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 4. 归一化
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 5. 全部转成tensor
    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()
    y_train = torch.tensor(y_train.to_numpy())
    y_test = torch.tensor(y_test.to_numpy())

    return x_train, x_test, y_train, y_test


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_house_data():
    # 1. 加载数据
    data = pd.read_csv('../data/house_prices.csv')
    data.drop("Id", axis=1, inplace=True)
    # 2. 区分特征和标签
    x = data.drop("SalePrice", axis=1)
    y = data["SalePrice"]
    # 3. 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 4. 特征工程
    # 4.1 数值型特征: 先补充缺失值  然后 数值型归一化
    # 4.2 类别型特征: 先补充缺失值 然后 类别型 独热编码

    num_features = x.select_dtypes(exclude="object").columns
    cat_features = x.select_dtypes(include="object").columns

    num_pipline = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),  # 用均值填充缺失的数值
        ("Scalar", StandardScaler()),
    ])

    cat_pipline = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="NaN")),  # 用均值填充缺失的数值
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])


    ctf = ColumnTransformer([
        ("num", num_pipline, num_features),
        ("cat", cat_pipline, cat_features)
    ])

    x_train = ctf.fit_transform(x_train)
    x_test = ctf.transform(x_test)


    # 5. 统一转成tensor返回
    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()

    y_train = torch.tensor(y_train.values).float()
    y_test = torch.tensor(y_test.values).float()
    return x_train, x_test, y_train, y_test



if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_house_data()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

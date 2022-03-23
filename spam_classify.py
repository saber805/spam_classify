import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score

class MLPclassifica(nn.Module):
    def __init__(self):
        super(MLPclassifica, self).__init__()

        self.hidden1 = nn.Sequential(
            nn.Linear(
                in_features=57,
                out_features=30,
                bias=True,
            ),
            nn.ReLU()
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(30, 10),
            nn.ReLU()
        )

        self.classifica = nn.Sequential(
            nn.Linear(10,2),
            nn.Sigmoid()
        )

    def forward(self, x):
        fc1 = self.hidden1(x)
        fc2 = self.hidden2(fc1)
        output = self.classifica(fc2)

        return fc1, fc2, output

filename = "G:\data\spambase.csv"
mlpc = torch.load('spam_model.pkl')  # 保存下来的模型和参数不能在没有类定义时直接使用

spam = pd.read_csv(filename)  # 直接拿全部数据集进行验证
X = spam.iloc[:, 0:57].values
y = spam.spam.values

scales = MinMaxScaler(feature_range=(0, 1))
X = scales.fit_transform(X)

X = torch.from_numpy(X.astype(np.float32))
y = torch.from_numpy(y.astype(np.int64))


_, _, output = mlpc(X)
_, pre_index = torch.max(output, 1)
accuracy = accuracy_score(y, pre_index)
print(accuracy)

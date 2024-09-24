from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from config import COL_2_2ap, COL_2_3ap, idx2mn
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from collections import Counter
ap = 2

# 定义模型
model = RandomForestClassifier(n_estimators=100)

df = pd.read_csv(f'./data/post/{ap}ap.csv')
df_test = pd.read_csv(f'./data/post/test_set_2_{ap}ap.csv')

COL = COL_2_2ap if ap == 2 else COL_2_3ap
COL = list(COL)
df = df[COL]

COL.pop()
df_test = df_test[COL]

df_test_bk = df_test.copy()

std = df.std()
assert std.all() != 0
corr = df.corr()


X = df[df.columns[:-1]]
y = df[df.columns[-1]]


mean = X.mean()
std = X.std()

X = (X - mean) / std

test_X = (df_test - mean) / std

test_X = test_X.values

X = X.values
y = y.values

# 查看原始数据集的类别分布
print(f"原始数据集类别分布: {Counter(y)}")

# Tomek Links 欠采样
tl = TomekLinks()
X_tl, y_tl = tl.fit_resample(X, y)
print(f"Tomek Links 欠采样后数据集类别分布: {Counter(y_tl)}")

# Edited Nearest Neighbours 欠采样
enn = EditedNearestNeighbours()
X_enn, y_enn = enn.fit_resample(X, y)
print(f"ENN 欠采样后数据集类别分布: {Counter(y_enn)}")




model.fit(X_tl, y_tl)
y_pred = model.predict(X)

print(f"训练集准确率: {accuracy_score(y, y_pred)}")
print(f"训练集混淆矩阵:\n{confusion_matrix(y, y_pred)}")
print(f"训练集分类报告:\n{classification_report(y, y_pred)}")

y_pred = model.predict(test_X)

print(y_pred)


df_test_bk['cls'] = y_pred
df_test_bk[['nss', 'mcs']] = df_test_bk['cls'].apply(idx2mn).apply(pd.Series)

df_test_bk.to_csv(f'test_set_2_{ap}ap.csv', index=False)


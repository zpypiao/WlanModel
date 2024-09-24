from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from config import COL_2_2ap, COL_2_3ap
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from collections import Counter


# 定义模型
models = {
    'logistic_regression': LogisticRegression(max_iter=1000, solver='lbfgs'),
    'random_forest': RandomForestClassifier(n_estimators=100),
    'svm': SVC(kernel='linear', C=1.0)
}

df = pd.read_csv('./data/post/3ap.csv')

df = df[COL_2_3ap]

std = df.std()
assert std.all() != 0
corr = df.corr()
# plt.figure(figsize=(12, 12))
# sns.heatmap(corr, annot=True)
# plt.show()

X = df[df.columns[:-1]]
y = df[df.columns[-1]]

mean = X.mean()
std = X.std()

X = (X - mean) / std

X = X.values
y = y.values

# 查看原始数据集的类别分布
print(f"原始数据集类别分布: {Counter(y)}")

# 随机欠采样
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
print(f"随机欠采样后数据集类别分布: {Counter(y_resampled)}")

# Tomek Links 欠采样
tl = TomekLinks()
X_tl, y_tl = tl.fit_resample(X, y)
print(f"Tomek Links 欠采样后数据集类别分布: {Counter(y_tl)}")

# Edited Nearest Neighbours 欠采样
enn = EditedNearestNeighbours()
X_enn, y_enn = enn.fit_resample(X, y)
print(f"ENN 欠采样后数据集类别分布: {Counter(y_enn)}")





X_train, X_test, y_train, y_test = train_test_split(X_tl, y_tl, test_size=0.2, random_state=42)

model_inds = ['logistic_regression', 'random_forest', 'svm']

res = {}

codes = ['tomek', 'enn', '']

for code in codes:
    if code == 'random':
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    elif code == 'tomek':
        X_train, X_test, y_train, y_test = train_test_split(X_tl, y_tl, test_size=0.2, random_state=42)
    elif code == 'enn':
        X_train, X_test, y_train, y_test = train_test_split(X_enn, y_enn, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for ind in model_inds:
        model = models[ind]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"Model: {ind}")
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

        accuracy = cr['accuracy']

        precision = cr['weighted avg']['precision']
        recall = cr['weighted avg']['recall']
        f1 = cr['weighted avg']['f1-score']


        res[ind+'_'+code] = [accuracy, precision, recall, f1]

        title = f'{ind}_{code.upper()}' if code else f'{ind}'

        plt.figure(figsize=(12, 12))
        sns.heatmap(cm, annot=True, fmt='d', color =  'blue', cmap='Blues')
        plt.title(title)
        plt.savefig(f'D:/Desktop/others/2/{title}.png')


res = pd.DataFrame(res, index=['accuracy', 'precision', 'recall', 'f1']).T


res.to_csv('res_3ap_cls.csv')


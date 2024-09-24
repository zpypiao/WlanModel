from utils import revert_std, plot_fig, calculate_error
from models import get_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import COL_1_2ap, COL_1_3ap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


from sklearn.kernel_ridge import KernelRidge

# 读取训练数据
df = pd.read_csv('./data/post/3ap.csv')

df = df[list(COL_1_3ap)]

# 计算标准差和均值，用于标准化
std = df.std()
mean = df.mean()
# 保证标准差不为0
assert std.all() != 0

# 提取标签的均值和标准差
y_mean = mean.iloc[-1]
y_std = std.iloc[-1]

# 标准化
df = (df - mean) / std


# 转化为numpy数组
data = df.values

# 提取特征和标签
X = data[:, :-1]
y = data[:, -1]

# # 将y复原
# y = (y * y_std) + y_mean

# 划分训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_test = revert_std(y_test, y_mean, y_std)

models = ['ols', 'ridge', 'svr', 'krr', 'rf', 'xgb', 'net']

# ind = 'krr'

# model = get_model(ind)

# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)


# y_test = revert_std(y_test, y_mean, y_std)
# y_pred = revert_std(y_pred, y_mean, y_std)

# cdf, mse, mae, mape = calculate_error(y_test, y_pred)

# print(f'CDF:{cdf}')
# print(f'MSE:{mse}')
# print(f'MAE:{mae}')
# print(f'MAPE:{mape}')
# plot_fig(y_test, y_pred, ind)



res = {}

error = {}

pre = {}

for ind in models:
    model = get_model(ind)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = revert_std(y_pred, y_mean, y_std)

    # y_pred = y_pred.astype(int)
    # y_test = y_test.astype(int)
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))






    # 计算各项误差指标
    cdf, mse, mae, mape = calculate_error(y_test, y_pred)

    # 计算绝对误差
    abs_error = np.abs(y_test - y_pred)
    abs_error = np.sort(abs_error)
    error[ind] = abs_error

    pre[ind] = y_pred


    res[ind] = (cdf, mse, mae, mape)
    # plot_fig(y_test, y_pred, ind)
    print(f'ind:{ind}, CDF:{cdf}, MSE:{mse}, MAE:{mae}, MAPE:{mape}')

res = pd.DataFrame(res).T

plt.figure()

for ind in models:
    # plt.plot(error[ind], label=ind)
    if error[ind][0] < 1000:
        sns.kdeplot(error[ind], label=ind.upper())

plt.legend()
plt.title('Error Distribution')
plt.show()


res.to_csv('3ap_res.csv')


pre['true'] = y_test

pre = pd.DataFrame(pre)

pre.to_csv('3ap_pre.csv', index=False)
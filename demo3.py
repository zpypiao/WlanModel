from utils import revert_std, plot_fig, calculate_error
from models import get_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import COL_3_2ap, COL_3_3ap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.kernel_ridge import KernelRidge

ap = 2

# 读取训练数据
df = pd.read_csv(f'./data/post/{ap}ap.csv')

if ap == 2:
    COL = COL_3_2ap
else:
    COL = COL_3_3ap

COL = list(COL)

# 选取特征
df = df[COL]

# 计算标准差和均值，用于标准化
std = df.std()
mean = df.mean()
# 保证标准差不为0
assert std.all() != 0

# 提取标签的均值和标准差
y_mean = mean.values[-2:]
y_std = std.values[-2:]


# 标准化
df = (df - mean) / std

# 转化为numpy数组
data = df.values



# 提取特征和标签
X = data[:, :-2]

y = data[:, -2:]

# # 将y复原
# y = (y * y_std) + y_mean

# 划分训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



back_df = pd.DataFrame(np.concatenate([X_test, y_test], axis=1), columns=df.columns)

back_df = back_df * std + mean


# y_test复原
y_test[:,0] = revert_std(y_test[:,0], y_mean[0], y_std[0])
y_test[:,1] = revert_std(y_test[:,1], y_mean[1], y_std[1])


# 直接以吞吐量为标签
y1_train = y_train[:, 1]
y1_test = y_test[:, 1]
# 以seq-TIME为标签
y2_train = y_train[:, 0]
y2_test = y_test[:, 0]

# # models = ['ols', 'ridge', 'svr', 'krr', 'rf', 'xgb', 'net']
# models = ['ols', 'ridge', 'svr', 'krr', 'rf', 'xgb', 'net']

# res = {}

# res['true'] = y1_test

# error = {}

# pre = {}

# for ind in models:
#     # 吞吐量预测部分    
#     model = get_model(ind)
#     model.fit(X_train, y1_train)
#     y_pred = model.predict(X_test)
#     y1_train = revert_std(y1_train, y_mean[1], y_std[1])
#     y_pred = revert_std(y_pred, y_mean[1], y_std[1])



#     # seq-TIME预测部分
#     model = get_model(ind)
#     model.fit(X_train, y2_train)
#     y_pred_ = model.predict(X_test)
#     y_pred_ = revert_std(y_pred_, y_mean[0], y_std[0])
#     back_df['seq_time_'] = y_pred_
#     back_df['pre'] = back_df['seq_time_'] * back_df['phy'] * (1-back_df['per'])/60
#     y_pred_ = back_df['pre'].values
#     cdf, mse, mae, mape = calculate_error(back_df['throughput'].values, back_df['pre'].values)
#     print(f'ind:{ind}, CDF:{cdf}, MSE:{mse}, MAE:{mae}, MAPE:{mape}')

#     back_df['pree'] = y_pred


#     # 计算各项误差指标
#     cdf, mse, mae, mape = calculate_error(y1_test, y_pred)

#     # 计算绝对误差
#     abs_error = np.abs(y1_test - y_pred)
#     abs_error = np.abs(back_df['throughput'].values - back_df['pre'].values)
#     abs_error = np.sort(abs_error)
#     error[ind] = abs_error

#     pre[ind] = y_pred


#     res[ind] = (cdf, mse, mae, mape)
#     # plot_fig(y_test, y_pred, ind)
#     print(f'ind:{ind}, CDF:{cdf}, MSE:{mse}, MAE:{mae}, MAPE:{mape}')

#     res[ind] = back_df['pre'].values

# # back_df.to_csv('3ap_back.csv', index=False)
# res = pd.DataFrame(res)
# res.to_csv('2ap_pre.csv', index=False)
# assert False

# plt.figure()


# for ind in models:
#     # plt.plot(error[ind], label=ind)
#     if error[ind][-1] < 1000:
#         sns.kdeplot(error[ind], label=ind.upper())

# plt.legend()
# plt.title('Error Distribution(3ap)')
# plt.show()



test = pd.read_csv(f'./data/post/test_set_1_{ap}ap.csv')
COL.pop()
COL.pop()

test = test[list(COL)]

means = mean.iloc[:-2]
stds = std.iloc[:-2]

tst = test.copy()

test = (test - means) / stds

test = test.values


X_test = test

model = get_model('svr')
model.fit(X, y[:,0])
the_y = y[:,0]*y_std[0] + y_mean[0]
y_pred = model.predict(X)
y_pred = revert_std(y_pred, y_mean[0], y_std[0])
cdf, mse, mae, mape = calculate_error(the_y, y_pred)
print(f'ind:svr, CDF:{cdf}, MSE:{mse}, MAE:{mae}, MAPE:{mape}')


y_pred = model.predict(X_test)
y_pred = revert_std(y_pred, y_mean[0], y_std[0])

tst['seq_time_'] = y_pred

tst['thhh'] = tst['seq_time_'] * tst['phy'] * (1-tst['per'])/60

tst.to_csv(f'{ap}ap_test.csv', index=False)
from utils import revert_std, plot_fig, calculate_error, rrf
from models import get_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import COL_11


from sklearn.kernel_ridge import KernelRidge

# 读取训练数据
df = pd.read_csv('./data/post/3ap.csv')

df = df[list(COL_11)]

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


models = ['ols', 'ridge', 'svr', 'krr', 'rf', 'xgb']

inds = ['rf', 'xgb']

res = []
for ind in inds:

    model = get_model(ind)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_pred = revert_std(y_pred, y_mean, y_std)

    res.append(model.get_importances(df.columns[:-1]))

    cdf, mse, mae, mape = calculate_error(y_test, y_pred)

    print(f'cdf: {cdf}, mse: {mse}, mae: {mae}, mape: {mape}')

    # plot_fig(y_test, y_pred, ind)



imp = rrf(res)

error = pd.DataFrame()

print(len(imp))

best = 10000
for i in range(1, len(imp)+1):
    col = [co[0] for co in imp[:i]]
    X, y = df[col].values, df['seq_time'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = get_model('svr')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = revert_std(y_pred, y_mean, y_std)
    y_test = revert_std(y_test, y_mean, y_std)
    cdf, mse, mae, mape = calculate_error(y_test, y_pred)
    error[str(i)] = [cdf, mse, mae, mape]

    if mae < best:
        best = mae
        best_col = col

print(best, best_col)

print(len(best_col))
error = error.T
error.columns = ['cdf', 'mse', 'mae', 'mape']
error.to_csv('error.csv')



# rrf_score = dict(rrf(res))

# print(rrf_score)
# # 生成一个特征表格
# # 将元组转换为字典
# for i in range(len(res)):
#     res[i] = dict(res[i])
# res.append(rrf_score)
# # 将字典转换为DataFrame
# df_imp = pd.DataFrame(res)
# df_imp = df_imp[df.columns[:-1]]
# df_imp = df_imp.T
# df_imp.columns = ['rf_score', 'xgb_score', 'rrf_score']
# df_imp['rf_rank'] = df_imp['rf_score'].rank(axis=0, ascending=False).astype(int)
# df_imp['xgb_rank'] = df_imp['xgb_score'].rank(axis=0, ascending=False).astype(int)
# df_imp['rrf_rank'] = df_imp['rrf_score'].rank(axis=0, ascending=False).astype(int)
# new = ['rf_score', 'rf_rank', 'xgb_score', 'xgb_rank', 'rrf_score', 'rrf_rank']
# df_imp = df_imp[new]

# df_imp = df_imp.sort_values(by='rrf_rank')


# df_imp.to_csv('imp.csv')

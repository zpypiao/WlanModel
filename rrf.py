import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from config import COL_1
from utils import calculate_error

df = pd.read_csv('./data/post/2ap.csv')

df = df[list(COL_1)]

# 计算标准差和均值，用于标准化
std = df.std()
mean = df.mean()
# 保证标准差不为0
assert std.all() != 0

X = df[df.columns[:-1]]
y = df[df.columns[-1]]

# 划分训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


n_iterations = 100       # 迭代次数
lambda_reg = 0.1       # 正则化参数
used_features = set()  # 记录已使用的特征
feature_importances = np.zeros(X_train.shape[1])  # 初始化特征重要性
feature_names = X_train.columns.tolist()



for iteration in range(n_iterations):
    # 训练随机森林
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # 获取特征重要性
    importances = rf.feature_importances_

    # 对新特征施加惩罚
    penalties = np.array([lambda_reg if f not in used_features else 0 for f in feature_names])
    adjusted_importances = importances - penalties

    # 确保特征重要性不为负数
    adjusted_importances[adjusted_importances < 0] = 0

    # 归一化特征重要性
    if adjusted_importances.sum() != 0:
        adjusted_importances /= adjusted_importances.sum()

    # 更新已使用的特征
    used_features.update([f for f, imp in zip(feature_names, adjusted_importances) if imp > 0])

    # 累积特征重要性
    feature_importances += adjusted_importances

    # 输出每次迭代的调整后特征重要性
    print(f"Iteration {iteration + 1}")
    print("Adjusted Feature Importances:")
    for feature, importance in zip(feature_names, adjusted_importances):
        print(f"{feature}: {importance:.4f}")
    print("\n")


# 平均每个特征的重要性
average_importances = feature_importances / n_iterations

# 根据平均特征重要性选择最终的特征集（可设定阈值）
threshold = np.mean(average_importances)
selected_features = [f for f, imp in zip(feature_names, average_importances) if imp >= threshold]

print("Selected Features after Regularization:")
print(selected_features)

# 使用选择的特征重新训练模型
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# 训练最终的随机森林模型
final_model = RandomForestRegressor(n_estimators=100, random_state=42)
final_model.fit(X_train_selected, y_train)

# 评估模型
predictions = final_model.predict(X_test_selected)
cdf, mse, mae = calculate_error(y_test, predictions)

print(f'CDF:{cdf}')
print(f'MSE:{mse}')
print(f'MAE:{mae}')
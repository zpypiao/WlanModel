import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('3ap_pre.csv')

cols = [col for col in df.columns if col != 'true']


target = df['true']

# # 展示所有预测结果在多张子图

# # 创建子图，分成3行3列（7个模型结果）
# fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=False, sharey=True)
# fig.suptitle('Model Predictions vs True Values(3ap)')

# # Flatten the axs array for easier iteration
# axs = axs.flatten()

# # 绘制每个模型的预测结果
# for i, model in enumerate(cols):  # 遍历所有模型（不包含真实值）
#     axs[i].plot(df.index, df['true'], label='True Values', color='red', marker='o')
#     axs[i].plot(df.index, df[model], label=model.upper(), color='blue', linestyle='--', marker='x')
#     axs[i].set_title(f'{model.upper()} model')
#     # axs[i].set_xlabel('Sample Index')
#     axs[i].set_ylabel('seq_time')
#     axs[i].legend()

# # 移除多余的子图框
# for i in range(len(cols), len(axs)):
#     fig.delaxes(axs[i])

# # 调整布局
# plt.tight_layout()
# plt.subplots_adjust(top=0.9)
# plt.show()


# 画CDF图
def plot_cdf(pre, label, the_label):
    errors = np.abs(pre - label)
    sorted_errors = np.sort(errors)
    yvals = np.arange(len(sorted_errors)) / float(len(sorted_errors) - 1)
    plt.plot(sorted_errors, yvals, label=the_label.upper())



for i, model in enumerate(cols):
    plot_cdf(df[model], target, model)

plt.axhline(y=0.9, color='r', linestyle='--', label='90%')
plt.legend()
plt.xlabel('Error')
plt.ylabel('CDF')
plt.title('CDF of Model Predictions vs True Values')
plt.show()
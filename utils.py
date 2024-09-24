import pandas as pd
import numpy as np
import json
import math
import matplotlib.pyplot as plt
from config import PHY
import seaborn as sns

class DataPreprocessor:

    def check_rssi(self, x):
        if type(x) == str:
            x = json.loads(x)
            x = list(filter(lambda k: k >= -82, x))
            return len(x)
        elif type(x) == float and not math.isnan(x):
            return 1
        else:
            return 0
    
    def load_data(self, path):
        df = pd.read_csv(path)
        for i in df.columns:
            if 'rssi' in i:
                df[i] = df[i].apply(self.check_rssi)

        return df
    
def CDF(label:np.ndarray, pred:np.ndarray, p:float=0.9) -> float:
    assert len(label) == len(pred)
    total = len(label)
    errors = np.abs(label - pred)
    errors = sorted(errors)
    # label = label[np.argsort(pred)]
    # cdf = np.arrange(1, total + 1) / total
    
    error_90_percentile = np.percentile(errors, 90)
    return error_90_percentile

def revert_std(data:np.ndarray, mean:np.ndarray, std:np.ndarray) -> np.ndarray:
    return data * std + mean

def plot_importances(importances:np.ndarray, cols:np.ndarray):
    fig, ax = plt.subplots()
    ax.barh(cols, importances)
    plt.show()

def plot_fig(y_test:np.ndarray, y_pred:np.ndarray, ind:str='') -> None:
    plt.plot(y_test, 'r')
    plt.plot(y_pred, 'b')
    plt.title(ind)
    plt.legend(['y_test', 'y_pred'])
    plt.show()


def mse(y_true:np.ndarray, y_pred:np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true:np.ndarray, y_pred:np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))

def phy(x:np.ndarray) -> float:
    return PHY[x[0]-1][x[1]]


def calculate_error(y_true:np.ndarray, y_pred:np.ndarray) -> tuple:
    cdf = CDF(y_true, y_pred)
    mse_ = mse(y_true, y_pred)
    mae_ = mae(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred)/y_true))
    return (cdf, mse_, mae_, mape)

def rrf(imps:list) -> list:
    ranks = []
    for imp in imps:
        rank = {}
        for i, v in enumerate(imp):
            rank[v[0]] = i+1
        ranks.append(rank)
    res = {}
    for rank in ranks:
        for item in rank.items():
            score = 1/(60+item[1])
            if item[0] not in res:
                res[item[0]] = score
            else:
                res[item[0]] += score

    res = sorted(res.items(), key=lambda x: x[1], reverse=True)

    return res

def plot_confuse(data):
    # 使用seaborn绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(data, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix(3ap)')
    plt.show()

def plot_roc():
    pass

if __name__ == '__main__':
    
    print(phy(np.array([2, 11])))
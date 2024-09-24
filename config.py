import optuna

# phy映射表
PHY = (
    (8.6, 7.2, 25.8, 34.4, 51.6, 68.8, 77.4, 86.0, 103.2, 114.7, 129.0, 143.4),
    (17.2, 34.4, 51.6, 68.8, 103.2, 137.6, 154.9, 172.1, 206.5, 229.4, 258.1, 286.8)
)


# 问题1使用的特征与标签

# 问题1，2ap筛选前特征
COL_1 = ('protocol', 'nav', 'eirp', 'ap_from_another_ap_max_ant_rssi_ed', 'ap_from_another_ap_max_ant_rssi_pd',\
'ap_from_another_ap_mean_ant_rssi_nav', 'sta_to_ap_max_ant_rssi_ed', 'sta_to_ap_max_ant_rssi_pd', \
'sta_to_ap_mean_ant_rssi_nav', 'sta_from_ap_max_ant_rssi_ed', 'sta_from_ap_max_ant_rssi_pd', \
'sta_from_ap_mean_ant_rssi_nav', 'sta_to_another_ap_max_ant_rssi_ed', 'sta_to_another_ap_max_ant_rssi_pd', \
'sta_to_another_ap_mean_ant_rssi_nav', 'sta_from_another_ap_max_ant_rssi_ed', 
'sta_from_another_ap_max_ant_rssi_pd', 'sta_from_another_ap_mean_ant_rssi_nav', 'seq_time')
# 问题1，2ap筛选后特征
COL_1_2ap = ['ap_from_another_ap_mean_ant_rssi_nav', 'sta_to_another_ap_mean_ant_rssi_nav', 'eirp', 'sta_to_ap_max_ant_rssi_ed', 'sta_from_another_ap_mean_ant_rssi_nav', 'protocol', 'sta_from_another_ap_max_ant_rssi_pd', 'sta_to_ap_mean_ant_rssi_nav', 'sta_to_ap_max_ant_rssi_pd', 'nav', 'ap_from_another_ap_max_ant_rssi_pd', 'seq_time']

# 问题1，3ap筛选前特征
COL_11 = ('protocol', 'nav', 'eirp', 'ap_from_another_ap1_max_ant_rssi_ed', 'ap_from_another_ap2_max_ant_rssi_ed', \
          'ap_from_another_ap1_max_ant_rssi_pd', 'ap_from_another_ap2_max_ant_rssi_pd', \
'ap_from_another_ap1_mean_ant_rssi_nav', 'ap_from_another_ap2_mean_ant_rssi_nav',\
    'sta_to_ap_max_ant_rssi_ed',\
        'sta_to_ap_max_ant_rssi_pd',\
'sta_to_ap_mean_ant_rssi_nav', \
    'sta_from_ap_max_ant_rssi_ed',\
        'sta_from_ap_max_ant_rssi_pd',\
'sta_from_ap_mean_ant_rssi_nav' ,\
    'sta_to_another_ap1_max_ant_rssi_ed', 'sta_to_another_ap2_max_ant_rssi_ed',\
        'sta_to_another_ap1_max_ant_rssi_pd', 'sta_to_another_ap2_max_ant_rssi_pd',\
'sta_to_another_ap1_mean_ant_rssi_nav', 'sta_to_another_ap2_mean_ant_rssi_nav',\
    'sta_from_another_ap1_max_ant_rssi_ed', 'sta_from_another_ap2_max_ant_rssi_ed',\
'sta_from_another_ap1_max_ant_rssi_pd', 'sta_from_another_ap2_max_ant_rssi_pd',\
    'sta_from_another_ap1_mean_ant_rssi_nav', 'sta_from_another_ap2_mean_ant_rssi_nav', 'seq_time')
# 问题1，3ap筛选后特征
COL_1_3ap = ('ap_from_another_ap2_mean_ant_rssi_nav', 'ap_from_another_ap1_mean_ant_rssi_nav', 'sta_from_ap_max_ant_rssi_ed', 'sta_to_another_ap1_mean_ant_rssi_nav', 'sta_from_another_ap1_max_ant_rssi_pd', 'sta_to_another_ap1_max_ant_rssi_pd', 'sta_to_another_ap2_mean_ant_rssi_nav', 'ap_from_another_ap2_max_ant_rssi_pd', 'protocol', 'sta_to_ap_max_ant_rssi_ed', 'sta_from_another_ap2_mean_ant_rssi_nav', 'sta_from_another_ap1_mean_ant_rssi_nav', 'sta_to_another_ap2_max_ant_rssi_pd', 'sta_to_ap_max_ant_rssi_pd', 'ap_from_another_ap1_max_ant_rssi_pd', 'eirp', 'sta_to_ap_mean_ant_rssi_nav', 'sta_from_another_ap2_max_ant_rssi_pd', 'sta_to_another_ap2_max_ant_rssi_ed', 'ap_from_another_ap2_max_ant_rssi_ed', 'sta_to_another_ap1_max_ant_rssi_ed', 'nav', 'seq_time')


# 问题2使用的特征与标签
COL_2_2ap = ['ap_from_another_ap_mean_ant_rssi_nav', 'sta_to_another_ap_mean_ant_rssi_nav', 'eirp', 'sta_to_ap_max_ant_rssi_ed', 'sta_from_another_ap_mean_ant_rssi_nav', 'protocol', 'sta_from_another_ap_max_ant_rssi_pd', 'sta_to_ap_mean_ant_rssi_nav', 'sta_to_ap_max_ant_rssi_pd', 'nav', 'ap_from_another_ap_max_ant_rssi_pd', 'snr', 'cls']
COL_2_3ap = ['ap_from_another_ap2_mean_ant_rssi_nav', 'ap_from_another_ap1_mean_ant_rssi_nav', 'sta_from_ap_max_ant_rssi_ed', 'sta_to_another_ap1_mean_ant_rssi_nav', 'sta_from_another_ap1_max_ant_rssi_pd', 'sta_to_another_ap1_max_ant_rssi_pd', 'sta_to_another_ap2_mean_ant_rssi_nav', 'ap_from_another_ap2_max_ant_rssi_pd', 'protocol', 'sta_to_ap_max_ant_rssi_ed', 'sta_from_another_ap2_mean_ant_rssi_nav', 'sta_from_another_ap1_mean_ant_rssi_nav', 'sta_to_another_ap2_max_ant_rssi_pd', 'sta_to_ap_max_ant_rssi_pd', 'ap_from_another_ap1_max_ant_rssi_pd', 'eirp', 'sta_to_ap_mean_ant_rssi_nav', 'sta_from_another_ap2_max_ant_rssi_pd', 'sta_to_another_ap2_max_ant_rssi_ed', 'ap_from_another_ap2_max_ant_rssi_ed', 'sta_to_another_ap1_max_ant_rssi_ed', 'nav', 'snr', 'cls']

# 问题3使用的特征与标签
COL_3_2ap = ['ap_from_another_ap_mean_ant_rssi_nav', 'sta_to_another_ap_mean_ant_rssi_nav', 'eirp', 'sta_to_ap_max_ant_rssi_ed', 'sta_from_another_ap_mean_ant_rssi_nav', 'protocol', 'sta_from_another_ap_max_ant_rssi_pd', 'sta_to_ap_mean_ant_rssi_nav', 'sta_to_ap_max_ant_rssi_pd', 'nav', 'ap_from_another_ap_max_ant_rssi_pd', 'snr', 'phy', 'per', 'seq_time', 'throughput']
COL_3_3ap =('ap_from_another_ap2_mean_ant_rssi_nav', 'ap_from_another_ap1_mean_ant_rssi_nav', 'sta_from_ap_max_ant_rssi_ed', 'sta_to_another_ap1_mean_ant_rssi_nav', 'sta_from_another_ap1_max_ant_rssi_pd', 'sta_to_another_ap1_max_ant_rssi_pd', 'sta_to_another_ap2_mean_ant_rssi_nav', 'ap_from_another_ap2_max_ant_rssi_pd', 'protocol', 'sta_to_ap_max_ant_rssi_ed', 'sta_from_another_ap2_mean_ant_rssi_nav', 'sta_from_another_ap1_mean_ant_rssi_nav', 'sta_to_another_ap2_max_ant_rssi_pd', 'sta_to_ap_max_ant_rssi_pd', 'ap_from_another_ap1_max_ant_rssi_pd', 'eirp', 'sta_to_ap_mean_ant_rssi_nav', 'sta_from_another_ap2_max_ant_rssi_pd', 'sta_to_another_ap2_max_ant_rssi_ed', 'ap_from_another_ap2_max_ant_rssi_ed', 'sta_to_another_ap1_max_ant_rssi_ed', 'nav', 'snr', 'phy', 'per', 'seq_time', 'throughput')


# protocal的映射
PROTOCOL = {
    'tcp': 0,
    'udp': 1
}

# mcs与nss的映射
def mn2idx(x) -> int:
    n = x.iloc[0]
    m = x.iloc[1]
    assert n in [1, 2]
    assert m in range(12)
    MN2IDX = []
    MN = []
    for nss in range(1, 3):
        for mcs in range(12):
            MN.append((nss, mcs))
    for i, (nss, mcs) in enumerate(MN):
        MN2IDX.append(((nss, mcs), i))
    MN2IDX = dict(MN2IDX)
    return MN2IDX[(n, m)]

def idx2mn(idx) -> tuple:
    MN = []
    for nss in range(1, 3):
        for mcs in range(12):
            MN.append((nss, mcs))
    return MN[idx]

# 模型默认参数
DEFAULT_PARAMS = {
    'xgb':{
        'n_estimators': 1000,
    },
    'rf':{
        'n_estimators': 100
    },
    'ridge':{
        'alpha': 1.0
    },
    'net':{
        'input_size': 14,
        'hidden_size': 64,
        'epochs': 100,
        'num_classes': 1
    },
    'krr':{
        'alpha': 1.0
    },
    'svr':{
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale'
    },
    'ols':{}
}

def xgb_search(trial):
    objective = trial.suggest_categorical('objective', ['reg:squarederror', 'reg:logistic'])
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    reg_alpha = trial.suggest_float('reg_alpha', 0.0, 1.0)
    reg_lambda = trial.suggest_float('reg_lambda', 0.0, 1.0)
    n_jobs = -1
    return {
        'objective': objective,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'n_jobs': n_jobs
    }

def rf_search(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    n_jobs = -1
    return {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'n_jobs': n_jobs
    }

def ridge_search(trial):
    alpha = trial.suggest_float('alpha', 0.0, 1.0)
    return {
        'alpha': alpha
    }

def net_search(trial):
    input_size = 18
    hidden_size = trial.suggest_int('hidden_size', 10, 1000)
    epochs = trial.suggest_int('epochs', 100, 10000)
    optimizer = trial.suggest_categorical('optimizer', ['RMSprop', 'Adam', 'SGD'])
    lr = trial.suggest_float('lr', 0.001, 0.01)
    return {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'epochs': epochs,
        'optimizer': optimizer,
        'lr': lr
    }

def krr_search(trial):
    alpha = trial.suggest_float('alpha', 0.0, 1.0)
    return {
        'alpha': alpha
    }

def svr_search(trial):
    kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid'])
    C = trial.suggest_float('C', 0.1, 10.0)
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    return {
        'kernel': kernel,
        'C': C,
        'gamma': gamma
    }

def ols_search(trial):
    return {}


def get_search_func(model_name):
    if model_name == 'xgb':
        return xgb_search
    elif model_name == 'rf':
        return rf_search
    elif model_name == 'ridge':
        return ridge_search
    elif model_name == 'net':
        return net_search
    elif model_name == 'krr':
        return krr_search
    elif model_name == 'svr':
        return svr_search
    elif model_name == 'ols':
        return ols_search







if __name__ == '__main__':
    print(mn2idx(1, 0))
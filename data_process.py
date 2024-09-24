import numpy as np
import pandas as pd
import math
import os
from functools import partial
import json
from config import PHY, PROTOCOL, mn2idx
from typing import Tuple

# 计算信噪比
def snr(p_inf:list, p_sig:np.ndarray, p_noise:float=10 ** (-174 / 10)) -> float:
    p_infer = 0.0
    for each in p_inf:
        k = 10 ** (each / 10) / 1000
        k = np.mean(k)
        p_infer += k

    p_sig = np.mean(10 ** (p_sig / 10) / 1000)

    snr_val = p_sig / (p_infer + p_noise)

    return 10 * np.log10(snr_val)

# 计算信噪比的df版本
def snr_df(x:pd.Series) -> float:
    p_inf = [np.array(json.loads(x.iloc[i])) for i in range(1, x.shape[0])]
    p_sig = np.array(json.loads(x.iloc[0]))
    return snr(p_inf, p_sig)

# 对rssi进行计数
def check_rssi(x, flag:int=-82) -> int:
    if type(x) == str:

        x = json.loads(x)
        x = list(filter(lambda k: k >= flag, x))
        return len(x)
    elif type(x) == float and not math.isnan(x):
        return 1
    else:
        return 0


# 加载数据
def load_data(ap_scenario:str) -> pd.DataFrame:
    # 针对2ap的情况
    if ap_scenario == '2ap':
        # 读取所有的文件
        all_files = os.listdir('data')
        # 选取2ap的文件
        files = [file for file in all_files if '2ap_' in file]
        # 读取2ap文件
        dfs = [pd.read_csv(f'data/{file}') for file in files]
        # 对文件进行处理
        dfs = [merge_data(df, '2ap') for df in dfs]
        # 合并所有的文件
        df = pd.concat(dfs, ignore_index=True)
        # 对protocol进行编码
        df['protocol'] = df['protocol'].apply(lambda x: PROTOCOL[x])
        return df
    
    # 针对3ap的情况
    elif ap_scenario == '3ap':
        all_files = os.listdir('data')
        files = [file for file in all_files if '3ap_' in file]
        dfs = [pd.read_csv(f'data/{file}') for file in files]
        dfs = [df.drop(columns=['predict throughput', 'error%']) if 'predict throughput' in df.columns else df for df in dfs]
        dfs = [merge_data(df, '3ap') for df in dfs]
        df = pd.concat(dfs, ignore_index=True)
        # 对protocol进行编码
        df['protocol'] = df['protocol'].apply(lambda x: PROTOCOL[x])
        return df
    else:
        raise ValueError('Invalid ap_scenario')


# 加载数据
def load_test_data(ap_scenario:int, sett:int) -> Tuple[pd.DataFrame, str]:
    paths = [
        ['data/test_set_1_2ap.csv', 'data/test_set_2_2ap.csv'],
        ['data/test_set_1_3ap.csv', 'data/test_set_2_3ap.csv']
    ]
    path = paths[ap_scenario-2][sett-1]
    
    df = pd.read_csv(path)
    df = merge_data(df, f'{ap_scenario}ap')
    df['protocol'] = df['protocol'].apply(lambda x: PROTOCOL[x])
    name = path.split('/')[-1]
    return (df, name)
    
def merge_data(df:pd.DataFrame, ap_scenario:str='2ap') -> pd.DataFrame:
    if ap_scenario == '2ap':

        # 将ap_from_ap的特征合并
        df['ap_from_another_ap_sum_ant_rssi'] = df['ap_from_ap_0_sum_ant_rssi'].combine_first(df['ap_from_ap_1_sum_ant_rssi'])
        df['ap_from_another_ap_max_ant_rssi'] = df['ap_from_ap_0_max_ant_rssi'].combine_first(df['ap_from_ap_1_max_ant_rssi'])
        df['ap_from_another_ap_mean_ant_rssi'] = df['ap_from_ap_0_mean_ant_rssi'].combine_first(df['ap_from_ap_1_mean_ant_rssi'])

        # 将sta_from_sta的特征合并
        df['sta_from_another_sta_rssi'] = df['sta_from_sta_0_rssi'].combine_first(df['sta_from_sta_1_rssi'])

        # 将sta_to/from_ap_0/1的特征重定向为to/from_ap/to/from_another_ap
        df['sta_to_ap_sum_ant_rssi'] = df[['sta_to_ap_0_sum_ant_rssi', 'sta_to_ap_1_sum_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[0] if x.iloc[2] == 'sta_0' else x.iloc[1], axis=1)
        df['sta_to_ap_max_ant_rssi'] = df[['sta_to_ap_0_max_ant_rssi', 'sta_to_ap_1_max_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[0] if x.iloc[2] == 'sta_0' else x.iloc[1], axis=1)
        df['sta_to_ap_mean_ant_rssi'] = df[['sta_to_ap_0_mean_ant_rssi', 'sta_to_ap_1_mean_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[0] if x.iloc[2] == 'sta_0' else x.iloc[1], axis=1)
        df['sta_from_ap_sum_ant_rssi'] = df[['sta_from_ap_0_sum_ant_rssi', 'sta_from_ap_1_sum_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[0] if x.iloc[2] == 'sta_0' else x.iloc[1], axis=1)
        df['sta_from_ap_max_ant_rssi'] = df[['sta_from_ap_0_max_ant_rssi', 'sta_from_ap_1_max_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[0] if x.iloc[2] == 'sta_0' else x.iloc[1], axis=1)
        df['sta_from_ap_mean_ant_rssi'] = df[['sta_from_ap_0_mean_ant_rssi', 'sta_from_ap_1_mean_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[0] if x.iloc[2] == 'sta_0' else x.iloc[1], axis=1)
        df['sta_to_another_ap_sum_ant_rssi'] = df[['sta_to_ap_0_sum_ant_rssi', 'sta_to_ap_1_sum_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[0] if x.iloc[2] == 'sta_1' else x.iloc[1], axis=1)
        df['sta_to_another_ap_max_ant_rssi'] = df[['sta_to_ap_0_max_ant_rssi', 'sta_to_ap_1_max_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[0] if x.iloc[2] == 'sta_1' else x.iloc[1], axis=1)
        df['sta_to_another_ap_mean_ant_rssi'] = df[['sta_to_ap_0_mean_ant_rssi', 'sta_to_ap_1_mean_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[0] if x.iloc[2] == 'sta_1' else x.iloc[1], axis=1)
        df['sta_from_another_ap_sum_ant_rssi'] = df[['sta_from_ap_0_sum_ant_rssi', 'sta_from_ap_1_sum_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[0] if x.iloc[2] == 'sta_1' else x.iloc[1], axis=1)
        df['sta_from_another_ap_max_ant_rssi'] = df[['sta_from_ap_0_max_ant_rssi', 'sta_from_ap_1_max_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[0] if x.iloc[2] == 'sta_1' else x.iloc[1], axis=1)
        df['sta_from_another_ap_mean_ant_rssi'] = df[['sta_from_ap_0_mean_ant_rssi', 'sta_from_ap_1_mean_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[0] if x.iloc[2] == 'sta_1' else x.iloc[1], axis=1)

        # 丢掉无用的列
        df.drop(columns=['sta_to_ap_0_sum_ant_rssi', 'sta_to_ap_1_sum_ant_rssi', 'sta_to_ap_0_max_ant_rssi', 'sta_to_ap_1_max_ant_rssi', 'sta_to_ap_0_mean_ant_rssi',
                         'sta_to_ap_1_mean_ant_rssi', 'sta_from_ap_0_sum_ant_rssi', 'sta_from_ap_1_sum_ant_rssi', 'sta_from_ap_0_max_ant_rssi', 'sta_from_ap_1_max_ant_rssi',
                         'sta_from_ap_0_mean_ant_rssi', 'sta_from_ap_1_mean_ant_rssi', 'sta_from_sta_0_rssi', 'sta_from_sta_1_rssi', 'ap_from_ap_0_sum_ant_rssi'\
                            , 'ap_from_ap_1_sum_ant_rssi', 'ap_from_ap_0_max_ant_rssi', 'ap_from_ap_1_max_ant_rssi', 'ap_from_ap_0_mean_ant_rssi',\
                                'ap_from_ap_1_mean_ant_rssi', ], inplace=True)
        
        # 读取每一个文件的nav和ed
        nav = df['nav'][0]
        ed = df['ed'][0]
        pd = df['pd'][0]

        # 计算信干燥比
        df['snr'] = df[['sta_from_ap_max_ant_rssi', 'sta_from_another_ap_max_ant_rssi']].apply(snr_df, axis=1)
        # 计算phy， PHY是一个映射表
        # 将mcs和nss转换为整型
        df['mcs'] = df['mcs'].astype(int)
        df['nss'] = df['nss'].astype(int)
        df['phy'] = df[['nss', 'mcs']].apply(lambda x: PHY[x.iloc[0]-1][x.iloc[1]], axis=1)

        # 对大于nav和大于ed的列进行计数统计，并生成新的列
        for i in df.columns:
            if 'rssi' in i:
                df[i+'_nav'] = df[i].apply(partial(check_rssi, flag=nav))
                df[i+'_ed'] = df[i].apply(partial(check_rssi, flag=ed))
                df[i+'_pd'] = df[i].apply(partial(check_rssi, flag=pd))
                df.drop(columns=[i], inplace=True)

        # 对列顺序进行重排
        cols = df.columns.tolist()
        rssi_cols = [i for i in cols if 'rssi' in i]
        no_rssi_cols = [i for i in cols if 'rssi' not in i]
        cols = no_rssi_cols[:-10] + rssi_cols + no_rssi_cols[-10:]


        # mean用来判断nav门限， max用来判断ed和pd门限， 其他rssi丢弃
        cols = [col for col in cols if (('rssi' not in col) or (('mean' in col) and ('nav' in col)) or (('max' in col) and (('ed' in col) or ('pd' in col))))]

        df = df[cols]

        ssnr = df.pop('snr')
        df.insert(30, 'snr', ssnr)


        phy = df.pop('phy')
        df.insert(33, 'phy', phy)

        df['cls'] = df[['nss', 'mcs']].apply(mn2idx, axis=1)

        return df

    elif ap_scenario == '3ap':
        # 将ap_from_ap的特征合并
        df['ap_from_another_ap1_max_ant_rssi'] = df[['ap_from_ap_0_max_ant_rssi', 'ap_from_ap_1_max_ant_rssi', 'ap_from_ap_2_max_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[1] if x.iloc[3] == 'sta_0' else x.iloc[0], axis=1)
        df['ap_from_another_ap1_sum_ant_rssi'] = df[['ap_from_ap_0_sum_ant_rssi', 'ap_from_ap_1_sum_ant_rssi', 'ap_from_ap_2_sum_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[1] if x.iloc[3] == 'sta_0' else x.iloc[0], axis=1)
        df['ap_from_another_ap1_mean_ant_rssi'] = df[['ap_from_ap_0_mean_ant_rssi', 'ap_from_ap_1_mean_ant_rssi', 'ap_from_ap_2_mean_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[1] if x.iloc[3] == 'sta_0' else x.iloc[0], axis=1)
        df['ap_from_another_ap2_max_ant_rssi'] = df[['ap_from_ap_0_max_ant_rssi', 'ap_from_ap_1_max_ant_rssi', 'ap_from_ap_2_max_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[1] if x.iloc[3] == 'sta_2' else x.iloc[2], axis=1)
        df['ap_from_another_ap2_sum_ant_rssi'] = df[['ap_from_ap_0_sum_ant_rssi', 'ap_from_ap_1_sum_ant_rssi', 'ap_from_ap_2_sum_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[1] if x.iloc[3] == 'sta_2' else x.iloc[2], axis=1)
        df['ap_from_another_ap2_mean_ant_rssi'] = df[['ap_from_ap_0_mean_ant_rssi', 'ap_from_ap_1_mean_ant_rssi', 'ap_from_ap_2_mean_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[1] if x.iloc[3] == 'sta_2' else x.iloc[2], axis=1)

        # 将sta_from_sta的特征合并
        df['sta_from_another_sta1_rssi'] = df[['sta_from_sta_0_rssi', 'sta_from_sta_1_rssi', 'sta_from_sta_2_rssi', 'sta_id']].apply(lambda x: x.iloc[1] if x.iloc[3] == 'sta_0' else x.iloc[0], axis=1)
        df['sta_from_another_sta2_rssi'] = df[['sta_from_sta_0_rssi', 'sta_from_sta_1_rssi', 'sta_from_sta_2_rssi', 'sta_id']].apply(lambda x: x.iloc[1] if x.iloc[3] == 'sta_2' else x.iloc[2], axis=1)

        # 将sta_to/from_ap_0/1/2的特征重定向为to/from_ap/to/from_another_ap
        df['sta_to_ap_max_ant_rssi'] = df[['sta_to_ap_0_max_ant_rssi', 'sta_to_ap_1_max_ant_rssi', 'sta_to_ap_2_max_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[0] if x.iloc[3] == 'sta_0' else x.iloc[1] if x.iloc[3] == 'sta_1' else x.iloc[2], axis=1)
        df['sta_to_ap_sum_ant_rssi'] = df[['sta_to_ap_0_sum_ant_rssi', 'sta_to_ap_1_sum_ant_rssi', 'sta_to_ap_2_sum_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[0] if x.iloc[3] == 'sta_0' else x.iloc[1] if x.iloc[3] == 'sta_1' else x.iloc[2], axis=1)
        df['sta_to_ap_mean_ant_rssi'] = df[['sta_to_ap_0_mean_ant_rssi', 'sta_to_ap_1_mean_ant_rssi', 'sta_to_ap_2_mean_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[0] if x.iloc[3] == 'sta_0' else x.iloc[1] if x.iloc[3] == 'sta_1' else x.iloc[2], axis=1)

        df['sta_from_ap_max_ant_rssi'] = df[['sta_from_ap_0_max_ant_rssi', 'sta_from_ap_1_max_ant_rssi', 'sta_from_ap_2_max_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[0] if x.iloc[3] == 'sta_0' else x.iloc[1] if x.iloc[3] == 'sta_1' else x.iloc[2], axis=1)
        df['sta_from_ap_sum_ant_rssi'] = df[['sta_from_ap_0_sum_ant_rssi', 'sta_from_ap_1_sum_ant_rssi', 'sta_from_ap_2_sum_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[0] if x.iloc[3] == 'sta_0' else x.iloc[1] if x.iloc[3] == 'sta_1' else x.iloc[2], axis=1)
        df['sta_from_ap_mean_ant_rssi'] = df[['sta_from_ap_0_mean_ant_rssi', 'sta_from_ap_1_mean_ant_rssi', 'sta_from_ap_2_mean_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[0] if x.iloc[3] == 'sta_0' else x.iloc[1] if x.iloc[3] == 'sta_1' else x.iloc[2], axis=1)

        df['sta_to_another_ap1_max_ant_rssi'] = df[['sta_to_ap_0_max_ant_rssi', 'sta_to_ap_1_max_ant_rssi', 'sta_to_ap_2_max_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[1] if x.iloc[3] == 'sta_0' else x.iloc[0], axis=1)
        df['sta_to_another_ap1_sum_ant_rssi'] = df[['sta_to_ap_0_sum_ant_rssi', 'sta_to_ap_1_sum_ant_rssi', 'sta_to_ap_2_sum_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[1] if x.iloc[3] == 'sta_0' else x.iloc[0], axis=1)
        df['sta_to_another_ap1_mean_ant_rssi'] = df[['sta_to_ap_0_mean_ant_rssi', 'sta_to_ap_1_mean_ant_rssi', 'sta_to_ap_2_mean_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[1] if x.iloc[3] == 'sta_0' else x.iloc[0], axis=1)
        df['sta_to_another_ap2_max_ant_rssi'] = df[['sta_to_ap_0_max_ant_rssi', 'sta_to_ap_1_max_ant_rssi', 'sta_to_ap_2_max_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[1] if x.iloc[3] == 'sta_2' else x.iloc[2], axis=1)
        df['sta_to_another_ap2_sum_ant_rssi'] = df[['sta_to_ap_0_sum_ant_rssi', 'sta_to_ap_1_sum_ant_rssi', 'sta_to_ap_2_sum_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[1] if x.iloc[3] == 'sta_2' else x.iloc[2], axis=1)
        df['sta_to_another_ap2_mean_ant_rssi'] = df[['sta_to_ap_0_mean_ant_rssi', 'sta_to_ap_1_mean_ant_rssi', 'sta_to_ap_2_mean_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[1] if x.iloc[3] == 'sta_2' else x.iloc[2], axis=1)

        df['sta_from_another_ap1_max_ant_rssi'] = df[['sta_from_ap_0_max_ant_rssi', 'sta_from_ap_1_max_ant_rssi', 'sta_from_ap_2_max_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[1] if x.iloc[3] == 'sta_0' else x.iloc[0], axis=1)
        df['sta_from_another_ap1_sum_ant_rssi'] = df[['sta_from_ap_0_sum_ant_rssi', 'sta_from_ap_1_sum_ant_rssi', 'sta_from_ap_2_sum_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[1] if x.iloc[3] == 'sta_0' else x.iloc[0], axis=1)
        df['sta_from_another_ap1_mean_ant_rssi'] = df[['sta_from_ap_0_mean_ant_rssi', 'sta_from_ap_1_mean_ant_rssi', 'sta_from_ap_2_mean_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[1] if x.iloc[3] == 'sta_0' else x.iloc[0], axis=1)
        df['sta_from_another_ap2_max_ant_rssi'] = df[['sta_from_ap_0_max_ant_rssi', 'sta_from_ap_1_max_ant_rssi', 'sta_from_ap_2_max_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[1] if x.iloc[3] == 'sta_2' else x.iloc[2], axis=1)
        df['sta_from_another_ap2_sum_ant_rssi'] = df[['sta_from_ap_0_sum_ant_rssi', 'sta_from_ap_1_sum_ant_rssi', 'sta_from_ap_2_sum_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[1] if x.iloc[3] == 'sta_2' else x.iloc[2], axis=1)
        df['sta_from_another_ap2_mean_ant_rssi'] = df[['sta_from_ap_0_mean_ant_rssi', 'sta_from_ap_1_mean_ant_rssi', 'sta_from_ap_2_mean_ant_rssi', 'sta_id']].apply(lambda x: x.iloc[1] if x.iloc[3] == 'sta_2' else x.iloc[2], axis=1)

        # 丢掉无用的列
        df.drop(columns=['sta_to_ap_0_max_ant_rssi', 'sta_to_ap_1_max_ant_rssi', 'sta_to_ap_2_max_ant_rssi', 'sta_to_ap_0_sum_ant_rssi', 'sta_to_ap_1_sum_ant_rssi', 'sta_to_ap_2_sum_ant_rssi', 'sta_to_ap_0_mean_ant_rssi', 'sta_to_ap_1_mean_ant_rssi', 'sta_to_ap_2_mean_ant_rssi', 'sta_from_ap_0_max_ant_rssi', 'sta_from_ap_1_max_ant_rssi', 'sta_from_ap_2_max_ant_rssi', 'sta_from_ap_0_sum_ant_rssi', 'sta_from_ap_1_sum_ant_rssi', 'sta_from_ap_2_sum_ant_rssi', 'sta_from_ap_0_mean_ant_rssi', 'sta_from_ap_1_mean_ant_rssi', 'sta_from_ap_2_mean_ant_rssi', 'sta_from_sta_0_rssi', 'sta_from_sta_1_rssi', 'sta_from_sta_2_rssi', 'ap_from_ap_0_max_ant_rssi', 'ap_from_ap_1_max_ant_rssi', 'ap_from_ap_2_max_ant_rssi', 'ap_from_ap_0_sum_ant_rssi', 'ap_from_ap_1_sum_ant_rssi', 'ap_from_ap_2_sum_ant_rssi', 'ap_from_ap_0_mean_ant_rssi', 'ap_from_ap_1_mean_ant_rssi', 'ap_from_ap_2_mean_ant_rssi'], inplace=True)
        

        # 计算信干燥比
        df['snr'] = df[['sta_from_ap_max_ant_rssi', 'sta_from_another_ap1_max_ant_rssi', 'sta_from_another_ap2_max_ant_rssi']].apply(snr_df, axis=1)

        # 计算phy， PHY是一个映射表
        # 将mcs和nss转换为整型
        df['mcs'] = df['mcs'].astype(int)
        df['nss'] = df['nss'].astype(int)
        df['phy'] = df[['nss', 'mcs']].apply(lambda x: PHY[x.iloc[0]-1][x.iloc[1]], axis=1)


        # 读取每一个文件的nav和ed
        nav = df['nav'][0]
        ed = df['ed'][0]
        pd = df['pd'][0]


        # assert nav == pd

        # 对大于nav和大于ed的列进行计数统计，并生成新的列
        for i in df.columns:
            if 'rssi' in i:
                df[i+'_nav'] = df[i].apply(partial(check_rssi, flag=nav))
                df[i+'_ed'] = df[i].apply(partial(check_rssi, flag=ed))
                df[i+'_pd'] = df[i].apply(partial(check_rssi, flag=pd))
                df.drop(columns=[i], inplace=True)

        # 对列顺序进行重排
        cols = df.columns.tolist()
        rssi_cols = [i for i in cols if 'rssi' in i]
        no_rssi_cols = [i for i in cols if 'rssi' not in i]
        cols = no_rssi_cols[:-10] + rssi_cols + no_rssi_cols[-10:]

    

        cols = [col for col in cols if (('rssi' not in col) or (('mean' in col) and ('nav' in col)) or (('max' in col) and (('ed' in col) or ('pd' in col))))]

        # assert False
        df = df[cols]


        ssnr = df.pop('snr')
        df.insert(39, 'snr', ssnr)


        phy = df.pop('phy')
        df.insert(42, 'phy', phy)


        try:
            df['cls'] = df[['nss', 'mcs']].apply(mn2idx, axis=1)
        except:
            print(df['nav'][0], df['loc_id'][0])

        return df


    else:
        raise ValueError('Invalid ap_scenario')



if __name__ == '__main__':


    # df = load_data('2ap')

    # df.to_csv('./data/post/2ap.csv', index=False)

    # df = load_data('3ap')
    # df.to_csv('./data/post/3ap.csv', index=False)

    for sett in [2]:
        for ap_scenario in [2, 3]:
        
            df, name = load_test_data(ap_scenario, sett)
            df.to_csv(f'./data/post/{name}', index=False)
            print(f'{name} saved')


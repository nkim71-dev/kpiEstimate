import os, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import findFiles


if __name__ == "__main__":
    src_dir = "data"
    dest_dir = "processedData"


    # 데이터 로드
    fileList = findFiles(os.path.join(os.getcwd(), src_dir), suffix='csv')
    dfs = [pd.read_csv(os.path.join(os.getcwd(), src_dir, f)) for f in fileList]
    df = pd.concat(dfs, axis=0).reset_index(drop=True)
    cols = df.columns.tolist()
    validCols = ['Timestamp', 'CellID', 'RAWCELLID', 'RSRP', 'RSRQ', 'SNR', 'CQI', 'RSSI', 'DL_bitrate', 'UL_bitrate', 'NRxRSRP', 'NRxRSRQ']

    # 데이터 전처리
    infoCols = ['Timestamp', 'CellID', 'RAWCELLID']
    kpiCols = ['DL_bitrate'] #'UL_bitrate'
    inputCols = ['RSRP', 'RSRQ', 'SNR', 'CQI', 'RSSI', 'NRxRSRP', 'NRxRSRQ']

    df = df[infoCols+inputCols+kpiCols].replace('-',np.nan)
    df[inputCols+kpiCols] = df[inputCols+kpiCols].replace('-',np.nan).astype('float64')
    df = df.dropna(axis=0).reset_index(drop=True)
    df = df.sort_values(by=['CellID', 'Timestamp']).reset_index(drop=True)
    info = df[infoCols]
    
    # 데이터 정규화
    scalers = dict()
    scaler = MinMaxScaler()
    scalers['X'] = scaler.fit(df[inputCols])
    scaler = StandardScaler()
    scalers['Y'] = scaler.fit(df[kpiCols])

    with open(f'./{dest_dir}/scalers.pickle', 'wb') as handle:
        pickle.dump(scalers, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    normalized_df = df.copy()
    normalized_df[inputCols] = scalers['X'].transform(df[inputCols])
    normalized_df[kpiCols] = scalers['Y'].transform(df[kpiCols])

    x = normalized_df[inputCols]
    y = normalized_df[kpiCols]

    # 데이터 스플릿 (Train:0.8, Valid:0.2, Test:0.2)
    train_index = normalized_df['CellID'].sample(frac=0.8, random_state=42)
    valid_index = train_index.sample(frac=0.2/0.8, random_state=42)

    train_index = train_index.index.tolist()
    valid_index = valid_index.index.tolist()
    test_index = list(set(np.arange(len(normalized_df)))-set(train_index+valid_index))

    xDataTrain = np.array(x.loc[train_index])
    xDataValid = np.array(x.loc[valid_index])
    xDataTest = np.array(x.loc[test_index])
    yDataTrain = np.array(y.loc[train_index])
    yDataValid = np.array(y.loc[valid_index])
    yDataTest = np.array(y.loc[test_index])

    data_dict = {'xDataTrain': xDataTrain,
                 'yDataTrain': yDataTrain,
                'xDataValid': xDataValid,
                'yDataValid': yDataValid,
                'xDataTest': xDataTest,
                'yDataTest': yDataTest}
    
    with open(f'./{dest_dir}/processedData.pickle', 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    

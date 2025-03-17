import numpy as np
import pandas as pd
import os, pickle
from datetime import datetime
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from utils import makeDir, setGpuUsage
from model import Dense, Transformer, Predictor
import argparse
      
# 코드 입력 인자
parser = argparse.ArgumentParser(description='Parameters for model inference')
parser.add_argument('--model-name', type=str, required=False, default=None, help='모델 이름')
parser.add_argument('--gpu-memory', type=int, required=False, default=4096, help='GPU 사용량')

if __name__ == "__main__":
    
    # 코드 실행 준비
    dt_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") # 성능 확인 코드 시작 일시
    args = parser.parse_args() # 추론에 필요한 argment 파싱
    setGpuUsage(args.gpu_memory) # 추론에 사용할 GPU메모리 설정
    makeDir('./figures') # 추론 결과 저장 디렉토리 생성

    # 전처리 데이터 로드
    print("Load preprocessed data")
    datapath = 'processedData'
    with open(f'./{datapath}/processedData.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)  
    xDataTrain = data_dict['xDataTrain']
    yDataTrain = data_dict['yDataTrain']
    xDataValid = data_dict['xDataValid']
    yDataValid = data_dict['yDataValid']
    xDataTest = data_dict['xDataTest']
    yDataTest = data_dict['yDataTest']

    # 추론 모델 미지정 시 최신 버전의 모델 추론
    model_list = os.listdir('./models')
    if args.model_name is not None:
        model_list = [m for m in model_list if args.model_name in m]
    model_list.sort()
    model_name = model_list[-1]
    model_dir = os.path.join(os.getcwd(),'models', model_name)
    # model_path = [dir for dir in os.listdir(model_dir)]

    test_proba_list = list()
    test_bov_igv_list = list()
    results_df = list()
    print(f"Inference models in '{model_name}'")
 
    # Test 데이터 추론
    # 모델 선언
    # predictor = Predictor(inputDim=xDataTrain.shape[-1], outputDim=1, modelType=model_name.split('_')[0])    
    if 'transformer' in model_name:
        predictor = Transformer(inputDim=xDataTrain.shape[-1])  
    else:
        predictor = Dense(inputDim=xDataTrain.shape[-1])    
    predictor(xDataTrain[:1])
    predictor.load_weights(model_dir, skip_mismatch=False) # 모델 가중치 로드
    yPredicted = predictor.predict(xDataTest, verbose=0) # 추론
    
    # 결과 출력을 위한 비정규화
    with open(f'./{datapath}/scalers.pickle', 'rb') as handle:
        scalers = pickle.load(handle)
    yPredictedRescaled = scalers['Y'].inverse_transform(yPredicted).reshape(-1)
    yDataTestRescaled = scalers['Y'].inverse_transform(yDataTest).reshape(-1)
    df_results = pd.DataFrame({'Ground Truth': yDataTestRescaled, 'Predicted': yPredictedRescaled})

    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(1,1,figsize=(8,4))
    ax.plot(yDataTestRescaled, alpha=1, label='Ground Truth')
    ax.plot(yPredictedRescaled, alpha=0.6, label='Predicted')
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(f'./figures/predictionLine.png')
    plt.clf()

    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(1,1,figsize=(5,2))
    ax = sns.scatterplot(data=df_results, x='Ground Truth', y='Predicted', alpha=0.8)
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Predicted')
    ax.grid()
    fig.tight_layout()
    fig.savefig(f'./figures/predictionScatter.png')
    plt.clf()

    rmse = root_mean_squared_error(yDataTestRescaled, yPredictedRescaled)
    mae = mean_absolute_error(yDataTestRescaled, yPredictedRescaled)
    # mape = mean_absolute_percentage_error(yDataTestRescaled, yPredictedRescaled)

    print(f"\nPrediction Errors on the Test Dataset:")
    print(f"-> RMSE: {rmse:04f}")
    print(f"-> MAE:  {mae:04f}")
    # print(f"-> MAPE:\t{mape:04f}")
import os, pickle, shutil
import numpy as np
import pandas as pd
from tensorflow import keras
from datetime import datetime
import matplotlib.pyplot as plt
from utils import makeDir, setGpuUsage
from model import Dense, Transformer, Predictor
import argparse

# 코드 입력 인자
parser = argparse.ArgumentParser(description='Parameters for model training')
parser.add_argument('--model-name', type=str, required=False, default='dense', help='모델 이름')
parser.add_argument('--epochs', type=int, required=False, default=500, help='최대 epochs')
parser.add_argument('--batch-size', type=int, required=False, default=64, help='배치 사이즈')
parser.add_argument('--gpu-memory', type=int, required=False, default=4096, help='GPU 사용량')


if __name__ == "__main__":

    # 코드 실행 준비
    dt_string_init = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") # 학습 코드 시작 일시
    args = parser.parse_args() # 학습에 필요한 argment 파싱
    setGpuUsage(args.gpu_memory) # 학습에 사용할 GPU메모리 설정
    makeDir('./models') # 학습 모델 저장 디렉토리 생성

    # 전처리 데이터 로드
    datapath = 'processedData'
    with open(f'./{datapath}/processedData.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)
    xDataTrain = data_dict['xDataTrain']
    yDataTrain = data_dict['yDataTrain']
    xDataValid = data_dict['xDataValid']
    yDataValid = data_dict['yDataValid']
    
    # 현재 디렉토리 및 모델 저장 디렉토리 설정
    curr_dir = os.getcwd()
    dest_dir = f"models"
    makeDir(dest_dir)        

    # 학습 수행
        
    # 모델 선언 및 컴파일
    # predictor = Predictor(inputDim=xDataTrain.shape[-1], outputDim=1, modelType=args.model_name)
    if 'transformer' in args.model_name:
        predictor = Transformer(inputDim=xDataTrain.shape[-1])  
    else:
        predictor = Dense(inputDim=xDataTrain.shape[-1])  
    predictor.compile(optimizer=keras.optimizers.Adam(1e-2, clipnorm=1.0), loss='MSE')
    predictor(xDataTrain[:1])

    # 학습을 위한 하이퍼파라미터 설정 (learning rate scheduler, early stopping)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    earlyStop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, 
                                            restore_best_weights=True, start_from_epoch=30)
    # 학습
    history = predictor.fit(x=xDataTrain, y=yDataTrain,  
                            validation_data = (xDataValid, yDataValid),
                            epochs=args.epochs, batch_size=args.batch_size,
                            callbacks=[reduce_lr, earlyStop], verbose=1)
    
    # 학습 모델 저장
    dt_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") # 모델 학습 완료 일시
    predictor.save_weights(os.path.join(dest_dir, f'{args.model_name}_{dt_string}.weights.h5')) # 모델 저장


    

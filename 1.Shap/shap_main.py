#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import lightgbm as lgb
import shap

from sklearn.preprocessing import MinMaxScaler

from shap_preprocessing import Preprocessing_Data
from shap_model import model_produce
from utills import *


def main():
    
    # 학습할 데이터와 모델 생성 #
    X, y, df = Preprocessing_Data()
    df_shap = model_produce(X, y)
    
    # 데이터 스케일링 #
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_shap = scaler.fit_transform(df_shap)
    df_shap = pd.DataFrame(df_shap, columns=X.columns)
    df_shap.columns = '기여도_' + df_shap.columns
    
    # 데이터 인덱스 맞추기 #
    df.reset_index(drop=True, inplace=True)
    df_shap.reset_index(drop=True, inplace=True)

    cc_df = pd.concat([df, df_shap], axis=1)
    cc_df.to_csv('shap_output.csv', index=False)
    
    return cc_df
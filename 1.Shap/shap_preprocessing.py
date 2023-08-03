#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from utills import *


def Preprocessing_Data():
    
    df = pd.read_csv('real_shap_df.csv', sep='|')
    
    # data type 변경
    df['점검진단시작일'] = pd.to_datetime(df['점검진단시작일'], format='%Y%m%d')
    df['점검진단종료일'] = pd.to_datetime(df['점검진단종료일'], format='%Y%m%d')
    df['점검진단구분'] = df['점검진단구분'].astype(str)
    df['시설물종별'] = df['시설물종별'].astype(str)
    
    if '보고년도' in df.columns:
        df['보고년도'] = df['보고년도'].astype(str)
        
    if '계약일자' in df.columns:
        df['계약일자'] = pd.to_datetime(df['계약일자'], format='%Y%m%d').astype(str)
        df['계약일자'] = df['계약일자'].astype(str)
        
    if '준공일' in df.columns:
        df['준공일'] = pd.to_datetime(df['준공일'], format='%Y%m%d').astype(str)
        df['준공일'] = df['준공일'].astype(str)
    
    if '승인상태' in df.columns:
        df['승인상태'] = df['승인상태'].astype(str)        
        
    if '분담구분' in df.columns:
        df['분담구분'] = df['분담구분'].astype(str)

    if '기술등급' in df.columns:
        df['기술등급'] = df['기술등급'].astype(str)
    
    if '정기점검구분' in df.columns:
        df['정기점검구분'] = df['정기점검구분'].astype(str)
    
    ## 안전등급 ##
    if '안전등급' in df.columns:
        df = df[df['안전등급'] != '3']
    
    ## 종합평가등급 ##
    if '종합평가등급' in df.columns:
        df = df[df['종합평가등급'] != '3']
        df = df[df['종합평가등급'] != '2']
    
    ## 내진설계적용여부 ##
    if '내진설계적용여부' in df.columns:
        df['내진설계적용여부'] = df['내진설계적용여부'].replace('X', np.nan).replace(' ', np.nan)
    
    ## 보고년도 ##
    if '보고년도' in df.columns:
        values_to_remove = ['2033', '2112', '220', '3021', '2101', '2104', '1912', '2026', '2031', '1014', '2202', '1010', '2103', '2323', '2025', '2121']
        df.loc[df['보고년도'].isin(values_to_remove), '보고년도'] = df.loc[df['보고년도'].isin(values_to_remove)]['보고년도'].map(lambda x: np.nan)
    
    # 모델에 넣기전 데이터 타입 변환
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
        
    ## 평가결과 ##
    df['평가결과'] = df['평가결과'].apply(result_transform)
    df['평가결과'].fillna('적정', inplace=True)
    
    X = df.drop(['시설물번호', '용역번호', '시설물구분', '점검진단구분', '시설물종별', '점검진단시작일', '점검진단종료일', '평가결과'], axis=1)
    y = df['평가결과'].astype('category')
    y = y.replace('적정', 0).replace('부실', 1)
    
    return X, y, df

if __name__ == "__main__":
    Preprocessing_Data()
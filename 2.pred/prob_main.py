#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import lightgbm as lgb

from prob_preprocessing import Preprocessing_Data
from prob_model import model_produce
from utills import *


def main():
    
    train, X_test, df = Preprocessing_Data()
    
    X_train = train.drop(['평가결과'], axis=1)
    y_train = train['평가결과']
    y_train = y_train.map(result_transform)
    
    lgb_model = model_produce(X_train, y_train)
    
    lgb_proba = lgb_model.predict_proba(X_test)
    one_proba = lgb_proba[:, 1]
    
    df = df.loc[X_test.index, :]
    
    sel_df_prob = pd.DataFrame(one_proba, columns=['단계별 부실 정도'], index=X_test.index).astype('float')
    sel_df_prob.sort_values(by='단계별 부실 정도', ascending=False, inplace=True)
    
    fin_sel_df = pd.concat([sel_df_prob, df], axis=1)
    fin_sel_df.drop(['평가결과'], axis=1, inplace=True)
    
    ex_columns = ['시설물명', '점검진단시작일', '점검진단종료일', '관리주체코드', '시설물종별', '시설물구분', '단계별 부실 정도']
    fin_sel_df.columns = ['분석_' + col if col not in ex_columns else col for col in fin_sel_df.columns]
    selected_columns = ex_columns + [col for col in fin_sel_df.columns if col not in ex_columns]
    fin_sel_df = fin_sel_df[selected_columns]

    fin_sel_df.to_csv('prob_output_final.csv', index=False)
    
    return fin_sel_df

if __name__ == "__main__":
    main()
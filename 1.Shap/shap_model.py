#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import shap

def model_produce(X, y): 
    
    # lightGBM으로 학습 #
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X, y)
    
    # shap 모델 학습 및 데이터 프레임으로 변환 #
    explainer = shap.TreeExplainer(model)
    shap_value = explainer.shap_values(X, check_additivity=False)
    df_shap = pd.DataFrame(shap_value[1], columns=X.columns)
    
    return df_shap
    
    
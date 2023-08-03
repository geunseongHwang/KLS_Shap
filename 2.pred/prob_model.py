#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb

from prob_preprocessing import Preprocessing_Data

def model_produce(X_train, y_train): 
    
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    return model
    
    
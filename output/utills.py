#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import re
import csv
import itertools
from datetime import datetime

def count_y_n(column):
    
    if isinstance(column, str):
        column = column.split('|')
        count_y = column.count('Y')
        count_n = column.count('N')
        return count_y - count_n
    else:
        return np.nan  # NaN 값이거나 문자열이 아닌 경우


def unify_results(x):
    if '실시' in x:
        return '실시'
    elif '미실시' in x:
        return '미실시'
    else:
        return np.nan

def trans_results(x):
    if 'A' in x:
        return 5
    elif 'B' in x:
        return 4
    elif 'C' in x:
        return 3
    elif 'D' in x:
        return 2
    elif 'E' in x:
        return 1
    else:
        return np.nan

def replace_outlier_with_median(col):
    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    col = col.mask(col < lower_bound, col.median())
    col = col.mask(col > upper_bound, col.median())
    return col

def excel_date_to_datetime(excel_date):
    base_date = datetime(1900, 1, 1)
    delta = timedelta(days=int(excel_date) - 1)  # 날짜를 정수로 변환하여 계산
    target_date = base_date + delta
    return target_date
    
    return df

def result_transform(df):
        
    if df in ['R2', 'R3', 'R4', 'D', 'D2']:
        return '부실'
    elif df in ['O2', 'O', 'R']:
        return '적정'
    else:
        return np.nan
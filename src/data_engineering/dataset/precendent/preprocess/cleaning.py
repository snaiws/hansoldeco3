import datetime

import numpy as np


def preprocess_baseline(df):
    df_ = df.copy()
    
    df_['공사종류(대분류)'] = df_['공사종류'].str.split(' / ').str[0]
    df_['공사종류(중분류)'] = df_['공사종류'].str.split(' / ').str[1]
    df_['공종(대분류)'] = df_['공종'].str.split(' > ').str[0]
    df_['공종(중분류)'] = df_['공종'].str.split(' > ').str[1]
    df_['사고객체(대분류)'] = df_['사고객체'].str.split(' > ').str[0]
    df_['사고객체(중분류)'] = df_['사고객체'].str.split(' > ').str[1]
    return df_

def preprocess_id(df):
    col = 'ID'
    df_ = df.copy()
    df_ = df_.set_index(col)
    return df_

def preprocess_datetime(s):
    d, m, t = s.split()
    dt = d+" "+t
    dt = datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M")
    if m == "오후" and dt.hour != 12:
        dt = dt + datetime.timedelta(hours = 12)
    return dt

def preprocess_dt(df, col):
    df_ = df.copy()
    df_[col] = df_[col].map(preprocess_datetime, na_action = 'ignore')
    return df_

def preprocess_tr(df):
    col1 = "사고인지 시간"
    col2 = "근무종류"
    df_ = df.copy()
    wttr = df_[col1].str.split('-', n=1, expand=True)
    wttr.columns = ['wt','tr']
    wt = wttr['wt']
    tr = wttr['tr']
    df_[col1] = tr.str.strip().replace('',np.nan)
    wt= wt.str.strip().replace('',np.nan)
    df_.insert(df.columns.get_loc(col1) + 1, col2, wt)
    return df_

def preprocess_temp_1(df):
    col = "기온"
    df_ = df.copy()
    df_[col] = df_[col].str.replace("℃","")
    df_[col].replace("", np.nan, inplace = True)
    df_[col] = df_[col].astype(float)
    return df_

def preprocess_humid_1(df):
    col = "습도"
    df_ = df.copy()
    df_[col] = df_[col].str.replace("%","")
    df_[col].replace("", np.nan, inplace = True)
    df_[col] = df_[col].astype(float)
    return df_

def preprocess_area_1(df):
    col = "연면적"
    df_ = df.copy()
    df_[col] = df_[col].str.replace(",","")
    df_[col] = df_[col].str.replace("㎡","")
    df_[col].replace("", np.nan, inplace = True)
    df_[col].replace("-", np.nan, inplace = True)
    df_[col] = df_[col].astype(float)
    return df_

def preprocess_floor_1(df):
    col = "층 정보"
    col1 = "지상"
    col2 = "지하"
    df_ = df.copy()
    df_[col].replace("-", np.nan, inplace = True)
    df_ = preprocess_hierarchy(df_, col, [col1, col2], sep = ",")
    
    df_[col1] = df_[col1].str.replace(r'\D', '', regex = True)
    df_[col2] = df_[col2].str.replace(r'\D', '', regex = True)

    df_[col1] = df_[col1].astype(float)
    df_[col2] = df_[col2].astype(float)
    return df_

def preprocess_cause_1(df):
    col = "사고원인"
    df_ = df.copy()
    df_[col] = df_[col].map(lambda x: np.nan if len(x)==1 else x, na_action = "ignore")
    return df_
    

def preprocess_hierarchy(df, col:str, levels:list, sep:str):
    level = len(levels)
    df_ = df.copy()
    frac = df_[col].str.split(sep, n=level-1, expand=True)
    frac.columns = levels
    frac = frac.apply(lambda x: x.str.strip())
    
    for i, column in enumerate(frac.columns):
        df_.insert(df.columns.get_loc(col) + i + 1, column, frac[column])
    df_.drop(col, axis=1, inplace = True)
    return df_


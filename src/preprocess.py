import datetime

import pandas as pd
import numpy as np



def preprocess_id(df):
    """
    ID col 인덱스화
    """
    col = 'ID'
    df_ = df.copy()
    df_ = df_.set_index(col)
    return df_

def preprocess_datetime(s:str):
    """
    입력(s) : 한솔의 datetime 형식(string)
    출력 : datetime 객체체
    """
    d, m, t = s.split()
    dt = d+" "+t
    dt = datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M")
    if m == "오후" and dt.hour != 12:
        dt = dt + datetime.timedelta(hours = 12)
    return dt

def preprocess_dt(df, col:str):
    """
    data frame과 column의 이름을 받아 해당 column의 타입을 datetime객체로 변경
    """
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
    """
    기온 column의 형태를 바꾸고 결측을 드러내고 숫자 타입으로 바꾸기
    """
    col = "기온"
    df_ = df.copy()
    df_[col] = df_[col].str.replace("℃","")
    df_[col].replace("", np.nan, inplace = True)
    df_[col] = df_[col].astype(float)
    return df_

def preprocess_humid_1(df):
    """
    습도 column의 형태를 바꾸고 결측을 드러내고 숫자 타입으로 바꾸기
    """
    col = "습도"
    df_ = df.copy()
    df_[col] = df_[col].str.replace("%","")
    df_[col].replace("", np.nan, inplace = True)
    df_[col] = df_[col].astype(float)
    return df_

def preprocess_area_1(df):
    """
    연면적 column의 형태를 바꾸고 결측을 드러내고 숫자 타입으로 바꾸기
    """
    col = "연면적"
    df_ = df.copy()
    df_[col] = df_[col].str.replace(",","")
    df_[col] = df_[col].str.replace("㎡","")
    df_[col].replace("", np.nan, inplace = True)
    df_[col].replace("-", np.nan, inplace = True)
    df_[col] = df_[col].astype(float)
    return df_

def preprocess_floor_1(df):
    """
    층 정보 column을 지상과 지하로 나누고 결측을 드러내고 숫자column으로 만들기기
    """
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
    """
    사고원인 column의 길이가 1인 이상값 결측처리
    """
    col = "사고원인"
    df_ = df.copy()
    df_[col] = df_[col].map(lambda x: np.nan if len(x)==1 else x, na_action = "ignore")
    return df_
    

def preprocess_hierarchy(df, col:str, levels:list, sep:str):
    """
    한솔의 형식을 가진 분류 column들을 분리하기
    """
    level = len(levels)
    df_ = df.copy()
    frac = df_[col].str.split(sep, n=level-1, expand=True)
    frac.columns = levels
    frac = frac.apply(lambda x: x.str.strip())
    
    for i, column in enumerate(frac.columns):
        df_.insert(df.columns.get_loc(col) + i + 1, column, frac[column])
    df_.drop(col, axis=1, inplace = True)
    return df_



def pipeline1(df):
    """
    바로 분석 할 수 있도록 형태 바꾸고 결측 드러내는 전처리
    """
    df = preprocess_dt(df, "발생일시")
    df = preprocess_tr(df)
    df = preprocess_dt(df, "사고인지 시간")
    
    df = preprocess_temp_1(df)
    df = preprocess_humid_1(df)
    df = preprocess_area_1(df)
    df = preprocess_cause_1(df)
    df = preprocess_floor_1(df)
    df = preprocess_hierarchy(df, "공사종류", ["공사종류_대분류", "공사종류_중분류","공사종류_소분류"], "/")
    df = preprocess_hierarchy(df, "공종", ["공종_대분류", "공종_소분류"], ">")
    df = preprocess_hierarchy(df, "사고객체", ["사고객체_대분류", "사고객체_소분류"], ">")
    df = preprocess_hierarchy(df, "부위", ["부위1", "부위2"], "/")

    df = preprocess_id(df)
    return df



if __name__ == "__main__":
    import os

    path_train = "/workspace/Storage/hansoldeco3/Data/train.csv"
    path_test = "/workspace/Storage/hansoldeco3/Data/test.csv"

    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)
    train_processed_1 = pipeline1(train)
    print(train_processed_1)
    test_processed_1 = pipeline1(test)
    print(test_processed_1)
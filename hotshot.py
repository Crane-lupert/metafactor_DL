import yfinance as yf
import pandas as pd
import glob
from tqdm import tqdm
import os
import warnings
import gc
import numpy as np
import itertools
from tqdm import tqdm
# from causalnex.structure.notears import from_pandas
# from causalnex.network import BayesianNetwork
# from econml.dr import DRLearner
# import dowhy
# from dowhy import CausalModel
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 코스피/코스닥 지수 데이터 로드
def load_market_indices():
    """
    코스피와 코스닥 지수의 월별 종가지수를 반환하는 함수입니다.
    
    Returns:
    - market_indices (pd.DataFrame): 코스피와 코스닥 지수의 월별 종가지수 데이터프레임
    """
    try:
        market_df = pd.read_csv(
            'data/kor_market.csv',
            skiprows=8,
            header=[0, 1, 2, 3, 4, 5],
            index_col=0,
            encoding='cp949',
            parse_dates=True
        )
    except UnicodeDecodeError:
        market_df = pd.read_csv(
            'data/kor_market.csv',
            skiprows=8,
            header=[0, 1, 2, 3, 4, 5],
            index_col=0,
            encoding='euc-kr',
            parse_dates=True
        )

    # 멀티인덱스 컬럼 이름 지정
    market_df.columns.names = ['Symbol', 'Symbol Name', 'Kind', 'item', 'item Name', 'Frequency']
    market_df.index.name = 'Date'

    # 'Kind', 'Frequency' 레벨 제거
    market_df.columns = market_df.columns.droplevel(['Kind', 'Frequency'])

    # '종가지수(포인트)' 데이터 추출
    index_mask = market_df.columns.get_level_values('item Name') == '종가지수(포인트)'
    index_df = market_df.loc[:, index_mask]

    # '코스피'와 '코스닥' 지수만 선택
    symbol_names = index_df.columns.get_level_values('Symbol Name')
    kospi_kosdaq_mask = (symbol_names == '코스피') | (symbol_names == '코스닥')
    kospi_kosdaq_indices = index_df.loc[:, kospi_kosdaq_mask]
    kospi_kosdaq_indices.columns = kospi_kosdaq_indices.columns.droplevel(['item', 'item Name'])

    # 데이터 정제: 쉼표 제거 및 숫자 변환
    kospi_kosdaq_indices = (
        kospi_kosdaq_indices
        .astype(str)
        .replace(',', '', regex=True)
        .replace(['', 'None', 'nan', 'NaN', 'N/A', ''], np.nan)
        .apply(pd.to_numeric, errors='coerce')
    )

    # 월말 종가지수로 리샘플링
    market_indices = kospi_kosdaq_indices

    return market_indices

# 위험 무시 수익률 데이터 로드
def load_risk_free_rate():
    """
    위험 무시 수익률(rf)을 계산하는 함수입니다.
    국고채 1년물 금리를 사용합니다.
    
    Returns:
    - rf_monthly_returns (pd.Series): 월별 위험 무시 수익률 시리즈
    """
    try:
        rf_df = pd.read_csv(
            'data/rf.csv',
            skiprows=8,
            header=[0, 1, 2, 3, 4, 5],
            index_col=0,
            encoding='cp949',
            parse_dates=True
        )
    except UnicodeDecodeError:
        rf_df = pd.read_csv(
            'data/rf.csv',
            skiprows=8,
            header=[0, 1, 2, 3, 4, 5],
            index_col=0,
            encoding='euc-kr',
            parse_dates=True
        )

    # 멀티인덱스 컬럼 이름 지정
    rf_df.columns.names = ['Symbol', 'Symbol Name', 'Kind', 'item', 'item Name', 'Frequency']
    rf_df.index.name = 'Date'

    # 'Kind', 'Frequency' 레벨 제거
    rf_df.columns = rf_df.columns.droplevel(['Kind', 'Frequency'])

    # '시장금리:국고1년(%)' 데이터 추출
    index_mask = rf_df.columns.get_level_values('item Name') == '시장금리:국고1년(%)'
    index_df = rf_df.loc[:, index_mask]

    # 'ECO' 심볼만 선택
    symbol_names = index_df.columns.get_level_values('Symbol Name')
    eco_mask = (symbol_names == 'ECO')
    rf_indices = index_df.loc[:, eco_mask]
    rf_indices.columns = rf_indices.columns.droplevel(['item', 'item Name'])

    # 데이터 정제: 쉼표 제거 및 숫자 변환
    rf_indices = (
        rf_indices
        .astype(str)
        .replace(',', '', regex=True)
        .replace(['', 'None', 'nan', 'NaN', 'N/A', ''], np.nan)
        .apply(pd.to_numeric, errors='coerce')
    )

    # 월말 금리로 리샘플링
    rf_monthly = rf_indices
    rf_monthly.columns = ['Risk_Free_Rate']

    return rf_monthly

# 환율 데이터 로드
def load_exchange_rate():
    """
    원/달러 환율 데이터를 로드하는 함수입니다.
    
    Returns:
    - exchange_rate (pd.DataFrame): 월별 환율 데이터프레임
    """
    exchange_df = pd.read_csv(
        'KRW_USD_Exchange.csv',
        index_col=0,
        parse_dates=True
    )

    # 월말 환율로 리샘플링
    exchange_rate = exchange_df
    exchange_rate.columns = ['Exchange_Rate']

    return exchange_rate
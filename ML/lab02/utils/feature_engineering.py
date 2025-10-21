# feature_engineering.py
import pandas as pd
import numpy as np

def create_temporal_features(df):
    df_features = df.copy()
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['month'] = df_features.index.month
    df_features['quarter'] = df_features.index.quarter
    df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
    return df_features

def create_lag_features(df, target_col='number_sold'):
    df_lags = df.copy()
    df_lags['lag_1'] = df_lags[target_col].shift(1)
    df_lags['lag_7'] = df_lags[target_col].shift(7)
    df_lags['lag_30'] = df_lags[target_col].shift(30)
    return df_lags

def create_rolling_features(df, target_col='number_sold'):
    df_rolling = df.copy()
    windows = [7, 30, 90]
    for window in windows:
        df_rolling[f'rolling_mean_{window}'] = df_rolling[target_col].rolling(window=window).mean()
        df_rolling[f'rolling_std_{window}'] = df_rolling[target_col].rolling(window=window).std()
        df_rolling[f'rolling_min_{window}'] = df_rolling[target_col].rolling(window=window).min()
        df_rolling[f'rolling_max_{window}'] = df_rolling[target_col].rolling(window=window).max()
    return df_rolling

def create_volatility_features(df, target_col='number_sold'):
    df_vol = df.copy()
    windows = [7, 30, 90]
    for window in windows:
        df_vol[f'rolling_cv_{window}'] = (df_vol[target_col].rolling(window=window).std() / 
                                        df_vol[target_col].rolling(window=window).mean())
    return df_vol

def main_feature_engineering(df, target_col='number_sold'):
    df_features = df.copy()
    df_features = create_temporal_features(df_features)
    df_features = create_lag_features(df_features, target_col)
    df_features = create_rolling_features(df_features, target_col)
    df_features = create_volatility_features(df_features, target_col)
    df_features = df_features.dropna()
    return df_features
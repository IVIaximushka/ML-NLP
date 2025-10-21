# decomposition.py
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats

def perform_decomposition(series, period, model_type):
    try:
        decomposition = seasonal_decompose(series, model=model_type, period=period)
        return decomposition
    except Exception as e:
        print(f"Ошибка декомпозиции: {e}")
        return None

def analyze_residuals(residuals, period, model_type):
    residuals_clean = residuals.dropna()
    
    # Тесты стационарности
    adf_result = adfuller(residuals_clean)
    kpss_result = kpss(residuals_clean, regression='c')
    
    # Тест нормальности
    shapiro_test = stats.shapiro(residuals_clean)
    
    return {
        'period': period,
        'model_type': model_type,
        'adf_pvalue': adf_result[1],
        'kpss_pvalue': kpss_result[1],
        'shapiro_pvalue': shapiro_test[1],
        'mean': residuals_clean.mean(),
        'std': residuals_clean.std()
    }
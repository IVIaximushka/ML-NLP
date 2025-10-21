# stationarity.py
import pandas as pd
import numpy as np
from scipy.stats import boxcox, boxcox_normmax
from statsmodels.tsa.stattools import adfuller, kpss

def test_stationarity(series, test_type='adf'):
    if test_type == 'adf':
        result = adfuller(series.dropna())
        return result[1] < 0.05
    elif test_type == 'kpss':
        result = kpss(series.dropna(), regression='c')
        return result[1] > 0.05

def main_stationarity_transformation(df, target_col='number_sold'):
    series = df[target_col]
    results = []
    
    # Лог-трансформация
    if (series > 0).all():
        log_series = np.log(series)
        results.append({
            'transformation': 'Лог-трансформация',
            'adf_stationary': test_stationarity(log_series, 'adf'),
            'kpss_stationary': test_stationarity(log_series, 'kpss')
        })
    
    # Дифференцирование
    diff_series = series.diff().dropna()
    results.append({
        'transformation': 'Дифференцирование 1-го порядка',
        'adf_stationary': test_stationarity(diff_series, 'adf'),
        'kpss_stationary': test_stationarity(diff_series, 'kpss')
    })
    
    return pd.DataFrame(results)
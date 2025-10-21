# exponential_smoothing.py
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

def main_exponential_smoothing(df, target_col='number_sold', h=7):
    split_idx = int(len(df) * 0.8)
    train = df[target_col].iloc[:split_idx]
    test = df[target_col].iloc[split_idx:split_idx + h]
    
    models = []
    
    # SES
    try:
        ses_model = ExponentialSmoothing(train, trend=None, seasonal=None, optimized=True).fit()
        ses_forecast = ses_model.forecast(h)
        ses_mae = mean_absolute_error(test, ses_forecast)
        ses_rmse = np.sqrt(mean_squared_error(test, ses_forecast))
        models.append({
            'model': 'SES',
            'mae': ses_mae,
            'rmse': ses_rmse,
            'forecast': ses_forecast
        })
    except:
        pass
    
    # Holt Additive
    try:
        holt_add_model = ExponentialSmoothing(train, trend='add', seasonal=None, optimized=True).fit()
        holt_add_forecast = holt_add_model.forecast(h)
        holt_add_mae = mean_absolute_error(test, holt_add_forecast)
        holt_add_rmse = np.sqrt(mean_squared_error(test, holt_add_forecast))
        models.append({
            'model': 'Holt Additive',
            'mae': holt_add_mae,
            'rmse': holt_add_rmse,
            'forecast': holt_add_forecast
        })
    except:
        pass
    
    return pd.DataFrame(models)
# cross_validation.py
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

def sliding_window_cv(X, y, model, n_splits=5):
    metrics = []
    for i in range(n_splits):
        train_size = len(X) // (n_splits + 1)
        start_train = i * train_size
        end_train = start_train + train_size
        start_test = end_train
        end_test = start_test + train_size
        
        if end_test > len(X):
            break
            
        X_train, X_test = X.iloc[start_train:end_train], X.iloc[start_test:end_test]
        y_train, y_test = y.iloc[start_train:end_train], y.iloc[start_test:end_test]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics.append({'fold': i, 'mae': mae, 'rmse': rmse})
    
    return pd.DataFrame(metrics)

def main_cross_validation(df, target_col='number_sold'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    model = LinearRegression()
    
    cv_results = sliding_window_cv(X, y, model)
    return cv_results
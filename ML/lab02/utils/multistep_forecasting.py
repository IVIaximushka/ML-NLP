# multistep_forecasting.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import time

def recursive_strategy(model, X_train, y_train, h):
    model.fit(X_train, y_train)
    predictions = []
    current_features = X_train.iloc[-1:].copy()
    
    for i in range(h):
        pred = model.predict(current_features)[0]
        predictions.append(pred)
        
        # Обновляем признаки для следующего шага
        if i < h - 1:
            # Сдвигаем лаговые признаки
            for col in ['lag_1', 'lag_7', 'lag_30']:
                if col in current_features.columns:
                    current_features[col] = pred
            
            # Обновляем скользящие статистики (упрощенно)
            for col in current_features.columns:
                if 'rolling_mean' in col:
                    current_features[col] = pred
    
    return np.array(predictions)

def direct_strategy(model_class, X_train, y_train, h):
    predictions = []
    for step in range(1, h + 1):
        model = model_class()
        y_train_shifted = y_train.shift(-step).dropna()
        X_train_aligned = X_train.iloc[:len(y_train_shifted)]
        model.fit(X_train_aligned, y_train_shifted)
        pred = model.predict(X_train_aligned.iloc[-1:])[0]
        predictions.append(pred)
    return np.array(predictions)

def hybrid_strategy(model_class, X_train, y_train, h):
    switch_point = h // 2
    predictions = []
    
    # Рекурсивная для первых шагов
    recursive_preds = recursive_strategy(model_class(), X_train, y_train, switch_point)
    predictions.extend(recursive_preds)
    
    # Прямая для оставшихся шагов
    for step in range(switch_point + 1, h + 1):
        model = model_class()
        y_train_shifted = y_train.shift(-step).dropna()
        X_train_aligned = X_train.iloc[:len(y_train_shifted)]
        model.fit(X_train_aligned, y_train_shifted)
        pred = model.predict(X_train_aligned.iloc[-1:])[0]
        predictions.append(pred)
    
    return np.array(predictions)

def compare_strategies(X, y, h=7):
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:split_idx + h]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:split_idx + h]
    
    strategies = {
        'Рекурсивная': lambda: recursive_strategy(LinearRegression(), X_train, y_train, h),
        'Прямая': lambda: direct_strategy(LinearRegression, X_train, y_train, h),
        'Гибридная': lambda: hybrid_strategy(LinearRegression, X_train, y_train, h)
    }
    
    results = {}
    
    for name, strategy in strategies.items():
        start_time = time.time()
        predictions = strategy()
        execution_time = time.time() - start_time
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        results[name] = {
            'predictions': predictions,
            'mae': mae,
            'rmse': rmse,
            'execution_time': execution_time
        }
    
    return results
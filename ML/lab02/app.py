# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Добавляем путь к текущей директории для импорта наших модулей
sys.path.append(os.path.dirname(__file__))

from utils.decomposition import perform_decomposition, analyze_residuals
from utils.feature_engineering import main_feature_engineering
from utils.multistep_forecasting import compare_strategies
from utils.cross_validation import main_cross_validation
from utils.stationarity import main_stationarity_transformation
from utils.exponential_smoothing import main_exponential_smoothing

def load_data():
    """Загрузка данных через интерфейс"""
    st.sidebar.header("Загрузка данных")
    uploaded_file = st.sidebar.file_uploader("Выберите CSV/Parquet файл", type=['csv', 'parquet'])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_parquet(uploaded_file)
        return df
    return None

def main():
    st.title("📈 Интерактивная система прогнозирования временных рядов")
    
    # Загрузка данных
    df = load_data()
    
    if df is not None:
        st.sidebar.header("Настройки анализа")
        
        # Выбор переменных
        target_col = st.sidebar.selectbox("Выберите целевую переменную", df.columns)
        date_col = st.sidebar.selectbox("Выберите столбец с датой", df.columns)
        
        # Преобразование даты
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        
        # Настройки прогноза
        h = st.sidebar.selectbox("Горизонт прогноза h", [7, 30, 90], index=0)
        
        # Основные вкладки
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Данные", "🔍 Декомпозиция", "⚙️ Признаки", 
            "📈 Прогнозирование", "✅ Валидация", "📉 Стационарность", "🔄 Сглаживание"
        ])
        
        with tab1:
            st.header("Обзор данных")
            st.write("Первые 10 строк:")
            st.dataframe(df.head(10))
            st.write(f"Размер данных: {df.shape}")
            st.write("Описательная статистика:")
            st.dataframe(df.describe())
            
            # Визуализация ряда
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df[target_col], name=target_col))
            fig.update_layout(title=f"Временной ряд: {target_col}", xaxis_title="Дата", yaxis_title=target_col)
            st.plotly_chart(fig)
        
        with tab2:
            st.header("Декомпозиция временного ряда")
            model_type = st.radio("Тип декомпозиции", ["additive", "multiplicative"])
            period = st.selectbox("Период сезонности", [7, 30, 365], index=0)
            
            if st.button("Выполнить декомпозицию"):
                decomposition = perform_decomposition(df[target_col], period, model_type)
                
                if decomposition:
                    # Визуализация компонентов
                    fig = make_subplots(rows=4, cols=1, subplot_titles=['Исходный ряд', 'Тренд', 'Сезонность', 'Остатки'])
                    
                    fig.add_trace(go.Scatter(x=df.index, y=decomposition.observed, name='Исходный'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=decomposition.trend, name='Тренд'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=decomposition.seasonal, name='Сезонность'), row=3, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=decomposition.resid, name='Остатки'), row=4, col=1)
                    
                    fig.update_layout(height=800, title_text="Декомпозиция временного ряда")
                    st.plotly_chart(fig)
                    
                    # Анализ остатков
                    residuals_analysis = analyze_residuals(decomposition.resid, period, model_type)
                    st.write("Анализ остатков:")
                    st.json(residuals_analysis)
        
        with tab3:
            st.header("Feature Engineering")
            if st.button("Создать признаки"):
                df_with_features = main_feature_engineering(df, target_col)
                st.write("Данные с признаками:")
                st.dataframe(df_with_features.head())
                st.write(f"Количество признаков: {len(df_with_features.columns)}")
        
        with tab4:
            st.header("Многопшаговое прогнозирование")
            if st.button("Сравнить стратегии"):
                df_with_features = main_feature_engineering(df, target_col)
                results = compare_strategies(
                    df_with_features.drop(columns=[target_col]),
                    df_with_features[target_col],
                    h
                )
                
                # Визуализация прогнозов
                fig = go.Figure()
                for strategy, result in results.items():
                    fig.add_trace(go.Scatter(
                        x=list(range(1, h+1)),  # Исправлено: преобразование range в list
                        y=result['predictions'],
                        name=strategy,
                        mode='lines+markers'
                    ))
                
                fig.update_layout(title="Сравнение прогнозов по стратегиям", 
                                xaxis_title="Шаг прогноза", 
                                yaxis_title=target_col)
                st.plotly_chart(fig)
                
                # Таблица метрик
                metrics_data = []
                for strategy, result in results.items():
                    metrics_data.append({
                        'Стратегия': strategy,
                        'MAE': result['mae'],
                        'RMSE': result['rmse'],
                        'Время (сек)': result['execution_time']
                    })
                
                st.dataframe(pd.DataFrame(metrics_data))
        
        with tab5:
            st.header("Кросс-валидация временных рядов")
            if st.button("Выполнить кросс-валидацию"):
                df_with_features = main_feature_engineering(df, target_col)
                cv_results = main_cross_validation(df_with_features, target_col)
                st.write("Результаты кросс-валидации:")
                st.dataframe(cv_results)
        
        with tab6:
            st.header("Приведение к стационарности")
            if st.button("Анализировать стационарность"):
                transformation_results = main_stationarity_transformation(df, target_col)
                st.write("Результаты преобразований:")
                st.dataframe(transformation_results)
        
        with tab7:
            st.header("Модели экспоненциального сглаживания")
            if st.button("Обучить модели сглаживания"):
                smoothing_results = main_exponential_smoothing(df, target_col, h)
                
                # Визуализация прогнозов
                fig = go.Figure()
                
                # Добавляем исторические данные
                split_idx = int(len(df) * 0.8)
                train = df[target_col].iloc[:split_idx]
                test = df[target_col].iloc[split_idx:split_idx + h]
                
                fig.add_trace(go.Scatter(x=train.index, y=train.values, name='Обучающие данные'))
                fig.add_trace(go.Scatter(x=test.index, y=test.values, name='Фактические значения', line=dict(dash='dash')))
                
                # Добавляем прогнозы
                for _, model_result in smoothing_results.iterrows():
                    if model_result['forecast'] is not None:
                        fig.add_trace(go.Scatter(
                            x=test.index,
                            y=model_result['forecast'],
                            name=f"{model_result['model']} прогноз",
                            mode='lines+markers'
                        ))
                
                fig.update_layout(title="Сравнение моделей экспоненциального сглаживания")
                st.plotly_chart(fig)
                
                # Таблица метрик
                st.write("Метрики моделей:")
                st.dataframe(smoothing_results[['model', 'mae', 'rmse']])
    
    else:
        st.info("👈 Загрузите CSV или Parquet файл через боковую панель для начала анализа")

if __name__ == "__main__":
    main()
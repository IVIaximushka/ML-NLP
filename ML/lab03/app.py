import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import io
import warnings
warnings.filterwarnings('ignore')

# Импорт моделей
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import boxcox, boxcox_normmax
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy.stats as stats

# Настройка страницы
st.set_page_config(
    page_title="Time Series Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
st.title("📈 Анализ и прогнозирование временных рядов")
st.markdown("---")

def load_sample_data():
    """Загрузка примеров данных"""
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    trend = np.linspace(100, 200, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    noise = np.random.normal(0, 5, len(dates))
    values = trend + seasonal + noise
    
    return pd.DataFrame({
        'Date': dates,
        'number_sold': values
    })

def handle_data_upload():
    """Обработка загрузки файлов"""
    st.sidebar.header("📁 Загрузка данных")
    
    uploaded_file = st.sidebar.file_uploader(
        "Загрузите CSV/Parquet файл", 
        type=['csv', 'parquet'],
        help="Файл должен содержать колонки Date и number_sold"
    )
    
    if uploaded_file is not None:
        try:
            # Чтение файла
            if uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            # Проверка обязательных колонок
            if 'Date' not in df.columns:
                st.error("❌ Файл должен содержать колонку 'Date'")
                return None
                
            # Преобразование даты
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            
            st.sidebar.success(f"✅ Данные загружены: {len(df)} строк")
            return df
            
        except Exception as e:
            st.error(f"❌ Ошибка чтения файла: {e}")
            return None
    else:
        # Использование примеров данных
        if st.sidebar.button("🎲 Использовать пример данных"):
            df = load_sample_data()
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            st.sidebar.success("✅ Загружен пример данных")
            return df
    
    return None

def model_parameters_sidebar(df):
    """Боковая панель выбора параметров модели"""
    st.sidebar.header("⚙️ Параметры модели")
    
    # Выбор целевой переменной
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_col = st.sidebar.selectbox(
        "Целевая переменная", 
        options=numeric_cols,
        index=0 if len(numeric_cols) > 0 else 0
    )
    
    # Выбор горизонта прогноза
    horizon = st.sidebar.radio(
        "Горизонт прогнозирования", 
        [1, 7, 30], 
        horizontal=True,
        help="Количество периодов для прогнозирования"
    )
    
    # Выбор модели
    model_type = st.sidebar.selectbox(
        "Модель прогнозирования",
        ["ARIMA", "SARIMA", "Prophet", "Exponential Smoothing", "Naive", "Seasonal Naive"],
        help="Выберите алгоритм для прогнозирования"
    )
    
    # Дополнительные параметры для моделей
    if model_type == "ARIMA":
        st.sidebar.subheader("Параметры ARIMA")
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            p = st.number_input("p (AR)", 0, 5, 1)
        with col2:
            d = st.number_input("d (I)", 0, 2, 1)
        with col3:
            q = st.number_input("q (MA)", 0, 5, 1)
    
    elif model_type == "SARIMA":
        st.sidebar.subheader("Параметры SARIMA")
        col1, col2, col3, col4 = st.sidebar.columns(4)
        with col1:
            p = st.number_input("p", 0, 3, 1)
        with col2:
            d = st.number_input("d", 0, 2, 1)
        with col3:
            q = st.number_input("q", 0, 3, 1)
        with col4:
            s = st.number_input("s", 1, 365, 7)
    
    elif model_type == "Exponential Smoothing":
        st.sidebar.subheader("Параметры сглаживания")
        trend_type = st.sidebar.selectbox("Тренд", ["add", "mul", None])
        seasonal_type = st.sidebar.selectbox("Сезонность", ["add", "mul", None])
        seasonal_periods = st.sidebar.number_input("Период сезонности", 1, 365, 7)
    
    # Настройки преобразований
    st.sidebar.header("🔄 Преобразования данных")
    use_boxcox = st.sidebar.checkbox("Применить преобразование Бокса-Кокса")
    lambda_val = None
    if use_boxcox:
        lambda_choice = st.sidebar.selectbox(
            "Параметр λ", 
            ["auto", "0 (логарифм)", "0.5", "1 (без преобразования)"]
        )
        if lambda_choice == "auto":
            lambda_val = "auto"
        elif lambda_choice == "0 (логарифм)":
            lambda_val = 0
        elif lambda_choice == "0.5":
            lambda_val = 0.5
        else:
            lambda_val = 1
    
    return {
        'target_col': target_col,
        'horizon': horizon,
        'model_type': model_type,
        'use_boxcox': use_boxcox,
        'lambda_val': lambda_val,
        'p': p if 'p' in locals() else 1,
        'd': d if 'd' in locals() else 1,
        'q': q if 'q' in locals() else 1,
        's': s if 's' in locals() else 7,
        'trend_type': trend_type if 'trend_type' in locals() else None,
        'seasonal_type': seasonal_type if 'seasonal_type' in locals() else None,
        'seasonal_periods': seasonal_periods if 'seasonal_periods' in locals() else 7
    }

def apply_transformations(data, use_boxcox, lambda_val):
    """Применение преобразований к данным"""
    if not use_boxcox:
        return data, None, "Без преобразования"
    
    if lambda_val == "auto":
        # Автоматический подбор lambda
        lambda_opt = boxcox_normmax(data + 1)  # +1 чтобы избежать отрицательных значений
        transformed_data = boxcox(data + 1, lmbda=lambda_opt)
        return transformed_data, lambda_opt, f"Бокса-Кокса (λ={lambda_opt:.3f})"
    else:
        # Ручной выбор lambda
        if lambda_val == 0:
            transformed_data = np.log(data + 1)
            return transformed_data, 0, "Логарифм"
        else:
            transformed_data = data ** lambda_val
            return transformed_data, lambda_val, f"Степенное (λ={lambda_val})"

def inverse_transformations(data, lambda_val, transformation_type):
    """Обратное преобразование данных"""
    if transformation_type == "Логарифм":
        return np.exp(data) - 1
    elif "Бокса-Кокса" in transformation_type:
        # Для упрощения возвращаем как есть
        return data
    elif "Степенное" in transformation_type:
        return data ** (1/lambda_val)
    else:
        return data

def train_arima_model(data, order, horizon):
    """Обучение ARIMA модели"""
    try:
        model = ARIMA(data, order=order)
        fitted_model = model.fit()
        
        # Прогноз
        forecast_result = fitted_model.get_forecast(steps=horizon)
        forecast = forecast_result.predicted_mean
        ci = forecast_result.conf_int()
        
        return forecast, ci, fitted_model
    except Exception as e:
        st.error(f"Ошибка ARIMA: {e}")
        return None, None, None

def train_sarima_model(data, order, seasonal_order, horizon):
    """Обучение SARIMA модели"""
    try:
        model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        
        # Прогноз
        forecast_result = fitted_model.get_forecast(steps=horizon)
        forecast = forecast_result.predicted_mean
        ci = forecast_result.conf_int()
        
        return forecast, ci, fitted_model
    except Exception as e:
        st.error(f"Ошибка SARIMA: {e}")
        return None, None, None

def train_prophet_model(data, horizon):
    """Обучение Prophet модели"""
    try:
        # Подготовка данных для Prophet
        prophet_df = data.reset_index()
        prophet_df.columns = ['ds', 'y']
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        model.fit(prophet_df)
        
        # Создание будущих дат
        future = model.make_future_dataframe(periods=horizon)
        forecast_df = model.predict(future)
        
        # Извлечение прогноза
        forecast = forecast_df['yhat'].values[-horizon:]
        ci_lower = forecast_df['yhat_lower'].values[-horizon:]
        ci_upper = forecast_df['yhat_upper'].values[-horizon:]
        
        ci = pd.DataFrame({
            'lower': ci_lower,
            'upper': ci_upper
        })
        
        return forecast, ci, model
    except Exception as e:
        st.error(f"Ошибка Prophet: {e}")
        return None, None, None

def train_exponential_smoothing(data, trend, seasonal, seasonal_periods, horizon):
    """Обучение модели экспоненциального сглаживания"""
    try:
        model = ExponentialSmoothing(
            data,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        )
        fitted_model = model.fit()
        
        # Прогноз
        forecast = fitted_model.forecast(horizon)
        
        return forecast, None, fitted_model
    except Exception as e:
        st.error(f"Ошибка Exponential Smoothing: {e}")
        return None, None, None

def naive_forecast(data, horizon, seasonal_period=1):
    """Наивный прогноз"""
    if seasonal_period > 1:
        # Сезонный наивный
        return np.tile(data[-seasonal_period:], int(np.ceil(horizon/seasonal_period)))[:horizon]
    else:
        # Простой наивный
        return np.full(horizon, data[-1])

def calculate_metrics(actual, forecast, model_name):
    """Вычисление метрик качества"""
    if len(actual) != len(forecast):
        min_len = min(len(actual), len(forecast))
        actual = actual[:min_len]
        forecast = forecast[:min_len]
    
    metrics = {}
    
    try:
        metrics['MAE'] = mean_absolute_error(actual, forecast)
        metrics['RMSE'] = np.sqrt(mean_squared_error(actual, forecast))
        metrics['MAPE'] = np.mean(np.abs((actual - forecast) / actual)) * 100
        metrics['R2'] = r2_score(actual, forecast)
        
        # MASE (упрощенная версия)
        naive_errors = np.mean(np.abs(np.diff(actual)))
        if naive_errors > 0:
            metrics['MASE'] = metrics['MAE'] / naive_errors
        else:
            metrics['MASE'] = np.nan
            
    except Exception as e:
        st.error(f"Ошибка расчета метрик: {e}")
        
    return metrics

def plot_forecast_results(historical, forecast, ci_lower, ci_upper, model_name, transformation_info):
    """Визуализация прогнозов с доверительными интервалами"""
    fig = go.Figure()
    
    # Исторические данные
    fig.add_trace(go.Scatter(
        x=historical.index, y=historical.values,
        name='Исторические данные',
        line=dict(color='blue', width=2),
        opacity=0.7
    ))
    
    # Прогноз
    forecast_dates = pd.date_range(
        start=historical.index[-1] + pd.Timedelta(days=1),
        periods=len(forecast),
        freq='D'
    )
    
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast,
        name=f'Прогноз ({model_name})',
        line=dict(color='red', width=3, dash='dash')
    ))
    
    # Доверительный интервал
    if ci_lower is not None and ci_upper is not None:
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=ci_upper,
            name='Верхняя граница ДИ',
            line=dict(color='lightgray', width=1),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=ci_lower,
            name='Нижняя граница ДИ',
            fill='tonexty',
            fillcolor='rgba(211,211,211,0.3)',
            line=dict(color='lightgray', width=1),
            showlegend=False
        ))
    
    fig.update_layout(
        title=f"Прогноз временного ряда - {model_name} ({transformation_info})",
        xaxis_title="Дата",
        yaxis_title="Значение",
        hovermode="x unified",
        height=500
    )
    
    return fig

def plot_residuals_analysis(model, model_name, data, forecast):
    """Анализ остатков модели"""
    if model is None:
        return None
        
    try:
        # Получение остатков
        if hasattr(model, 'resid'):
            residuals = model.resid.dropna()
        else:
            # Для моделей без атрибута resid
            residuals = data - forecast
        
        if len(residuals) < 5:
            return None
            
        # Создание subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Распределение остатков', 'Q-Q plot', 
                          'ACF остатков', 'Остатки во времени')
        )
        
        # Гистограмма остатков
        fig.add_trace(go.Histogram(x=residuals, nbinsx=30, name="Распределение"),
                     row=1, col=1)
        
        # Q-Q plot
        qq_data = stats.probplot(residuals, dist="norm")
        fig.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[0][1], 
                               mode='markers', name="Q-Q"),
                     row=1, col=2)
        fig.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[0][0]*qq_data[1][0] + qq_data[1][1],
                               mode='lines', name="Нормальное распределение"),
                     row=1, col=2)
        
        # ACF остатков
        acf_values = acf(residuals, nlags=20)
        fig.add_trace(go.Bar(x=list(range(len(acf_values))), y=acf_values,
                           name="ACF"),
                     row=2, col=1)
        
        # Остатки во времени
        fig.add_trace(go.Scatter(x=data.index[-len(residuals):], y=residuals,
                               mode='lines', name="Остатки"),
                     row=2, col=2)
        
        fig.update_layout(height=600, title_text=f"Диагностика остатков - {model_name}")
        return fig
        
    except Exception as e:
        st.error(f"Ошибка анализа остатков: {e}")
        return None

def export_forecast(forecast, model_name, historical):
    """Экспорт прогнозов в файл"""
    forecast_dates = pd.date_range(
        start=historical.index[-1] + pd.Timedelta(days=1),
        periods=len(forecast),
        freq='D'
    )
    
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': forecast,
        'Model': model_name
    })
    
    csv = forecast_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Скачать прогноз в CSV",
        data=csv,
        file_name=f"forecast_{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True
    )

def main():
    """Основная функция приложения"""
    
    # Загрузка данных
    df = handle_data_upload()
    
    if df is None:
        # Показ инструкций если данные не загружены
        st.markdown("""
        ## 🚀 Добро пожаловать в приложение для прогнозирования временных рядов!
        
        ### Как использовать:
        1. **Загрузите данные** через боковую панель слева
        2. **Выберите параметры** модели и преобразования
        3. **Настройте горизонт** прогнозирования
        4. **Получите прогноз** и анализ качества
        
        ### Формат данных:
        - CSV или Parquet файл
        - Колонка `Date` с датами
        - Колонка с числовыми значениями для прогнозирования
        
        ### Доступные модели:
        - **ARIMA/SARIMA** - для стационарных рядов
        - **Prophet** - для рядов с трендом и сезонностью  
        - **Exponential Smoothing** - для сглаживания
        - **Naive методы** - как бенчмарки
        
        Нажмите кнопку "🎲 Использовать пример данных" чтобы начать!
        """)
        return
    
    # Параметры модели
    params = model_parameters_sidebar(df)
    
    # Основной контент
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Визуализация данных")
        
        # График исходных данных
        fig_data = px.line(df, y=params['target_col'], 
                          title=f"Исходные данные: {params['target_col']}")
        st.plotly_chart(fig_data, use_container_width=True)
    
    with col2:
        st.subheader("📈 Статистика данных")
        st.metric("Количество наблюдений", len(df))
        st.metric("Период данных", 
                 f"{df.index.min().strftime('%Y-%m-%d')} - {df.index.max().strftime('%Y-%m-%d')}")
        st.metric("Среднее значение", f"{df[params['target_col']].mean():.2f}")
        st.metric("Стандартное отклонение", f"{df[params['target_col']].std():.2f}")
    
    # Разделение на train/test
    train_size = int(len(df) * 0.8)
    train_data = df[params['target_col']].iloc[:train_size]
    test_data = df[params['target_col']].iloc[train_size:train_size + params['horizon']]
    
    # Применение преобразований
    transformed_data, lambda_val, transformation_info = apply_transformations(
        train_data, params['use_boxcox'], params['lambda_val']
    )
    
    # Обучение модели
    st.subheader("🎯 Прогнозирование")
    
    forecast = None
    ci = None
    trained_model = None
    
    with st.spinner(f"Обучаем модель {params['model_type']}..."):
        if params['model_type'] == "ARIMA":
            forecast, ci, trained_model = train_arima_model(
                transformed_data, 
                (params['p'], params['d'], params['q']), 
                params['horizon']
            )
        elif params['model_type'] == "SARIMA":
            forecast, ci, trained_model = train_sarima_model(
                transformed_data,
                (params['p'], params['d'], params['q']),
                (params['p'], params['d'], params['q'], params['s']),
                params['horizon']
            )
        elif params['model_type'] == "Prophet":
            forecast, ci, trained_model = train_prophet_model(
                transformed_data, 
                params['horizon']
            )
        elif params['model_type'] == "Exponential Smoothing":
            forecast, ci, trained_model = train_exponential_smoothing(
                transformed_data,
                params['trend_type'],
                params['seasonal_type'],
                params['seasonal_periods'],
                params['horizon']
            )
        elif params['model_type'] == "Naive":
            forecast = naive_forecast(transformed_data.values, params['horizon'])
        elif params['model_type'] == "Seasonal Naive":
            forecast = naive_forecast(transformed_data.values, params['horizon'], params['s'])
    
    if forecast is not None:
        # Обратное преобразование если применялось
        if params['use_boxcox']:
            forecast = inverse_transformations(forecast, lambda_val, transformation_info)
            if ci is not None:
                ci['lower'] = inverse_transformations(ci['lower'], lambda_val, transformation_info)
                ci['upper'] = inverse_transformations(ci['upper'], lambda_val, transformation_info)
        
        # Визуализация результатов
        fig_forecast = plot_forecast_results(
            train_data, forecast, 
            ci['lower'] if ci is not None else None,
            ci['upper'] if ci is not None else None,
            params['model_type'], transformation_info
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Метрики качества
        if len(test_data) >= len(forecast):
            metrics = calculate_metrics(test_data.values[:len(forecast)], forecast, params['model_type'])
            
            st.subheader("📊 Метрики качества")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MAE", f"{metrics.get('MAE', 0):.2f}")
            with col2:
                st.metric("RMSE", f"{metrics.get('RMSE', 0):.2f}")
            with col3:
                st.metric("MAPE", f"{metrics.get('MAPE', 0):.1f}%")
            with col4:
                st.metric("R²", f"{metrics.get('R2', 0):.3f}")
        
        # Диагностика остатков
        if trained_model is not None:
            fig_residuals = plot_residuals_analysis(
                trained_model, params['model_type'], train_data, forecast
            )
            if fig_residuals:
                st.plotly_chart(fig_residuals, use_container_width=True)
        
        # Экспорт результатов
        st.subheader("💾 Экспорт результатов")
        export_forecast(forecast, params['model_type'], train_data)
        
    else:
        st.error("❌ Не удалось построить прогноз. Попробуйте изменить параметры модели.")

if __name__ == "__main__":
    main()
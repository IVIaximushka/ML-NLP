import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
import plotly.subplots as sp
from io import StringIO

# Конфигурация страницы
st.set_page_config(
    page_title="Анализ временных рядов",
    page_icon="📈",
    layout="wide"
)

# Заголовок приложения
st.title("📈 Интерактивный анализ временных рядов")
st.markdown("Загрузите ваш CSV-файл и настройте параметры анализа")

# Боковая панель для загрузки данных и настроек
with st.sidebar:
    st.header("📁 Загрузка данных")
    
    uploaded_file = st.file_uploader(
        "Выберите CSV файл", 
        type=['csv'],
        help="Файл должен содержать столбец с датой и числовые столбцы"
    )
    
    st.header("⚙️ Параметры анализа")
    
    # Выбор переменных
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        date_col = st.selectbox("Выберите столбец с датой", df.columns)
        target_col = st.selectbox("Выберите целевую переменную", df.columns)
        
        # Преобразование даты
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        except:
            st.error("Ошибка при преобразовании даты")
    
    # Параметры анализа
    seasonality_period = st.number_input(
        "Период сезонности", 
        min_value=2, 
        value=7,
        help="Например: 7 для недельной, 30 для месячной, 365 для годовой сезонности"
    )
    
    max_lags = st.number_input(
        "Максимальное количество лагов для ACF/PACF",
        min_value=5,
        value=40
    )
    
    window_size = st.number_input(
        "Окно для скользящего среднего",
        min_value=2,
        value=30
    )
    
    decomposition_model = st.radio(
        "Модель декомпозиции",
        ["additive", "multiplicative"]
    )
    
    # Кнопка запуска анализа
    analyze_btn = st.button("🚀 Запустить анализ", type="primary")

# Основная область приложения
if uploaded_file is not None and analyze_btn:
    try:
        # Вкладки для различных анализов
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Основной график", 
            "🔍 Декомпозиция", 
            "📈 ACF/PACF",
            "📉 Стационарность",
            "📋 Отчет"
        ])
        
        with tab1:
            st.header("Основной временной ряд")
            
            # График исходного ряда
            fig_main = go.Figure()
            fig_main.add_trace(go.Scatter(
                x=df.index, 
                y=df[target_col],
                name="Исходный ряд",
                line=dict(color='blue')
            ))
            
            # Скользящее среднее
            rolling_mean = df[target_col].rolling(window=window_size).mean()
            fig_main.add_trace(go.Scatter(
                x=df.index, 
                y=rolling_mean,
                name=f"Скользящее среднее ({window_size})",
                line=dict(color='red', dash='dash')
            ))
            
            fig_main.update_layout(
                title="Временной ряд со скользящим средним",
                xaxis_title="Дата",
                yaxis_title=target_col
            )
            
            st.plotly_chart(fig_main, use_container_width=True)
            
            # Базовая статистика
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Среднее", f"{df[target_col].mean():.2f}")
            with col2:
                st.metric("Стандартное отклонение", f"{df[target_col].std():.2f}")
            with col3:
                st.metric("Минимум", f"{df[target_col].min():.2f}")
            with col4:
                st.metric("Максимум", f"{df[target_col].max():.2f}")
        
        with tab2:
            st.header("Декомпозиция временного ряда")
            
            # Декомпозиция
            if len(df) >= 2 * seasonality_period:
                decomposition = seasonal_decompose(
                    df[target_col].dropna(),
                    model=decomposition_model,
                    period=seasonality_period
                )
                
                # График декомпозиции
                fig_decomp = sp.make_subplots(
                    rows=4, cols=1,
                    subplot_titles=['Исходный ряд', 'Тренд', 'Сезонность', 'Остатки'],
                    vertical_spacing=0.05
                )
                
                fig_decomp.add_trace(
                    go.Scatter(x=df.index, y=decomposition.observed, name="Исходный"),
                    row=1, col=1
                )
                
                fig_decomp.add_trace(
                    go.Scatter(x=df.index, y=decomposition.trend, name="Тренд"),
                    row=2, col=1
                )
                
                fig_decomp.add_trace(
                    go.Scatter(x=df.index, y=decomposition.seasonal, name="Сезонность"),
                    row=3, col=1
                )
                
                fig_decomp.add_trace(
                    go.Scatter(x=df.index, y=decomposition.resid, name="Остатки"),
                    row=4, col=1
                )
                
                fig_decomp.update_layout(height=800, showlegend=False)
                st.plotly_chart(fig_decomp, use_container_width=True)
            else:
                st.warning(f"Для декомпозиции нужно как минимум {2 * seasonality_period} наблюдений")
        
        with tab3:
            st.header("Автокорреляционный анализ")
            
            # ACF и PACF
            series_clean = df[target_col].dropna()
            
            acf_values = acf(series_clean, nlags=max_lags)
            pacf_values = pacf(series_clean, nlags=max_lags)
            
            fig_acf = sp.make_subplots(rows=1, cols=2, subplot_titles=['ACF', 'PACF'])
            
            # ACF plot
            fig_acf.add_trace(
                go.Bar(x=list(range(len(acf_values))), y=acf_values, name="ACF"),
                row=1, col=1
            )
            
            # PACF plot
            fig_acf.add_trace(
                go.Bar(x=list(range(len(pacf_values))), y=pacf_values, name="PACF"),
                row=1, col=2
            )
            
            # Добавление доверительных интервалов
            conf_int = 1.96 / np.sqrt(len(series_clean))
            
            fig_acf.add_hline(y=conf_int, line_dash="dash", line_color="red", row=1, col=1)
            fig_acf.add_hline(y=-conf_int, line_dash="dash", line_color="red", row=1, col=1)
            fig_acf.add_hline(y=conf_int, line_dash="dash", line_color="red", row=1, col=2)
            fig_acf.add_hline(y=-conf_int, line_dash="dash", line_color="red", row=1, col=2)
            
            fig_acf.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_acf, use_container_width=True)
        
        with tab4:
            st.header("Тесты на стационарность")
            
            # Тест Дики-Фуллера
            adf_result = adfuller(df[target_col].dropna())
            
            # Тест KPSS
            try:
                kpss_result = kpss(df[target_col].dropna())
            except:
                kpss_result = [None, None, None, None]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Тест Дики-Фуллера (ADF)")
                st.metric("ADF Statistic", f"{adf_result[0]:.4f}")
                st.metric("p-value", f"{adf_result[1]:.4f}")
                st.metric(
                    "Стационарен", 
                    "Да" if adf_result[1] < 0.05 else "Нет",
                    delta="Стационарен" if adf_result[1] < 0.05 else "Нестационарен",
                    delta_color="normal" if adf_result[1] < 0.05 else "inverse"
                )
            
            with col2:
                st.subheader("Тест KPSS")
                if kpss_result[0] is not None:
                    st.metric("KPSS Statistic", f"{kpss_result[0]:.4f}")
                    st.metric("p-value", f"{kpss_result[1]:.4f}")
                    st.metric(
                        "Стационарен", 
                        "Да" if kpss_result[1] > 0.05 else "Нет",
                        delta="Стационарен" if kpss_result[1] > 0.05 else "Нестационарен",
                        delta_color="normal" if kpss_result[1] > 0.05 else "inverse"
                    )
                else:
                    st.error("KPSS тест не удалось выполнить")
            
            # Интерпретация
            st.info("""
            **Интерпретация:**
            - **ADF:** p-value < 0.05 → ряд стационарен
            - **KPSS:** p-value > 0.05 → ряд стационарен
            """)
        
        with tab5:
            st.header("Полный отчет")
            
            # Создание отчета
            report = f"""
            # Отчет анализа временного ряда
            
            ## Основные характеристики
            - **Переменная:** {target_col}
            - **Период:** {df.index.min()} - {df.index.max()}
            - **Количество наблюдений:** {len(df)}
            - **Пропущенные значения:** {df[target_col].isna().sum()}
            
            ## Статистические тесты
            - **Тест ADF:** p-value = {adf_result[1]:.4f} ({'стационарен' if adf_result[1] < 0.05 else 'нестационарен'})
            - **Тест KPSS:** p-value = {kpss_result[1] if kpss_result[1] is not None else 'N/A':.4f} ({'стационарен' if kpss_result[1] is not None and kpss_result[1] > 0.05 else 'нестационарен'})
            
            ## Параметры анализа
            - **Модель декомпозиции:** {decomposition_model}
            - **Период сезонности:** {seasonality_period}
            - **Окно скользящего среднего:** {window_size}
            """
            
            st.markdown(report)
            
            # Экспорт отчета
            html_report = f"""
            <html>
            <head>
                <title>Анализ временного ряда</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <h1>Отчет анализа временного ряда</h1>
                {report.replace('\n', '<br>')}
            </body>
            </html>
            """
            
            st.download_button(
                label="📥 Скачать HTML отчет",
                data=html_report,
                file_name="time_series_report.html",
                mime="text/html"
            )
    
    except Exception as e:
        st.error(f"Ошибка при анализе: {str(e)}")
else:
    # Демонстрационный режим
    st.info("👈 Загрузите CSV файл и настройте параметры в боковой панели")
    
    # Пример структуры данных
    st.subheader("Пример ожидаемой структуры данных:")
    sample_data = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'Sales': np.random.randn(100).cumsum() + 100,
        'Temperature': np.sin(np.linspace(0, 4*np.pi, 100)) * 10 + 20
    })
    st.dataframe(sample_data.head(10))

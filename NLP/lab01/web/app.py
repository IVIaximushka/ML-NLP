import streamlit as st
import pandas as pd
import json
import plotly.express as px
from utils.tokenization import apply_tokenization_pipeline
from utils.metrics import calculate_basic_metrics, calculate_oov_rate, calculate_processing_speed
from utils.visualization import (create_token_length_distribution, create_frequency_plot,
                               create_top_tokens_chart, create_metrics_comparison, create_wordcloud)
import tempfile
import os

# Настройка страницы
st.set_page_config(
    page_title="Анализатор токенизации текста",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
st.title("📊 Интерактивный анализатор токенизации текста")
st.markdown("""
Это приложение позволяет анализировать различные методы токенизации и нормализации текста.
Загрузите свой датасет или используйте примеры данных для начала анализа.
""")

# Боковая панель с настройками
st.sidebar.header("⚙️ Настройки обработки")

# Выбор источника данных
data_source = st.sidebar.radio("Источник данных:", 
                              ["Пример данных", "Загрузить файл"])

texts = []
if data_source == "Пример данных":
    # Генерируем пример данных
    sample_texts = [
        "Президент Казахстана Касым-Жомарт Токаев провел встречу с президентом Украины Владимиром Зеленским.",
        "Главы государств обсудили вопросы экономического сотрудничества между двумя странами.",
        "Встреча состоялась в рамках заседания Генеральной Ассамблеи ООН в Нью-Йорке.",
        "Стороны договорились активизировать торгово-экономические отношения."
    ]
    texts = sample_texts
    st.sidebar.info(f"Используется пример данных: {len(texts)} текстов")

else:
    uploaded_file = st.sidebar.file_uploader("Загрузите JSONL файл", type=['jsonl', 'json'])
    if uploaded_file is not None:
        try:
            # Чтение загруженного файла
            lines = uploaded_file.getvalue().decode('utf-8').splitlines()
            for line in lines:
                data = json.loads(line)
                text = data.get('header', '') + ' ' + data.get('text', '')
                texts.append(text)
            st.sidebar.success(f"Загружено {len(texts)} текстов")
        except Exception as e:
            st.sidebar.error(f"Ошибка загрузки файла: {e}")

# Настройки токенизации
st.sidebar.subheader("Методы обработки")

col1, col2 = st.sidebar.columns(2)

with col1:
    tokenizer_method = st.selectbox(
        "Токенизация:",
        ["razdel", "spacy", "nltk", "naive", "regex"],
        help="Выберите метод токенизации текста"
    )

with col2:
    normalizer_method = st.selectbox(
        "Нормализация:",
        ["none", "pymorphy2", "spacy", "porter", "snowball"],
        help="Выберите метод нормализации токенов"
    )

# Дополнительные настройки
max_texts = st.sidebar.slider("Максимальное количество текстов для анализа:", 
                             min_value=10, max_value=1000, value=100 if len(texts) > 100 else len(texts))

# Кнопка запуска анализа
analyze_button = st.sidebar.button("🚀 Запустить анализ", type="primary")

# Основная область контента
if analyze_button and texts:
    # Ограничиваем количество текстов
    analysis_texts = texts[:max_texts]
    
    # Прогресс-бар
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Обработка текстов
    status_text.text("Токенизация текстов...")
    all_tokens = []
    
    for i, text in enumerate(analysis_texts):
        tokens = apply_tokenization_pipeline(text, tokenizer_method, normalizer_method)
        all_tokens.append(tokens)
        progress_bar.progress((i + 1) / len(analysis_texts))
    
    # Расчет метрик
    status_text.text("Расчет метрик...")
    metrics = calculate_basic_metrics(all_tokens)
    
    # Расчет OOV rate (если есть базовый метод для сравнения)
    base_tokens = []
    for text in analysis_texts:
        base_tokens.append(apply_tokenization_pipeline(text, "razdel", "none"))
    
    oov_rate = calculate_oov_rate(base_tokens, all_tokens)
    
    # Расчет скорости
    speed = calculate_processing_speed(analysis_texts[:10], 
                                     lambda x: apply_tokenization_pipeline(x, tokenizer_method, normalizer_method))
    
    status_text.text("Генерация визуализаций...")
    
    # Отображение результатов
    st.header("📈 Результаты анализа")
    
    # Ключевые метрики
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Общее количество токенов", f"{metrics.get('total_tokens', 0):,}")
    
    with col2:
        st.metric("Размер словаря", f"{metrics.get('vocabulary_size', 0):,}")
    
    with col3:
        st.metric("Доля OOV", f"{oov_rate:.2%}")
    
    with col4:
        st.metric("Скорость (текстов/сек)", f"{speed:.1f}")
    
    # Визуализации
    tab1, tab2, tab3, tab4 = st.tabs(["📏 Распределение длин", "📊 Частотность", "🏆 Топ токены", "☁️ Облако слов"])
    
    with tab1:
        if metrics.get('token_lengths'):
            fig_length = create_token_length_distribution(metrics['token_lengths'])
            st.plotly_chart(fig_length, use_container_width=True)
    
    with tab2:
        if metrics.get('frequencies'):
            fig_freq = create_frequency_plot(metrics['frequencies'])
            st.plotly_chart(fig_freq, use_container_width=True)
    
    with tab3:
        if metrics.get('most_common_tokens'):
            fig_top = create_top_tokens_chart(metrics['most_common_tokens'])
            st.plotly_chart(fig_top, use_container_width=True)
    
    with tab4:
        wordcloud_img = create_wordcloud(all_tokens)
        if wordcloud_img:
            st.image(wordcloud_img, use_column_width=True)
    
    # Детальная таблица с метриками
    st.subheader("📋 Детальная статистика")
    
    metrics_df = pd.DataFrame([{
        'Метод токенизации': tokenizer_method,
        'Метод нормализации': normalizer_method,
        'Количество текстов': len(analysis_texts),
        'Всего токенов': metrics.get('total_tokens', 0),
        'Уникальные токены': metrics.get('unique_tokens', 0),
        'Средняя длина токена': f"{metrics.get('avg_token_length', 0):.2f}",
        'Лексическое разнообразие': f"{metrics.get('lexical_diversity', 0):.4f}",
        'Доля OOV': f"{oov_rate:.2%}",
        'Скорость обработки': f"{speed:.1f} текстов/сек"
    }])
    
    st.dataframe(metrics_df, use_container_width=True)
    
    # Сравнение методов (если нужно сравнить несколько)
    st.subheader("🔄 Сравнение методов")
    
    compare_methods = st.multiselect(
        "Выберите методы для сравнения:",
        ["razdel+none", "razdel+pymorphy2", "spacy+spacy", "nltk+porter"],
        default=["razdel+none", "razdel+pymorphy2"]
    )
    
    if st.button("Сравнить методы") and compare_methods:
        comparison_metrics = {}
        
        for method in compare_methods:
            tokenizer, normalizer = method.split('+')
            comp_tokens = []
            
            for text in analysis_texts[:50]:  # Используем подмножество для скорости
                tokens = apply_tokenization_pipeline(text, tokenizer, normalizer)
                comp_tokens.append(tokens)
            
            comp_metrics = calculate_basic_metrics(comp_tokens)
            comparison_metrics[method] = comp_metrics
        
        fig_compare = create_metrics_comparison(comparison_metrics)
        st.plotly_chart(fig_compare, use_container_width=True)
    
    # Экспорт результатов
    st.subheader("💾 Экспорт результатов")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Экспорт метрик в CSV
        csv = metrics_df.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label="📥 Скачать метрики (CSV)",
            data=csv,
            file_name=f"tokenization_metrics_{tokenizer_method}_{normalizer_method}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Экспорт токенов
        tokens_export = "\n".join([" ".join(tokens) for tokens in all_tokens])
        st.download_button(
            label="📥 Скачать токены (TXT)",
            data=tokens_export,
            file_name=f"tokens_{tokenizer_method}_{normalizer_method}.txt",
            mime="text/plain"
        )
    
    status_text.text("✅ Анализ завершен!")

elif not texts:
    st.info("👈 Загрузите данные или используйте пример данных для начала анализа")

else:
    # Демонстрационный контент при первом запуске
    st.markdown("""
    ## 🎯 Возможности приложения
    
    ### 📊 Анализируемые метрики:
    - **Распределение длин токенов** - гистограмма длин обработанных токенов
    - **Частотность токенов** - закон Ципфа для анализа распределения частот
    - **Топ-20 токенов** - самые частые слова после обработки
    - **Облако слов** - визуализация наиболее значимых токенов
    
    ### ⚙️ Поддерживаемые методы:
    **Токенизация:**
    - `razdel` - высококачественная токенизация для русского языка
    - `spacy` - промышленная библиотека NLP
    - `nltk` - классическая библиотека обработки текста
    - `naive` - наивная токенизация по пробелам
    - `regex` - регулярные выражения
    
    **Нормализация:**
    - `none` - без нормализации
    - `pymorphy2` - морфологический анализ для русского
    - `spacy` - лемматизация средствами spaCy
    - `porter` - стемминг Портера
    - `snowball` - стемминг Сноуболл
    
    ### 🚀 Чтобы начать:
    1. Выберите источник данных в боковой панели
    2. Настройте параметры токенизации и нормализации
    3. Нажмите кнопку "Запустить анализ"
    4. Изучите результаты на интерактивных графиках
    """)

# Футер
st.markdown("---")
st.markdown("*Анализатор токенизации текста v1.0 | Разработано для сравнительного анализа методов NLP*")

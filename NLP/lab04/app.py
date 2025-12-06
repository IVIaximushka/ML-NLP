import streamlit as st
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from itertools import islice

# --- Конфигурация страницы ---
st.set_page_config(
    page_title="Анализ кластеризации текстов",
    page_icon="📊",
    layout="wide"
)

# --- Заголовок и описание ---
st.title("📊 Веб-интерфейс для анализа кластеризации текстов")
st.markdown("""
Это приложение объединяет этапы лабораторной работы:
1. **Загрузка и предобработка** корпуса из `corpus.jsonl`
2. **Векторизация текстов** с помощью TF-IDF
3. **Кластеризация** различными алгоритмами
4. **Визуализация** и **анализ** результатов
""")

# --- Боковая панель: выбор параметров ---
st.sidebar.header("⚙️ Параметры обработки")

# 1. Загрузка данных
st.sidebar.subheader("1. Данные")
sample_size = st.sidebar.slider("Число документов для анализа", 10, 200, 50)

# 2. Векторизация
st.sidebar.subheader("2. Векторизация")
vectorization_method = st.sidebar.radio(
    "Метод векторизации",
    ["TF-IDF"]
)
max_features = st.sidebar.slider("Макс. число признаков (TF-IDF)", 50, 500, 100)

# 3. Кластеризация
st.sidebar.subheader("3. Алгоритм кластеризации")
clustering_algorithm = st.sidebar.selectbox(
    "Выберите алгоритм",
    ["K-Means", "DBSCAN", "Иерархическая кластеризация", "Gaussian Mixture"]
)

# Параметры в зависимости от алгоритма
if clustering_algorithm == "K-Means":
    n_clusters = st.sidebar.slider("Число кластеров", 2, 10, 5)
    params = {"n_clusters": n_clusters, "random_state": 42}
elif clustering_algorithm == "DBSCAN":
    eps = st.sidebar.slider("eps", 0.1, 1.0, 0.5, step=0.1)
    min_samples = st.sidebar.slider("min_samples", 2, 10, 5)
    params = {"eps": eps, "min_samples": min_samples}
elif clustering_algorithm == "Иерархическая кластеризация":
    n_clusters = st.sidebar.slider("Число кластеров", 2, 10, 5)
    linkage = st.sidebar.selectbox("Связь", ["ward", "complete", "average", "single"])
    params = {"n_clusters": n_clusters, "linkage": linkage}
else:  # Gaussian Mixture
    n_components = st.sidebar.slider("Число компонент", 2, 10, 5)
    params = {"n_components": n_components, "random_state": 42}

# --- Основная часть приложения ---
@st.cache_data
def load_corpus(limit):
    """Загрузка и предобработка корпуса"""
    with open('corpus.jsonl', 'r', encoding='utf-8') as f:
        corpus = []
        for i, line in enumerate(f):
            if i >= limit:
                break
            try:
                corpus.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    if not corpus:
        st.error("Не удалось загрузить данные из файла corpus.jsonl")
        return pd.DataFrame()
    
    df = pd.DataFrame(corpus)
    
    # Безопасная очистка текста
    if 'text' in df.columns:
        df['text_clean'] = df['text'].apply(
            lambda x: str(x).lower() if pd.notnull(x) else ''
        ).str.replace(r'[^\w\s]', ' ', regex=True)
    else:
        st.error("В данных отсутствует колонка 'text'")
        return pd.DataFrame()
    
    return df

@st.cache_data
def vectorize_texts(texts, method="TF-IDF", max_features=100):
    """Векторизация текстов"""
    if method == "TF-IDF":
        vectorizer = TfidfVectorizer(max_features=max_features, max_df=0.8, min_df=2)
        vectors = vectorizer.fit_transform(texts).toarray()
        feature_names = vectorizer.get_feature_names_out()
        return vectors, feature_names, vectorizer
    return None, None, None

def apply_clustering(X, algorithm, params):
    """Применение выбранного алгоритма кластеризации"""
    if algorithm == "K-Means":
        model = KMeans(**params, n_init=10)
    elif algorithm == "DBSCAN":
        model = DBSCAN(**params)
    elif algorithm == "Иерархическая кластеризация":
        model = AgglomerativeClustering(**params)
    else:  # Gaussian Mixture
        model = GaussianMixture(**params)
    
    if algorithm == "Gaussian Mixture":
        labels = model.fit_predict(X)
    else:
        labels = model.fit_predict(X)
    
    return labels, model

def get_top_tfidf_words(X_tfidf, feature_names, labels, cluster_id, top_n=10):
    """Получение топ-N слов для кластера по TF-IDF"""
    cluster_indices = np.where(labels == cluster_id)[0]
    if len(cluster_indices) == 0:
        return []
    
    cluster_tfidf = X_tfidf[cluster_indices].mean(axis=0)
    top_indices = np.argsort(cluster_tfidf)[-top_n:][::-1]
    return [(feature_names[i], cluster_tfidf[i]) for i in top_indices]

# --- Загрузка данных ---
st.header("📁 Загруженные данные")
df = load_corpus(sample_size)

if df.empty:
    st.stop()

st.write(f"Загружено **{len(df)}** документов")

# Отображение данных в компактном формате
col1, col2 = st.columns(2)
with col1:
    st.metric("Документов", len(df))
with col2:
    if 'category' in df.columns:
        st.metric("Категорий", df['category'].nunique())

if st.checkbox("Показать таблицу данных"):
    display_cols = ['title', 'publication_date', 'category'] if 'category' in df.columns else ['title', 'publication_date']
    st.dataframe(df[display_cols].head(), use_container_width=True)

# --- Векторизация ---
st.header("🔡 Векторизация текстов")
with st.spinner("Векторизация..."):
    X, feature_names, vectorizer = vectorize_texts(
        df['text_clean'].tolist(), 
        method=vectorization_method, 
        max_features=max_features
    )
    
if X is None:
    st.error("Не удалось выполнить векторизацию")
    st.stop()

st.success(f"Созданы векторы размерности: {X.shape}")

# --- Кластеризация ---
st.header("🎯 Кластеризация")
with st.spinner("Выполняется кластеризация..."):
    labels, model = apply_clustering(X, clustering_algorithm, params)
    
    # Статистика кластеров
    unique_labels = np.unique(labels)
    n_clusters_detected = len(unique_labels[unique_labels != -1])
    n_noise = np.sum(labels == -1) if -1 in labels else 0
    
    df['cluster'] = labels

# Отображение статистики
col1, col2, col3 = st.columns(3)
col1.metric("Алгоритм", clustering_algorithm)
col2.metric("Обнаружено кластеров", n_clusters_detected)
if n_noise > 0:
    col3.metric("Точек шума", n_noise)
else:
    col3.metric("Документов", len(df))

# Визуализация распределения по кластерам
st.write("Распределение по кластерам:")
cluster_counts = df['cluster'].value_counts().sort_index()
fig_dist = go.Figure(data=[
    go.Bar(x=[str(k) for k in cluster_counts.index], 
           y=cluster_counts.values,
           marker_color='lightseagreen')
])
fig_dist.update_layout(
    title="Распределение документов по кластерам",
    xaxis_title="Кластер",
    yaxis_title="Количество документов",
    height=400
)
st.plotly_chart(fig_dist, use_container_width=True)

# --- Визуализация ---
st.header("📊 Визуализация кластеров")

# t-SNE для снижения размерности до 2D
with st.spinner("Создание визуализации t-SNE..."):
    perplexity_val = min(30, max(5, len(df) // 5))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val)
    X_tsne = tsne.fit_transform(X)
    
    df['tsne_x'] = X_tsne[:, 0]
    df['tsne_y'] = X_tsne[:, 1]
    
    # Интерактивный scatter plot с Plotly
    fig = px.scatter(
        df, 
        x='tsne_x', 
        y='tsne_y', 
        color=df['cluster'].astype(str),
        hover_data=['title', 'cluster'],
        title='t-SNE визуализация кластеров',
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig.update_traces(marker=dict(size=10, opacity=0.7))
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

# --- Анализ кластеров ---
st.header("🔍 Анализ содержимого кластеров")

# Выбор кластера для анализа
available_clusters = sorted(df['cluster'].unique())
selected_cluster = st.selectbox(
    "Выберите кластер для детального анализа",
    available_clusters,
    format_func=lambda x: f"Кластер {x}" if x != -1 else "Шум (кластер -1)"
)

cluster_docs = df[df['cluster'] == selected_cluster]
st.write(f"**Документов в кластере {selected_cluster}:** {len(cluster_docs)}")

# Топ слова для выбранного кластера
if vectorization_method == "TF-IDF" and feature_names is not None and len(cluster_docs) > 0:
    st.subheader(f"Ключевые слова кластера {selected_cluster}")
    top_words = get_top_tfidf_words(X, feature_names, labels, selected_cluster, top_n=15)
    
    if top_words:
        words, scores = zip(*top_words)
        fig_bar = go.Figure(go.Bar(
            x=scores,
            y=words,
            orientation='h',
            marker=dict(color='lightseagreen')
        ))
        fig_bar.update_layout(
            title=f"Топ-15 слов по TF-IDF (кластер {selected_cluster})",
            xaxis_title="Средний вес TF-IDF",
            yaxis_title="Слово",
            height=400
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Не удалось извлечь ключевые слова для этого кластера.")

# Примеры документов из кластера
if len(cluster_docs) > 0:
    st.subheader(f"Примеры документов (кластер {selected_cluster})")
    
    # Ограничиваем количество отображаемых документов
    max_docs_to_show = min(3, len(cluster_docs))
    
    for idx, row in cluster_docs.head(max_docs_to_show).iterrows():
        with st.expander(f"{row['title'][:80]}..."):
            if 'publication_date' in row:
                st.write(f"**Дата:** {row['publication_date']}")
            if 'category' in row:
                st.write(f"**Категория:** {row['category']}")
            if 'text' in row:
                st.write(f"**Текст (первые 300 символов):** {row['text'][:300]}...")

# --- Сводная таблица по всем кластерам ---
st.header("📋 Сводная информация по кластерам")

summary_data = []
for cluster_id in sorted(df['cluster'].unique()):
    cluster_docs = df[df['cluster'] == cluster_id]
    
    # Ключевые слова для кластера
    top_words_list = []
    if vectorization_method == "TF-IDF" and feature_names is not None and len(cluster_docs) > 0:
        top_words = get_top_tfidf_words(X, feature_names, labels, cluster_id, top_n=5)
        top_words_list = [word for word, _ in top_words] if top_words else []
    
    # Пример заголовка
    sample_title = "Нет данных"
    if len(cluster_docs) > 0 and 'title' in cluster_docs.columns:
        sample_title = cluster_docs.iloc[0]['title']
        if len(sample_title) > 60:
            sample_title = sample_title[:57] + "..."
    
    summary_data.append({
        "Кластер": cluster_id,
        "Документов": len(cluster_docs),
        "Ключевые слова": ", ".join(top_words_list),
        "Пример документа": sample_title
    })

summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

# --- Экспорт результатов ---
st.header("💾 Экспорт результатов")

csv_data = df[['title', 'publication_date', 'category', 'cluster']].to_csv(index=False) if 'category' in df.columns else df[['title', 'publication_date', 'cluster']].to_csv(index=False)

st.download_button(
    label="Скачать результаты кластеризации (CSV)",
    data=csv_data,
    file_name="clustering_results.csv",
    mime="text/csv",
    use_container_width=True
)

# --- Информация о выбранных параметрах ---
with st.expander("📋 Сводка параметров выполнения"):
    st.write(f"**Алгоритм кластеризации:** {clustering_algorithm}")
    st.write(f"**Параметры:** {params}")
    st.write(f"**Метод векторизации:** {vectorization_method}")
    st.write(f"**Размер выборки:** {sample_size} документов")
    st.write(f"**Размерность векторов:** {X.shape}")
    st.write(f"**Вычислено кластеров:** {n_clusters_detected}")

st.success("Анализ завершен! Измените параметры в боковой панели для нового анализа.")
st.info("💡 **Совет:** Для лучшей визуализации попробуйте разные значения параметров алгоритмов.")
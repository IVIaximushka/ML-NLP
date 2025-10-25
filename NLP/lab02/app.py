# web_interface.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from scipy.spatial.distance import cosine

st.set_page_config(page_title="Анализ векторных пространств", layout="wide")

st.title("🔍 Веб-интерфейс для анализа векторных пространств")

# Загрузка моделей
@st.cache_resource
def load_selected_models():
    # Загружаем ранее сохраненные модели
    from gensim.models import Word2Vec, FastText, Doc2Vec
    import os
    
    models = {}
    model_files = [f for f in os.listdir('saved_models') if f.endswith('.model')][:3]
    
    for model_file in model_files:
        model_path = os.path.join('saved_models', model_file)
        model_name = model_file.replace('.model', '')
        
        if 'word2vec' in model_name:
            models[model_name] = Word2Vec.load(model_path)
        elif 'fasttext' in model_name:
            models[model_name] = FastText.load(model_path)
        elif 'doc2vec' in model_name:
            models[model_name] = Doc2Vec.load(model_path)
    
    return models

models = load_selected_models()

# Сайдбар для выбора модели
st.sidebar.header("Настройки")
selected_model_name = st.sidebar.selectbox(
    "Выберите модель:",
    list(models.keys())
)
model = models[selected_model_name]

# Основные вкладки
tab1, tab2, tab3, tab4 = st.tabs([
    "🧮 Векторная арифметика", 
    "📊 Семантическое сходство", 
    "📈 Семантические оси", 
    "📋 Отчёт"
])

with tab1:
    st.header("Интерактивная векторная арифметика")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Введите выражение")
        expression = st.text_input(
            "Векторное выражение:",
            "король - мужчина + женщина",
            help="Формат: слово1 + слово2 - слово3 ..."
        )
        
        if st.button("Вычислить"):
            try:
                # Парсим выражение
                tokens = expression.split()
                positive_words = []
                negative_words = []
                current_sign = 1
                
                for token in tokens:
                    if token == '+':
                        current_sign = 1
                    elif token == '-':
                        current_sign = -1
                    else:
                        if current_sign == 1:
                            positive_words.append(token)
                        else:
                            negative_words.append(token)
                
                # Выполняем векторную арифметику
                if hasattr(model, 'wv'):
                    result = model.wv.most_similar(
                        positive=positive_words, 
                        negative=negative_words, 
                        topn=10
                    )
                    
                    st.success("Результат вычисления:")
                    result_df = pd.DataFrame(result, columns=['Слово', 'Сходство'])
                    st.dataframe(result_df, use_container_width=True)
                    
                    # Визуализация
                    words_to_visualize = positive_words + negative_words + [word for word, _ in result[:5]]
                    vectors = []
                    labels = []
                    
                    for word in words_to_visualize:
                        if word in model.wv:
                            vectors.append(model.wv[word])
                            labels.append(word)
                    
                    if len(vectors) >= 3:
                        # Применяем PCA для визуализации
                        pca = PCA(n_components=2)
                        vectors_2d = pca.fit_transform(vectors)
                        
                        fig = go.Figure()
                        
                        # Исходные слова
                        fig.add_trace(go.Scatter(
                            x=vectors_2d[:len(positive_words + negative_words), 0],
                            y=vectors_2d[:len(positive_words + negative_words), 1],
                            mode='markers+text',
                            marker=dict(size=15, color='blue'),
                            text=labels[:len(positive_words + negative_words)],
                            textposition="top center",
                            name="Исходные слова"
                        ))
                        
                        # Результаты
                        fig.add_trace(go.Scatter(
                            x=vectors_2d[len(positive_words + negative_words):, 0],
                            y=vectors_2d[len(positive_words + negative_words):, 1],
                            mode='markers+text',
                            marker=dict(size=15, color='red'),
                            text=labels[len(positive_words + negative_words):],
                            textposition="top center",
                            name="Результаты"
                        ))
                        
                        fig.update_layout(
                            title="Визуализация векторной арифметики (PCA)",
                            xaxis_title="Компонента 1",
                            yaxis_title="Компонента 2"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Ошибка: {e}")

with tab2:
    st.header("Эксперименты с семантическим сходством")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Калькулятор сходства")
        word1 = st.text_input("Первое слово:", "компьютер")
        word2 = st.text_input("Второе слово:", "ноутбук")
        
        if st.button("Вычислить сходство"):
            if hasattr(model, 'wv'):
                if word1 in model.wv and word2 in model.wv:
                    similarity = 1 - cosine(model.wv[word1], model.wv[word2])
                    st.metric("Косинусное сходство", f"{similarity:.4f}")
                else:
                    st.warning("Одно или оба слова отсутствуют в словаре модели")
    
    with col2:
        st.subheader("Ближайшие соседи")
        target_word = st.text_input("Слово для анализа:", "программа")
        n_neighbors = st.slider("Количество соседей:", 5, 20, 10)
        
        if st.button("Найти соседей"):
            if hasattr(model, 'wv') and target_word in model.wv:
                neighbors = model.wv.most_similar(target_word, topn=n_neighbors)
                neighbors_df = pd.DataFrame(neighbors, columns=['Слово', 'Сходство'])
                st.dataframe(neighbors_df, use_container_width=True)
    
    # Визуализация графа семантических связей
    st.subheader("Граф семантических связей")
    central_word = st.text_input("Центральное слово для графа:", "искусственный")
    graph_depth = st.slider("Глубина графа:", 1, 3, 2)
    
    if st.button("Построить граф"):
        if hasattr(model, 'wv') and central_word in model.wv:
            # Собираем слова для графа
            graph_words = [central_word]
            graph_edges = []
            
            # Первый уровень связей
            level1_neighbors = model.wv.most_similar(central_word, topn=10)
            for neighbor, similarity in level1_neighbors:
                graph_words.append(neighbor)
                graph_edges.append((central_word, neighbor, similarity))
            
            # Второй уровень связей (если нужно)
            if graph_depth >= 2:
                for neighbor, _ in level1_neighbors[:5]:
                    if neighbor in model.wv:
                        level2_neighbors = model.wv.most_similar(neighbor, topn=5)
                        for neighbor2, similarity2 in level2_neighbors:
                            if neighbor2 not in graph_words:
                                graph_words.append(neighbor2)
                            graph_edges.append((neighbor, neighbor2, similarity2))
            
            # Создаем визуализацию графа
            edge_x = []
            edge_y = []
            edge_text = []
            
            node_x = []
            node_y = []
            node_text = []
            
            # Простая круговая компоновка
            n_nodes = len(graph_words)
            angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
            
            for i, word in enumerate(graph_words):
                node_x.append(np.cos(angles[i]))
                node_y.append(np.sin(angles[i]))
                node_text.append(word)
            
            for edge in graph_edges:
                source_idx = graph_words.index(edge[0])
                target_idx = graph_words.index(edge[1])
                
                edge_x.extend([node_x[source_idx], node_x[target_idx], None])
                edge_y.extend([node_y[source_idx], node_y[target_idx], None])
                edge_text.append(f"{edge[2]:.3f}")
            
            fig = go.Figure()
            
            # Рёбра
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='gray'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))
            
            # Узлы
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(size=20, color='lightblue'),
                text=node_text,
                textposition="middle center",
                hoverinfo='text',
                showlegend=False
            ))
            
            fig.update_layout(
                title="Граф семантических связей",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Визуализация семантических осей")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Определение оси")
        axis_word1 = st.text_input("Первое слово оси:", "мужчина")
        axis_word2 = st.text_input("Второе слово оси:", "женщина")
        
        if st.button("Создать ось"):
            if (hasattr(model, 'wv') and 
                axis_word1 in model.wv and 
                axis_word2 in model.wv):
                
                axis_vector = model.wv[axis_word1] - model.wv[axis_word2]
                st.session_state.axis_vector = axis_vector
                st.session_state.axis_name = f"{axis_word1}-{axis_word2}"
                st.success(f"Ось '{axis_word1}-{axis_word2}' создана!")
    
    with col2:
        st.subheader("Проекция слов на ось")
        if 'axis_vector' in st.session_state:
            test_words = st.text_area(
                "Слова для проекции (через запятую):",
                "король, королева, принц, принцесса, врач, учитель, программист"
            )
            
            if st.button("Спроецировать"):
                words_list = [w.strip() for w in test_words.split(',')]
                projections = {}
                
                for word in words_list:
                    if word in model.wv:
                        projection = np.dot(model.wv[word], st.session_state.axis_vector)
                        projections[word] = projection
                
                if projections:
                    # Сортируем по значению проекции
                    sorted_projections = dict(sorted(projections.items(), key=lambda x: x[1]))
                    
                    # Создаем визуализацию
                    fig = go.Figure()
                    
                    words = list(sorted_projections.keys())
                    values = list(sorted_projections.values())
                    
                    fig.add_trace(go.Scatter(
                        x=values,
                        y=words,
                        mode='markers',
                        marker=dict(size=15, color=values, colorscale='RdYlBu_r'),
                        hovertemplate='<b>%{y}</b><br>Проекция: %{x:.3f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title=f"Проекция слов на ось '{st.session_state.axis_name}'",
                        xaxis_title="Значение проекции",
                        yaxis_title="Слова",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Динамический отчёт")
    
    if st.button("Сгенерировать отчёт"):
        # Собираем статистику
        if hasattr(model, 'wv'):
            # Основная информация о модели
            st.subheader("📊 Основная информация")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Размер словаря", len(model.wv.key_to_index))
            with col2:
                st.metric("Размерность векторов", model.vector_size)
            with col3:
                # Примерная точность аналогий
                st.metric("Модель", selected_model_name)
            
            # Анализ распределения сходств
            st.subheader("📈 Распределение семантических сходств")
            
            # Берем случайную выборку слов
            sample_words = list(model.wv.key_to_index.keys())[:1000]
            similarities = []
            
            for i in range(len(sample_words)):
                for j in range(i+1, min(i+10, len(sample_words))):
                    if (sample_words[i] in model.wv and 
                        sample_words[j] in model.wv):
                        sim = 1 - cosine(model.wv[sample_words[i]], model.wv[sample_words[j]])
                        similarities.append(sim)
            
            if similarities:
                fig = px.histogram(
                    x=similarities, 
                    nbins=50,
                    title="Распределение косинусных сходств между словами"
                )
                fig.update_layout(xaxis_title="Косинусное сходство", yaxis_title="Частота")
                st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap семантических близостей
            st.subheader("🔥 Heatmap семантических близостей")
            
            common_words = ['компьютер', 'программа', 'данные', 'информация', 'система', 'сеть']
            available_words = [w for w in common_words if w in model.wv]
            
            if len(available_words) >= 3:
                vectors = [model.wv[word] for word in available_words]
                similarity_matrix = np.zeros((len(available_words), len(available_words)))
                
                for i in range(len(available_words)):
                    for j in range(len(available_words)):
                        similarity_matrix[i,j] = 1 - cosine(vectors[i], vectors[j])
                
                fig = px.imshow(
                    similarity_matrix,
                    x=available_words,
                    y=available_words,
                    color_continuous_scale='viridis',
                    title="Матрица семантических близостей"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 2D проекция слов
            st.subheader("🎯 2D проекция слов")
            
            projection_words = st.text_input(
                "Слова для проекции (через запятую):",
                "компьютер, программа, данные, информация, система, человек, работа, время, город, страна"
            )
            
            words_list = [w.strip() for w in projection_words.split(',')]
            available_words = [w for w in words_list if w in model.wv]
            
            if len(available_words) >= 3:
                vectors = [model.wv[word] for word in available_words]
                
                # Применяем t-SNE
                tsne = TSNE(n_components=2, random_state=42)
                vectors_2d = tsne.fit_transform(vectors)
                
                fig = px.scatter(
                    x=vectors_2d[:, 0],
                    y=vectors_2d[:, 1],
                    text=available_words,
                    title="t-SNE проекция слов"
                )
                
                fig.update_traces(
                    marker=dict(size=12),
                    textposition='top center'
                )
                
                fig.update_layout(
                    xaxis_title="Компонента 1",
                    yaxis_title="Компонента 2"
                )
                
                st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Советы:**\n"
    "- Используйте векторную арифметику для семантических аналогий\n"
    "- Исследуйте семантические оси для анализа смещений\n"
    "- Визуализируйте связи между словами через графы"
)
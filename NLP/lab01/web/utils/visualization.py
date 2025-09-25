import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def create_token_length_distribution(token_lengths):
    """Создает график распределения длин токенов"""
    fig = px.histogram(x=token_lengths, nbins=20, 
                      title='Распределение длин токенов',
                      labels={'x': 'Длина токена', 'y': 'Количество'})
    fig.update_layout(showlegend=False)
    return fig

def create_frequency_plot(frequencies):
    """Создает график частотности токенов (Zipf's law)"""
    sorted_freq = sorted(frequencies, reverse=True)
    ranks = range(1, len(sorted_freq) + 1)
    
    fig = px.line(x=ranks, y=sorted_freq, log_x=True, log_y=True,
                 title='Закон Ципфа: Ранг vs Частота',
                 labels={'x': 'Ранг (логарифмическая шкала)', 
                        'y': 'Частота (логарифмическая шкала)'})
    return fig

def create_top_tokens_chart(most_common_tokens):
    """Создает барчарт самых частых токенов"""
    tokens, counts = zip(*most_common_tokens) if most_common_tokens else ([], [])
    
    fig = px.bar(x=counts, y=tokens, orientation='h',
                title='20 самых частых токенов',
                labels={'x': 'Частота', 'y': 'Токен'})
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

def create_metrics_comparison(metrics_dict):
    """Создает сравнительную визуализацию метрик"""
    methods = list(metrics_dict.keys())
    metrics_data = []
    
    for method, metrics in metrics_dict.items():
        metrics_data.append({
            'Method': method,
            'Vocabulary Size': metrics.get('vocabulary_size', 0),
            'Avg Token Length': metrics.get('avg_token_length', 0),
            'Lexical Diversity': metrics.get('lexical_diversity', 0)
        })
    
    df = pd.DataFrame(metrics_data)
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=('Размер словаря', 'Средняя длина токена', 'Лексическое разнообразие'))
    
    fig.add_trace(go.Bar(x=df['Method'], y=df['Vocabulary Size'], name='Размер словаря'), 1, 1)
    fig.add_trace(go.Bar(x=df['Method'], y=df['Avg Token Length'], name='Средняя длина'), 1, 2)
    fig.add_trace(go.Bar(x=df['Method'], y=df['Lexical Diversity'], name='Лексическое разнообразие'), 1, 3)
    
    fig.update_layout(height=400, title_text='Сравнение методов токенизации', showlegend=False)
    return fig

def create_wordcloud(tokens_list):
    """Создает облако слов"""
    all_tokens = ' '.join([' '.join(tokens) for tokens in tokens_list])
    
    if not all_tokens.strip():
        return None
    
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                         colormap='viridis', max_words=50).generate(all_tokens)
    
    # Конвертируем в base64 для отображения в Streamlit
    img_buffer = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Облако токенов', size=16)
    plt.tight_layout()
    plt.savefig(img_buffer, format='png', bbox_inches='tight')
    plt.close()
    
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.read()).decode()
    return f"data:image/png;base64,{img_data}"

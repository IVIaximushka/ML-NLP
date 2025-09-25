import re
import nltk
from razdel import tokenize as razdel_tokenize
import spacy
import pymorphy3
from nltk.stem import SnowballStemmer, PorterStemmer
from collections import Counter
import time

# Инициализация компонентов
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Загрузка моделей spaCy
try:
    nlp_spacy = spacy.load("ru_core_news_sm")
except OSError:
    print("Модель spaCy для русского не найдена. Запустите: python -m spacy download ru_core_news_sm")
    nlp_spacy = None

# Инициализация стеммеров и морфологического анализатора
stemmer_porter = PorterStemmer()
stemmer_snowball = SnowballStemmer(language='russian')
morph_analyzer = pymorphy3.MorphAnalyzer()

def tokenize_naive(text):
    """Наивная токенизация по пробелам"""
    return text.split()

def tokenize_regex(text):
    """Токенизация с помощью регулярных выражений"""
    pattern = r'\b[а-яё]+\b|[!?.,;:]'
    return re.findall(pattern, text.lower())

def tokenize_nltk(text):
    """Токенизация с помощью NLTK"""
    return nltk.word_tokenize(text.lower(), language='russian')

def tokenize_razdel(text):
    """Токенизация с помощью razdel"""
    return [token.text for token in razdel_tokenize(text)]

def tokenize_spacy(text):
    """Токенизация с помощью spaCy"""
    if nlp_spacy is None:
        return tokenize_razdel(text)  # Fallback
    doc = nlp_spacy(text)
    return [token.text for token in doc]

def stem_porter(tokens):
    """Стемминг Porter"""
    return [stemmer_porter.stem(token) for token in tokens]

def stem_snowball(tokens):
    """Стемминг Snowball"""
    return [stemmer_snowball.stem(token) for token in tokens]

def lemmatize_pymorphy2(tokens):
    """Лемматизация с помощью pymorphy2"""
    lemmas = []
    for token in tokens:
        parsed = morph_analyzer.parse(token)[0]
        lemmas.append(parsed.normal_form)
    return lemmas

def lemmatize_spacy(tokens):
    """Лемматизация с помощью spaCy"""
    if nlp_spacy is None:
        return lemmatize_pymorphy2(tokens)  # Fallback
    text = ' '.join(tokens)
    doc = nlp_spacy(text)
    return [token.lemma_ for token in doc]

def apply_tokenization_pipeline(text, tokenizer_name, normalizer_name):
    """Применяет полный пайплайн токенизации и нормализации"""
    tokenizers = {
        'naive': tokenize_naive,
        'regex': tokenize_regex,
        'nltk': tokenize_nltk,
        'razdel': tokenize_razdel,
        'spacy': tokenize_spacy
    }
    
    normalizers = {
        'none': lambda x: x,
        'porter': stem_porter,
        'snowball': stem_snowball,
        'pymorphy2': lemmatize_pymorphy2,
        'spacy': lemmatize_spacy
    }
    
    tokenizer = tokenizers.get(tokenizer_name, tokenize_razdel)
    normalizer = normalizers.get(normalizer_name, lambda x: x)
    
    # Применяем токенизацию
    tokens = tokenizer(text)
    
    # Применяем нормализацию
    if normalizer_name != 'none':
        tokens = normalizer(tokens)
    
    return tokens

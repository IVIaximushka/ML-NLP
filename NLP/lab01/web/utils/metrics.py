import time
import numpy as np
from collections import Counter

def calculate_basic_metrics(tokens_list):
    """Рассчитывает базовые метрики для списка токенизированных текстов"""
    all_tokens = []
    for tokens in tokens_list:
        all_tokens.extend(tokens)
    
    if not all_tokens:
        return {}
    
    token_counts = Counter(all_tokens)
    total_tokens = len(all_tokens)
    unique_tokens = len(token_counts)
    
    # Длины токенов
    token_lengths = [len(token) for token in all_tokens]
    
    # Частотность
    frequencies = list(token_counts.values())
    
    return {
        'total_tokens': total_tokens,
        'unique_tokens': unique_tokens,
        'vocabulary_size': unique_tokens,
        'avg_token_length': np.mean(token_lengths),
        'std_token_length': np.std(token_lengths),
        'token_lengths': token_lengths,
        'frequencies': frequencies,
        'most_common_tokens': token_counts.most_common(20),
        'lexical_diversity': unique_tokens / total_tokens if total_tokens > 0 else 0
    }

def calculate_oov_rate(original_tokens, processed_tokens):
    """Рассчитывает долю OOV токенов"""
    if not original_tokens or not processed_tokens:
        return 0
    
    original_vocab = set()
    for tokens in original_tokens:
        original_vocab.update(tokens)
    
    oov_count = 0
    total_count = 0
    
    for tokens in processed_tokens:
        for token in tokens:
            total_count += 1
            if token not in original_vocab:
                oov_count += 1
    
    return oov_count / total_count if total_count > 0 else 0

def calculate_processing_speed(texts, pipeline_func, iterations=3):
    """Измеряет скорость обработки"""
    if not texts:
        return 0
    
    start_time = time.time()
    for _ in range(iterations):
        for text in texts:
            _ = pipeline_func(text)
    
    end_time = time.time()
    total_texts = len(texts) * iterations
    return total_texts / (end_time - start_time) if (end_time - start_time) > 0 else 0

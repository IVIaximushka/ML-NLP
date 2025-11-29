import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def load_data():
    train_df = pd.read_json('train.jsonl', lines=True)
    test_df = pd.read_json('test.jsonl', lines=True)
    return train_df, test_df

def train_models(X_train, y_train):
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    
    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    
    return lr_model, rf_model

def predict_text(text, models, vectorizer, encoder):
    text_vectorized = vectorizer.transform([text])
    
    lr_pred = models[0].predict(text_vectorized)[0]
    rf_pred = models[1].predict(text_vectorized)[0]
    
    lr_proba = models[0].predict_proba(text_vectorized)[0]
    rf_proba = models[1].predict_proba(text_vectorized)[0]
    
    return {
        'lr_class': encoder.inverse_transform([lr_pred])[0],
        'rf_class': encoder.inverse_transform([rf_pred])[0],
        'lr_proba': lr_proba,
        'rf_proba': rf_proba
    }

def plot_probabilities(proba_lr, proba_rf, class_names):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.barh(class_names, proba_lr)
    ax1.set_title('Logistic Regression Probabilities')
    ax1.set_xlim(0, 1)
    
    ax2.barh(class_names, proba_rf)
    ax2.set_title('Random Forest Probabilities')
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    return fig

def main():
    st.set_page_config(page_title="Text Classifier Analysis", layout="wide")
    st.title("📊 Анализ классификаторов текстов")
    
    train_df, test_df = load_data()
    
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(train_df['text'])
    X_test = vectorizer.transform(test_df['text'])
    
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df['category'])
    y_test = encoder.transform(test_df['category'])
    
    lr_model, rf_model = train_models(X_train, y_train)
    models = (lr_model, rf_model)
    
    tab1, tab2, tab3 = st.tabs(["🔍 Классификация текста", "📈 Сравнение моделей", "📊 Анализ ошибок"])
    
    with tab1:
        st.header("Интерактивная классификация текста")
        
        text_input = st.text_area("Введите текст для классификации:", 
                                "Введите текст новости здесь...")
        
        if st.button("Классифицировать"):
            if text_input.strip():
                result = predict_text(text_input, models, vectorizer, encoder)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Logistic Regression")
                    st.write(f"**Класс:** {result['lr_class']}")
                    st.write(f"**Вероятность:** {max(result['lr_proba']):.3f}")
                
                with col2:
                    st.subheader("Random Forest")
                    st.write(f"**Класс:** {result['rf_class']}")
                    st.write(f"**Вероятность:** {max(result['rf_proba']):.3f}")
                
                fig = plot_probabilities(result['lr_proba'], result['rf_proba'], encoder.classes_)
                st.pyplot(fig)
    
    with tab2:
        st.header("Сравнение моделей")
        
        y_pred_lr = lr_model.predict(X_test)
        y_pred_rf = rf_model.predict(X_test)
        
        lr_accuracy = accuracy_score(y_test, y_pred_lr)
        rf_accuracy = accuracy_score(y_test, y_pred_rf)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Logistic Regression Accuracy", f"{lr_accuracy:.3f}")
        
        with col2:
            st.metric("Random Forest Accuracy", f"{rf_accuracy:.3f}")
        
        metrics_df = pd.DataFrame({
            'Model': ['Logistic Regression', 'Random Forest'],
            'Accuracy': [lr_accuracy, rf_accuracy]
        })
        
        st.bar_chart(metrics_df.set_index('Model'))
        
        st.subheader("Отчет по классификации - Logistic Regression")
        st.text(classification_report(y_test, y_pred_lr, target_names=encoder.classes_))
    
    with tab3:
        st.header("Анализ ошибок")
        
        y_pred_lr = lr_model.predict(X_test)
        errors_mask = y_test != y_pred_lr
        error_indices = np.where(errors_mask)[0]
        
        error_df = pd.DataFrame({
            'text': test_df['text'].iloc[error_indices].values,
            'true_class': [encoder.classes_[i] for i in y_test[error_indices]],
            'predicted_class': [encoder.classes_[i] for i in y_pred_lr[error_indices]]
        })
        
        st.write(f"Всего ошибок: {len(error_df)}")
        st.dataframe(error_df.head(10))
        
        error_counts = error_df.groupby(['true_class', 'predicted_class']).size().reset_index(name='count')
        st.subheader("Частые ошибки")
        st.dataframe(error_counts.sort_values('count', ascending=False))

if __name__ == "__main__":
    main()
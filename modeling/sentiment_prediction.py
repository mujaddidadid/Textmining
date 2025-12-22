import pickle
import os
import streamlit as st

def load_prediction_resources(model_path='best_model.pkl', vec_path='vectorizer.pkl'):
    """
    Memuat model dan vectorizer dari file pickle.
    """
    try:
        if not os.path.exists(model_path) or not os.path.exists(vec_path):
            return None, None, "File model (.pkl) tidak ditemukan."
            
        with open(model_path, 'rb') as m_file:
            model = pickle.load(m_file)
        with open(vec_path, 'rb') as v_file:
            vectorizer = pickle.load(v_file)
            
        return model, vectorizer, None
    except Exception as e:
        return None, None, str(e)

def perform_prediction(raw_text):
    """
    Logika utama: Preprocessing -> Lexicon Labeling -> ML Predicting
    """
    from preprocessing.preprocessing import case_folding, cleaning, tokenizing, stopword_removal, stemming
    from labeling.lexicon_labeling import label_sentiment # Import fungsi lexicon Anda
    
    # 1. Load Resources (Model & Vectorizer)
    model, vectorizer, error = load_prediction_resources()
    if error:
        return None, None, None, error

    # 2. Preprocessing Tunggal (Konsisten dengan pipeline training)
    text = case_folding(raw_text)
    text = cleaning(text)
    tokens = tokenizing(text)
    tokens = stopword_removal(tokens)
    tokens = stemming(tokens)
    cleaned_text = " ".join(tokens)

    # 3. Analisa Otomatis via Lexicon
    # Memberi label berdasarkan skor kata positif/negatif
    lexicon_result = label_sentiment(cleaned_text)

    # 4. Transformasi & Prediksi via Machine Learning (Pickle)
    vectorized_input = vectorizer.transform([cleaned_text])
    ml_prediction = model.predict(vectorized_input)[0]

    # Return: Hasil ML, Hasil Lexicon, Teks Bersih, Error
    return ml_prediction, lexicon_result, cleaned_text, None
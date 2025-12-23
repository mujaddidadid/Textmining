import pickle
import os
import streamlit as st
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@st.cache_resource
def load_prediction_resources(model_path='best_model.pkl', vec_path='vectorizer.pkl'):
    try:
        if not os.path.exists(model_path):
            return None, None, f"Error: File model '{model_path}' tidak ditemukan."
        if not os.path.exists(vec_path):
            return None, None, f"Error: File vectorizer '{vec_path}' tidak ditemukan."
            
        with open(model_path, 'rb') as m_file:
            model = pickle.load(m_file)
        with open(vec_path, 'rb') as v_file:
            vectorizer = pickle.load(v_file)
            
        return model, vectorizer, None
    except Exception as e:
        return None, None, f"Gagal memuat resource: {str(e)}"

def convert_label(prediction):
    label_str = str(prediction)
    
    clean_label = label_str.strip().capitalize()
    
    if clean_label == "1":
        return "Positif"
    elif clean_label == "0":
        return "Negatif"
    elif clean_label == "-1":
        return "Negatif"
        
    return clean_label

def perform_prediction(raw_text):
    try:
        from preprocessing.preprocessing import case_folding, cleaning, tokenizing, stopword_removal, stemming
        from labeling.lexicon_labeling import label_sentiment 
        
        model, vectorizer, error = load_prediction_resources()
        if error: return None, None, None, error

        text = case_folding(raw_text)
        text = cleaning(text)
        tokens = tokenizing(text)
        tokens = stopword_removal(tokens)
        tokens = stemming(tokens)
        cleaned_text = " ".join(tokens)

        if not cleaned_text.strip():
            return None, None, None, "Teks kosong setelah cleaning."

        lexicon_result = label_sentiment(cleaned_text)

        vectorized_input = vectorizer.transform([cleaned_text])
        prediction_raw = model.predict(vectorized_input)[0]

        ml_prediction = convert_label(prediction_raw)

        return ml_prediction, lexicon_result, cleaned_text, None

    except Exception as e:
        return None, None, None, f"System Error: {str(e)}"
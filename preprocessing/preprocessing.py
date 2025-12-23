import re
import nltk
import pandas as pd
import os
import ast  

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords")

factory = StemmerFactory()
stemmer = factory.create_stemmer()

stop_words = set(stopwords.words("indonesian"))


def case_folding(text):
    return text.lower()

def cleaning(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)  
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenizing(text):
    return word_tokenize(text)

def stopword_removal(tokens):
    return [word for word in tokens if word not in stop_words]

def stemming(tokens):
    return [stemmer.stem(word) for word in tokens]


def preprocess_stepwise(text_list):
    folder_path = "uploaded_files"
    file_name = "data_preprocessing_result.xlsx"
    file_path = os.path.join(folder_path, file_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


    if os.path.exists(file_path):
        try:
            df_existing = pd.read_excel(file_path)
            
            if 'Original' in df_existing.columns:
                existing_original = df_existing['Original'].astype(str).tolist()
                input_original = [str(t) for t in text_list]

                if existing_original == input_original:
                    print("Data konsisten ditemukan! Memuat dari file (Skip Stemming)...")
                    
                    return {
                        "original": df_existing['Original'].tolist(),
                        "case_folding": df_existing['Case Folding'].tolist(),
                        "cleaning": df_existing['Cleaning'].tolist(),
                        "tokenizing": [ast.literal_eval(x) for x in df_existing['Tokenizing']],
                        "stopword": [ast.literal_eval(x) for x in df_existing['Stopword Removal']],
                        "stemming": [ast.literal_eval(x) for x in df_existing['Stemming']],
                        "final": df_existing['Final Text'].tolist()
                    }
        except Exception as e:
            print(f"Gagal memuat file lama, melakukan proses ulang. Error: {e}")


    print("Memulai Preprocessing baru...")
    original = text_list
    
    casefolded = [case_folding(t) for t in original]
    
    cleaned = [cleaning(t) for t in casefolded]
    
    tokenized = [tokenizing(t) for t in cleaned]
    
    no_stopwords = [stopword_removal(t) for t in tokenized]
    
    stemmed = [stemming(t) for t in no_stopwords]
    
    final_text = [" ".join(t) for t in stemmed]


    df_result = pd.DataFrame({
        "Original": original,
        "Case Folding": casefolded,
        "Cleaning": cleaned,
        "Tokenizing": [str(t) for t in tokenized],
        "Stopword Removal": [str(t) for t in no_stopwords],
        "Stemming": [str(t) for t in stemmed],
        "Final Text": final_text
    })

    df_result.to_excel(file_path, index=False)
    print(f"Hasil preprocessing disimpan di: {file_path}")

    return {
        "original": original,
        "case_folding": casefolded,
        "cleaning": cleaned,
        "tokenizing": tokenized,
        "stopword": no_stopwords,
        "stemming": stemmed,
        "final": final_text
    }
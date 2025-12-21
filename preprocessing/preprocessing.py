import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# GANTI: Import Sastrawi Factory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ==============================
# DOWNLOAD RESOURCE NLTK
# ==============================
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# ==============================
# INISIALISASI
# ==============================
# GANTI: Inisialisasi Sastrawi Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# GANTI: Menggunakan stopwords bahasa Indonesia agar cocok dengan Sastrawi
stop_words = set(stopwords.words("indonesian"))

# ==============================
# 1. CASE FOLDING
# ==============================
def case_folding(text):
    return text.lower()

# ==============================
# 2. CLEANING
# ==============================
def cleaning(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # hapus emoji & simbol
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ==============================
# 3. TOKENIZING
# ==============================
def tokenizing(text):
    return word_tokenize(text)

# ==============================
# 4. STOPWORD REMOVAL
# ==============================
def stopword_removal(tokens):
    return [word for word in tokens if word not in stop_words]

# ==============================
# 5. STEMMING (SASTRAWI)
# ==============================
def stemming(tokens):
    # Sastrawi sebenarnya bisa memproses kalimat langsung, 
    # tapi untuk menjaga konsistensi input (list of tokens), kita loop per kata.
    return [stemmer.stem(word) for word in tokens]

# ==============================
# PREPROCESSING PER TAHAP
# ==============================
def preprocess_stepwise(text_list):
    original = text_list
    
    # 1. Case Folding
    casefolded = [case_folding(t) for t in original]
    
    # 2. Cleaning
    cleaned = [cleaning(t) for t in casefolded]
    
    # 3. Tokenizing
    tokenized = [tokenizing(t) for t in cleaned]
    
    # 4. Stopword Removal
    no_stopwords = [stopword_removal(t) for t in tokenized]
    
    # 5. Stemming (Proses ini mungkin agak lama karena Sastrawi kompleks)
    stemmed = [stemming(t) for t in no_stopwords]
    
    # Gabung kembali menjadi kalimat
    final_text = [" ".join(t) for t in stemmed]

    return {
        "original": original,
        "case_folding": casefolded,
        "cleaning": cleaned,
        "tokenizing": tokenized,
        "stopword": no_stopwords,
        "stemming": stemmed,
        "final": final_text
    }
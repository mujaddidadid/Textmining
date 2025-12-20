import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ==============================
# DOWNLOAD RESOURCE NLTK
# ==============================
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# ==============================
# INISIALISASI
# ==============================
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

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
# 5. STEMMING
# ==============================
def stemming(tokens):
    return [stemmer.stem(word) for word in tokens]

# ==============================
# PREPROCESSING PER TAHAP
# ==============================
def preprocess_stepwise(text_list):
    original = text_list
    casefolded = [case_folding(t) for t in original]
    cleaned = [cleaning(t) for t in casefolded]
    tokenized = [tokenizing(t) for t in cleaned]
    no_stopwords = [stopword_removal(t) for t in tokenized]
    stemmed = [stemming(t) for t in no_stopwords]
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

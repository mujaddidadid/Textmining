from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def tfidf_transform(corpus, max_features=1000):
    """
    corpus : list teks hasil preprocessing
    return  :
      - tfidf_df        : DataFrame TF-IDF
      - tfidf_scores    : rata-rata skor TF-IDF tiap kata
      - vectorizer      : Objek vectorizer yang sudah di-fit (Penting untuk Pickle)
    """

    # Inisialisasi Vectorizer
    vectorizer = TfidfVectorizer(max_features=max_features)
    
    # Melakukan fit dan transform pada corpus
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Membuat DataFrame dari matriks TF-IDF
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    # Menghitung rata-rata skor TF-IDF untuk setiap fitur (kata)
    tfidf_scores = tfidf_df.mean(axis=0).sort_values(ascending=False)

    # Mengembalikan tiga objek: DataFrame, Skor, dan Objek Vectorizer asli
    return tfidf_df, tfidf_scores, vectorizer
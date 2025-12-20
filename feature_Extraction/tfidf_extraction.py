from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def tfidf_transform(corpus, max_features=1000):
    """
    corpus : list teks hasil preprocessing
    return  :
      - tfidf_df        : DataFrame TF-IDF
      - tfidf_scores    : rata-rata skor TF-IDF tiap kata
      - vectorizer
    """

    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    # RATA-RATA TF-IDF (SAMA PERSIS DENGAN COLAB)
    tfidf_scores = tfidf_df.mean(axis=0).sort_values(ascending=False)

    return tfidf_df, tfidf_scores, vectorizer

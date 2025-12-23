from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np 

def tfidf_transform(corpus, max_features=1000):
    if not isinstance(corpus, list):
        if hasattr(corpus, 'tolist'):
            corpus = corpus.tolist()
        else:
            corpus = list(corpus)


    clean_corpus = [str(x) if pd.notna(x) and x is not None else "" for x in corpus]


    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(clean_corpus)

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    tfidf_scores = tfidf_df.mean(axis=0).sort_values(ascending=False)
    return tfidf_df, tfidf_scores, vectorizer
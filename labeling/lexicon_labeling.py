# ==============================
# LEXICON-BASED AUTOMATIC LABELING
# ==============================

# Kamus sentimen sederhana (Bahasa Indonesia)
POSITIVE_WORDS = [
    "bagus", "baik", "mantap", "hebat", "puas", "senang",
    "cepat", "ramah", "murah", "nyaman", "keren", "mantul"
]

NEGATIVE_WORDS = [
    "buruk", "jelek", "lambat", "kecewa", "parah", "error",
    "rusak", "lama", "mahal", "ribet", "lemot"
]


def label_sentiment(text):
    """
    Memberi label sentimen pada satu teks (hasil preprocessing)
    """

    score = 0
    words = text.split()

    for word in words:
        if word in POSITIVE_WORDS:
            score += 1
        elif word in NEGATIVE_WORDS:
            score -= 1

    if score > 0:
        return "Positif"
    elif score < 0:
        return "Negatif"
    else:
        return "Netral"


def label_corpus(text_list):
    """
    Memberi label sentimen pada banyak teks
    """

    labels = []
    for text in text_list:
        label = label_sentiment(text)
        labels.append(label)

    return labels

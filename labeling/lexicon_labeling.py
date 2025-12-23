import os

def load_lexicon_from_file(filename):
    words = set()
    
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        words.add(word)
        except Exception as e:
            print(f"Gagal membaca file {filename}: {e}")
    else:
        print(f"PERINGATAN: File '{filename}' tidak ditemukan di folder proyek.")
    
    return words
POSITIVE_WORDS = load_lexicon_from_file("labeling/positive.txt")
NEGATIVE_WORDS = load_lexicon_from_file("labeling/negative.txt")


def label_sentiment(text): 
    if not isinstance(text, str):
        return "Netral"
    
    if not text.strip():
        return "Netral"

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
    labels = []
    for text in text_list:
        label = label_sentiment(text)
        labels.append(label)

    return labels
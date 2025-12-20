# ==============================
# LEXICON-BASED AUTOMATIC LABELING
# ==============================

# Kamus sentimen sederhana (Bahasa Indonesia)
POSITIVE_WORDS = [
    "bagus", "baik", "mantap", "hebat", "puas", "senang",
    "cepat", "ramah", "murah", "nyaman", "keren", "mantul", "memuaskan", "puasin", "terbaik", "unggul", "fantastis", "istimewa",
    "menakjubkan", "excellent", "awesome", "top", "bagus banget", "oke",
    "ok", "okay", "wow", "wih", "alhamdulillah", "makasih", "terima kasih", "terjangkau", "murah meriah", "worth it", "worth", "worth the price",
    "harga pas", "harga oke", "harga bersahabat", "promo", "diskon",
    "gratis", "free", "murah banget", "murahnya", "happy", "bahagia", "seneng", "senangnya", "asyik", "enak", "legit",
    "manis", "menyenangkan", "memuaskan", "sesuai harapan", "sangat baik", "sangat bagus", "sangat puas", "sangat membantu",
    "super", "super", "fantastic", "amazing", "terlalu bagus",
    "perfect", "paripurna", "flawless", "no complain", "no komplain", "mantep", "mantap jiwa", "top markotop", "jos", "josss", "joss",
    "gemes", "gemesh", "gemas", "kece", "keceh", "keren abis",
    "nendang", "nendang banget", "sip", "sipp", "oke sip", "huuh luar biasah"
]

NEGATIVE_WORDS = [
    "buruk", "jelek", "lambat", "kecewa", "parah", "error",
    "rusak", "lama", "mahal", "ribet", "lemot",  "jelek banget", "buruk banget", "parah banget", "sangat buruk",
    "sangat jelek", "sangat kecewa", "sangat mengecewakan", "menyesal",
    "nyesel", "kapok", "tidak akan kembali", "tidak akan repeat", "kualitas buruk", "kw", "kw1", "kw2", "kw3", "palsu", "fake",
    "imitasi", "copasan", "aspal", "aspalan", "tiruan", "tidak original",
    "tidak asli", "murahan", "murah meriah tapi jelek","terlalu mahal", "overpriced", "mahal banget", "mahal sekali",
    "tidak worth", "tidak worth it", "rugi", "merugi", "kemahalan",
    "mahal ga worth it", "mahal gak worth it","marah", "kesal", "jengkel", "kesel", "kecewa berat",
    "frustasi", "frustrated", "stress", "stres", "bete",
    "bad mood", "mood rusak", "sedih", "kecewa hati",
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

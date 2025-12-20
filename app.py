import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ==============================
# IMPORT MODULE INTERNAL
# ==============================
from preprocessing.preprocessing import preprocess_stepwise
from labeling.lexicon_labeling import label_corpus
from feature_Extraction.tfidf_extraction import tfidf_transform
from modeling.modeling import train_and_compare_models

# ==============================
# KONFIGURASI HALAMAN
# ==============================
st.set_page_config(
    page_title="Text Mining App",
    page_icon="💬",
    layout="wide"
)

# ==============================
# HEADER
# ==============================
st.markdown(
    """
    <h1 style="text-align:center;">💬 Aplikasi Text Mining</h1>
    <p style="text-align:center;">
        Preprocessing → Pelabelan → TF-IDF → Split Data → Modeling → Evaluation
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ==============================
# SIDEBAR
# ==============================
menu = st.sidebar.radio(
    "Tahapan",
    (
        "Input Komentar",
        "Data Preprocessing",
        "Pelabelan Otomatis",
        "Feature Extraction (TF-IDF)",
        "Modeling & Evaluation"
    )
)

# ==============================
# SESSION STATE
# ==============================
for key in ["komentar", "clean_text", "labels", "tfidf_df"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "tfidf_df" else []

# ==============================
# 1️⃣ INPUT KOMENTAR
# ==============================
if menu == "Input Komentar":
    st.subheader("📝 Input Komentar")

    text = st.text_area(
        "Masukkan komentar (1 baris = 1 komentar)",
        height=200,
        placeholder="Contoh:\nAplikasinya sangat bagus\nPelayanannya buruk sekali"
    )

    if st.button("💾 Simpan Komentar"):
        st.session_state["komentar"] = [
            t.strip() for t in text.split("\n") if t.strip()
        ]
        st.success(f"{len(st.session_state['komentar'])} komentar disimpan")

    if st.session_state["komentar"]:
        st.dataframe(pd.DataFrame({"Komentar": st.session_state["komentar"]}))

# ==============================
# 2️⃣ DATA PREPROCESSING
# ==============================
elif menu == "Data Preprocessing":
    st.subheader("📍 Data Preprocessing")

    if not st.session_state["komentar"]:
        st.warning("Belum ada komentar")
    else:
        result = preprocess_stepwise(st.session_state["komentar"])
        st.session_state["clean_text"] = result["final"]

        # ==============================
        # ORIGINAL TEXT
        # ==============================
        st.markdown("## 🔹 Original Text")
        st.dataframe(pd.DataFrame({
            "Original Text": result["original"]
        }))

        # ==============================
        # CASE FOLDING
        # ==============================
        st.markdown("## 🔹 Case Folding")
        st.dataframe(pd.DataFrame({
            "Case Folding": result["case_folding"]
        }))

        # ==============================
        # CLEANING
        # ==============================
        st.markdown("## 🔹 Cleaning")
        st.dataframe(pd.DataFrame({
            "Cleaning": result["cleaning"]
        }))

        # ==============================
        # TOKENIZING
        # ==============================
        st.markdown("## 🔹 Tokenizing")
        st.dataframe(pd.DataFrame({
            "Tokenizing": result["tokenizing"]
        }))

        # ==============================
        # STOPWORD REMOVAL
        # ==============================
        st.markdown("## 🔹 Stopword Removal")
        st.dataframe(pd.DataFrame({
            "Stopword Removal": result["stopword"]
        }))

        # ==============================
        # STEMMING
        # ==============================
        st.markdown("## 🔹 Stemming")
        st.dataframe(pd.DataFrame({
            "Stemming": result["stemming"]
        }))

        # ==============================
        # FINAL TEXT
        # ==============================
        st.markdown("## 🔹 Final Text (Siap Modeling)")
        st.dataframe(pd.DataFrame({
            "Final Text": result["final"]
        }))


# ==============================
# 3️⃣ PELABELAN OTOMATIS
# ==============================
elif menu == "Pelabelan Otomatis":
    st.subheader("📍 Pelabelan Otomatis (Lexicon-Based)")

    if not st.session_state["clean_text"]:
        st.warning("Lakukan preprocessing terlebih dahulu")
    else:
        labels = label_corpus(st.session_state["clean_text"])
        st.session_state["labels"] = labels

        df_label = pd.DataFrame({
            "Teks (Preprocessing)": st.session_state["clean_text"],
            "Label Sentimen": labels
        })

        st.dataframe(df_label)
        st.bar_chart(df_label["Label Sentimen"].value_counts())

# ==============================
# 4️⃣ FEATURE EXTRACTION (TF-IDF)
# ==============================
elif menu == "Feature Extraction (TF-IDF)":
    st.subheader("📍 Feature Extraction – TF-IDF")

    if not st.session_state["clean_text"]:
        st.warning("Lakukan preprocessing terlebih dahulu")
    else:
        tfidf_df, tfidf_scores, _ = tfidf_transform(
            st.session_state["clean_text"]
        )
        st.session_state["tfidf_df"] = tfidf_df

        st.dataframe(tfidf_df.head())

        st.markdown("### Top 20 Kata (TF-IDF)")
        st.table(tfidf_scores.head(20))

        st.markdown("### ☁️ WordCloud TF-IDF")
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate_from_frequencies(tfidf_scores.to_dict())

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

# ==============================
# 5️⃣ MODELING & EVALUATION
# ==============================
elif menu == "Modeling & Evaluation":
    st.subheader("📍 Perbandingan Model Klasifikasi")

    if st.session_state["tfidf_df"] is None or not st.session_state["labels"]:
        st.warning("TF-IDF dan label belum tersedia")
    else:
        from modeling.modeling import train_and_compare_models

        result = train_and_compare_models(
            st.session_state["tfidf_df"],
            st.session_state["labels"]
        )

        acc_df = result["accuracy_df"]

        # ==============================
        # BAR CHART AKURASI
        # ==============================
        st.markdown("### 📊 Perbandingan Akurasi Model")
        st.bar_chart(
            acc_df.set_index("Model")
        )

        # ==============================
        # INFO MODEL TERBAIK
        # ==============================
        st.markdown("### 🏆 Model Terbaik")
        st.write(
            f"**{result['best_model']}** "
            f"(Accuracy = {result['best_accuracy']:.4f})"
        )

        # ==============================
        # CONFUSION MATRIX
        # ==============================
        st.markdown("### 🔢 Confusion Matrix")
        st.dataframe(result["confusion_matrix"])

        # ==============================
        # CLASSIFICATION REPORT
        # ==============================
        st.markdown("### 📄 Classification Report")
        st.text(result["classification_report"])


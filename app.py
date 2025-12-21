# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud

# # ==============================
# # IMPORT MODULE INTERNAL
# # ==============================
# from preprocessing.preprocessing import preprocess_stepwise
# from labeling.lexicon_labeling import label_corpus
# from feature_Extraction.tfidf_extraction import tfidf_transform
# from modeling.modeling import train_and_compare_models

# # ==============================
# # KONFIGURASI HALAMAN
# # ==============================
# st.set_page_config(
#     page_title="Text Mining App",
#     page_icon="💬",
#     layout="wide"
# )
# # ==============================
# # CSS MODERN
# # ==============================
# st.markdown("""
# <style>
# .stApp {
#     background: radial-gradient(circle at top, #0f172a, #020617);
#     color: #e5e7eb;
#     font-family: 'Inter', sans-serif;
# }
# section[data-testid="stSidebar"] {
#     background: #020617;
#     border-right: 1px solid #1e293b;
# }
# .app-header {
#     text-align: center;
#     padding: 40px 0 30px 0;
# }
# .app-header h1 {
#     font-size: 42px;
#     font-weight: 800;
#     background: linear-gradient(90deg, #38bdf8, #22c55e);
#     -webkit-background-clip: text;
#     -webkit-text-fill-color: transparent;
# }
# .app-header p {
#     font-size: 17px;
#     color: #cbd5f5;
# }
# .card {
#     background: #020617;
#     border: 1px solid #1e293b;
#     border-radius: 18px;
#     padding: 22px;
#     margin-bottom: 24px;
#     box-shadow: 0 15px 40px rgba(0,0,0,.4);
# }
# .stButton > button {
#     width: 100%;
#     height: 46px;
#     border-radius: 14px;
#     font-weight: 700;
#     color: white;
#     background: linear-gradient(90deg, #0ea5e9, #22c55e);
#     border: none;
# }
# thead tr th {
#     background: #020617 !important;
#     color: #38bdf8 !important;
# }
# tbody tr td {
#     background: #020617 !important;
#     color: #e5e7eb !important;
# }
# [data-testid="stMetric"] {
#     background: #020617;
#     border-radius: 16px;
#     padding: 16px;
#     border: 1px solid #1e293b;
# }
# </style>
# """, unsafe_allow_html=True)

# # ==============================
# # HEADER
# # ==============================
# st.markdown(
#     """
#     <h1 style="text-align:center;">💬 Aplikasi Text Mining</h1>
#     <p style="text-align:center;">
#         Preprocessing → Pelabelan → TF-IDF → Split Data → Modeling → Evaluation
#     </p>
#     <hr>
#     """,
#     unsafe_allow_html=True
# )

# # ==============================
# # SIDEBAR
# # ==============================
# menu = st.sidebar.radio(
#     "Tahapan",
#     (
#         "Input Komentar",
#         "Data Preprocessing",
#         "Pelabelan Otomatis",
#         "Feature Extraction (TF-IDF)",
#         "Modeling & Evaluation"
#     )
# )

# # ==============================
# # SESSION STATE
# # ==============================
# for key in ["komentar", "clean_text", "labels", "tfidf_df"]:
#     if key not in st.session_state:
#         st.session_state[key] = None if key == "tfidf_df" else []

# # ==============================
# # 1️⃣ INPUT KOMENTAR
# # ==============================
# if menu == "Input Komentar":
#     st.subheader("📝 Input Komentar")

#     text = st.text_area(
#         "Masukkan komentar (1 baris = 1 komentar)",
#         height=200,
#         placeholder="Contoh:\nAplikasinya sangat bagus\nPelayanannya buruk sekali"
#     )

#     if st.button("💾 Simpan Komentar"):
#         st.session_state["komentar"] = [
#             t.strip() for t in text.split("\n") if t.strip()
#         ]
#         st.success(f"{len(st.session_state['komentar'])} komentar disimpan")

#     if st.session_state["komentar"]:
#         st.dataframe(pd.DataFrame({"Komentar": st.session_state["komentar"]}))

# # ==============================
# # 2️⃣ DATA PREPROCESSING
# # ==============================
# elif menu == "Data Preprocessing":
#     st.subheader("📍 Data Preprocessing")

#     if not st.session_state["komentar"]:
#         st.warning("Belum ada komentar")
#     else:
#         result = preprocess_stepwise(st.session_state["komentar"])
#         st.session_state["clean_text"] = result["final"]

#         # ==============================
#         # ORIGINAL TEXT
#         # ==============================
#         st.markdown("## 🔹 Original Text")
#         st.dataframe(pd.DataFrame({
#             "Original Text": result["original"]
#         }))

#         # ==============================
#         # CASE FOLDING
#         # ==============================
#         st.markdown("## 🔹 Case Folding")
#         st.dataframe(pd.DataFrame({
#             "Case Folding": result["case_folding"]
#         }))

#         # ==============================
#         # CLEANING
#         # ==============================
#         st.markdown("## 🔹 Cleaning")
#         st.dataframe(pd.DataFrame({
#             "Cleaning": result["cleaning"]
#         }))

#         # ==============================
#         # TOKENIZING
#         # ==============================
#         st.markdown("## 🔹 Tokenizing")
#         st.dataframe(pd.DataFrame({
#             "Tokenizing": result["tokenizing"]
#         }))

#         # ==============================
#         # STOPWORD REMOVAL
#         # ==============================
#         st.markdown("## 🔹 Stopword Removal")
#         st.dataframe(pd.DataFrame({
#             "Stopword Removal": result["stopword"]
#         }))

#         # ==============================
#         # STEMMING
#         # ==============================
#         st.markdown("## 🔹 Stemming")
#         st.dataframe(pd.DataFrame({
#             "Stemming": result["stemming"]
#         }))

#         # ==============================
#         # FINAL TEXT
#         # ==============================
#         st.markdown("## 🔹 Final Text (Siap Modeling)")
#         st.dataframe(pd.DataFrame({
#             "Final Text": result["final"]
#         }))


# # ==============================
# # 3️⃣ PELABELAN OTOMATIS
# # ==============================
# elif menu == "Pelabelan Otomatis":
#     st.subheader("📍 Pelabelan Otomatis (Lexicon-Based)")

#     if not st.session_state["clean_text"]:
#         st.warning("Lakukan preprocessing terlebih dahulu")
#     else:
#         labels = label_corpus(st.session_state["clean_text"])
#         st.session_state["labels"] = labels

#         df_label = pd.DataFrame({
#             "Teks (Preprocessing)": st.session_state["clean_text"],
#             "Label Sentimen": labels
#         })

#         st.dataframe(df_label)
#         st.bar_chart(df_label["Label Sentimen"].value_counts())

# # ==============================
# # 4️⃣ FEATURE EXTRACTION (TF-IDF)
# # ==============================
# elif menu == "Feature Extraction (TF-IDF)":
#     st.subheader("📍 Feature Extraction – TF-IDF")

#     if not st.session_state["clean_text"]:
#         st.warning("Lakukan preprocessing terlebih dahulu")
#     else:
#         tfidf_df, tfidf_scores, _ = tfidf_transform(
#             st.session_state["clean_text"]
#         )
#         st.session_state["tfidf_df"] = tfidf_df

#         st.dataframe(tfidf_df.head())

#         st.markdown("### Top 20 Kata (TF-IDF)")
#         st.table(tfidf_scores.head(20))

#         st.markdown("### ☁️ WordCloud TF-IDF")
#         wc = WordCloud(
#             width=800,
#             height=400,
#             background_color="white"
#         ).generate_from_frequencies(tfidf_scores.to_dict())

#         fig, ax = plt.subplots(figsize=(10, 5))
#         ax.imshow(wc, interpolation="bilinear")
#         ax.axis("off")
#         st.pyplot(fig)

# # ==============================
# # 5️⃣ MODELING & EVALUATION
# # ==============================
# elif menu == "Modeling & Evaluation":
#     st.subheader("📍 Perbandingan Model Klasifikasi")

#     if st.session_state["tfidf_df"] is None or not st.session_state["labels"]:
#         st.warning("⚠️ TF-IDF dan label belum tersedia. Silakan lakukan preprocessing terlebih dahulu.")
#     else:
#         # ==============================
#         # VALIDASI SEBELUM MODELING
#         # ==============================
#         st.markdown("### 📋 Validasi Data Sebelum Modeling")
        
#         tfidf_data = st.session_state["tfidf_df"]
#         labels_data = st.session_state["labels"]
        
#         # Hitung statistik
#         n_samples = len(tfidf_data)
#         unique_labels = sorted(list(set(labels_data))) # Sort agar urutan label konsisten
#         n_classes = len(unique_labels)
        
#         # Tampilkan informasi
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric("Jumlah Sampel", n_samples)
#         with col2:
#             st.metric("Jumlah Kelas", n_classes)
#         with col3:
#             st.metric("Label", ", ".join(unique_labels))
        
#         # Validasi data sederhana (Validation logic lengkap ada di modeling.py)
#         if n_samples < 5:
#             st.error("❌ **Data terlalu sedikit** (Min 5 sampel).")
#         elif n_classes < 2:
#             st.error("❌ **Hanya ada 1 jenis label**. Butuh minimal 2 variasi sentimen.")
#         else:
#             # DATA VALID → LANJUT MODELING
#             st.success("✅ **Data valid! Memulai proses training...**")
            
#             # Tombol untuk memulai training agar user siap
#             if st.button("🚀 Mulai Training Model"):
                
#                 with st.spinner("⏳ Sedang melatih model (Naive Bayes, DT, RF, & Stacking)..."):
#                     try:
#                         # Import di dalam sini untuk menghindari circular import jika ada
#                         from modeling.modeling import train_and_compare_models

#                         result = train_and_compare_models(
#                             st.session_state["tfidf_df"],
#                             st.session_state["labels"]
#                         )
                        
#                         # Jika hasil None (artinya error di dalam modeling.py)
#                         if result is None:
#                             st.error("❌ Terjadi kesalahan saat training. Cek console/terminal.")
#                             st.stop()

#                         acc_df = result["accuracy_df"]

#                         # ==============================
#                         # BAR CHART AKURASI
#                         # ==============================
#                         st.markdown("### 📊 Perbandingan Akurasi Model")
#                         st.bar_chart(acc_df.set_index("Model"))
                        
#                         # Tampilkan tabel dengan highlight
#                         st.markdown("### 📋 Detail Akurasi")
                        
#                         def highlight_best(row):
#                             if row['Model'] == result['best_model']:
#                                 return ['background-color: #d4edda; color: #155724; font-weight: bold'] * len(row)
#                             return [''] * len(row)
                        
#                         st.dataframe(
#                             acc_df.style.apply(highlight_best, axis=1).format({"Accuracy": "{:.4f}"}),
#                             use_container_width=True
#                         )

#                         # ==============================
#                         # INFO MODEL TERBAIK
#                         # ==============================
#                         st.markdown("---")
#                         st.markdown("### 🏆 Analisis Model Terbaik")
                        
#                         best_model_name = result['best_model']
#                         best_acc = result['best_accuracy']

#                         # Logika Improvement Stacking
#                         base_models = ['Naive Bayes', 'Decision Tree', 'Random Forest']
#                         base_acc_list = []
                        
#                         for model in base_models:
#                             if model in acc_df['Model'].values:
#                                 val = acc_df.loc[acc_df['Model'] == model, 'Accuracy'].values[0]
#                                 base_acc_list.append(val)
                        
#                         col1, col2 = st.columns(2)
                        
#                         with col1:
#                             st.info(f"**Model Pemenang:**\n# {best_model_name}")
                        
#                         with col2:
#                             if best_model_name == 'Stacking' and base_acc_list:
#                                 best_base = max(base_acc_list)
#                                 improvement = best_acc - best_base
#                                 if improvement > 0:
#                                     st.success(f"**Stacking Boost:**\n# +{improvement:.2%}")
#                                     st.caption("Peningkatan akurasi dibanding single model terbaik.")
#                                 else:
#                                     st.warning("Stacking sama kuat dengan model terbaik.")
#                             else:
#                                 st.metric("Akurasi Tertinggi", f"{best_acc:.2%}")

#                         # ==============================
#                         # CONFUSION MATRIX (DIPERBAIKI)
#                         # ==============================
#                         st.markdown("### 🔢 Confusion Matrix")
                        
#                         # Ubah array menjadi DataFrame agar ada Labelnya
#                         cm_array = result["confusion_matrix"]
#                         cm_df = pd.DataFrame(
#                             cm_array, 
#                             index=[f"Actual {l}" for l in unique_labels], 
#                             columns=[f"Pred {l}" for l in unique_labels]
#                         )
#                         st.dataframe(cm_df, use_container_width=True)

#                         # ==============================
#                         # CLASSIFICATION REPORT (DIPERBAIKI)
#                         # ==============================
#                         st.markdown("### 📄 Classification Report")
                        
#                         # Karena output modeling.py sekarang DICTIONARY, kita convert ke DataFrame
#                         report_dict = result["classification_report"]
#                         report_df = pd.DataFrame(report_dict).transpose()
                        
#                         # Format angka agar rapi (2 desimal)
#                         st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
                        
#                     except Exception as e:
#                         st.error(f"❌ **ERROR Runtime:** {str(e)}")
#                         st.code(e) # Tampilkan kode error untuk debugging

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
    page_title="Text Mining",
    layout="wide"
)
# ==============================
# CSS MODERN
# ==============================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e5e7eb;
    font-family: 'Inter', sans-serif;
}
section[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid #1e293b;
}
.app-header {
    text-align: center;
    padding: 40px 0 30px 0;
}
.app-header h1 {
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-header p {
    font-size: 17px;
    color: #cbd5f5;
}
.card {
    background: #020617;
    border: 1px solid #1e293b;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 24px;
    box-shadow: 0 15px 40px rgba(0,0,0,.4);
}
.stButton > button {
    width: 100%;
    height: 46px;
    border-radius: 14px;
    font-weight: 700;
    color: white;
    background: linear-gradient(90deg, #0ea5e9, #22c55e);
    border: none;
}
thead tr th {
    background: #020617 !important;
    color: #38bdf8 !important;
}
tbody tr td {
    background: #020617 !important;
    color: #e5e7eb !important;
}
[data-testid="stMetric"] {
    background: #020617;
    border-radius: 16px;
    padding: 16px;
    border: 1px solid #1e293b;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# CUSTOM CSS (FUTURISTIC & GLASSMORPHISM)
# ==============================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    /* Font Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar Futuristik */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #161b22 100%);
        border-right: 1px solid #30363d;
    }
    
    /* Header Styling */
    .main-title {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    
    .sub-title {
        text-align: center;
        color: #8892b0;
        font-size: 0.9rem;
        letter-spacing: 3px;
        margin-bottom: 2rem;
        text-transform: uppercase;
    }

    /* Container Styling (Glassmorphism) */
    div.stDataFrame, div.stTable, .stPlotlyChart {
        background: rgba(22, 27, 34, 0.7) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(48, 54, 61, 0.8) !important;
        padding: 10px;
    }

    /* Neon Radio Buttons (Sidebar) */
    .stRadio > label {
        color: #00f2fe !important;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.8rem !important;
        letter-spacing: 1px;
    }

    /* Styling Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        padding: 12px;
        border-radius: 10px;
        font-weight: bold;
        text-transform: uppercase;
        font-family: 'Orbitron', sans-serif;
        transition: 0.4s ease;
    }
    
    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.6);
        transform: scale(1.02);
    }

    /* Status System Indicator */
    .status-box {
        background: rgba(0, 255, 170, 0.1);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #00ffaa;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================
st.markdown('<p class="main-title">TEXT MINING</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Tugas Akhir Text Mining</p>', unsafe_allow_html=True)

# ==============================
# SIDEBAR (NAVBAR)
# ==============================
with st.sidebar:
    st.markdown('<h2 style="text-align:center; color:#00f2fe; font-family:Orbitron, sans-serif;">TEXT MINING</h2>', unsafe_allow_html=True)
    st.markdown("---")
    
    menu = st.radio(
        "Pilih lah :",
        (
            "Input Komentar",
            "Data Preprocessing",
            "Pelabelan Otomatis",
            "Feature Extraction",
            "Modeling & Evaluation"
        )
    )
    
    st.markdown("---")
    st.markdown("""
        <div class="status-box">
            <p style="margin:0; font-size: 0.8rem; color: #00ffaa; font-weight:bold;">● SYSTEM ONLINE</p>
            <p style="margin:0; font-size: 0.7rem; color: #8892b0;">Engine: Lexicon + Stacking</p>
        </div>
    """, unsafe_allow_html=True)

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
    st.subheader("Input Komentar")
    
    col_in, col_info = st.columns([2, 1])
    
    with col_in:
        text = st.text_area(
            "Input raw text (1 baris = 1 data):",
            height=200,
            placeholder="Contoh:\nAplikasi sangat membantu...\nRespon lambat..."
        )
        if st.button("COBA"):
            st.session_state["komentar"] = [
                t.strip() for t in text.split("\n") if t.strip()
            ]
            st.success(f"Data komentar : {len(st.session_state['komentar'])} Data.")

    with col_info:
        if st.session_state["komentar"]:
            st.metric("Total Data", len(st.session_state["komentar"]))

    if st.session_state["komentar"]:
        st.markdown("### Preview Raw Data")
        st.dataframe(pd.DataFrame({"Komentar": st.session_state["komentar"]}), use_container_width=True)

# ==============================
# 2️⃣ DATA PREPROCESSING (PERBAIKAN)
# ==============================
elif menu == "Data Preprocessing":
    st.subheader("Data Preprocessing")

    if not st.session_state["komentar"]:
        st.warning("No data found. Please go to Input Module.")
    else:
        result = preprocess_stepwise(st.session_state["komentar"])
        st.session_state["clean_text"] = result["final"]

        # Futuristik Tabs
        t1, t2, t3 = st.tabs(["PREPROCESSING", "Hasil Cleaning", "Data Asli"])
        
        with t1:
            # Bagian Atas: Case Folding & Lexicon Cleaning
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Case Folding**")
                st.dataframe(pd.DataFrame(result["case_folding"]), use_container_width=True)
            with c2:
                st.write("**Lexicon Cleaning**")
                st.dataframe(pd.DataFrame(result["cleaning"]), use_container_width=True)
            
            # Bagian Bawah: Tokenizing & Stopword Removal (Perbaikan Tampilan)
            c3, c4 = st.columns(2)
            with c3:
                st.write("**Tokenizing**")
                # Menggunakan DataFrame agar list token terlihat jelas per baris
                st.dataframe(pd.DataFrame({"Tokens": result["tokenizing"]}), use_container_width=True)
            with c4:
                st.write("**Stopword Removal**")
                # Menggunakan DataFrame agar list hasil pembersihan stopword terlihat jelas
                st.dataframe(pd.DataFrame({"After Stopword": result["stopword"]}), use_container_width=True)

        with t2:
            st.success("Preprocessing Berhasil")
            st.dataframe(pd.DataFrame({"Cleaning Data": result["final"]}), use_container_width=True)

        with t3:
            st.write("**Data Asli :**")
            st.dataframe(pd.DataFrame({"Original Text": result["original"]}), use_container_width=True)

# ==============================
# 3️⃣ PELABELAN OTOMATIS
# ==============================
elif menu == "Pelabelan Otomatis":
    st.subheader("Lexicon Labeling")

    if not st.session_state["clean_text"]:
        st.error("Sequence Error: Preprocessing required.")
    else:
        labels = label_corpus(st.session_state["clean_text"])
        st.session_state["labels"] = labels

        df_label = pd.DataFrame({
            "Cleaned Text": st.session_state["clean_text"],
            "Sentiment Class": labels
        })

        # Perubahan: Tabel di atas, Grafik di bawah
        st.dataframe(df_label, use_container_width=True)
        
        st.markdown("#### Sentiment Distribution")
        st.bar_chart(df_label["Sentiment Class"].value_counts())

# ==============================
# 4️⃣ FEATURE EXTRACTION
# ==============================
elif menu == "Feature Extraction":
    st.subheader("Feature Vectorization (TF-IDF)")

    if not st.session_state["clean_text"]:
        st.warning("Feature Source Missing.")
    else:
        tfidf_df, tfidf_scores, _ = tfidf_transform(st.session_state["clean_text"])
        st.session_state["tfidf_df"] = tfidf_df

        st.markdown("#### Vector Matrix")
        st.dataframe(tfidf_df.iloc[:1000], use_container_width=True)

        st.markdown("---")
        cx, cy = st.columns(2)
        with cx:
            st.markdown("#### Feature Significance")
            st.table(tfidf_scores.head(50))
        with cy:
            st.markdown("#### WordCloud")
            wc = WordCloud(
                width=800, 
                height=450, 
                background_color="#0e1117", 
                colormap="cool"
            ).generate_from_frequencies(tfidf_scores.to_dict())
            
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

# # ==============================
# 5️⃣ MODELING & EVALUATION (PERBAIKAN)
# ==============================
elif menu == "Modeling & Evaluation":
    st.subheader("Model Benchmarking & Stacking")

    # Pastikan data TF-IDF dan Label tersedia sebelum training
    if st.session_state["tfidf_df"] is None or not st.session_state["labels"]:
        st.error("Modeling Blocked: Prepare features and labels first.")
    else:
        try:
            with st.spinner('Synchronizing Models & Running Cross-Validation...'):
                # Memanggil fungsi yang sudah dioptimasi di modeling.py
                result = train_and_compare_models(
                    st.session_state["tfidf_df"],
                    st.session_state["labels"]
                )

            # Jika data terlalu sedikit atau terjadi error saat split, result bisa None
            if result is not None:
                # Winner Card - Menampilkan model terbaik secara visual
                st.markdown(f"""
                    <div style="background: rgba(0, 242, 254, 0.05); padding: 30px; border-radius: 20px; border: 1px solid #4facfe; text-align: center; margin-bottom:30px;">
                        <p style="color: #4facfe; font-family: Orbitron; margin:0;">OPTIMAL CLASSIFIER FOUND</p>
                        <h1 style="color: white; margin:10px 0;">{result['best_model']}</h1>
                        <p style="font-size: 1.5rem; color: #00ffaa; margin:0; font-family: Orbitron;">Accuracy: {result['best_accuracy']:.4f}</p>
                    </div>
                """, unsafe_allow_html=True)

                # Visualisasi Perbandingan Akurasi Semua Model
                st.markdown("### Hasil Metrics")
                st.bar_chart(result["accuracy_df"].set_index("Model"))

                # Tampilan Detail Evaluasi Model Terbaik (Secara Vertikal)
                st.markdown("#### Confusion Matrix")
                # Confusion matrix ditampilkan dalam DataFrame agar rapi
                st.dataframe(result["confusion_matrix"], use_container_width=True)
                
                st.markdown("#### Classification Report")
                # Menampilkan laporan teks (Precision, Recall, F1-Score)
                st.text(result["classification_report"])
            else:
                st.error("Gagal melatih model. Pastikan jumlah sampel data mencukupi untuk proses Cross-Validation.")

<<<<<<< HEAD
                        result = train_and_compare_models(
                            st.session_state["tfidf_df"],
                            st.session_state["labels"]
                        )
                        
                        # Jika hasil None (artinya error di dalam modeling.py)
                        if result is None:
                            st.error("❌ Terjadi kesalahan saat training. Cek console/terminal.")
                            st.stop()

                        acc_df = result["accuracy_df"]

                        # ==============================
                        # BAR CHART AKURASI
                        # ==============================
                        st.markdown("### 📊 Perbandingan Akurasi Model")
                        st.bar_chart(acc_df.set_index("Model"))
                        
                        # Tampilkan tabel dengan highlight
                        st.markdown("### 📋 Detail Akurasi")
                        
                        def highlight_best(row):
                            if row['Model'] == result['best_model']:
                                return ['background-color: #d4edda; color: #155724; font-weight: bold'] * len(row)
                            return [''] * len(row)
                        
                        st.dataframe(
                            acc_df.style.apply(highlight_best, axis=1).format({"Accuracy": "{:.4f}"}),
                            use_container_width=True
                        )

                        # ==============================
                        # INFO MODEL TERBAIK
                        # ==============================
                        st.markdown("---")
                        st.markdown("### 🏆 Analisis Model Terbaik")
                        
                        best_model_name = result['best_model']
                        best_acc = result['best_accuracy']

                        # Logika Improvement Stacking
                        base_models = ['Naive Bayes', 'Decision Tree', 'Random Forest']
                        base_acc_list = []
                        
                        for model in base_models:
                            if model in acc_df['Model'].values:
                                val = acc_df.loc[acc_df['Model'] == model, 'Accuracy'].values[0]
                                base_acc_list.append(val)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.info(f"**Model Pemenang:**\n# {best_model_name}")
                        
                        with col2:
                            if best_model_name == 'Stacking' and base_acc_list:
                                best_base = max(base_acc_list)
                                improvement = best_acc - best_base
                                if improvement > 0:
                                    st.success(f"**Stacking Boost:**\n# +{improvement:.2%}")
                                    st.caption("Peningkatan akurasi dibanding single model terbaik.")
                                else:
                                    st.warning("Stacking sama kuat dengan model terbaik.")
                            else:
                                st.metric("Akurasi Tertinggi", f"{best_acc:.2%}")

                        # ==============================
                        # CONFUSION MATRIX (DIPERBAIKI)
                        # ==============================
                        st.markdown("### 🔢 Confusion Matrix")
                        
                        # Ubah array menjadi DataFrame agar ada Labelnya
                        cm_array = result["confusion_matrix"]
                        cm_df = pd.DataFrame(
                            cm_array, 
                            index=[f"Actual {l}" for l in unique_labels], 
                            columns=[f"Pred {l}" for l in unique_labels]
                        )
                        st.dataframe(cm_df, use_container_width=True)

                        # ==============================
                        # CLASSIFICATION REPORT (DIPERBAIKI)
                        # ==============================
                        st.markdown("### 📄 Classification Report")
                        
                        # Karena output modeling.py sekarang DICTIONARY, kita convert ke DataFrame
                        report_dict = result["classification_report"]
                        report_df = pd.DataFrame(report_dict).transpose()
                        
                        # Format angka agar rapi (2 desimal)
                        st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ **ERROR Runtime:** {str(e)}")
                        st.code(e) # Tampilkan kode error untuk debugging

=======
        except Exception as e:
            st.error(f"Terjadi kesalahan saat pemrosesan model: {e}")
>>>>>>> 9abf1560027b71f3254d67db6f5a3eccf3d10167

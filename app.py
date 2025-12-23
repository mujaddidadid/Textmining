import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from preprocessing.preprocessing import preprocess_stepwise
from labeling.lexicon_labeling import label_corpus
from feature_Extraction.tfidf_extraction import tfidf_transform
from modeling.modeling import train_and_compare_models
from preprocessing.preprocessing import case_folding, cleaning, tokenizing, stopword_removal, stemming
from modeling.sentiment_prediction import perform_prediction

st.set_page_config(
    page_title="Text Mining",
    layout="wide"
)

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

# HEADER

st.markdown('<p class="main-title">TEXT MINING</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Tugas Akhir Text Mining</p>', unsafe_allow_html=True)

# SIDEBAR (NAVBAR)
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
            "Modeling & Evaluation",
            "Sentiment Prediction"
        )
    )
    
    st.markdown("---")
    st.markdown("""
        <div class="status-box">
            <p style="margin:0; font-size: 0.8rem; color: #00ffaa; font-weight:bold;">● SYSTEM ONLINE</p>
            <p style="margin:0; font-size: 0.7rem; color: #8892b0;">Engine: Lexicon + Stacking</p>
        </div>
    """, unsafe_allow_html=True)


for key in ["komentar", "clean_text", "labels", "tfidf_df"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "tfidf_df" else []

def save_uploaded_file(uploaded_file):
    """Fungsi untuk menyimpan file upload ke folder lokal"""
    folder_path = "uploaded_files"
    
    # Buat folder jika belum ada
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Simpan file
    file_path = os.path.join(folder_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    return file_path

def analisa_statistik_data(komentar_list):
    """Fungsi untuk menampilkan statistik data understanding"""
    if not komentar_list:
        return

    jumlah = len(komentar_list)
    # Convert ke string untuk keamanan
    chars = [len(str(k)) for k in komentar_list]
    words = [len(str(k).split()) for k in komentar_list]

    st.markdown("---")
    st.subheader("📊 Data Understanding")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Data", f"{jumlah}")
    
    avg_char = round(sum(chars)/jumlah, 2) if jumlah else 0
    avg_word = round(sum(words)/jumlah, 2) if jumlah else 0
    
    c2.metric("Avg Karakter", f"{avg_char}")
    c3.metric("Avg Kata", f"{avg_word}")

#  DATA UNDERSTANDING
if menu == "Input Komentar":
    st.subheader("Input Data Excel & Data Understanding")

    uploaded_file = st.file_uploader("Upload File Excel (.xlsx)", type=['xlsx'])
    
    if uploaded_file is not None:
        try:
            saved_path = save_uploaded_file(uploaded_file)
            st.toast(f"File tersimpan: {saved_path}")

            # 2. Baca Excel
            df = pd.read_excel(saved_path)
            
            # 3. Pilih Kolom
            col_up1, col_up2 = st.columns([3, 1])
            with col_up1:
                nama_kolom = st.selectbox("Pilih Kolom Komentar:", df.columns)
            
            with col_up2:
                st.write("") 
                st.write("") 
                if st.button("Proses Data", key="btn_excel"):
                    data_excel = df[nama_kolom].dropna().astype(str).tolist()
                    st.session_state["komentar"] = data_excel
                    st.success(f"Berhasil memuat {len(data_excel)} baris data.")

            with st.expander("Lihat Isi File Excel (Raw)"):
                st.dataframe(df.head())

        except Exception as e:
            st.error(f"Error saat membaca file: {e}")

    # OUTPUT DATA UNDERSTANDING

    
    st.divider()
    
    if st.session_state["komentar"]:
        analisa_statistik_data(st.session_state["komentar"])
        
        st.markdown("### Preview Data Terpilih")
        st.dataframe(pd.DataFrame({"Komentar": st.session_state["komentar"]}), use_container_width=True)
    else:
        st.info("Belum ada data. Silakan upload file Excel di atas.")

# DATA PREPROCESSING
elif menu == "Data Preprocessing":
    st.subheader(" Data Preprocessing")
    if not st.session_state["komentar"]:
        st.warning("Data kosong. Silakan input data di menu 'Input Komentar' terlebih dahulu.")
    else:
        with st.spinner('Sedang memproses data (Cleaning, Tokenizing, Stemming)... Harap tunggu.'):
            
            result = preprocess_stepwise(st.session_state["komentar"])
            
            st.session_state["clean_text"] = result["final"]

        t1, t2, t3 = st.tabs([" Tahapan Preprocessing", " Hasil Final & Download", " Data Asli"])
        
        # --- TAB 1: TAHAPAN DETIL ---
        with t1:
            st.info("Berikut adalah detail perubahan teks pada setiap tahapan:")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**1. Case Folding**")
                st.caption("Mengubah huruf menjadi kecil semua.")
                st.dataframe(pd.DataFrame({"Result": result["case_folding"]}), use_container_width=True)
            with c2:
                st.markdown("**2. Cleaning**")
                st.caption("Menghapus angka, simbol, dan link.")
                st.dataframe(pd.DataFrame({"Result": result["cleaning"]}), use_container_width=True)
            
            c3, c4 = st.columns(2)
            with c3:
                st.markdown("**3. Tokenizing**")
                st.caption("Memecah kalimat menjadi kata-kata.")
                st.dataframe(pd.DataFrame({"Result": result["tokenizing"]}), use_container_width=True)
            with c4:
                st.markdown("**4. Stopword Removal**")
                st.caption("Menghapus kata umum yang tidak bermakna.")
                st.dataframe(pd.DataFrame({"Result": result["stopword"]}), use_container_width=True)

            st.divider() 
            st.markdown("**5. Stemming (Sastrawi)**")
            st.caption("Mengubah kata berimbuhan menjadi kata dasar (Proses terberat).")
            st.dataframe(pd.DataFrame({"Result": result["stemming"]}), use_container_width=True)

        with t2:
            st.success(" Preprocessing Selesai! Data siap untuk tahap Labeling.")
            
            st.write("**Preview Data Bersih (Final Text):**")
            st.dataframe(pd.DataFrame({"Final Text": result["final"]}), use_container_width=True)

            st.divider()
            

            file_path = "uploaded_files/data_preprocessing_result.xlsx"
            
            if os.path.exists(file_path):
                with open(file_path, "rb") as file:
                    st.download_button(
                        label="⬇ Download Hasil Preprocessing (.xlsx)",
                        data=file,
                        file_name="hasil_preprocessing_lengkap.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.warning("File hasil preprocessing belum ditemukan di server.")

        # --- TAB 3: DATA ASLI ---
        with t3:
            st.write("**Data Asli (Raw Data):**")
            st.dataframe(pd.DataFrame({"Original Text": result["original"]}), use_container_width=True)


# PELABELAN OTOMATIS
elif menu == "Pelabelan Otomatis":
    st.subheader(" Pelabelan Otomatis (Lexicon)")

    if not st.session_state["clean_text"]:
        st.warning(" Data preprocessing belum tersedia. Silakan jalankan 'Data Preprocessing' terlebih dahulu.")
    else:
        with st.spinner('Sedang melakukan pelabelan sentimen dengan Lexicon...'):
            
            # 1. Proses Labeling
            labels = label_corpus(st.session_state["clean_text"])
            st.session_state["labels"] = labels
            
            # 2. Buat DataFrame Hasil
            df_label = pd.DataFrame({
                "Teks Bersih (Stemmed)": st.session_state["clean_text"],
                "Prediksi Sentimen": labels
            })
            
            file_path_label = "uploaded_files/hasil_labeling.xlsx"
            if not os.path.exists("uploaded_files"):
                os.makedirs("uploaded_files")
                
            df_label.to_excel(file_path_label, index=False)

        t_view, t_stats = st.tabs([" Tabel Data", " Statistik & Download"])

        with t_view:
            st.success(f"Labeling selesai! Total Data: {len(df_label)}")
            st.dataframe(df_label, use_container_width=True)

        with t_stats:
            counts = df_label["Prediksi Sentimen"].value_counts()
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("**Jumlah per Sentimen:**")
                st.dataframe(counts, use_container_width=True)
            with c2:
                st.bar_chart(counts, color=["#34d399"]) 

            st.divider()
            
            if os.path.exists(file_path_label):
                with open(file_path_label, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Hasil Labeling (.xlsx)",
                        data=f,
                        file_name="hasil_labeling_lexicon.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.warning("Gagal menyimpan file hasil labeling.")


#  FEATURE EXTRACTION
elif menu == "Feature Extraction":
    st.subheader("Feature Vectorization (TF-IDF)")

    if not st.session_state["clean_text"]:
        st.warning("Feature Source Missing.")
    else:
        tfidf_df, tfidf_scores, vectorizer_obj = tfidf_transform(st.session_state["clean_text"])
        
        st.session_state["tfidf_df"] = tfidf_df
        st.session_state["vectorizer"] = vectorizer_obj 
        
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


#  MODELING & EVALUATION 
elif menu == "Modeling & Evaluation":
    st.subheader("Model Benchmarking & Stacking")

    if st.session_state["tfidf_df"] is None or not st.session_state["labels"]:
        st.error("Modeling Blocked: Prepare features and labels first.")
    else:
        try:
            with st.spinner('Synchronizing Models & Running Cross-Validation...'):
                result = train_and_compare_models(
                    st.session_state["tfidf_df"],
                    st.session_state["labels"]
                )

            if result is not None:
                st.markdown(f"""
                    <div style="background: rgba(0, 242, 254, 0.05); padding: 30px; border-radius: 20px; border: 1px solid #4facfe; text-align: center; margin-bottom:30px;">
                        <p style="color: #4facfe; font-family: Orbitron; margin:0;">OPTIMAL CLASSIFIER FOUND</p>
                        <h1 style="color: white; margin:10px 0;">{result['best_model']}</h1>
                        <p style="font-size: 1.5rem; color: #00ffaa; margin:0; font-family: Orbitron;">Accuracy: {result['best_accuracy']:.4f}</p>
                    </div>
                """, unsafe_allow_html=True)

                import pickle 
                
                with open('best_model.pkl', 'wb') as f:
                    pickle.dump(result['model_object'], f)
                
                if "vectorizer" in st.session_state:
                    with open('vectorizer.pkl', 'wb') as f:
                        pickle.dump(st.session_state["vectorizer"], f)
                
                st.success("Berhasil: Model (best_model.pkl) & Vectorizer (vectorizer.pkl) telah disimpan otomatis ke folder project.")

                st.markdown("### Hasil Metrics")
                st.bar_chart(result["accuracy_df"].set_index("Model"))

                st.markdown("#### Confusion Matrix")
                st.dataframe(result["confusion_matrix"], use_container_width=True)
                
                st.markdown("#### Classification Report")
                st.text(result["classification_report"])
            else:
                st.error("Gagal melatih model. Pastikan jumlah sampel data mencukupi.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat pemrosesan model: {e}")

#  SENTIMENT PREDICTION guanakan PICKLE

elif menu == "Sentiment Prediction":
    st.subheader("🔮 Analisis Sentimen Otomatis")

    user_input = st.text_area("Masukkan komentar:", placeholder="Contoh: telaga sarangan bagus sekali")

    show_steps = st.checkbox("Tampilkan proses preprocessing (step-by-step)", value=True)

    if st.button("ANALISA SEKARANG", use_container_width=True):
        if user_input:
            with st.spinner('Sedang memproses prediksi...'):
                
                from modeling.sentiment_prediction import perform_prediction
                ml_res, lex_res, cleaned, error = perform_prediction(user_input)

            if error:
                st.error(f"Terjadi kesalahan: {error}")
            else:
                if show_steps:
                    st.info("✅ Preprocessing Selesai")
                    from preprocessing.preprocessing import case_folding, cleaning, tokenizing, stopword_removal, stemming
                    
                    s_casefold = case_folding(user_input)
                    s_cleaning = cleaning(s_casefold)
                    s_tokens = tokenizing(s_cleaning)
                    s_stopword = stopword_removal(s_tokens)
                    s_stemming = stemming(s_stopword)

                    tab1, tab2, tab3 = st.tabs(["Cleaning", "Stemming", "Final Text"])
                    
                    with tab1:
                        c1, c2 = st.columns(2)
                        with c1: 
                            st.write("**Case Folding**")
                            st.code(s_casefold, language="text")
                        with c2: 
                            st.write("**Cleaning**")
                            st.code(s_cleaning, language="text")
                    
                    with tab2:
                        st.write("**Stopword & Stemming**")
                        st.write(s_stemming)
                    
                    with tab3:
                        st.code(cleaned, language='text')

                st.markdown("---")
                st.subheader("HASIL PREDIKSI SENTIMEN")
                
                if "Positif" in str(lex_res):
                    st.success(f"**Lexicon Based**\n\n# {lex_res}", icon="📖")
                elif "Negatif" in str(lex_res):
                    st.error(f"**Lexicon Based**\n\n# {lex_res}", icon="📖")
                else:
                    st.warning(f"**Lexicon Based**\n\n# {lex_res}", icon="📖")

                with st.expander("Lihat Teks Bersih yang Diproses", expanded=True):
                    st.caption("Teks ini yang digunakan untuk menghitung skor sentimen:")
                    st.code(cleaned, language="text")

        else:
            st.warning("⚠️ Mohon masukkan teks komentar terlebih dahulu.")
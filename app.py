import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from preprocessing.preprocessing import preprocess_stepwise
from labeling.lexicon_labeling import label_corpus
from feature_Extraction.tfidf_extraction import tfidf_transform
from modeling.modeling import train_and_compare_models

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


for key in ["komentar", "clean_text", "labels", "tfidf_df"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "tfidf_df" else []

#  INPUT KOMENTAR
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

# DATA PREPROCESSING (PERBAIKAN)
elif menu == "Data Preprocessing":
    st.subheader("Data Preprocessing")

    if not st.session_state["komentar"]:
        st.warning("No data found. Please go to Input Module.")
    else:
        result = preprocess_stepwise(st.session_state["komentar"])
        st.session_state["clean_text"] = result["final"]
        t1, t2, t3 = st.tabs(["PREPROCESSING", "Hasil Cleaning", "Data Asli"])
        
        with t1:
            c1, c2 = st.columns(2)
            with c1:
                st.write("**1. Case Folding**")
                st.dataframe(pd.DataFrame({"Case Folding": result["case_folding"]}), use_container_width=True)
            with c2:
                st.write("**2. Lexicon Cleaning**")
                st.dataframe(pd.DataFrame({"Cleaning": result["cleaning"]}), use_container_width=True)
            
            c3, c4 = st.columns(2)
            with c3:
                st.write("**3. Tokenizing**")
                st.dataframe(pd.DataFrame({"Tokens": result["tokenizing"]}), use_container_width=True)
            with c4:
                st.write("**4. Stopword Removal**")
                st.dataframe(pd.DataFrame({"Stopword Removal": result["stopword"]}), use_container_width=True)

            st.divider() 
            st.write("**5. Stemming (Sastrawi)**")
            st.caption("Mengubah kata berimbuhan menjadi kata dasar bahasa Indonesia.")
            st.dataframe(pd.DataFrame({"Stemmed Tokens": result["stemming"]}), use_container_width=True)

        with t2:
            st.success("Preprocessing Berhasil Disimpan!")
            st.write("Data teks final yang siap untuk klasifikasi:")
            st.dataframe(pd.DataFrame({"Final Text": result["final"]}), use_container_width=True)

        with t3:
            st.write("**Data Asli (Raw Data):**")
            st.dataframe(pd.DataFrame({"Original Text": result["original"]}), use_container_width=True)

#  PELABELAN OTOMATIS (UPDATED)
elif menu == "Pelabelan Otomatis":
    st.subheader("Lexicon Labeling")

    if not st.session_state["clean_text"]:
        st.error("Sequence Error: Preprocessing required.")
    else:
        labels = label_corpus(st.session_state["clean_text"])
        st.session_state["labels"] = labels
        df_label = pd.DataFrame({
            "Teks Bersih (Stemmed)": st.session_state["clean_text"],
            "Prediksi Sentimen": labels
        })
        st.dataframe(df_label, use_container_width=True)
        st.markdown("#### Sentiment Distribution")
        st.bar_chart(df_label["Prediksi Sentimen"].value_counts())

#  FEATURE EXTRACTION
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

# MODELING & EVALUATION 
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

                st.markdown("### Hasil Metrics")
                st.bar_chart(result["accuracy_df"].set_index("Model"))

                st.markdown("#### Confusion Matrix")
                st.dataframe(result["confusion_matrix"], use_container_width=True)
                
                st.markdown("#### Classification Report")
                st.text(result["classification_report"])
            else:
                st.error("Gagal melatih model. Pastikan jumlah sampel data mencukupi untuk proses Cross-Validation.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat pemrosesan model: {e}")
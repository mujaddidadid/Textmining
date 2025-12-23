import streamlit as st
import pandas as pd
import os

# 1. FUNGSI UNTUK MENYIMPAN FILE
def save_uploaded_file(uploaded_file):
    folder_path = "uploaded_files"    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        file_path = os.path.join(folder_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    return file_path

# 2. FUNGSI DATA UNDERSTANDING (Milik Anda)
def show_data_understanding(komentar_list):
    st.subheader(" Data Understanding")

    if len(komentar_list) == 0:
        st.warning("Belum ada komentar yang diinput.")
        return

    jumlah_komentar = len(komentar_list)
    panjang_karakter = [len(str(k)) for k in komentar_list] 
    jumlah_kata = [len(str(k).split()) for k in komentar_list]

    st.markdown("###  Statistik Data Teks")

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Komentar", jumlah_komentar)
    
    avg_char = round(sum(panjang_karakter) / jumlah_komentar, 2) if jumlah_komentar > 0 else 0
    avg_word = round(sum(jumlah_kata) / jumlah_komentar, 2) if jumlah_komentar > 0 else 0
    
    col2.metric("Rata-rata Karakter", avg_char)
    col3.metric("Rata-rata Kata", avg_word)

    st.markdown("###  Contoh Komentar (5 Data Awal)")
    for i, k in enumerate(komentar_list[:5], start=1):
        st.write(f"{i}. {k}")

def run_data_understanding_page():
    st.title("Data Understanding Module")
    st.write("Silakan upload data Excel untuk dianalisis.")

    # Widget Upload File
    uploaded_file = st.file_uploader("Upload File Excel (.xlsx)", type=['xlsx'])

    if uploaded_file is not None:
        try:
            saved_path = save_uploaded_file(uploaded_file)
            st.success(f" File berhasil disimpan di: {saved_path}")

            df = pd.read_excel(uploaded_file)
            
            with st.expander("Lihat Data Mentah (Preview)"):
                st.dataframe(df.head())


            st.info(" Pilih kolom yang berisi teks komentar:")
            list_kolom = df.columns.tolist()
            nama_kolom = st.selectbox("Nama Kolom Komentar", list_kolom)

            komentar_list = df[nama_kolom].tolist()

            st.divider()
            show_data_understanding(komentar_list)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")


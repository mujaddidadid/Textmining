import streamlit as st

def show_data_understanding(komentar_list):
    st.subheader("📊 Data Understanding")

    if len(komentar_list) == 0:
        st.warning("Belum ada komentar yang diinput.")
        return

    # Statistik dasar
    jumlah_komentar = len(komentar_list)
    panjang_karakter = [len(k) for k in komentar_list]
    jumlah_kata = [len(k.split()) for k in komentar_list]

    st.markdown("### 📌 Statistik Data Teks")

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Komentar", jumlah_komentar)
    col2.metric("Rata-rata Karakter", round(sum(panjang_karakter) / jumlah_komentar, 2))
    col3.metric("Rata-rata Kata", round(sum(jumlah_kata) / jumlah_komentar, 2))

    st.markdown("### 📝 Contoh Komentar")
    for i, k in enumerate(komentar_list[:5], start=1):
        st.write(f"{i}. {k}")

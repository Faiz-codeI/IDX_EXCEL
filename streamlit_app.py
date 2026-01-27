import streamlit as st

st.set_page_config(page_title="IDX Excel", page_icon="📊")

st.title("IDX Excel")
st.write(
    "Gunakan aplikasi ini untuk mengunggah dan memeriksa data Excel. "
    "File utama untuk Streamlit ada di `streamlit_app.py`."
)

uploaded_file = st.file_uploader("Unggah file Excel", type=["xlsx", "xls"])

if uploaded_file:
    st.success("File diterima. Tambahkan logika pemrosesan sesuai kebutuhan.")

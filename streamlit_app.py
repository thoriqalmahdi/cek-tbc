# app_streamlit.py
import os
import json
import numpy as np
import joblib
import streamlit as st
from PIL import Image

from feature_extractor_vgg16 import build_vgg16_extractor, extract_features

# -----------------------------
# Konfigurasi halaman
# -----------------------------
st.set_page_config(page_title="Klasifikasi CXR (VGG16 + SVM)", layout="centered")
st.title("🩺 Klasifikasi CXR (VGG16 + SVM)")
st.caption("Unggah gambar, diekstrak dengan VGG16, lalu diklasifikasikan oleh SVM.")

# -----------------------------
# Cache model agar tidak reload tiap interaksi
# -----------------------------
@st.cache_resource
def load_models():
    svm_model = joblib.load("svm_model.pkl")
    scaler = joblib.load("scaler.pkl")
    vgg = build_vgg16_extractor()
    return svm_model, scaler, vgg

svm_model, scaler, vgg_model = load_models()

# Nama kelas — bisa hardcode atau baca dari file
DEFAULT_CLASS_NAMES = ["Normal", "Pneumonia", "Tuberkulosis"]
class_names = DEFAULT_CLASS_NAMES
if os.path.exists("class_names.json"):
    try:
        with open("class_names.json", "r") as f:
            class_names = json.load(f)
    except Exception:
        pass

# -----------------------------
# UI upload gambar
# -----------------------------
img_file = st.file_uploader("Unggah gambar JPG/PNG", type=["jpg", "jpeg", "png"])

if img_file is not None:
    # Tampilkan preview
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    # Simpan sementara agar extractor (yang menerima path) bisa membaca
    tmp_path = "tmp_upload.png"
    img.save(tmp_path)

    # Ekstraksi fitur
    with st.spinner("🔍 Mengekstraksi fitur dengan VGG16..."):
        features = extract_features(tmp_path, vgg_model)

    # Siapkan (1, n_features) dan scaling
    feat_2d = np.asarray(features).reshape(1, -1)
    feat_scaled = scaler.transform(feat_2d)

    # Prediksi
    pred_idx = int(svm_model.predict(feat_scaled)[0])
    pred_name = class_names[pred_idx] if pred_idx < len(class_names) else f"kelas_{pred_idx}"

    st.success(f"✅ Prediksi: **{pred_name}**")

    # (Opsional) probabilitas jika SVC dilatih dengan probability=True
    if hasattr(svm_model, "predict_proba"):
        try:
            proba = svm_model.predict_proba(feat_scaled)[0]
            # tampilkan top-3
            order = sorted(list(enumerate(proba)), key=lambda x: x[1], reverse=True)
            st.subheader("Probabilitas:")
            for i, p in order:
                label = class_names[i] if i < len(class_names) else f"kelas_{i}"
                st.write(f"- {label}: {p:.4f}")
        except Exception:
            st.info("Model tidak dikonfigurasi untuk probabilitas (butuh SVC(probability=True) saat training).")


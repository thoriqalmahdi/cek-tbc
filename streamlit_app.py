# streamlit_app.py
import os, json, tempfile, numpy as np, joblib
import streamlit as st
from PIL import Image
import pandas as pd

from feature_extractor_vgg16 import build_vgg16_extractor, extract_features

# -----------------------------
# 🎨 Global page config + CSS
# -----------------------------
st.set_page_config(page_title="Klasifikasi CXR (VGG16 + SVM)", page_icon="🩺", layout="wide")

CUSTOM_CSS = """
<style>
/* center main column a bit */
.main .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1100px;}
/* pretty cards */
.card {
  background: #ffffff;
  border: 1px solid rgba(0,0,0,.06);
  border-radius: 16px;
  padding: 18px 20px;
  box-shadow: 0 8px 24px rgba(0,0,0,.06);
}
.badge {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 0.78rem;
  background: #EEF2FF; color: #3730A3;
  border: 1px solid #C7D2FE;
}
.pred-label {
  font-size: 1.4rem; font-weight: 700; margin: 4px 0 0 0;
}
.subtle { color: #6b7280; }
.footer { text-align:center; color:#9ca3af; padding-top:12px; font-size:0.9rem;}
hr {border: none; border-top: 1px solid #eee; margin: 0.8rem 0 1.2rem;}
.upload-tip {font-size:0.92rem; color:#6b7280;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# 🧠 Cache model & assets
# -----------------------------
@st.cache_resource
def load_models():
    svm = joblib.load("svm_model.pkl")
    scaler = joblib.load("scaler.pkl")
    vgg = build_vgg16_extractor()
    return svm, scaler, vgg

svm_model, scaler, vgg_model = load_models()

# class names
DEFAULT_CLASSES = ["Normal", "Pneumonia", "Tuberkulosis"]
if os.path.exists("class_names.json"):
    try:
        CLASS_NAMES = json.load(open("class_names.json"))
    except Exception:
        CLASS_NAMES = DEFAULT_CLASSES
else:
    CLASS_NAMES = DEFAULT_CLASSES

# -----------------------------
# 🧭 Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")
    show_probs = st.toggle("Tampilkan probabilitas", value=True)
    conf_warn = st.slider("Peringatan jika probabilitas tertinggi < ", 0.0, 1.0, 0.60, 0.05)
    st.markdown("---")
    st.markdown("### ℹ️ Info Model")
    st.caption("• Ekstraktor: **VGG16 (ImageNet, GAP)**\n\n• Klasifier: **SVM**\n\n• Preprocess: **StandardScaler**")
    st.markdown("---")
    st.markdown("Made with ❤️ Streamlit")

# -----------------------------
# 🏷️ Header
# -----------------------------
st.markdown("""
<div class='card'>
  <span class='badge'>VGG16 ➜ Feature Vector ➜ SVM</span>
  <h2 style='margin:6px 0 4px 0'>Klasifikasi CXR</h2>
  <div class='subtle'>Unggah gambar X-Ray (JPG/PNG). Aplikasi mengekstrak fitur via VGG16 lalu mengklasifikasikan dengan SVM.</div>
</div>
""", unsafe_allow_html=True)
st.write("")

# -----------------------------
# 📤 Upload + Layout
# -----------------------------
left, right = st.columns([5, 7], gap="large")

with left:
    st.markdown("#### 1) Unggah Gambar")
    img_file = st.file_uploader("Tarik & lepas atau pilih file", type=["jpg", "jpeg", "png"])
    st.markdown("<div class='upload-tip'>Tips: gunakan gambar ~224×224 atau lebih besar. Format warna RGB otomatis.</div>", unsafe_allow_html=True)

    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, caption="Preview", use_column_width=True)
    else:
        st.info("Belum ada gambar. Unggah untuk mulai klasifikasi.")

with right:
    st.markdown("#### 2) Hasil")
    result_box = st.empty()

    if img_file:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name

        # Progress simulasi UX
        p = st.progress(0, text="Mempersiapkan...")
        p.progress(20, text="Ekstraksi fitur VGG16…")
        feats = extract_features(tmp_path, vgg_model)
        p.progress(55, text="Normalisasi fitur…")
        feat_2d = np.asarray(feats).reshape(1, -1)
        feat_scaled = scaler.transform(feat_2d)
        p.progress(75, text="Inferensi SVM…")
        pred_idx = int(svm_model.predict(feat_scaled)[0])
        pred_name = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"kelas_{pred_idx}"

        # Probabilitas (jika ada)
        probs = None
        if show_probs and hasattr(svm_model, "predict_proba"):
            try:
                probs = svm_model.predict_proba(feat_scaled)[0]
            except Exception:
                probs = None
        p.progress(100, text="Selesai ✅")

        # ---- Kartu hasil
        with result_box.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("**Prediksi**")
            st.markdown(f"<div class='pred-label'>{pred_name}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='subtle'>Index: {pred_idx}</div>", unsafe_allow_html=True)
            st.markdown("<hr/>", unsafe_allow_html=True)

            if probs is not None:
                df = pd.DataFrame({
                    "Kelas": CLASS_NAMES,
                    "Probabilitas": probs
                }).sort_values("Probabilitas", ascending=False).reset_index(drop=True)

                top_p = float(df.loc[0, "Probabilitas"])
                if top_p < conf_warn:
                    st.warning(f"Keyakinan model rendah (p={top_p:.3f} < {conf_warn:.2f}). Pertimbangkan verifikasi manual.")

                colA, colB = st.columns([4, 5])
                with colA:
                    st.dataframe(df.style.format({"Probabilitas": "{:.4f}"}), use_container_width=True, hide_index=True)
                with colB:
                    st.bar_chart(df.set_index("Kelas"))

            # Detail teknis (expandable)
            with st.expander("Detail teknis"):
                st.write(f"- Panjang vektor fitur: **{len(feats)}**")
                st.write("- Preprocess: **StandardScaler** dari data latih")
                st.write("- Ekstraktor: **VGG16 include_top=False, GlobalAveragePooling2D**")
            st.markdown("</div>", unsafe_allow_html=True)

        # Clean temp
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# -----------------------------
# 🧾 Footer
# -----------------------------
st.markdown("<div class='footer'>© 2025 – Demo VGG16 + SVM • Streamlit</div>", unsafe_allow_html=True)

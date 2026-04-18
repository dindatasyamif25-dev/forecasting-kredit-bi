
# ============================================================
# app.py — Fixed Version (tanpa unsafe HTML)
# ============================================================

import streamlit as st
import pandas as pd
import sys, os

sys.path.append("/content/forecasting_kredit_bi")

from modules.preprocessing       import proses_data
from modules.model_arima         import jalankan_arima
from modules.model_sarima        import jalankan_sarima
from modules.model_random_forest import jalankan_random_forest
from modules.evaluation          import jalankan_evaluasi
from modules.dashboard           import (
    grafik_historis, grafik_train_test,
    grafik_forecast_vs_aktual, grafik_forecast_depan,
    grafik_metrik, grafik_feature_importance,
    tampil_tabel_forecast, tampil_insight,
)

# ── KONFIGURASI HALAMAN ──────────────────────────────────────
st.set_page_config(
    page_title="BI Forecasting Kredit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Pengaturan")
    st.divider()

    st.markdown("### 📁 Upload Data")
    uploaded = st.file_uploader(
        "Upload file CSV atau Excel",
        type=["csv", "xlsx", "xls"],
    )

    st.markdown("atau")
    pakai_sample = st.button(
        "📊 Gunakan Data Sample",
        use_container_width=True,
        type="primary",
    )

    st.divider()
    st.markdown("### 🔧 Pengaturan Model")
    n_periods = st.slider(
        "Periode Forecast (bulan)",
        min_value=1, max_value=24, value=6,
    )
    train_ratio = st.slider(
        "Rasio Training (%)",
        min_value=60, max_value=90, value=80, step=5,
    ) / 100

    st.divider()
    st.info(
        "Aplikasi membandingkan 3 metode:\n\n"
        "📘 ARIMA — Statistik klasik\n\n"
        "📗 SARIMA — Dengan seasonal\n\n"
        "📙 Random Forest — Machine Learning"
    )
    st.caption("© 2024 BI Forecasting Kredit v1.0")

# ── HEADER ───────────────────────────────────────────────────
st.title("🏦 BI Forecasting Kredit Perbankan")
st.caption("Perbandingan Metode ARIMA · SARIMA · Random Forest "
           "untuk Mendukung Business Intelligence")
st.divider()

# ── TENTUKAN SUMBER DATA ──────────────────────────────────────
sumber_data = None

if pakai_sample:
    sumber_data = "data/sample_data.csv"
    st.success("📊 Menggunakan data sample kredit perbankan "
               "(Jan 2015 — Des 2023)")
elif uploaded is not None:
    sumber_data = uploaded
    st.success(f"📁 File diupload: **{uploaded.name}**")

# ── HALAMAN AWAL (belum ada data) ────────────────────────────
if sumber_data is None:
    st.markdown("## 👆 Mulai dengan Upload Data")
    st.markdown(
        "Upload file CSV/Excel di sidebar kiri, "
        "atau klik **Gunakan Data Sample** untuk mencoba."
    )

    with st.expander("📋 Lihat Format Data yang Diperlukan"):
        st.markdown("**Kolom yang diperlukan:**")
        st.markdown("- Kolom **tanggal** (Date, Tanggal, Periode, dll)")
        st.markdown("- Kolom **nilai kredit** (Total_Kredit, Kredit, dll)")
        st.markdown("**Contoh format CSV:**")
        contoh = pd.DataFrame({
            "Tanggal"     : ["2020-01-01","2020-02-01","2020-03-01"],
            "Total_Kredit": [5234.5, 5312.8, 5289.3],
        })
        st.dataframe(contoh, hide_index=True, use_container_width=True)
    st.stop()

# ── 1. PREPROCESSING ─────────────────────────────────────────
st.markdown("## 📥 1. Data & Preprocessing")

with st.spinner("⏳ Memproses data..."):
    hasil_prep, error = proses_data(sumber_data, train_ratio)

if error:
    st.error(error)
    st.stop()

stats = hasil_prep["stats"]

# Metric cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("📅 Total Data",    f"{stats['jumlah_data']} bulan")
col2.metric("🏋️ Training",     f"{hasil_prep['n_train']} bulan")
col3.metric("🧪 Testing",      f"{hasil_prep['n_test']} bulan")
col4.metric("🔍 Missing Value", hasil_prep["jumlah_missing"])

# Grafik
st.plotly_chart(
    grafik_historis(hasil_prep["ts"], hasil_prep["kolom_nilai"]),
    use_container_width=True,
)
st.plotly_chart(
    grafik_train_test(hasil_prep["train"], hasil_prep["test"]),
    use_container_width=True,
)

with st.expander("🔎 Preview Data (10 baris pertama)"):
    st.dataframe(
        hasil_prep["df"].head(10),
        use_container_width=True,
        hide_index=True,
    )

# ── 2. PEMODELAN ─────────────────────────────────────────────
st.divider()
st.markdown("## 🤖 2. Pemodelan — ARIMA · SARIMA · Random Forest")

col1, col2, col3 = st.columns(3)

with col1:
    with st.spinner("⏳ Melatih ARIMA..."):
        hasil_arima, err = jalankan_arima(
            hasil_prep["train"], hasil_prep["test"], n_periods
        )
    if err:
        st.error(f"ARIMA gagal: {err}")
        hasil_arima = None
    else:
        st.success(f"✅ ARIMA {hasil_arima['order']}")

with col2:
    with st.spinner("⏳ Melatih SARIMA..."):
        hasil_sarima, err = jalankan_sarima(
            hasil_prep["train"], hasil_prep["test"], n_periods
        )
    if err:
        st.error(f"SARIMA gagal: {err}")
        hasil_sarima = None
    else:
        st.success(f"✅ SARIMA {hasil_sarima['order']}")

with col3:
    with st.spinner("⏳ Melatih Random Forest..."):
        hasil_rf, err = jalankan_random_forest(
            hasil_prep["train"], hasil_prep["test"], n_periods
        )
    if err:
        st.error(f"RF gagal: {err}")
        hasil_rf = None
    else:
        st.success(f"✅ Random Forest ({hasil_rf['n_estimators']} trees)")

if not any([hasil_arima, hasil_sarima, hasil_rf]):
    st.error("Semua model gagal. Periksa data Anda.")
    st.stop()

# ── 3. EVALUASI ───────────────────────────────────────────────
st.divider()
st.markdown("## 📊 3. Evaluasi & Perbandingan Model")

hasil_eval, error = jalankan_evaluasi(
    hasil_prep["test"],
    hasil_arima, hasil_sarima, hasil_rf,
    stats,
)

if error:
    st.error(error)
    st.stop()

# KPI
col1, col2, col3, col4 = st.columns(4)
col1.metric("📅 Total Data",      f"{stats['jumlah_data']} bulan")
col2.metric("📈 Pertumbuhan",     f"{stats['growth_pct']}%")
col3.metric("🏆 Model Terbaik",   hasil_eval["model_terbaik"])
col4.metric("⭐ MAPE Terbaik",    f"{hasil_eval['mape_terbaik']}%")

# Tabel evaluasi
st.markdown("### 🏆 Tabel Perbandingan Akurasi")
st.dataframe(
    hasil_eval["df_eval"],
    use_container_width=True,
    hide_index=True,
)

# Grafik metrik & forecast
st.plotly_chart(
    grafik_metrik(hasil_eval["df_eval"]),
    use_container_width=True,
)
st.plotly_chart(
    grafik_forecast_vs_aktual(
        hasil_prep["test"],
        hasil_arima, hasil_sarima, hasil_rf,
    ),
    use_container_width=True,
)

# ── 4. DASHBOARD BI ──────────────────────────────────────────
st.divider()
st.markdown("## 🌐 4. Business Intelligence Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Forecast ke Depan",
    "📋 Tabel Forecast",
    "🌲 Feature Importance",
    "💡 Insight & Rekomendasi",
])

with tab1:
    st.plotly_chart(
        grafik_forecast_depan(
            hasil_prep["ts"],
            hasil_arima, hasil_sarima, hasil_rf,
            hasil_eval["model_terbaik"],
        ),
        use_container_width=True,
    )

with tab2:
    st.markdown("### 📋 Tabel Forecast Semua Model")
    tampil_tabel_forecast(
        hasil_arima, hasil_sarima, hasil_rf,
        hasil_eval["model_terbaik"],
    )

with tab3:
    if hasil_rf and hasil_rf.get("feature_importance") is not None:
        st.plotly_chart(
            grafik_feature_importance(
                hasil_rf["feature_importance"]
            ),
            use_container_width=True,
        )
    else:
        st.warning("Random Forest tidak berhasil dijalankan.")

with tab4:
    st.markdown("### 💡 Insight Otomatis")
    tampil_insight(hasil_eval["insights"])

# Footer
st.divider()
st.caption(
    "🏦 BI Forecasting Kredit Perbankan v1.0 | "
    "Magister Informatika | "
    "Powered by Streamlit & Python"
)

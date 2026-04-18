
# ============================================================
# dashboard.py — Komponen Visual Dashboard Streamlit
# ============================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st


# ── 1. LOAD CSS ──────────────────────────────────────────────
def load_css():
    """Memuat custom CSS untuk tampilan modern."""
    try:
        with open("assets/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>",
                        unsafe_allow_html=True)
    except:
        pass


# ── 2. HEADER UTAMA ──────────────────────────────────────────
def tampil_header():
    """Menampilkan header utama aplikasi."""
    st.markdown("""
    <div class="main-header">
        <h1>🏦 BI Forecasting Kredit Perbankan</h1>
        <p>Perbandingan Metode ARIMA · SARIMA · Random Forest
        untuk Mendukung Business Intelligence</p>
    </div>
    """, unsafe_allow_html=True)


# ── 3. KPI CARDS ─────────────────────────────────────────────
def tampil_kpi(stats, model_terbaik, mape_terbaik, kat_terbaik):
    """Menampilkan kartu KPI ringkasan di bagian atas."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📊 Total Data",
            value=f"{stats['jumlah_data']} Bulan",
            delta=f"{stats['periode_awal']} → {stats['periode_akhir']}"
        )
    with col2:
        st.metric(
            label="📈 Pertumbuhan Kredit",
            value=f"{stats['growth_pct']}%",
            delta="Naik" if stats["growth_pct"] > 0 else "Turun"
        )
    with col3:
        st.metric(
            label="🏆 Model Terbaik",
            value=model_terbaik,
            delta=f"MAPE {mape_terbaik}%"
        )
    with col4:
        st.metric(
            label="⭐ Akurasi",
            value=kat_terbaik,
            delta=f"MAPE < 10% = Sangat Baik"
        )


# ── 4. GRAFIK DATA HISTORIS ───────────────────────────────────
def grafik_historis(ts, kolom_nilai):
    """Grafik tren data kredit historis."""
    fig = go.Figure()

    # Konversi index ke datetime
    x_vals = [str(p) for p in ts.index]

    # Line utama
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=ts.values,
        mode="lines+markers",
        name="Kredit Aktual",
        line=dict(color="#1F4E79", width=2.5),
        marker=dict(size=4),
        fill="tozeroy",
        fillcolor="rgba(31,78,121,0.08)",
    ))

    # Moving average 6 bulan
    ma6 = pd.Series(ts.values).rolling(6).mean()
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=ma6.values,
        mode="lines",
        name="Moving Average (6 Bln)",
        line=dict(color="#F39C12", width=2, dash="dash"),
    ))

    fig.update_layout(
        title="📈 Tren Kredit Perbankan Historis",
        xaxis_title="Periode",
        yaxis_title=f"{kolom_nilai} (Triliun Rp)",
        template="plotly_white",
        height=420,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


# ── 5. GRAFIK TRAIN TEST SPLIT ────────────────────────────────
def grafik_train_test(train, test):
    """Grafik pembagian data training dan testing."""
    fig = go.Figure()

    # Training data
    fig.add_trace(go.Scatter(
        x=[str(p) for p in train.index],
        y=train.values,
        mode="lines",
        name="Training Data",
        line=dict(color="#2196F3", width=2),
        fill="tozeroy",
        fillcolor="rgba(33,150,243,0.08)",
    ))

    # Testing data
    fig.add_trace(go.Scatter(
        x=[str(p) for p in test.index],
        y=test.values,
        mode="lines",
        name="Testing Data",
        line=dict(color="#FF9800", width=2),
        fill="tozeroy",
        fillcolor="rgba(255,152,0,0.08)",
    ))

    fig.update_layout(
        title="📊 Pembagian Data Training vs Testing",
        xaxis_title="Periode",
        yaxis_title="Nilai Kredit (Triliun Rp)",
        template="plotly_white",
        height=350,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


# ── 6. GRAFIK FORECAST VS AKTUAL ─────────────────────────────
def grafik_forecast_vs_aktual(test, hasil_arima,
                               hasil_sarima, hasil_rf):
    """Grafik perbandingan prediksi ketiga model vs aktual."""
    fig = go.Figure()
    x_test = [str(p) for p in test.index]

    # Data aktual
    fig.add_trace(go.Scatter(
        x=x_test, y=test.values,
        mode="lines+markers",
        name="Aktual",
        line=dict(color="#1F4E79", width=3),
        marker=dict(size=7, symbol="circle"),
    ))

    # ARIMA
    if hasil_arima and hasil_arima.get("pred_test") is not None:
        fig.add_trace(go.Scatter(
            x=x_test,
            y=hasil_arima["pred_test"].values,
            mode="lines+markers",
            name="ARIMA",
            line=dict(color="#2196F3", width=2, dash="dash"),
            marker=dict(size=5, symbol="triangle-up"),
        ))

    # SARIMA
    if hasil_sarima and hasil_sarima.get("pred_test") is not None:
        fig.add_trace(go.Scatter(
            x=x_test,
            y=hasil_sarima["pred_test"].values,
            mode="lines+markers",
            name="SARIMA",
            line=dict(color="#4CAF50", width=2, dash="dot"),
            marker=dict(size=5, symbol="square"),
        ))

    # Random Forest
    if hasil_rf and hasil_rf.get("pred_test") is not None:
        fig.add_trace(go.Scatter(
            x=x_test,
            y=hasil_rf["pred_test"].values,
            mode="lines+markers",
            name="Random Forest",
            line=dict(color="#FF9800", width=2, dash="dashdot"),
            marker=dict(size=5, symbol="diamond"),
        ))

    fig.update_layout(
        title="🔍 Perbandingan Prediksi vs Aktual (Testing Period)",
        xaxis_title="Periode",
        yaxis_title="Nilai Kredit (Triliun Rp)",
        template="plotly_white",
        height=420,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


# ── 7. GRAFIK FORECAST MASA DEPAN ────────────────────────────
def grafik_forecast_depan(ts, hasil_arima,
                           hasil_sarima, hasil_rf,
                           model_terbaik):
    """Grafik forecast masa depan semua model."""
    fig = go.Figure()

    # Data historis (12 bulan terakhir)
    ts_tail  = ts.tail(24)
    x_hist   = [str(p) for p in ts_tail.index]

    fig.add_trace(go.Scatter(
        x=x_hist, y=ts_tail.values,
        mode="lines",
        name="Data Historis",
        line=dict(color="#1F4E79", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(31,78,121,0.06)",
    ))

    # Garis pemisah historis vs forecast
    last_hist = str(ts.index[-1])

    # Forecast ARIMA
    if hasil_arima and hasil_arima.get("forecast_future") is not None:
        ff = hasil_arima["forecast_future"]
        fig.add_trace(go.Scatter(
            x=[str(p) for p in ff.index],
            y=ff.values,
            mode="lines+markers",
            name="Forecast ARIMA",
            line=dict(color="#2196F3", width=2, dash="dash"),
            marker=dict(size=7),
        ))

    # Forecast SARIMA
    if hasil_sarima and hasil_sarima.get("forecast_future") is not None:
        ff = hasil_sarima["forecast_future"]
        fig.add_trace(go.Scatter(
            x=[str(p) for p in ff.index],
            y=ff.values,
            mode="lines+markers",
            name="Forecast SARIMA",
            line=dict(color="#4CAF50", width=2, dash="dot"),
            marker=dict(size=7),
        ))

    # Forecast Random Forest
    if hasil_rf and hasil_rf.get("forecast_future") is not None:
        ff = hasil_rf["forecast_future"]
        fig.add_trace(go.Scatter(
            x=[str(p) for p in ff.index],
            y=ff.values,
            mode="lines+markers",
            name="Forecast RF",
            line=dict(color="#FF9800", width=2, dash="dashdot"),
            marker=dict(size=7),
        ))

    # Garis vertikal pemisah
    fig.add_vline(
        x=last_hist,
        line_width=2,
        line_dash="solid",
        line_color="red",
        annotation_text="Mulai Forecast",
        annotation_position="top right",
    )

    fig.update_layout(
        title="🔮 Forecast Kredit Perbankan ke Depan",
        xaxis_title="Periode",
        yaxis_title="Nilai Kredit (Triliun Rp)",
        template="plotly_white",
        height=450,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


# ── 8. GRAFIK PERBANDINGAN METRIK ─────────────────────────────
def grafik_metrik(df_eval):
    """Grafik bar perbandingan MAE, RMSE, MAPE."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("MAE", "RMSE", "MAPE (%)"),
    )

    warna = {"ARIMA": "#2196F3",
             "SARIMA": "#4CAF50",
             "Random Forest": "#FF9800"}

    colors = [warna.get(m, "#888") for m in df_eval["model"]]

    # MAE
    fig.add_trace(go.Bar(
        x=df_eval["model"], y=df_eval["MAE"],
        marker_color=colors, showlegend=False,
        text=df_eval["MAE"].round(2),
        textposition="outside",
    ), row=1, col=1)

    # RMSE
    fig.add_trace(go.Bar(
        x=df_eval["model"], y=df_eval["RMSE"],
        marker_color=colors, showlegend=False,
        text=df_eval["RMSE"].round(2),
        textposition="outside",
    ), row=1, col=2)

    # MAPE
    fig.add_trace(go.Bar(
        x=df_eval["model"], y=df_eval["MAPE"],
        marker_color=colors, showlegend=False,
        text=df_eval["MAPE"].round(2),
        textposition="outside",
    ), row=1, col=3)

    fig.update_layout(
        title="📊 Perbandingan Metrik Evaluasi Model",
        template="plotly_white",
        height=380,
        margin=dict(l=0, r=0, t=80, b=0),
    )
    return fig


# ── 9. GRAFIK FEATURE IMPORTANCE ─────────────────────────────
def grafik_feature_importance(feature_importance):
    """Grafik feature importance Random Forest."""
    top10 = feature_importance.head(10)

    fig = go.Figure(go.Bar(
        x=top10["importance"] * 100,
        y=top10["fitur"],
        orientation="h",
        marker_color="#FF9800",
        text=(top10["importance"] * 100).round(1),
        texttemplate="%{text}%",
        textposition="outside",
    ))

    fig.update_layout(
        title="🌲 Feature Importance — Random Forest",
        xaxis_title="Importance (%)",
        yaxis_title="Fitur",
        template="plotly_white",
        height=380,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


# ── 10. TABEL FORECAST ───────────────────────────────────────
def tampil_tabel_forecast(hasil_arima, hasil_sarima,
                           hasil_rf, model_terbaik):
    """Menampilkan tabel forecast semua model."""
    data = {}

    if hasil_arima and hasil_arima.get("forecast_future") is not None:
        ff = hasil_arima["forecast_future"]
        data["Periode"]  = [str(p) for p in ff.index]
        data["ARIMA"]    = ff.values.round(2)

    if hasil_sarima and hasil_sarima.get("forecast_future") is not None:
        data["SARIMA"]   = hasil_sarima["forecast_future"].values.round(2)

    if hasil_rf and hasil_rf.get("forecast_future") is not None:
        data["Random Forest"] = hasil_rf["forecast_future"].values.round(2)

    if not data:
        st.warning("Tidak ada data forecast untuk ditampilkan.")
        return

    df = pd.DataFrame(data)

    # Highlight kolom model terbaik
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


# ── 11. TAMPIL INSIGHT ───────────────────────────────────────
def tampil_insight(insights):
    """Menampilkan insight otomatis."""
    for insight in insights:
        if insight.startswith("✅") or insight.startswith("📈"):
            st.success(insight)
        elif insight.startswith("⚠️") or insight.startswith("📉"):
            st.warning(insight)
        elif insight.startswith("🔮") or insight.startswith("💡"):
            st.info(insight)
        else:
            st.info(insight)

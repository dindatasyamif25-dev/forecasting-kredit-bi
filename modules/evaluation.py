
# ============================================================
# evaluation.py — Evaluasi dan Perbandingan Model
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ── 1. HITUNG METRIK ─────────────────────────────────────────
def hitung_mae(aktual, prediksi):
    """Mean Absolute Error — rata-rata selisih absolut."""
    return float(np.mean(np.abs(aktual - prediksi)))

def hitung_rmse(aktual, prediksi):
    """Root Mean Square Error — akar rata-rata kuadrat selisih."""
    return float(np.sqrt(np.mean((aktual - prediksi) ** 2)))

def hitung_mape(aktual, prediksi):
    """Mean Absolute Percentage Error — error dalam persen."""
    aktual   = np.array(aktual)
    prediksi = np.array(prediksi)
    # Hindari pembagian dengan nol
    mask = aktual != 0
    return float(np.mean(np.abs(
        (aktual[mask] - prediksi[mask]) / aktual[mask]
    )) * 100)


# ── 2. KATEGORI MAPE ─────────────────────────────────────────
def kategori_mape(mape):
    """
    Menentukan kategori akurasi berdasarkan nilai MAPE.
    """
    if mape < 10:
        return "Sangat Baik"
    elif mape < 20:
        return "Baik"
    elif mape < 50:
        return "Wajar"
    else:
        return "Tidak Akurat"


# ── 3. EVALUASI SATU MODEL ───────────────────────────────────
def evaluasi_model(nama_model, aktual, prediksi):
    """
    Menghitung semua metrik untuk satu model.
    """
    # Samakan panjang data
    min_len  = min(len(aktual), len(prediksi))
    aktual   = np.array(aktual[:min_len])
    prediksi = np.array(prediksi[:min_len])

    mae  = hitung_mae(aktual, prediksi)
    rmse = hitung_rmse(aktual, prediksi)
    mape = hitung_mape(aktual, prediksi)
    kat  = kategori_mape(mape)

    return {
        "model"    : nama_model,
        "MAE"      : round(mae, 4),
        "RMSE"     : round(rmse, 4),
        "MAPE"     : round(mape, 4),
        "Kategori" : kat,
    }


# ── 4. BANDINGKAN SEMUA MODEL ────────────────────────────────
def bandingkan_model(aktual, hasil_arima, hasil_sarima, hasil_rf):
    """
    Membandingkan ketiga model dan menentukan model terbaik.
    """
    hasil_list = []

    # Evaluasi ARIMA
    if hasil_arima and hasil_arima.get("pred_test") is not None:
        ev = evaluasi_model(
            "ARIMA",
            aktual.values,
            hasil_arima["pred_test"].values
        )
        hasil_list.append(ev)

    # Evaluasi SARIMA
    if hasil_sarima and hasil_sarima.get("pred_test") is not None:
        ev = evaluasi_model(
            "SARIMA",
            aktual.values,
            hasil_sarima["pred_test"].values
        )
        hasil_list.append(ev)

    # Evaluasi Random Forest
    if hasil_rf and hasil_rf.get("pred_test") is not None:
        ev = evaluasi_model(
            "Random Forest",
            aktual.values,
            hasil_rf["pred_test"].values
        )
        hasil_list.append(ev)

    # Buat DataFrame perbandingan
    df_eval = pd.DataFrame(hasil_list)

    # Tentukan model terbaik berdasarkan MAPE terkecil
    idx_terbaik  = df_eval["MAPE"].idxmin()
    model_terbaik = df_eval.loc[idx_terbaik, "model"]
    mape_terbaik  = df_eval.loc[idx_terbaik, "MAPE"]
    kat_terbaik   = df_eval.loc[idx_terbaik, "Kategori"]

    # Tambah kolom ranking
    df_eval["Rank"] = df_eval["MAPE"].rank().astype(int)
    df_eval = df_eval.sort_values("Rank").reset_index(drop=True)

    return df_eval, model_terbaik, mape_terbaik, kat_terbaik


# ── 5. GENERATE INSIGHT OTOMATIS ─────────────────────────────
def generate_insight(stats, model_terbaik, mape_terbaik,
                     kat_terbaik, forecast_terbaik):
    """
    Membuat teks insight otomatis berdasarkan hasil analisis.
    """
    insights = []

    # Insight 1: Tren data
    tren   = stats["tren"]
    growth = stats["growth_pct"]
    if tren == "naik":
        insights.append(
            f"📈 **Tren Kredit Naik** — Data historis menunjukkan "
            f"pertumbuhan kredit sebesar **{growth}%** selama periode analisis. "
            f"Ini mengindikasikan ekspansi kredit yang sehat."
        )
    elif tren == "turun":
        insights.append(
            f"📉 **Tren Kredit Turun** — Data historis menunjukkan "
            f"penurunan kredit sebesar **{abs(growth)}%** selama periode analisis. "
            f"Perlu strategi untuk mendorong penyaluran kredit."
        )
    else:
        insights.append(
            f"➡️ **Tren Kredit Stabil** — Data historis menunjukkan "
            f"pola kredit yang relatif stabil dengan variasi "
            f"**{stats['cv_pct']}%**."
        )

    # Insight 2: Model terbaik
    insights.append(
        f"✅ **Model Terbaik: {model_terbaik}** — Dipilih berdasarkan "
        f"MAPE terkecil sebesar **{mape_terbaik}%** "
        f"(kategori: {kat_terbaik}). "
        f"Model ini direkomendasikan untuk forecasting kredit ke depan."
    )

    # Insight 3: Proyeksi ke depan
    if forecast_terbaik is not None:
        nilai_awal  = round(float(forecast_terbaik.iloc[0]), 2)
        nilai_akhir = round(float(forecast_terbaik.iloc[-1]), 2)
        if nilai_akhir > nilai_awal:
            insights.append(
                f"🔮 **Proyeksi Meningkat** — Forecast {len(forecast_terbaik)} "
                f"bulan ke depan menunjukkan kecenderungan **peningkatan** "
                f"dari {nilai_awal:,.2f} menjadi {nilai_akhir:,.2f} "
                f"(Triliun Rp). Siapkan alokasi dana yang memadai."
            )
        else:
            insights.append(
                f"🔮 **Proyeksi Menurun** — Forecast {len(forecast_terbaik)} "
                f"bulan ke depan menunjukkan kecenderungan **penurunan** "
                f"dari {nilai_awal:,.2f} menjadi {nilai_akhir:,.2f} "
                f"(Triliun Rp). Perlu strategi akuisisi nasabah baru."
            )

    # Insight 4: Rekomendasi
    insights.append(
        f"💡 **Rekomendasi** — Gunakan model **{model_terbaik}** "
        f"sebagai dasar perencanaan kredit. Perbarui model setiap "
        f"bulan dengan data terbaru untuk menjaga akurasi forecasting."
    )

    return insights


# ── 6. FUNGSI UTAMA ──────────────────────────────────────────
def jalankan_evaluasi(test, hasil_arima, hasil_sarima,
                      hasil_rf, stats):
    """
    Fungsi utama evaluasi — menjalankan semua tahap:
    1. Bandingkan ketiga model
    2. Tentukan model terbaik
    3. Generate insight otomatis

    Return:
        hasil → dict semua output evaluasi
        error → pesan error jika gagal
    """
    try:
        # Step 1: Bandingkan model
        df_eval, model_terbaik, mape_terbaik, kat_terbaik = bandingkan_model(
            test, hasil_arima, hasil_sarima, hasil_rf
        )

        # Step 2: Ambil forecast dari model terbaik
        if model_terbaik == "ARIMA":
            forecast_terbaik = hasil_arima["forecast_future"]
        elif model_terbaik == "SARIMA":
            forecast_terbaik = hasil_sarima["forecast_future"]
        else:
            forecast_terbaik = hasil_rf["forecast_future"]

        # Step 3: Generate insight
        insights = generate_insight(
            stats, model_terbaik, mape_terbaik,
            kat_terbaik, forecast_terbaik
        )

        hasil = {
            "df_eval"         : df_eval,
            "model_terbaik"   : model_terbaik,
            "mape_terbaik"    : mape_terbaik,
            "kat_terbaik"     : kat_terbaik,
            "forecast_terbaik": forecast_terbaik,
            "insights"        : insights,
        }

        return hasil, None

    except Exception as e:
        return None, str(e)

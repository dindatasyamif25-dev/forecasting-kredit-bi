
# ============================================================
# model_arima.py — Model ARIMA untuk Forecasting
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA


# ── 1. CARI PARAMETER TERBAIK OTOMATIS ──────────────────────
def cari_parameter_arima(train):
    """
    Mencari parameter (p, d, q) terbaik secara otomatis
    menggunakan auto_arima berdasarkan nilai AIC terkecil.
    """
    try:
        model_auto = auto_arima(
            train,
            start_p=0, max_p=3,
            start_q=0, max_q=3,
            d=None,
            seasonal=False,
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )
        order = model_auto.order
        aic   = round(model_auto.aic(), 2)
        return order, aic

    except Exception:
        # Jika auto gagal, pakai default (1,1,1)
        return (1, 1, 1), None


# ── 2. TRAINING MODEL ────────────────────────────────────────
def training_arima(train, order=(1, 1, 1)):
    """
    Melatih model ARIMA dengan order (p, d, q).
    """
    try:
        model  = ARIMA(train, order=order)
        result = model.fit()
        return result, None
    except Exception as e:
        return None, str(e)


# ── 3. FORECAST PADA DATA TEST ───────────────────────────────
def forecast_test(result, n_test):
    """
    Membuat prediksi sejumlah n_test periode
    untuk dibandingkan dengan data aktual.
    """
    try:
        pred = result.forecast(steps=n_test)
        return pred, None
    except Exception as e:
        return None, str(e)


# ── 4. FORECAST KE DEPAN ────────────────────────────────────
def forecast_future(train, test, order, n_periods=6):
    """
    Melatih ulang model dengan SEMUA data (train+test)
    lalu forecast n_periods ke depan.
    """
    try:
        # Gabung train dan test untuk forecast masa depan
        ts_full = pd.concat([train, test])
        model   = ARIMA(ts_full, order=order)
        result  = model.fit()

        # Forecast ke depan
        forecast    = result.forecast(steps=n_periods)
        forecast_ci = result.get_forecast(steps=n_periods).conf_int()

        return forecast, forecast_ci, None
    except Exception as e:
        return None, None, str(e)


# ── 5. FUNGSI UTAMA ──────────────────────────────────────────
def jalankan_arima(train, test, n_periods=6):
    """
    Fungsi utama — menjalankan semua tahap ARIMA:
    1. Cari parameter terbaik
    2. Training
    3. Forecast test
    4. Forecast masa depan

    Return:
        hasil → dict semua output
        error → pesan error jika gagal
    """

    # Step 1: Cari parameter terbaik
    order, aic = cari_parameter_arima(train)

    # Step 2: Training
    result, error = training_arima(train, order)
    if error:
        return None, f"ARIMA Training Error: {error}"

    # Step 3: Forecast test
    pred_test, error = forecast_test(result, len(test))
    if error:
        return None, f"ARIMA Forecast Error: {error}"

    # Pastikan index pred_test sama dengan test
    pred_test = pd.Series(
        pred_test.values,
        index=test.index,
        name="ARIMA"
    )

    # Step 4: Forecast masa depan
    forecast_future_val, forecast_ci, error = forecast_future(
        train, test, order, n_periods
    )
    if error:
        return None, f"ARIMA Future Forecast Error: {error}"

    # Buat index periode masa depan
    last_period  = test.index[-1]
    future_index = pd.period_range(
        start=last_period + 1,
        periods=n_periods,
        freq="M"
    )
    forecast_future_series = pd.Series(
        forecast_future_val.values,
        index=future_index,
        name="ARIMA_Future"
    )

    hasil = {
        "model"           : "ARIMA",
        "order"           : order,
        "aic"             : aic,
        "result"          : result,
        "pred_test"       : pred_test,
        "forecast_future" : forecast_future_series,
        "forecast_ci"     : forecast_ci,
        "n_periods"       : n_periods,
    }

    return hasil, None


# ============================================================
# model_sarima.py — Model SARIMA untuk Forecasting
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ── 1. CARI PARAMETER TERBAIK OTOMATIS ──────────────────────
def cari_parameter_sarima(train, seasonal_period=12):
    """
    Mencari parameter (p,d,q)(P,D,Q,s) terbaik otomatis
    menggunakan auto_arima dengan seasonal=True.
    """
    try:
        model_auto = auto_arima(
            train,
            start_p=0, max_p=2,
            start_q=0, max_q=2,
            d=None,
            start_P=0, max_P=1,
            start_Q=0, max_Q=1,
            D=None,
            seasonal=True,
            m=seasonal_period,
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )
        order          = model_auto.order
        seasonal_order = model_auto.seasonal_order
        aic            = round(model_auto.aic(), 2)
        return order, seasonal_order, aic

    except Exception:
        # Jika auto gagal, pakai default
        return (1,1,1), (1,1,1,12), None


# ── 2. TRAINING MODEL ────────────────────────────────────────
def training_sarima(train, order, seasonal_order):
    """
    Melatih model SARIMA dengan parameter yang diberikan.
    """
    try:
        model  = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)
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
def forecast_future(train, test, order, seasonal_order, n_periods=6):
    """
    Melatih ulang dengan SEMUA data lalu forecast ke depan.
    """
    try:
        ts_full = pd.concat([train, test])
        model   = SARIMAX(
            ts_full,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result      = model.fit(disp=False)
        forecast    = result.forecast(steps=n_periods)
        forecast_ci = result.get_forecast(steps=n_periods).conf_int()

        return forecast, forecast_ci, None
    except Exception as e:
        return None, None, str(e)


# ── 5. FUNGSI UTAMA ──────────────────────────────────────────
def jalankan_sarima(train, test, n_periods=6, seasonal_period=12):
    """
    Fungsi utama — menjalankan semua tahap SARIMA:
    1. Cari parameter terbaik
    2. Training
    3. Forecast test
    4. Forecast masa depan

    Return:
        hasil → dict semua output
        error → pesan error jika gagal
    """

    # Step 1: Cari parameter terbaik
    order, seasonal_order, aic = cari_parameter_sarima(
        train, seasonal_period
    )

    # Step 2: Training
    result, error = training_sarima(train, order, seasonal_order)
    if error:
        return None, f"SARIMA Training Error: {error}"

    # Step 3: Forecast test
    pred_test, error = forecast_test(result, len(test))
    if error:
        return None, f"SARIMA Forecast Error: {error}"

    # Samakan index dengan test
    pred_test = pd.Series(
        pred_test.values,
        index=test.index,
        name="SARIMA"
    )

    # Step 4: Forecast masa depan
    forecast_future_val, forecast_ci, error = forecast_future(
        train, test, order, seasonal_order, n_periods
    )
    if error:
        return None, f"SARIMA Future Forecast Error: {error}"

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
        name="SARIMA_Future"
    )

    hasil = {
        "model"           : "SARIMA",
        "order"           : order,
        "seasonal_order"  : seasonal_order,
        "aic"             : aic,
        "result"          : result,
        "pred_test"       : pred_test,
        "forecast_future" : forecast_future_series,
        "forecast_ci"     : forecast_ci,
        "n_periods"       : n_periods,
    }

    return hasil, None

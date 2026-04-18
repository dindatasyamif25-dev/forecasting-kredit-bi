
# ============================================================
# model_random_forest.py — Model Random Forest untuk Forecasting
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


# ── 1. BUAT LAG FEATURES ────────────────────────────────────
def buat_lag_features(ts, n_lags=12):
    """
    Mengubah data time series menjadi format supervised learning
    dengan menambahkan lag features (nilai masa lalu sebagai input).

    Contoh dengan n_lags=3:
    Input : [lag_1, lag_2, lag_3]
    Target: [nilai_sekarang]
    """
    df = pd.DataFrame({"nilai": ts.values})

    # Tambah lag features
    for i in range(1, n_lags + 1):
        df[f"lag_{i}"] = df["nilai"].shift(i)

    # Tambah fitur rolling statistics
    df["roll_mean_3"]  = df["nilai"].shift(1).rolling(3).mean()
    df["roll_mean_6"]  = df["nilai"].shift(1).rolling(6).mean()
    df["roll_std_3"]   = df["nilai"].shift(1).rolling(3).std()

    # Tambah fitur waktu dari index
    if hasattr(ts.index, "month"):
        df["bulan"]    = ts.index.month
        df["kuartal"]  = ts.index.quarter

    # Hapus baris yang masih ada NaN
    df = df.dropna()

    # Pisahkan fitur (X) dan target (y)
    X = df.drop("nilai", axis=1)
    y = df["nilai"]

    return X, y, df


# ── 2. TRAINING MODEL ────────────────────────────────────────
def training_rf(X_train, y_train,
                n_estimators=200,
                max_depth=10,
                min_samples_split=3,
                random_state=42):
    """
    Melatih model Random Forest Regressor.
    """
    try:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model, None
    except Exception as e:
        return None, str(e)


# ── 3. FORECAST PADA DATA TEST ───────────────────────────────
def forecast_test(model, X_test, test_index):
    """
    Membuat prediksi pada data test.
    """
    try:
        pred = model.predict(X_test)
        pred_series = pd.Series(
            pred,
            index=test_index,
            name="RF"
        )
        return pred_series, None
    except Exception as e:
        return None, str(e)


# ── 4. FORECAST KE DEPAN ─────────────────────────────────────
def forecast_future(model, ts_full, n_periods=6, n_lags=12):
    """
    Forecast n_periods ke depan secara rekursif.
    Setiap prediksi baru dipakai sebagai input prediksi berikutnya.
    """
    try:
        # Ambil nilai terakhir dari data penuh
        history = list(ts_full.values)
        forecasts = []

        for i in range(n_periods):
            # Buat fitur dari history terbaru
            lag_vals = history[-(n_lags):][::-1]

            # Rolling stats
            roll_mean_3 = np.mean(history[-3:])
            roll_mean_6 = np.mean(history[-6:])
            roll_std_3  = np.std(history[-3:])

            # Fitur waktu
            last_period = ts_full.index[-1]
            next_month  = (last_period + (i + 1)).month
            next_quarter= (next_month - 1) // 3 + 1

            # Gabung semua fitur
            features = lag_vals + [
                roll_mean_3, roll_mean_6, roll_std_3,
                next_month, next_quarter
            ]

            # Prediksi
            pred = model.predict([features])[0]
            forecasts.append(pred)
            history.append(pred)

        # Buat index masa depan
        last_period  = ts_full.index[-1]
        future_index = pd.period_range(
            start=last_period + 1,
            periods=n_periods,
            freq="M"
        )
        forecast_series = pd.Series(
            forecasts,
            index=future_index,
            name="RF_Future"
        )
        return forecast_series, None

    except Exception as e:
        return None, str(e)


# ── 5. FEATURE IMPORTANCE ────────────────────────────────────
def get_feature_importance(model, feature_names):
    """
    Mengambil nilai feature importance dari model RF.
    """
    importance = pd.DataFrame({
        "fitur"      : feature_names,
        "importance" : model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return importance


# ── 6. FUNGSI UTAMA ──────────────────────────────────────────
def jalankan_random_forest(train, test, n_periods=6, n_lags=12):
    """
    Fungsi utama — menjalankan semua tahap Random Forest:
    1. Buat lag features
    2. Training
    3. Forecast test
    4. Forecast masa depan
    5. Feature importance

    Return:
        hasil → dict semua output
        error → pesan error jika gagal
    """

    # Gabung train dan test untuk buat fitur lengkap
    ts_full = pd.concat([train, test])

    # Step 1: Buat lag features dari data penuh
    X_full, y_full, df_full = buat_lag_features(ts_full, n_lags)

    # Pisahkan train dan test berdasarkan ukuran
    n_train_rows = len(train) - n_lags
    X_train = X_full.iloc[:n_train_rows]
    y_train = y_full.iloc[:n_train_rows]
    X_test  = X_full.iloc[n_train_rows:]
    y_test  = y_full.iloc[n_train_rows:]

    if len(X_train) < 10:
        return None, "Data training terlalu sedikit untuk Random Forest"

    # Step 2: Training
    model, error = training_rf(X_train, y_train)
    if error:
        return None, f"RF Training Error: {error}"

    # Step 3: Forecast test
    pred_test, error = forecast_test(model, X_test, test.index)
    if error:
        return None, f"RF Forecast Error: {error}"

    # Step 4: Forecast masa depan
    forecast_future_val, error = forecast_future(
        model, ts_full, n_periods, n_lags
    )
    if error:
        return None, f"RF Future Forecast Error: {error}"

    # Step 5: Feature importance
    feature_importance = get_feature_importance(
        model, X_train.columns.tolist()
    )

    hasil = {
        "model"            : "Random Forest",
        "n_estimators"     : 200,
        "n_lags"           : n_lags,
        "model_obj"        : model,
        "pred_test"        : pred_test,
        "forecast_future"  : forecast_future_val,
        "feature_importance": feature_importance,
        "n_periods"        : n_periods,
    }

    return hasil, None

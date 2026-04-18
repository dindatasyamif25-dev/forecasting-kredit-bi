
# config.py — Konfigurasi Global Project

# Identitas Aplikasi
APP_TITLE    = "BI Forecasting Kredit Perbankan"
APP_SUBTITLE = "Perbandingan ARIMA · SARIMA · Random Forest"
APP_ICON     = "🏦"
APP_VERSION  = "1.0.0"

# Warna per Model
COLOR_ARIMA  = "#2196F3"
COLOR_SARIMA = "#4CAF50"
COLOR_RF     = "#FF9800"
COLOR_ACTUAL = "#1F4E79"

# Parameter Model Default
ARIMA_ORDER_DEFAULT     = (1, 1, 1)
SARIMA_ORDER_DEFAULT    = (1, 1, 1)
SARIMA_SEASONAL_DEFAULT = (1, 1, 1, 12)

RF_N_ESTIMATORS      = 200
RF_MAX_DEPTH         = 10
RF_MIN_SAMPLES_SPLIT = 3
RF_RANDOM_STATE      = 42
RF_N_LAGS            = 12

# Pembagian Data
TRAIN_RATIO     = 0.80
MIN_DATA_POINTS = 24

# Periode Forecast
FORECAST_PERIODS_DEFAULT = 6
FORECAST_PERIODS_MAX     = 24

# Interpretasi MAPE
MAPE_THRESHOLDS = {
    "Sangat Baik"  : (0,   10),
    "Baik"         : (10,  20),
    "Wajar"        : (20,  50),
    "Tidak Akurat" : (50, 9999),
}

# Konfigurasi Grafik
CHART_HEIGHT       = 450
CHART_HEIGHT_SMALL = 300
CHART_TEMPLATE     = "plotly_white"

import os
BASE_DIR   = "/content/forecasting_kredit_bi"
DATA_DIR   = BASE_DIR + "/data"
ASSETS_DIR = BASE_DIR + "/assets"


# ============================================================
# preprocessing.py — Persiapan Data Time Series
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ── 1. BACA FILE ─────────────────────────────────────────────
def load_data(uploaded_file):
    """
    Membaca file CSV atau Excel.
    Bisa menerima path file atau file upload dari Streamlit.
    """
    try:
        if hasattr(uploaded_file, "name"):
            nama = uploaded_file.name
        else:
            nama = str(uploaded_file)

        if nama.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif nama.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Format file harus CSV atau Excel (.xlsx/.xls)")

        return df, None

    except Exception as e:
        return None, str(e)


# ── 2. DETEKSI KOLOM OTOMATIS ────────────────────────────────
def deteksi_kolom(df):
    """
    Otomatis mendeteksi kolom tanggal dan kolom nilai/kredit.
    """
    DATE_KEYWORDS  = ["date","tanggal","bulan","periode",
                      "month","waktu","time","tahun","year"]
    VALUE_KEYWORDS = ["kredit","credit","loan","pinjaman",
                      "total","amount","jumlah","value",
                      "outstanding","nominal","revenue"]

    kolom_tanggal = None
    kolom_nilai   = None

    for col in df.columns:
        col_lower = col.lower()

        # Cari kolom tanggal
        if kolom_tanggal is None:
            if any(k in col_lower for k in DATE_KEYWORDS):
                kolom_tanggal = col
                continue
            # Coba deteksi dari isi kolom
            try:
                pd.to_datetime(df[col].head(5))
                kolom_tanggal = col
                continue
            except:
                pass

        # Cari kolom nilai
        if kolom_nilai is None:
            if any(k in col_lower for k in VALUE_KEYWORDS):
                if pd.api.types.is_numeric_dtype(df[col]):
                    kolom_nilai = col

    return kolom_tanggal, kolom_nilai


# ── 3. CLEANING DATA ─────────────────────────────────────────
def clean_data(df, kolom_tanggal, kolom_nilai):
    """
    Membersihkan data:
    - Konversi tipe kolom
    - Hapus baris kosong
    - Urutkan berdasarkan tanggal
    - Hapus duplikasi
    """
    df = df.copy()

    # Konversi kolom tanggal
    df[kolom_tanggal] = pd.to_datetime(df[kolom_tanggal], errors="coerce")

    # Konversi kolom nilai ke numerik
    df[kolom_nilai] = pd.to_numeric(df[kolom_nilai], errors="coerce")

    # Hapus baris yang tanggal atau nilainya kosong
    df = df.dropna(subset=[kolom_tanggal, kolom_nilai])

    # Hapus duplikasi tanggal (simpan yang pertama)
    df = df.drop_duplicates(subset=[kolom_tanggal], keep="first")

    # Urutkan dari tanggal terlama ke terbaru
    df = df.sort_values(kolom_tanggal).reset_index(drop=True)

    return df


# ── 4. BUAT TIME SERIES ──────────────────────────────────────
def buat_time_series(df, kolom_tanggal, kolom_nilai):
    """
    Mengubah DataFrame menjadi Series time series
    dengan index tanggal.
    """
    ts = df.set_index(kolom_tanggal)[kolom_nilai]
    ts.index = pd.DatetimeIndex(ts.index).to_period("M")
    ts = ts.asfreq("M")
    return ts


# ── 5. HANDLE MISSING VALUES ─────────────────────────────────
def handle_missing(ts):
    """
    Mengisi nilai yang kosong di tengah data
    menggunakan interpolasi linier.
    """
    jumlah_missing = ts.isna().sum()
    if jumlah_missing > 0:
        ts = ts.interpolate(method="linear")
    return ts, jumlah_missing


# ── 6. SPLIT TRAIN / TEST ────────────────────────────────────
def split_data(ts, train_ratio=0.80):
    """
    Membagi data menjadi training set dan testing set.
    Default: 80% training, 20% testing.
    """
    n        = len(ts)
    n_train  = int(n * train_ratio)
    n_test   = n - n_train

    train = ts.iloc[:n_train]
    test  = ts.iloc[n_train:]

    return train, test, n_train, n_test


# ── 7. STATISTIK RINGKAS ─────────────────────────────────────
def statistik_data(ts):
    """
    Menghitung statistik ringkas untuk ditampilkan
    di dashboard.
    """
    nilai = ts.values

    # Hitung pertumbuhan (growth) awal vs akhir
    growth = ((nilai[-1] - nilai[0]) / nilai[0]) * 100

    # Koefisien variasi (seberapa stabil data)
    cv = (np.std(nilai) / np.mean(nilai)) * 100

    # Tentukan tren
    if growth > 5:
        tren = "naik"
    elif growth < -5:
        tren = "turun"
    else:
        tren = "stabil"

    return {
        "jumlah_data"   : len(ts),
        "periode_awal"  : str(ts.index[0]),
        "periode_akhir" : str(ts.index[-1]),
        "nilai_min"     : round(float(nilai.min()), 2),
        "nilai_max"     : round(float(nilai.max()), 2),
        "nilai_rata"    : round(float(nilai.mean()), 2),
        "std"           : round(float(nilai.std()), 2),
        "growth_pct"    : round(growth, 2),
        "cv_pct"        : round(cv, 2),
        "tren"          : tren,
    }


# ── 8. FUNGSI UTAMA ──────────────────────────────────────────
def proses_data(uploaded_file, train_ratio=0.80):
    """
    Fungsi utama yang menjalankan semua tahap preprocessing
    secara berurutan dari awal sampai akhir.

    Return:
        hasil  → dict berisi semua output preprocessing
        error  → pesan error jika gagal (None jika berhasil)
    """

    # Step 1: Baca file
    df, error = load_data(uploaded_file)
    if error:
        return None, f"❌ Gagal membaca file: {error}"

    # Step 2: Deteksi kolom
    kolom_tanggal, kolom_nilai = deteksi_kolom(df)
    if kolom_tanggal is None:
        return None, "❌ Kolom tanggal tidak ditemukan. Pastikan ada kolom berisi tanggal."
    if kolom_nilai is None:
        return None, "❌ Kolom nilai tidak ditemukan. Pastikan ada kolom berisi angka kredit."

    # Step 3: Cleaning
    df = clean_data(df, kolom_tanggal, kolom_nilai)
    if len(df) < 24:
        return None, f"❌ Data terlalu sedikit ({len(df)} baris). Minimal 24 bulan."

    # Step 4: Buat time series
    ts = buat_time_series(df, kolom_tanggal, kolom_nilai)

    # Step 5: Handle missing values
    ts, jumlah_missing = handle_missing(ts)

    # Step 6: Split data
    train, test, n_train, n_test = split_data(ts, train_ratio)

    # Step 7: Statistik
    stats = statistik_data(ts)

    # Kumpulkan semua hasil
    hasil = {
        "df"             : df,
        "ts"             : ts,
        "train"          : train,
        "test"           : test,
        "n_train"        : n_train,
        "n_test"         : n_test,
        "kolom_tanggal"  : kolom_tanggal,
        "kolom_nilai"    : kolom_nilai,
        "jumlah_missing" : jumlah_missing,
        "stats"          : stats,
    }

    return hasil, None

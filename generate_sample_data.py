
import pandas as pd
import numpy as np
import os

def generate_kredit_data():
    np.random.seed(42)
    dates = pd.date_range(start="2015-01-01", end="2023-12-01", freq="MS")
    n = len(dates)

    trend    = np.linspace(3800, 6800, n)
    months   = np.array([d.month for d in dates])
    seasonal = (
        120 * np.sin(2 * np.pi * (months - 3) / 12) +
         40 * np.sin(4 * np.pi * (months - 1) / 12)
    )

    covid_effect = np.zeros(n)
    for i, d in enumerate(dates):
        if d.year == 2020:
            if d.month in [4, 5, 6]:
                covid_effect[i] = -300 * (1 - (d.month - 4) / 3)
            elif d.month in [7, 8, 9]:
                covid_effect[i] = -150

    noise        = np.random.normal(0, 45, n)
    kredit_total = trend + seasonal + covid_effect + noise

    kredit_modal_kerja = kredit_total * 0.445 + np.random.normal(0, 20, n)
    kredit_investasi   = kredit_total * 0.275 + np.random.normal(0, 15, n)
    kredit_konsumsi    = kredit_total * 0.280 + np.random.normal(0, 18, n)

    npl_rate = 2.8 + 0.5 * np.sin(2 * np.pi * months / 12) + np.random.normal(0, 0.3, n)
    npl_rate[covid_effect < -100] += 1.2
    npl_rate = np.clip(npl_rate, 1.5, 5.5)

    bi_rate  = np.where([d.year >= 2022 for d in dates], 5.5, 4.5) + np.random.normal(0, 0.2, n)
    bi_rate  = np.clip(bi_rate, 3.5, 6.5)
    dpk      = kredit_total * 1.18 + np.random.normal(0, 50, n)

    df = pd.DataFrame({
        "Tanggal"           : dates,
        "Total_Kredit"      : np.round(kredit_total, 2),
        "Kredit_Modal_Kerja": np.round(kredit_modal_kerja, 2),
        "Kredit_Investasi"  : np.round(kredit_investasi, 2),
        "Kredit_Konsumsi"   : np.round(kredit_konsumsi, 2),
        "NPL_Gross_Pct"     : np.round(npl_rate, 2),
        "BI_Rate_Pct"       : np.round(bi_rate, 2),
        "DPK"               : np.round(dpk, 2),
    })
    return df

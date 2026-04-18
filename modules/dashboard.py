import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def load_css():
    try:
        with open("assets/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>",
                        unsafe_allow_html=True)
    except:
        pass


def grafik_historis(ts, kolom_nilai):
    x_vals = [str(p) for p in ts.index]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=ts.values,
        mode="lines+markers",
        name="Kredit Aktual",
        line=dict(color="#1F4E79", width=2.5),
        marker=dict(size=4),
        fill="tozeroy",
        fillcolor="rgba(31,78,121,0.08)",
    ))
    ma6 = pd.Series(ts.values).rolling(6).mean()
    fig.add_trace(go.Scatter(
        x=x_vals, y=ma6.values,
        mode="lines",
        name="Moving Average (6 Bln)",
        line=dict(color="#F39C12", width=2, dash="dash"),
    ))
    fig.update_layout(
        title="Tren Kredit Perbankan Historis",
        xaxis_title="Periode",
        yaxis_title=f"{kolom_nilai} (Triliun Rp)",
        template="plotly_white",
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=1,
                    yanchor="bottom", xanchor="right"),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def grafik_train_test(train, test):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[str(p) for p in train.index],
        y=train.values,
        mode="lines",
        name="Training Data",
        line=dict(color="#2196F3", width=2),
        fill="tozeroy",
        fillcolor="rgba(33,150,243,0.08)",
    ))
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
        title="Pembagian Data Training vs Testing",
        xaxis_title="Periode",
        yaxis_title="Nilai Kredit (Triliun Rp)",
        template="plotly_white",
        height=350,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=1,
                    yanchor="bottom", xanchor="right"),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def grafik_forecast_vs_aktual(test, hasil_arima,
                               hasil_sarima, hasil_rf):
    fig = go.Figure()
    x_test = [str(p) for p in test.index]

    fig.add_trace(go.Scatter(
        x=x_test, y=test.values,
        mode="lines+markers",
        name="Aktual",
        line=dict(color="#1F4E79", width=3),
        marker=dict(size=7, symbol="circle"),
    ))
    if hasil_arima and hasil_arima.get("pred_test") is not None:
        fig.add_trace(go.Scatter(
            x=x_test,
            y=hasil_arima["pred_test"].values,
            mode="lines+markers",
            name="ARIMA",
            line=dict(color="#2196F3", width=2, dash="dash"),
            marker=dict(size=5, symbol="triangle-up"),
        ))
    if hasil_sarima and hasil_sarima.get("pred_test") is not None:
        fig.add_trace(go.Scatter(
            x=x_test,
            y=hasil_sarima["pred_test"].values,
            mode="lines+markers",
            name="SARIMA",
            line=dict(color="#4CAF50", width=2, dash="dot"),
            marker=dict(size=5, symbol="square"),
        ))
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
        title="Perbandingan Prediksi vs Aktual (Testing Period)",
        xaxis_title="Periode",
        yaxis_title="Nilai Kredit (Triliun Rp)",
        template="plotly_white",
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=1,
                    yanchor="bottom", xanchor="right"),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def grafik_forecast_depan(ts, hasil_arima,
                           hasil_sarima, hasil_rf,
                           model_terbaik):
    fig = go.Figure()
    ts_tail = ts.tail(24)
    x_hist = [str(p) for p in ts_tail.index]

    fig.add_trace(go.Scatter(
        x=x_hist,
        y=ts_tail.values,
        mode="lines",
        name="Data Historis",
        line=dict(color="#1F4E79", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(31,78,121,0.06)",
    ))

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

    fig.add_shape(
        type="line",
        x0=x_hist[-1],
        x1=x_hist[-1],
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="red", width=2, dash="dot"),
    )
    fig.add_annotation(
        x=x_hist[-1],
        y=1,
        xref="x",
        yref="paper",
        text="Mulai Forecast",
        showarrow=False,
        font=dict(color="red", size=12),
        xanchor="left",
        yanchor="bottom",
    )

    fig.update_layout(
        title="Forecast Kredit Perbankan ke Depan",
        xaxis_title="Periode",
        yaxis_title="Nilai Kredit (Triliun Rp)",
        template="plotly_white",
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=1,
                    yanchor="bottom", xanchor="right"),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def grafik_metrik(df_eval):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("MAE", "RMSE", "MAPE (%)"),
    )
    warna = {
        "ARIMA": "#2196F3",
        "SARIMA": "#4CAF50",
        "Random Forest": "#FF9800",
    }
    colors = [warna.get(m, "#888") for m in df_eval["model"]]

    fig.add_trace(go.Bar(
        x=df_eval["model"], y=df_eval["MAE"],
        marker_color=colors, showlegend=False,
        text=df_eval["MAE"].round(2),
        textposition="outside",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df_eval["model"], y=df_eval["RMSE"],
        marker_color=colors, showlegend=False,
        text=df_eval["RMSE"].round(2),
        textposition="outside",
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        x=df_eval["model"], y=df_eval["MAPE"],
        marker_color=colors, showlegend=False,
        text=df_eval["MAPE"].round(2),
        textposition="outside",
    ), row=1, col=3)

    fig.update_layout(
        title="Perbandingan Metrik Evaluasi Model",
        template="plotly_white",
        height=380,
        margin=dict(l=0, r=0, t=80, b=0),
    )
    return fig


def grafik_feature_importance(feature_importance):
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
        title="Feature Importance — Random Forest",
        xaxis_title="Importance (%)",
        yaxis_title="Fitur",
        template="plotly_white",
        height=380,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def tampil_tabel_forecast(hasil_arima, hasil_sarima,
                           hasil_rf, model_terbaik):
    data = {}
    if hasil_arima and hasil_arima.get("forecast_future") is not None:
        ff = hasil_arima["forecast_future"]
        data["Periode"] = [str(p) for p in ff.index]
        data["ARIMA"] = ff.values.round(2)
    if hasil_sarima and hasil_sarima.get("forecast_future") is not None:
        data["SARIMA"] = hasil_sarima["forecast_future"].values.round(2)
    if hasil_rf and hasil_rf.get("forecast_future") is not None:
        data["Random Forest"] = hasil_rf["forecast_future"].values.round(2)
    if not data:
        st.warning("Tidak ada data forecast.")
        return
    st.dataframe(
        pd.DataFrame(data),
        use_container_width=True,
        hide_index=True,
    )


def tampil_insight(insights):
    for insight in insights:
        if any(x in insight for x in ["✅", "📈", "➡️"]):
            st.success(insight)
        elif any(x in insight for x in ["⚠️", "📉"]):
            st.warning(insight)
        else:
            st.info(insight)

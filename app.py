import textwrap
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def html_block(text):
    """Dedent a triple-quoted HTML string before handing it to st.markdown."""
    return textwrap.dedent(text).strip()


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# DESIGN TOKENS
# ============================================================

INK_NAVY = "#14213D"
SOOT_GREY = "#5B6472"
HAZE_AMBER = "#E8A33D"
SKY_TEAL = "#2E7D8C"
FOG_BG = "#F3EFE7"
CARD_BG = "#FFFFFF"
BORDER = "#E4DFD3"

AQI_SCALE = [
    (50,  "#4CAF62", "Good"),
    (100, "#E8C547", "Moderate"),
    (150, "#F0913F", "Unhealthy (Sensitive)"),
    (200, "#E15554", "Unhealthy"),
    (300, "#8E5FC2", "Very Unhealthy"),
    (500, "#6E2C2C", "Hazardous"),
]


def aqi_category(value):
    """Return (label, color) for a given AQI value."""
    for threshold, color, label in AQI_SCALE:
        if value <= threshold:
            return label, color

    return AQI_SCALE[-1][2], AQI_SCALE[-1][1]


# ============================================================
# GLOBAL STYLE
# ============================================================

st.markdown(
    html_block(f"""
    <link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">

    <style>
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background-color: {FOG_BG};
        }}

        h1, h2, h3, .app-title {{
            font-family: 'Sora', sans-serif !important;
            color: {INK_NAVY} !important;
        }}

        .eyebrow {{
            font-family: 'Inter', sans-serif;
            font-size: 0.78rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: {SKY_TEAL};
            font-weight: 600;
            margin-bottom: 0.25rem;
        }}

        .app-title {{
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.1rem;
        }}

        .app-subtitle {{
            color: {SOOT_GREY};
            font-size: 1.02rem;
            margin-bottom: 1.5rem;
        }}

        .section-label {{
            font-family: 'Sora', sans-serif;
            font-weight: 600;
            font-size: 1.15rem;
            color: {INK_NAVY};
            margin: 1.6rem 0 0.7rem 0;
            padding-bottom: 0.4rem;
            border-bottom: 2px solid {BORDER};
        }}

        div[data-testid="stMetric"] {{
            background-color: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 14px 16px 10px 16px;
            box-shadow: 0 1px 3px rgba(20, 33, 61, 0.06);
        }}

        div[data-testid="stMetricLabel"] {{
            color: {SOOT_GREY};
        }}

        .info-card {{
            background-color: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 18px 18px;
            min-height: 105px;
            box-shadow: 0 1px 3px rgba(20, 33, 61, 0.06);
        }}

        .info-label {{
            color: {SOOT_GREY};
            font-size: 0.82rem;
            font-weight: 600;
            margin-bottom: 8px;
        }}

        .info-value {{
            color: {INK_NAVY};
            font-family: 'Sora', sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            line-height: 1.1;
        }}

        .info-unit {{
            color: {SOOT_GREY};
            font-size: 0.75rem;
            margin-top: 5px;
        }}

        .forecast-card {{
            background-color: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 18px 16px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(20, 33, 61, 0.06);
        }}

        .forecast-card .day-label {{
            font-size: 0.82rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: {SOOT_GREY};
            font-weight: 600;
            margin-bottom: 6px;
        }}

        .forecast-card .aqi-value {{
            font-family: 'Sora', sans-serif;
            font-size: 2.1rem;
            font-weight: 700;
            color: {INK_NAVY};
            line-height: 1.1;
        }}

        .aqi-pill {{
            display: inline-block;
            margin-top: 8px;
            padding: 3px 12px;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 600;
            color: white;
        }}

        .legend-row {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 6px;
        }}

        .legend-chip {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.76rem;
            color: {SOOT_GREY};
        }}

        .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }}
    </style>
    """),
    unsafe_allow_html=True
)


# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = "data/processed/karachi_aqi_features.csv"
PREDICTION_PATH = "results/latest_aqi_prediction.csv"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(ttl=300)
def load_data(path):
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def load_prediction(path):
    return pd.read_csv(path)


try:
    # Historical/current environmental data
    df = load_data(DATA_PATH)

    # Prediction generated by GitHub Actions
    prediction_df = load_prediction(PREDICTION_PATH)

except FileNotFoundError as e:
    st.error(f"Could not find a required file: {e}")
    st.stop()


# ============================================================
# VALIDATE DATA
# ============================================================

required_data_columns = [
    "date",
    "AQI",
    "PM2.5",
    "PM10",
    "Temperature",
    "Humidity"
]

missing_data_columns = [
    col for col in required_data_columns
    if col not in df.columns
]

if missing_data_columns:
    st.error(
        f"Missing columns in {DATA_PATH}: "
        f"{missing_data_columns}"
    )
    st.stop()


required_prediction_columns = [
    "AQI_t+1",
    "AQI_t+2",
    "AQI_t+3"
]

missing_prediction_columns = [
    col for col in required_prediction_columns
    if col not in prediction_df.columns
]

if missing_prediction_columns:
    st.error(
        f"Missing prediction columns in {PREDICTION_PATH}: "
        f"{missing_prediction_columns}"
    )
    st.stop()


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown(
        '<div class="eyebrow">Karachi · Live Forecast</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="app-title">🌫️ Karachi AQI Predictor</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="app-subtitle">'
        'AI-powered 3-day Air Quality Index forecasting, '
        'with a look at which environmental factors are driving '
        'each prediction.'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# LATEST DATA
# ============================================================

latest = df.iloc[-1]

current_aqi = float(latest["AQI"])
current_label, current_color = aqi_category(current_aqi)

latest_date = pd.to_datetime(latest["date"])

with header_right:
    st.markdown(
        html_block(f"""
        <div style="text-align:right; padding-top: 18px;">
            <div class="eyebrow" style="text-align:right;">As of</div>
            <div style="font-family:'Sora',sans-serif;
                        font-weight:600;
                        color:{INK_NAVY};">
                {latest_date.strftime('%b %d, %Y')}
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )


# ============================================================
# GITHUB ACTIONS PREDICTION
# ============================================================

latest_prediction = prediction_df.iloc[-1]

aqi_1 = float(latest_prediction["AQI_t+1"])
aqi_2 = float(latest_prediction["AQI_t+2"])
aqi_3 = float(latest_prediction["AQI_t+3"])


# ============================================================
# TODAY'S AQI — GAUGE
# ============================================================

st.markdown(
    '<div class="section-label">Today\'s Air Quality</div>',
    unsafe_allow_html=True
)

gauge_col, detail_col = st.columns([1.1, 1.4])

with gauge_col:

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=current_aqi,
            number={
                "font": {
                    "size": 42,
                    "color": INK_NAVY,
                    "family": "Sora"
                }
            },
            gauge={
                "axis": {
                    "range": [0, 400],
                    "tickcolor": SOOT_GREY
                },

                "bar": {
                    "color": INK_NAVY,
                    "thickness": 0.25
                },

                "steps": [
                    {"range": [0, 50], "color": "#4CAF62"},
                    {"range": [50, 100], "color": "#E8C547"},
                    {"range": [100, 150], "color": "#F0913F"},
                    {"range": [150, 200], "color": "#E15554"},
                    {"range": [200, 300], "color": "#8E5FC2"},
                    {"range": [300, 400], "color": "#6E2C2C"},
                ],

                "threshold": {
                    "line": {
                        "color": INK_NAVY,
                        "width": 3
                    },

                    "thickness": 0.9,
                    "value": current_aqi
                }
            }
        )
    )

    gauge.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"}
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )


with detail_col:

    st.markdown(
        html_block(f"""
        <div style="padding-top: 30px;">

            <span class="aqi-pill"
                  style="background-color:{current_color};">
                {current_label}
            </span>

            <p style="color:{SOOT_GREY};
                      margin-top:14px;
                      font-size:0.95rem;
                      line-height:1.5;">

                Current AQI reading of

                <b style="color:{INK_NAVY};">
                    {current_aqi:.0f}
                </b>

                classifies today's air as

                <b>{current_label.lower()}</b>.

                Scroll down for the 3-day outlook
                and the factors the model weighed most heavily.

            </p>

            <div class="legend-row">
        """),
        unsafe_allow_html=True
    )

    legend_html = "".join(
        f'<div class="legend-chip">'
        f'<span class="legend-dot" '
        f'style="background-color:{c};"></span>'
        f'{l}</div>'
        for _, c, l in AQI_SCALE
    )

    st.markdown(
        legend_html + "</div></div>",
        unsafe_allow_html=True
    )


# ============================================================
# 3-DAY FORECAST CARDS
# ============================================================

st.markdown(
    '<div class="section-label">3-Day Forecast</div>',
    unsafe_allow_html=True
)

forecast_days = [
    (
        "Today",
        current_aqi,
        latest_date
    ),
    (
        "Tomorrow",
        aqi_1,
        latest_date + pd.Timedelta(days=1)
    ),
    (
        "Day 2",
        aqi_2,
        latest_date + pd.Timedelta(days=2)
    ),
    (
        "Day 3",
        aqi_3,
        latest_date + pd.Timedelta(days=3)
    ),
]


cols = st.columns(4)

for col, (day_label, value, date) in zip(
    cols,
    forecast_days
):

    label, color = aqi_category(value)

    with col:

        st.markdown(
            html_block(f"""
            <div class="forecast-card">

                <div class="day-label">
                    {day_label} · {date.strftime('%b %d')}
                </div>

                <div class="aqi-value">
                    {value:.0f}
                </div>

                <span class="aqi-pill"
                      style="background-color:{color};">
                    {label}
                </span>

            </div>
            """),
            unsafe_allow_html=True
        )


# ============================================================
# FORECAST TREND CHART
# ============================================================

st.markdown(
    '<div class="section-label">Forecast Trend</div>',
    unsafe_allow_html=True
)

forecast_dates = [
    d for _, _, d in forecast_days
]

forecast_values = [
    v for _, v, _ in forecast_days
]

point_colors = [
    aqi_category(v)[1]
    for v in forecast_values
]


fig = go.Figure()


# AQI category bands
band_edges = [0] + [
    threshold
    for threshold, _, _ in AQI_SCALE
]

for i in range(len(AQI_SCALE)):

    fig.add_hrect(
        y0=band_edges[i],
        y1=band_edges[i + 1],
        fillcolor=AQI_SCALE[i][1],
        opacity=0.08,
        line_width=0
    )


fig.add_trace(
    go.Scatter(
        x=forecast_dates,
        y=forecast_values,

        mode="lines+markers+text",

        line=dict(
            color=INK_NAVY,
            width=3
        ),

        marker=dict(
            size=14,
            color=point_colors,
            line=dict(
                color=INK_NAVY,
                width=1.5
            )
        ),

        text=[
            f"{v:.0f}"
            for v in forecast_values
        ],

        textposition="top center",

        textfont=dict(
            color=INK_NAVY,
            size=13,
            family="Sora"
        ),

        fill="tozeroy",

        fillcolor="rgba(20, 33, 61, 0.05)",

        name="AQI",

        hovertemplate=(
            "%{x|%b %d}"
            "<br>AQI: %{y:.0f}"
            "<extra></extra>"
        )
    )
)


fig.update_layout(
    height=380,

    margin=dict(
        l=10,
        r=10,
        t=10,
        b=10
    ),

    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",

    font={
        "family": "Inter",
        "color": SOOT_GREY
    },

    xaxis={
        "showgrid": False,
        "title": None
    },

    yaxis={
        "title": "AQI",

        "range": [
            0,
            max(
                300,
                max(forecast_values) * 1.2
            )
        ],

        "gridcolor": BORDER
    },

    showlegend=False
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# CURRENT CONDITIONS
# ============================================================

st.markdown(
    '<div class="section-label">Current Conditions</div>',
    unsafe_allow_html=True
)

conditions = [
    ("PM2.5", latest["PM2.5"], "μg/m³"),
    ("PM10", latest["PM10"], "μg/m³"),
    ("Temperature", latest["Temperature"], "°C"),
    ("Humidity", latest["Humidity"], "%"),
]


cond_cols = st.columns(4)


for col, (label, value, unit) in zip(
    cond_cols,
    conditions
):

    with col:

        card_html = (
            f'<div class="info-card">'
            f'<div class="info-label">{label}</div>'
            f'<div class="info-value">{float(value):.1f}</div>'
            f'<div class="info-unit">{unit}</div>'
            f'</div>'
        )

        st.markdown(
            card_html,
            unsafe_allow_html=True
        )


# ============================================================
# SHAP — AI EXPLAINABILITY
# ============================================================

st.markdown(
    '<div class="section-label">🧠 AI Explainability</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<p style="color:{SOOT_GREY}; margin-top:-8px;">'
    "Which environmental factors have the greatest influence "
    "on tomorrow's AQI prediction?</p>",
    unsafe_allow_html=True
)


importance_path = "outputs/shap/feature_importance.csv"


try:

    importance = pd.read_csv(
        importance_path
    )

    top_importance = (
        importance
        .head(10)
        .sort_values(
            "mean_abs_shap",
            ascending=True
        )
    )


    shap_bar = go.Figure(
        go.Bar(
            x=top_importance["mean_abs_shap"],
            y=top_importance["feature"],
            orientation="h",

            marker=dict(
                color=top_importance["mean_abs_shap"],
                colorscale=[
                    [0, SKY_TEAL],
                    [1, HAZE_AMBER]
                ],
            ),

            hovertemplate=(
                "%{y}: %{x:.3f}"
                "<extra></extra>"
            )
        )
    )


    shap_bar.update_layout(
        height=380,

        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        ),

        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",

        font={
            "family": "Inter",
            "color": SOOT_GREY
        },

        xaxis={
            "title": "Mean |SHAP value|",
            "gridcolor": BORDER
        },

        yaxis={
            "title": None
        }
    )


    st.plotly_chart(
        shap_bar,
        use_container_width=True
    )


except FileNotFoundError:

    st.info(
        "Run the SHAP analysis script to generate "
        "feature importance data."
    )


# ============================================================
# SHAP IMAGE
# ============================================================

shap_image_path = (
    "outputs/shap/shap_summary_AQI_t+1.png"
)

try:

    st.image(
        shap_image_path,
        caption=(
            "Feature impact distribution for "
            "tomorrow's AQI prediction"
        )
    )

except Exception:
    pass


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    html_block(f"""
    <div style="text-align:center;
                margin-top: 2.5rem;
                padding-top: 1.2rem;
                border-top: 1px solid {BORDER};
                color:{SOOT_GREY};
                font-size:0.85rem;">

        Karachi AQI Predictor ·
        AI-powered air quality forecasting

    </div>
    """),
    unsafe_allow_html=True
)
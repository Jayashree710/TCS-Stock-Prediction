
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="TCS | Stock Prediction AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LIQUID AURORA THEME
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --pink: #ff4ecd;
    --purple: #9b5cff;
    --orange: #ff9f43;
    --green: #35e6a0;
    --cyan: #42e8ff;
}

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(
        -45deg,
        #170e2e,
        #29113d,
        #102e35,
        #39203b,
        #172c29,
        #301b31
    );
    background-size: 400% 400%;
    animation: aurora 18s ease infinite;
    color: white;
}

@keyframes aurora {
    0% { background-position: 0% 50%; }
    25% { background-position: 100% 50%; }
    50% { background-position: 100% 100%; }
    75% { background-position: 0% 100%; }
    100% { background-position: 0% 50%; }
}

.stApp::before {
    content: "";
    position: fixed;
    width: 450px;
    height: 450px;
    top: 5%;
    left: 10%;
    background: radial-gradient(circle, #ff4ecd55, transparent 70%);
    filter: blur(65px);
    animation: floatOne 12s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}

.stApp::after {
    content: "";
    position: fixed;
    width: 500px;
    height: 500px;
    bottom: 0;
    right: 5%;
    background: radial-gradient(circle, #35e6a044, transparent 70%);
    filter: blur(75px);
    animation: floatTwo 15s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}

@keyframes floatOne {
    from { transform: translate(0, 0); }
    to { transform: translate(180px, 100px); }
}

@keyframes floatTwo {
    from { transform: translate(0, 0); }
    to { transform: translate(-150px, -100px); }
}

.main .block-container {
    padding-top: 2rem;
    position: relative;
    z-index: 1;
}

section[data-testid="stSidebar"] {
    background: rgba(20, 15, 38, 0.88);
    backdrop-filter: blur(25px);
    border-right: 1px solid #ffffff20;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

section[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.055);
    padding: 12px 15px;
    border-radius: 12px;
    margin-bottom: 8px;
    transition: 0.3s ease;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    background: linear-gradient(90deg, #ff4ecd55, #9b5cff55);
    transform: translateX(5px);
}

h1, h2, h3 {
    color: white !important;
    font-weight: 800 !important;
}

h1 {
    background: linear-gradient(90deg, #ff80df, #b99aff, #66f2dc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

p, label, span {
    color: #eeeeff;
}

div[data-testid="stMetric"],
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.075);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 18px;
    box-shadow: 0 8px 32px #00000020;
    transition: all 0.35s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-7px);
    border-color: #ff4ecd99;
    box-shadow: 0 12px 35px #ff4ecd25;
}

[data-testid="stMetricValue"] {
    color: white !important;
    font-weight: 800;
}

[data-testid="stMetricLabel"] {
    color: #e5c9ff !important;
}

.stButton button,
.stDownloadButton button {
    background: linear-gradient(110deg, #ff4ecd, #9b5cff, #42e8ff);
    color: white !important;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    padding: 10px 22px;
    transition: 0.3s ease;
}

.stButton button:hover,
.stDownloadButton button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 8px 25px #ff4ecd55;
}

input, textarea {
    background: rgba(255,255,255,0.09) !important;
    color: white !important;
    border-radius: 10px !important;
}

div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.09);
    border: 1px solid #ffffff30;
    border-radius: 10px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #ffffff25;
    border-radius: 15px;
    overflow: hidden;
}

button[data-baseweb="tab"] {
    color: white !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 3px solid #ff4ecd !important;
}

section.main div[data-testid="stPlotlyChart"] {
    background: rgba(255,255,255,0.045);
    border: 1px solid #ffffff18;
    border-radius: 18px;
    padding: 8px;
}

hr {
    border-color: #ffffff25;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

CSV_FILE = BASE_DIR / "TCS.csv"
MODEL_FILE = BASE_DIR / "TCS_Stock_Model.pkl"


# =========================================================
# LOAD STOCK DATA
# =========================================================

@st.cache_data
def load_data():

    if not CSV_FILE.exists():
        return None

    # Try Yahoo Finance CSV format
    try:
        df = pd.read_csv(
            CSV_FILE,
            skiprows=3,
            header=None
        )

        if df.shape[1] >= 7 and not df.empty:

            df = df.iloc[:, :7]

            df.columns = [
                "Date",
                "Adj Close",
                "Close",
                "High",
                "Low",
                "Open",
                "Volume"
            ]

            if pd.to_datetime(
                df["Date"],
                errors="coerce"
            ).notna().any():

                return df

    except Exception:
        pass

    # Try normal CSV format
    try:
        df = pd.read_csv(CSV_FILE)

        df.columns = [
            str(c).strip().replace(" ", "_")
            for c in df.columns
        ]

        if "Date" not in df.columns:
            st.error("Your CSV must contain a Date column.")
            return None

        if "Close" not in df.columns:
            st.error("Your CSV must contain a Close column.")
            return None

        return df

    except Exception as e:
        st.error(f"CSV loading error: {e}")
        return None


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not MODEL_FILE.exists():
        return None

    try:
        return joblib.load(MODEL_FILE)

    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None


df = load_data()
model = load_model()


# =========================================================
# DATA PREPROCESSING
# =========================================================

if df is not None:

    df.columns = [
        str(c).strip().replace(" ", "_")
        for c in df.columns
    ]

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj_Close",
        "Volume"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df = df.dropna(
        subset=["Date", "Close"]
    )

    df = df.sort_values("Date").reset_index(drop=True)

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    df["Daily_Return"] = df["Close"].pct_change() * 100

    if "High" in df.columns and "Low" in df.columns:
        df["Daily_Range"] = df["High"] - df["Low"]

    if "Volume" not in df.columns:
        df["Volume"] = 0

    if "Open" not in df.columns:
        df["Open"] = df["Close"]

    if "High" not in df.columns:
        df["High"] = df["Close"]

    if "Low" not in df.columns:
        df["Low"] = df["Close"]


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def money(value):

    return f"₹{value:,.2f}"


def hero(title, subtitle):

    st.markdown(
        f"""
        <div style="
            padding:28px;
            border-radius:24px;
            background:linear-gradient(
                120deg,
                rgba(255,78,205,0.22),
                rgba(155,92,255,0.20),
                rgba(53,230,160,0.15)
            );
            border:1px solid rgba(255,255,255,0.2);
            backdrop-filter:blur(20px);
            margin-bottom:25px;
        ">
            <h1 style="font-size:35px;margin:0;">
                {title}
            </h1>
            <p style="font-size:15px;color:#eeeeff;">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def card(title, value, color="#ff4ecd"):

    st.markdown(
        f"""
        <div style="
            padding:20px;
            border-radius:18px;
            background:rgba(255,255,255,0.07);
            border:1px solid {color}66;
            box-shadow:0 5px 25px {color}22;
        ">
            <p style="color:#ddddf5;font-size:14px;">
                {title}
            </p>
            <h2 style="color:{color};font-size:25px;">
                {value}
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )


def style_chart(fig, height=400):

    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.025)",
        font=dict(
            color="white",
            family="Inter"
        ),
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0)"
        ),
        hoverlabel=dict(
            bgcolor="#241735",
            font_color="white"
        )
    )

    return fig


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown("# ✦ TCS VISION")

    st.caption("MACHINE LEARNING STOCK ANALYTICS")

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Command Center",
            "📊 Stock Pulse",
            "🔮 Prediction Lab",
            "🚀 Trend Radar",
            "🌈 Market Behaviour",
            "🗂️ Data Explorer",
            "🧠 Model Studio"
        ]
    )

    st.markdown("---")

    st.markdown("### SYSTEM STATUS")

    if model is not None:
        st.success("Prediction Model Loaded")
    else:
        st.error("Prediction Model Missing")

    if df is not None:
        st.success("Stock Dataset Loaded")
    else:
        st.error("Stock Dataset Missing")

    st.markdown("---")

    st.caption("TCS STOCK PREDICTION SYSTEM")
    st.caption("Built using Machine Learning")


# =========================================================
# CHECK DATA AND MODEL
# =========================================================

if df is None:

    st.error("TCS.csv file was not found or could not be loaded.")

    st.info(
        "Keep TCS.csv in the same folder as app.py."
    )

    st.stop()


if df.empty:

    st.error("Your dataset contains no valid stock records.")
    st.stop()


latest = df.iloc[-1]
previous = df.iloc[-2] if len(df) > 1 else latest

latest_price = float(latest["Close"])

price_change = latest_price - float(previous["Close"])

change_percent = (
    price_change / float(previous["Close"]) * 100
    if float(previous["Close"]) != 0 else 0
)


# =========================================================
# PAGE 1: COMMAND CENTER
# =========================================================

if page == "🏠 Command Center":

    hero(
        "TCS STOCK COMMAND CENTER",
        "Your intelligent space for stock market insights and machine learning predictions."
    )

    st.markdown("## Market Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Latest Closing Price",
            money(latest_price),
            f"{change_percent:.2f}%"
        )

    with c2:
        st.metric(
            "Opening Price",
            money(float(latest["Open"]))
        )

    with c3:
        st.metric(
            "Highest Price",
            money(float(latest["High"]))
        )

    with c4:
        st.metric(
            "Trading Volume",
            f"{int(latest['Volume']):,}"
        )

    st.markdown("---")

    left, right = st.columns([2, 1])

    with left:

        st.markdown("### 📈 Stock Performance")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Close"],
                name="Closing Price",
                mode="lines",
                line=dict(
                    color="#ff4ecd",
                    width=3
                ),
                fill="tozeroy",
                fillcolor="rgba(255,78,205,0.10)"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["MA20"],
                name="20 Day Average",
                line=dict(
                    color="#42e8ff",
                    width=2
                )
            )
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    with right:

        st.markdown("### 🎯 Market Distribution")

        positive = int(
            (df["Daily_Return"] > 0).sum()
        )

        negative = int(
            (df["Daily_Return"] < 0).sum()
        )

        fig = px.pie(
            values=[positive, negative],
            names=["Positive Days", "Negative Days"],
            hole=0.65,
            color_discrete_sequence=[
                "#35e6a0",
                "#ff4ecd"
            ]
        )

        st.plotly_chart(
            style_chart(fig, 300),
            use_container_width=True
        )

    st.markdown("### 🗂️ Latest Stock Records")

    st.dataframe(
        df.tail(10).sort_values(
            "Date",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# PAGE 2: STOCK PULSE
# =========================================================

elif page == "📊 Stock Pulse":

    hero(
        "STOCK PULSE",
        "Explore TCS historical stock movements and trading activity."
    )

    st.markdown("## Candlestick Analysis")

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                increasing_line_color="#35e6a0",
                decreasing_line_color="#ff4ecd",
                name="TCS Stock"
            )
        ]
    )

    st.plotly_chart(
        style_chart(fig, 500),
        use_container_width=True
    )

    st.markdown("## Trading Volume")

    fig = px.area(
        df,
        x="Date",
        y="Volume",
        color_discrete_sequence=["#9b5cff"]
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    st.markdown("## Price Comparison")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["High"],
            name="High",
            line=dict(color="#35e6a0")
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Low"],
            name="Low",
            line=dict(color="#ff9f43")
        )
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )


# =========================================================
# PAGE 3: PREDICTION LAB
# =========================================================

elif page == "🔮 Prediction Lab":

    hero(
        "PREDICTION LAB",
        "Predict the next TCS closing price using your trained Random Forest model."
    )

    st.markdown("## Machine Learning Prediction")

    if model is None:

        st.error(
            "TCS_Stock_Model.pkl was not found. "
            "Place your trained model in the same folder as app.py."
        )

    else:

        st.markdown(
            "Enter the five previous closing prices to generate a prediction."
        )

        col1, col2 = st.columns(2)

        with col1:

            lag1 = st.number_input(
                "Previous Day Closing Price",
                min_value=0.0,
                value=latest_price,
                step=1.0
            )

            lag2 = st.number_input(
                "Two Days Before",
                min_value=0.0,
                value=float(df["Close"].iloc[-2]) if len(df) >= 2 else latest_price,
                step=1.0
            )

            lag3 = st.number_input(
                "Three Days Before",
                min_value=0.0,
                value=float(df["Close"].iloc[-3]) if len(df) >= 3 else latest_price,
                step=1.0
            )

        with col2:

            lag4 = st.number_input(
                "Four Days Before",
                min_value=0.0,
                value=float(df["Close"].iloc[-4]) if len(df) >= 4 else latest_price,
                step=1.0
            )

            lag5 = st.number_input(
                "Five Days Before",
                min_value=0.0,
                value=float(df["Close"].iloc[-5]) if len(df) >= 5 else latest_price,
                step=1.0
            )

        if st.button(
            "✨ GENERATE STOCK PREDICTION",
            use_container_width=True
        ):

            try:

                input_data = pd.DataFrame(
                    [[lag1, lag2, lag3, lag4, lag5]],
                    columns=[
                        "Close_Lag_1",
                        "Close_Lag_2",
                        "Close_Lag_3",
                        "Close_Lag_4",
                        "Close_Lag_5"
                    ]
                )

                # Match the feature order used during training
                if hasattr(model, "feature_names_in_"):

                    expected_features = list(
                        model.feature_names_in_
                    )

                    if set(expected_features) != set(input_data.columns):
                        raise ValueError(
                            "Model features do not match the five lag inputs. "
                            "Check your model training code."
                        )

                    input_data = input_data[expected_features]

                prediction = float(
                    model.predict(input_data)[0]
                )

                difference = prediction - lag1

                percentage = (
                    difference / lag1 * 100
                    if lag1 != 0 else 0
                )

                st.markdown("---")

                st.markdown("### 🎯 Prediction Result")

                st.markdown(
                    f"""
                    <div style="
                        padding:30px;
                        border-radius:25px;
                        text-align:center;
                        background:linear-gradient(
                            120deg,
                            #ff4ecd33,
                            #9b5cff44,
                            #35e6a033
                        );
                        border:1px solid #ffffff44;
                    ">
                        <p style="font-size:16px;">
                            PREDICTED NEXT CLOSING PRICE
                        </p>
                        <h1 style="
                            font-size:45px;
                            color:#ffffff !important;
                        ">
                            ₹{prediction:,.2f}
                        </h1>
                        <p style="font-size:18px;">
                            {percentage:+.2f}% compared with previous close
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if difference > 0:
                    st.success(
                        "The model predicts a higher closing price."
                    )

                elif difference < 0:
                    st.warning(
                        "The model predicts a lower closing price."
                    )

                else:
                    st.info(
                        "The model predicts an unchanged closing price."
                    )

                st.caption(
                    "This is an experimental machine learning estimate, "
                    "not a guaranteed market result or investment advice."
                )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )


# =========================================================
# PAGE 4: TREND RADAR
# =========================================================

elif page == "🚀 Trend Radar":

    hero(
        "TREND RADAR",
        "Discover stock trends using moving averages and technical indicators."
    )

    st.markdown("## Moving Average Analysis")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            name="Closing Price",
            line=dict(color="#ff4ecd", width=2)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MA20"],
            name="MA 20",
            line=dict(color="#42e8ff", width=2)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MA50"],
            name="MA 50",
            line=dict(color="#35e6a0", width=2)
        )
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    st.markdown("## Relative Strength Index")

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["RSI"],
            name="RSI",
            line=dict(color="#ff9f43", width=2)
        )
    )

    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="#ff4ecd"
    )

    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="#35e6a0"
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    st.caption(
        "RSI is a technical indicator. It does not independently "
        "predict future stock prices."
    )


# =========================================================
# PAGE 5: MARKET BEHAVIOUR
# =========================================================

elif page == "🌈 Market Behaviour":

    hero(
        "MARKET BEHAVIOUR",
        "Understand daily price movements, volatility and trading patterns."
    )

    st.markdown("## Daily Returns")

    fig = px.histogram(
        df.dropna(subset=["Daily_Return"]),
        x="Daily_Return",
        nbins=40,
        color_discrete_sequence=["#ff4ecd"]
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    st.markdown("## Daily Trading Activity")

    fig = px.bar(
        df.tail(60),
        x="Date",
        y="Volume",
        color="Volume",
        color_continuous_scale=[
            "#9b5cff",
            "#ff4ecd",
            "#ff9f43"
        ]
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    st.markdown("## Daily Price Range")

    fig = px.area(
        df,
        x="Date",
        y="Daily_Range",
        color_discrete_sequence=["#35e6a0"]
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )


# =========================================================
# PAGE 6: DATA EXPLORER
# =========================================================

elif page == "🗂️ Data Explorer":

    hero(
        "DATA EXPLORER",
        "Explore, filter and download your TCS historical stock dataset."
    )

    st.markdown("## Dataset Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Total Records",
            f"{len(df):,}"
        )

    with c2:
        st.metric(
            "Available Columns",
            len(df.columns)
        )

    with c3:
        st.metric(
            "Missing Values",
            int(df.isna().sum().sum())
        )

    st.markdown("---")

    st.markdown("## Filter Stock Records")

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    date_range = st.date_input(
        "Choose Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:

        start_date, end_date = date_range

        filtered_df = df[
            (df["Date"].dt.date >= start_date) &
            (df["Date"].dt.date <= end_date)
        ]

    else:
        filtered_df = df.copy()

    search = st.text_input(
        "Search date or stock record"
    )

    if search:
        filtered_df = filtered_df[
            filtered_df.astype(str).apply(
                lambda column: column.str.contains(
                    search,
                    case=False,
                    na=False
                )
            ).any(axis=1)
        ]

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("## Download Dataset")

    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ DOWNLOAD FILTERED CSV",
        data=csv_data,
        file_name="TCS_Filtered_Data.csv",
        mime="text/csv",
        use_container_width=True
    )


# =========================================================
# PAGE 7: MODEL STUDIO
# =========================================================

elif page == "🧠 Model Studio":

    hero(
        "MODEL STUDIO",
        "Understand the machine learning model powering your stock predictions."
    )

    st.markdown("## Model Information")

    st.markdown(
        """
        ### Random Forest Regression

        Random Forest Regression is an ensemble machine learning
        algorithm that combines multiple decision trees to generate
        a numerical prediction.

        ### How This Project Works

        1. Historical TCS stock data is loaded.
        2. Previous closing prices are used as input features.
        3. The trained Random Forest model processes the inputs.
        4. The model generates an estimated next closing price.
        5. The dashboard displays the prediction.
        """
    )

    st.markdown("---")

    st.markdown("## Input Features")

    features = [
        "Close_Lag_1",
        "Close_Lag_2",
        "Close_Lag_3",
        "Close_Lag_4",
        "Close_Lag_5"
    ]

    for i, feature in enumerate(features, start=1):

        st.markdown(
            f"""
            <div style="
                padding:14px;
                margin-bottom:10px;
                border-radius:12px;
                background:rgba(255,255,255,0.07);
                border-left:4px solid #ff4ecd;
            ">
                <b style="color:#ff80df;">
                    Feature {i}
                </b>
                <span style="margin-left:12px;">
                    {feature}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown("## Project Technology")

    technologies = [
        ("Python", "#42e8ff"),
        ("Streamlit", "#ff4ecd"),
        ("Pandas", "#9b5cff"),
        ("NumPy", "#35e6a0"),
        ("Random Forest", "#ff9f43"),
        ("Plotly", "#42e8ff"),
        ("Joblib", "#ff4ecd")
    ]

    cols = st.columns(4)

    for i, (name, color) in enumerate(technologies):

        with cols[i % 4]:
            card(
                "TECHNOLOGY",
                name,
                color
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        padding:20px;
        color:#d9c9ee;
    ">
        <h3 style="
            background:linear-gradient(
                90deg,
                #ff4ecd,
                #9b5cff,
                #35e6a0
            );
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
        ">
            ✦ TCS STOCK PREDICTION AI
        </h3>
        <p>
            Machine Learning Based Stock Analytics Dashboard
        </p>
        <p style="font-size:12px;">
            Built with Python • Streamlit • Random Forest • Plotly
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

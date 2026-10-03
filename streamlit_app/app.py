import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))


# ============================================================
# IMPORT PROJECT DATA FUNCTIONS
# ============================================================

from preprocessing import (
    load_and_filter_data,
    prepare_data,
    create_monthly_series,
)


# ============================================================
# FORECAST FILES
# ============================================================

FORECAST_FILES = {
    "Rice (local)": "rice_local_forecast.csv",
    "Beans (red)": "beans_red_forecast.csv",
    "Yam": "yam_forecast.csv",
    "Gari (white)": "gari_white_forecast.csv",
    "Oil (palm)": "oil_palm_forecast.csv",
    "Tomatoes": "tomatoes_forecast.csv",
}


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Food Price Forecasting System",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# CUSTOM DASHBOARD STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL APP
       ======================================================== */

    .stApp {
        background-color: #F5F7F2;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }


    /* ========================================================
       HEADER / TITLE
       ======================================================== */

    h1 {
        color: #16A34A;
        font-weight: 750;
        letter-spacing: -0.6px;
    }

    h2 {
        color: #16A34A;
        font-weight: 700;
        margin-top: 2rem;
        letter-spacing: -0.2px;
    }

    h3 {
        color: #15803D;
        font-weight: 650;
    }

    p {
        color: #334155;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #DCE5DC;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #16A34A;
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #DCE5DC;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 3px 10px rgba(22, 163, 74, 0.08);
        min-height: 110px;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748B;
        font-size: 0.82rem;
        font-weight: 650;
    }

    div[data-testid="stMetricValue"] {
        color: #16A34A;
        font-weight: 750;
    }


    /* ========================================================
       SELECTBOX
       ======================================================== */

    div[data-baseweb="select"] > div {
        background-color: #FFFFFF;
        border-radius: 10px;
        border-color: #C8D8CA;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: #16A34A;
    }


    /* ========================================================
       INFORMATION / ALERT BOXES
       ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid #DCE5DC;
    }


    /* ========================================================
       EXPANDERS
       ======================================================== */

    details {
        background-color: #FFFFFF;
        border: 1px solid #DCE5DC;
        border-radius: 12px;
        margin-top: 12px;
    }

    details summary {
        color: #16A34A;
        font-weight: 650;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        background-color: #16A34A;
        color: #FFFFFF;
        border: none;
        border-radius: 9px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #15803D;
        color: #FFFFFF;
    }


    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {
        border-color: #DCE5DC;
    }


    /* ========================================================
       CAPTIONS
       ======================================================== */

    .stCaption {
        color: #64748B;
    }


    /* ========================================================
       DATAFRAME
       ======================================================== */

    div[data-testid="stDataFrame"] {
        border: 1px solid #DCE5DC;
        border-radius: 10px;
    }


    /* ========================================================
       GOLD ACCENT
       ======================================================== */

    .gold-accent {
        height: 3px;
        width: 70px;
        background-color: #C49A3A;
        border-radius: 10px;
        margin-top: 8px;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

@st.cache_data
def load_monthly_data():

    df = load_and_filter_data()

    df = prepare_data(df)

    monthly = create_monthly_series(df)

    return monthly


@st.cache_data
def load_forecast(commodity):

    forecast_path = (
        PROJECT_ROOT
        / "reports"
        / "forecasts"
        / FORECAST_FILES[commodity]
    )

    forecast = pd.read_csv(forecast_path)

    forecast["date"] = pd.to_datetime(
        forecast["date"]
    )

    return forecast

@st.cache_data
def load_anomalies():
    anomaly_path = (
        PROJECT_ROOT
        / "reports"
        / "spikes"
        / "anomaly_detection_comparison.csv"
    )

    anomalies = pd.read_csv(anomaly_path)
    anomalies["month"] = pd.to_datetime(anomalies["month"])

    return anomalies

@st.cache_data
def load_model_evaluation():
    evaluation_path = (
        PROJECT_ROOT
        / "reports"
        / "model_evaluation.csv"
    )

    evaluation = pd.read_csv(evaluation_path)

    return evaluation
# ============================================================
# DATA COVERAGE FUNCTION
# ============================================================

def classify_data_coverage(observation_count):

    if observation_count >= 30:
        return "Good"

    elif observation_count >= 10:
        return "Moderate"

    else:
        return "Limited"


# ============================================================
# CURRENT MONTH DETECTION
# ============================================================

current_month = (
    pd.Timestamp.today()
    .to_period("M")
    .to_timestamp()
)


# ============================================================
# DASHBOARD TITLE
# ============================================================

st.title("📈 Food Price Forecasting System")

st.markdown(
    '<div class="gold-accent"></div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    This system uses historical retail food-price data to analyse
    price movements and generate forecasts for selected staple foods
    in Nigeria.
    """
)

st.caption(
    "Forecasts are model-based estimates and should not be interpreted "
    "as guaranteed future prices."
)


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

try:

    monthly_data = load_monthly_data()

except Exception as error:

    st.error(
        "The dashboard could not load the food-price data."
    )

    st.exception(error)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Dashboard Controls")

commodity = st.sidebar.selectbox(
    "Select a commodity",
    [
        "Rice (local)",
        "Beans (red)",
        "Yam",
        "Gari (white)",
        "Oil (palm)",
        "Tomatoes",
    ],
)

st.sidebar.markdown("---")

st.sidebar.subheader("🎨 Appearance")

theme = st.sidebar.radio(
    "Choose theme",
    ["☀️ Light", "🌙 Dark"],
    horizontal=True,
)

# ============================================================
# DARK THEME
# ============================================================

if theme == "🌙 Dark":

    st.markdown(
        """
        <style>

        .stApp {
            background-color: #111827;
        }

        .block-container {
            color: #E5E7EB;
        }

        h1,
        h2 {
            color: #4ADE80;
        }

        h3 {
            color: #86EFAC;
        }

        p {
            color: #D1D5DB;
        }

        section[data-testid="stSidebar"] {
            background-color: #1F2937;
            border-right: 1px solid #374151;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #4ADE80;
        }

        div[data-testid="stMetric"] {
            background-color: #1F2937;
            border: 1px solid #374151;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.25);
        }

        div[data-testid="stMetricLabel"] {
            color: #9CA3AF;
        }

        div[data-testid="stMetricValue"] {
            color: #4ADE80;
        }

        div[data-baseweb="select"] > div {
            background-color: #1F2937;
            border-color: #4B5563;
        }

        details {
            background-color: #1F2937;
            border-color: #374151;
        }

        details summary {
            color: #4ADE80;
        }

        hr {
            border-color: #374151;
        }

        .stCaption {
            color: #9CA3AF;
        }

        div[data-testid="stDataFrame"] {
            border-color: #374151;
        }

        /* Streamlit header / top bar */
header[data-testid="stHeader"] {
    background-color: #111827;
}

header[data-testid="stHeader"] button {
    color: #E5E7EB;
}

/* Top-right Streamlit controls */
header[data-testid="stHeader"] svg {
    fill: #D1D5DB;
}

/* Sidebar icons and controls */
section[data-testid="stSidebar"] svg {
    fill: #D1D5DB;
}

/* Radio buttons */
div[role="radiogroup"] label {
    color: #D1D5DB;
}

/* Selectbox text and icons */
div[data-baseweb="select"] {
    color: #E5E7EB;
}

div[data-baseweb="select"] svg {
    fill: #D1D5DB;
}

/* Input / widget text */
input {
    color: #E5E7EB !important;
}

/* Buttons and widget icons */
button svg {
    fill: currentColor;
}

        div[data-testid="stDataFrame"] {
            border-color: #374151;
        }

        /* Streamlit header / top bar */
        header[data-testid="stHeader"] {
            background-color: #111827;
        }

        header[data-testid="stHeader"] button {
            color: #E5E7EB;
        }

        header[data-testid="stHeader"] svg {
            fill: #D1D5DB;
        }

        section[data-testid="stSidebar"] svg {
            fill: #D1D5DB;
        }

        div[role="radiogroup"] label {
            color: #D1D5DB;
        }

        div[data-baseweb="select"] {
            color: #E5E7EB;
        }

        div[data-baseweb="select"] svg {
            fill: #D1D5DB;
        }

        input {
            color: #E5E7EB !important;
        }

        button svg {
            fill: currentColor;
        }


        </style>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")

st.sidebar.markdown("---")

st.sidebar.info(
    f"""
    **Current calendar month**

    {current_month.strftime("%B %Y")}

    The dashboard automatically detects the current month
    from the system date.
    """
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **About the prices**

    The prices shown are retail-market observations from
    available WFP-covered markets.

    They should not be interpreted as the price at every
    shop or market across Nigeria.
    """
)

# ============================================================
# FILTER SELECTED COMMODITY
# ============================================================

commodity_data = monthly_data[
    monthly_data["commodity"] == commodity
].copy()

commodity_data = commodity_data.sort_values("month")


if commodity_data.empty:

    st.warning(
        "No data is currently available for this commodity."
    )

    st.stop()


# ============================================================
# CURRENT SELECTION
# ============================================================

st.header(commodity)

st.write(
    "Use the sections below to understand the historical "
    "price pattern and the expected future price."
)


# ============================================================
# LATEST HISTORICAL DATA
# ============================================================

latest_row = commodity_data.iloc[-1]

latest_price = latest_row["price"]
latest_date = latest_row["month"]
latest_observations = latest_row["observation_count"]

coverage = classify_data_coverage(
    latest_observations
)


# ============================================================
# MAIN METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Latest Actual Price",
        f"₦{latest_price:,.2f}",
    )

with col2:

    st.metric(
        "Latest Actual Month",
        latest_date.strftime("%b %Y"),
    )

with col3:

    st.metric(
        "Market Observations",
        f"{int(latest_observations):,}",
    )

with col4:

    st.metric(
        "Data Coverage",
        coverage,
    )


# ============================================================
# HISTORICAL PRICE SECTION
# ============================================================

st.subheader("📊 Historical Retail Price")

st.write(
    f"""
    This chart shows the monthly median retail price for
    **{commodity}** based on available WFP market observations.
    """
)


fig = px.line(
    commodity_data,
    x="month",
    y="price",
    markers=True,
    labels={
        "month": "Month",
        "price": "Retail Price (₦)",
    },
    title="Historical Retail Price Trend",
)

if theme == "🌙 Dark":
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#1F2937",
        font_color="#E5E7EB",
    )

fig.update_layout(
    height=480,
    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20,
    ),
    hovermode="x unified",
)

fig.update_traces(
    hovertemplate=(
        "<b>%{x|%b %Y}</b><br>"
        "Price: ₦%{y:,.2f}"
        "<extra></extra>"
    )
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# HISTORICAL DATA EXPLANATION
# ============================================================

with st.expander("ℹ️ How is this price calculated?"):

    st.write(
        f"""
        The displayed monthly price is the **median retail
        price** calculated from the available WFP market
        observations for {commodity} during that month.

        For the latest month shown:

        - **Month:** {latest_date.strftime("%B %Y")}
        - **Median retail price:** ₦{latest_price:,.2f}
        - **Source observations:** {int(latest_observations):,}
        - **Coverage level:** {coverage}

        The coverage level describes how much source data
        contributed to the monthly price.

        It is **not a statistical confidence level**.
        """
    )


# ============================================================
# FORECAST SECTION
# ============================================================

st.subheader("🔮 Future Price Forecast")


try:

    forecast_data = load_forecast(commodity)

except Exception as error:

    st.error(
        "The forecast could not be loaded."
    )

    st.exception(error)

    forecast_data = None


if forecast_data is not None and not forecast_data.empty:

    forecast_data = forecast_data.sort_values("date")


    # ========================================================
    # KEEP CURRENT AND FUTURE FORECASTS
    # ========================================================

    selectable_forecasts = forecast_data[
        forecast_data["date"] >= current_month
    ].copy()


    # ========================================================
    # CHECK WHETHER A CURRENT/FUTURE FORECAST EXISTS
    # ========================================================

    if selectable_forecasts.empty:

        st.warning(
            f"""
            No current or future forecast is available for
            {commodity} from {current_month.strftime("%B %Y")}
            onward.
            """
        )

    else:

        # ====================================================
        # FORECAST MONTH SELECTOR
        # ====================================================

        forecast_options = selectable_forecasts["date"].tolist()

        selected_forecast_date = st.selectbox(
            "Select forecast month",
            forecast_options,
            format_func=lambda date: date.strftime(
                "%B %Y"
            ),
        )


        selected_forecast = selectable_forecasts[
            selectable_forecasts["date"]
            == selected_forecast_date
        ].iloc[0]


        selected_forecast_price = (
            selected_forecast["forecast"]
        )

        selected_lower = selected_forecast["lower_ci"]

        selected_upper = selected_forecast["upper_ci"]


        # ====================================================
        # FORECAST STATUS
        # ====================================================

        if selected_forecast_date == current_month:

            forecast_status = "Current Month Forecast"

            st.info(
                f"""
                **{selected_forecast_date.strftime("%B %Y")}**
                is the current calendar month.

                The price shown below is the model's forecast
                for the current month.
                """
            )

        else:

            months_ahead = (
                (selected_forecast_date.year - current_month.year)
                * 12
                + (
                    selected_forecast_date.month
                    - current_month.month
                )
            )

            forecast_status = (
                f"{months_ahead} month"
                f"{'s' if months_ahead != 1 else ''} ahead"
            )

            st.info(
                f"""
                **{selected_forecast_date.strftime("%B %Y")}**
                is a future forecast.

                This forecast is approximately
                **{forecast_status.lower()}** from the
                current month.
                """
            )


        # ====================================================
        # SELECTED FORECAST METRICS
        # ====================================================

        forecast_col1, forecast_col2, forecast_col3 = st.columns(3)

        with forecast_col1:

            st.metric(
                "Expected Price",
                f"₦{selected_forecast_price:,.2f}",
            )
            st.info(
    """
    **What does this forecast mean?**

    The forecasted price is the model's estimated retail price
    for the selected month, based on historical price patterns
    in the available data.

    The expected range shows the range of prices within which
    the actual price may fall. The range becomes wider when
    forecasting farther into the future because uncertainty
    increases.

    A forecast is an estimate, not a guaranteed future price.
    """
)

        with forecast_col2:

            st.metric(
                "Forecasted Price",
                forecast_status,
            )

        with forecast_col3:

            st.metric(
                "Expected Range",
                (
                    f"₦{selected_lower:,.0f}"
                    f" – "
                    f"₦{selected_upper:,.0f}"
                ),
            )


        # ====================================================
        # FORECAST EXPLANATION
        # ====================================================

        st.write(
            f"""
            For **{selected_forecast_date.strftime("%B %Y")}**,
            the ARIMA model estimates a retail price of
            approximately **₦{selected_forecast_price:,.2f}**
            for **{commodity}**.

            The model's expected range is approximately
            **₦{selected_lower:,.2f} to ₦{selected_upper:,.2f}**.
            """
        )


    # ========================================================
    # FULL HISTORICAL + FORECAST CHART
    # ========================================================

    st.markdown("#### Historical prices and model forecast")

    historical_plot = commodity_data[
        ["month", "price"]
    ].copy()

    historical_plot = historical_plot.rename(
        columns={
            "month": "date",
            "price": "value",
        }
    )

    historical_plot["type"] = "Historical"


    forecast_plot = forecast_data[
        ["date", "forecast"]
    ].copy()

    forecast_plot = forecast_plot.rename(
        columns={
            "forecast": "value",
        }
    )

    forecast_plot["type"] = "Forecast"


    combined_plot = pd.concat(
        [
            historical_plot,
            forecast_plot,
        ],
        ignore_index=True,
    )


    fig_forecast = px.line(
    combined_plot,
    x="date",
    y="value",
    color="type",
    labels={
        "date": "Month",
        "value": "Retail Price (₦)",
        "type": "Series",
    },
    title="Actual and Forecasted Retail Price",
)
    if theme == "🌙 Dark":
        fig_forecast.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#1F2937",
        font_color="#E5E7EB",
    )


    # ========================================================
    # EXPECTED RANGE
    # ========================================================

    fig_forecast.add_scatter(
        x=forecast_data["date"],
        y=forecast_data["upper_ci"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    )

    fig_forecast.add_scatter(
        x=forecast_data["date"],
        y=forecast_data["lower_ci"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        name="Expected range",
        hoverinfo="skip",
    )


    fig_forecast.update_layout(
        height=500,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
        hovermode="x unified",
    )


    st.plotly_chart(
        fig_forecast,
        use_container_width=True,
    )


    # ========================================================
    # EXPECTED RANGE EXPLANATION
    # ========================================================

    with st.expander("ℹ️ What does the expected range mean?"):

        if selectable_forecasts.empty:

            st.write(
                "No current or future forecast is available."
            )

        else:

            st.write(
    f"""
    The shaded area around the forecast represents the model's
    **95% expected range**.

    For the selected month:

    - **Forecasted price:** ₦{selected_forecast_price:,.2f}
    - **Lower boundary:** ₦{selected_lower:,.2f}
    - **Upper boundary:** ₦{selected_upper:,.2f}

    The expected range shows the uncertainty around the model's
    forecast. It generally becomes wider farther into the future
    because there is more uncertainty about what may happen.

    The range is **not a guarantee** that the actual market price
    will fall inside it.
    """
)

    # ========================================================
    # GARI DATA WARNING
    # ========================================================

    if commodity == "Gari (white)":

        st.warning(
            """
            **Data availability note:** The available Gari
            retail-price history ends in January 2023.

            Therefore, this forecast is based on historical
            data that is older than the other commodity
            forecasts and should be interpreted with caution.
            """
        )


# ============================================================
# UNUSUAL PRICE MOVEMENTS
# ============================================================

# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("📊 Model Performance")

try:

    evaluation_data = load_model_evaluation()

    commodity_evaluation = evaluation_data[
        evaluation_data["commodity"] == commodity
    ].copy()

except Exception as error:

    st.error(
        "The model evaluation results could not be loaded."
    )

    st.exception(error)

    commodity_evaluation = pd.DataFrame()


if commodity_evaluation.empty:

    st.info(
        f"No model evaluation results are available for "
        f"{commodity}."
    )

else:

    evaluation = commodity_evaluation.iloc[0]

    st.write(
    """
    The forecasting model was tested using historical data
    that was kept separate from the data used to train the
    model.

    These results show how closely the model's predictions
    matched actual prices during the historical test period.
    Lower error values generally indicate smaller prediction
    errors, but these results do not guarantee the same
    accuracy for future forecasts.
    """
)
    # ========================================================
    # PERFORMANCE METRICS
    # ========================================================

    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

    with perf_col1:

        st.metric(
            "MAE",
            f"₦{evaluation['mae']:,.2f}"
        )

    with perf_col2:

        st.metric(
            "RMSE",
            f"₦{evaluation['rmse']:,.2f}"
        )

    with perf_col3:

        st.metric(
            "MAPE",
            f"{evaluation['mape']:.2f}%"
        )

    with perf_col4:

        st.metric(
            "Test Observations",
            f"{int(evaluation['testing_observations'])}"
        )


    # ========================================================
    # PLAIN-LANGUAGE EXPLANATION
    # ========================================================

    st.info(
        """
        **How to read these results**

        **MAE** shows the average size of the model's prediction
        error in Nigerian Naira. For example, a MAE of ₦500 means
        the predictions were, on average, about ₦500 away from
        the actual prices during testing.

        **RMSE** is another measure of prediction error. It gives
        more weight to larger errors.

        **MAPE** expresses the average prediction error as a
        percentage of the actual price.

        These measurements describe historical test performance.
        They do not guarantee that future predictions will have
        the same level of accuracy.
        """
    )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.markdown("### Model Information")

    model_col1, model_col2, model_col3 = st.columns(3)

    with model_col1:

        st.write(
            f"**ARIMA model:** "
            f"ARIMA{evaluation['arima_order']}"
        )

    with model_col2:

        st.write(
            f"**Training observations:** "
            f"{int(evaluation['training_observations'])}"
        )

    with model_col3:

        st.write(
            f"**Testing observations:** "
            f"{int(evaluation['testing_observations'])}"
        )


    # ========================================================
    # DATA-SPECIFIC CAUTION
    # ========================================================

    if commodity == "Gari (white)":

        st.warning(
            """
            **Important:** The Gari model was evaluated using
            only 7 testing observations. Its evaluation result
            should therefore be interpreted cautiously.

            The available Gari data also ends in January 2023,
            so its current forecast is limited by the age of
            the source data.
            """
        )


    # ========================================================
    # TECHNICAL DETAILS
    # ========================================================

    with st.expander(
        "🔬 Technical details of model evaluation"
    ):

        technical_col1, technical_col2 = st.columns(2)

        with technical_col1:

            st.write(
                f"**ARIMA order:** "
                f"{evaluation['arima_order']}"
            )

            st.write(
                f"**Differencing order (d):** "
                f"{int(evaluation['differencing_order'])}"
            )

            st.write(
                f"**Training observations:** "
                f"{int(evaluation['training_observations'])}"
            )

            st.write(
                f"**Testing observations:** "
                f"{int(evaluation['testing_observations'])}"
            )

        with technical_col2:

            st.write(
                f"**MAE:** "
                f"₦{evaluation['mae']:,.2f}"
            )

            st.write(
                f"**RMSE:** "
                f"₦{evaluation['rmse']:,.2f}"
            )

            st.write(
                f"**MAPE:** "
                f"{evaluation['mape']:.2f}%"
            )

            st.write(
                f"**Ljung-Box p-value:** "
                f"{evaluation['ljung_box_pvalue']:.4f}"
            )

        st.markdown(
            """
            **Ljung-Box test**

            The Ljung-Box test checks whether the model residuals
            still contain evidence of autocorrelation at the
            tested lag.

            A p-value above 0.05 means there is not strong
            statistical evidence of residual autocorrelation
            at the tested lag.

            This does not mean that the model is perfect or that
            future forecasts are guaranteed to be accurate.
            """
        )


# ============================================================
# UNUSUAL PRICE MOVEMENTS
# ============================================================

st.subheader("⚠️ Unusual Price Movements")

try:

    anomaly_data = load_anomalies()

    commodity_anomalies = anomaly_data[
        anomaly_data["commodity"] == commodity
    ].copy()

    commodity_anomalies = commodity_anomalies.sort_values(
        "month",
        ascending=False
    )

except Exception as error:

    st.error(
        "The unusual price movement data could not be loaded."
    )

    st.exception(error)

    commodity_anomalies = pd.DataFrame()


if commodity_anomalies.empty:

    st.success(
        f"No unusual price movements were detected for "
        f"{commodity} in the available historical data."
    )

else:

    st.write(
        f"""
        The system detected **{len(commodity_anomalies)} unusual
        price movement{"s" if len(commodity_anomalies) != 1 else ""}
        ** for {commodity} in the available historical data.
        """
    )

    st.info(
    """
    An unusual movement means that the price changed in a way
    that was unusually large compared with the recent or
    model-based historical pattern.

    This detection does **not** identify the cause of the movement
    and does not automatically mean that the data is incorrect.
    It simply highlights a movement that deserves attention.
    """
)
    # ========================================================
    # LATEST ANOMALY SUMMARY
    # ========================================================

    latest_anomaly = commodity_anomalies.iloc[0]

    anomaly_col1, anomaly_col2, anomaly_col3, anomaly_col4 = (
        st.columns(4)
    )

    with anomaly_col1:

        st.metric(
            "Detected Events",
            f"{len(commodity_anomalies):,}"
        )

    with anomaly_col2:

        st.metric(
            "Latest Event",
            latest_anomaly["month"].strftime("%b %Y")
        )

    with anomaly_col3:

        st.metric(
            "Actual Price",
            f"₦{latest_anomaly['actual_price']:,.2f}"
        )

    with anomaly_col4:

        detection = latest_anomaly[
            "detection_classification"
        ]

        if detection == "ARIMA residual method only":

            metric_label = "Model Difference"

            if pd.isna(latest_anomaly["residual"]):

                metric_value = "Not available"

            else:

                metric_value = (
                    f"₦{abs(latest_anomaly['residual']):,.2f}"
                )

        else:

            metric_label = "Price Change"

            if pd.isna(latest_anomaly["pct_change"]):

                metric_value = "Not available"

            else:

                metric_value = (
                    f"{latest_anomaly['pct_change']:+.2f}%"
                )

        st.metric(
            metric_label,
            metric_value
        )


    # ========================================================
    # RECENT UNUSUAL MOVEMENTS
    # ========================================================

    st.markdown("### Recent Unusual Movements")

    for _, row in commodity_anomalies.head(5).iterrows():

        direction = str(row["rolling_direction"])

        if direction == "positive":

            direction_text = "Unusual increase"

        elif direction == "negative":

            direction_text = "Unusual decrease"

        else:

            direction_text = "Unusual movement"


        detection = row["detection_classification"]

        if detection == "Both methods":

            detection_text = "Both methods"

        elif detection == "Rolling method only":

            detection_text = "Recent-pattern method"

        elif detection == "ARIMA residual method only":

            detection_text = "ARIMA model method"

        else:

            detection_text = detection


    with st.expander(
        f"{row['month'].strftime('%B %Y')} — "
        f"{direction_text}"
):

        col1, col2, col3 = st.columns(3)

        with col1:

            st.write(
                f"**Actual retail price:** "
                f"₦{row['actual_price']:,.2f}"
        )

        if not pd.isna(row["pct_change"]):

            st.write(
                f"**Price movement:** "
                f"{row['pct_change']:+.2f}%"
            )

            if row["pct_change"] > 0:
                st.caption(
                    "The price increased unusually compared "
                    "with recent historical movements."
                )

            elif row["pct_change"] < 0:
                st.caption(
                    "The price decreased unusually compared "
                    "with recent historical movements."
                )

        if not pd.isna(row["residual"]):

            if row["residual"] > 0:

                st.write(
                    f"**Model comparison:** "
                    f"₦{abs(row['residual']):,.2f} above expectation"
                )

            elif row["residual"] < 0:

                st.write(
                    f"**Model comparison:** "
                    f"₦{abs(row['residual']):,.2f} below expectation"
                )

            else:

                st.write(
                    "**Model comparison:** "
                    "No difference from model expectation"
                )

    with col2:

        st.write(
            f"**Market observations:** "
            f"{int(row['observation_count'])}"
        )

        st.write(
            f"**Data coverage:** "
            f"{row['data_coverage']}"
        )

    with col3:

        st.write(
            f"**Detection method:** "
            f"{detection_text}"
        )

    st.caption(
        "This alert identifies a statistically unusual price "
        "movement. It does not determine why the price changed "
        "and does not automatically mean the data is incorrect."
    )

    # ========================================================
    # TECHNICAL DETAILS OF ANOMALY DETECTION
    # ========================================================

    with st.expander(
        "🔬 Technical details of the detection"
    ):

        technical_table = commodity_anomalies[
            [
                "month",
                "actual_price",
                "pct_change",
                "rolling_robust_z",
                "residual_robust_z",
                "detection_classification",
            ]
        ].copy()

        technical_table["month"] = technical_table[
            "month"
        ].dt.strftime("%Y-%m")

        technical_table = technical_table.rename(
            columns={
                "month": "Month",
                "actual_price": "Actual Price",
                "pct_change": "Price Change (%)",
                "rolling_robust_z": "Rolling Robust Z",
                "residual_robust_z": "ARIMA Residual Z",
                "detection_classification": "Detection",
            }
        )

        st.dataframe(
            technical_table,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            """
            **Detection methods**

            **Recent-pattern method:** compares the current
            month-to-month price movement with the recent
            historical pattern using a robust statistical
            measure.

            **ARIMA model method:** compares the actual price
            with the price expected by the ARIMA model.

            **Robust Z-score:** measures how unusual a movement
            is relative to its reference pattern. The system
            uses a threshold of approximately ±3.5.

            A detected event should be interpreted as a
            statistical alert, not as an explanation of the
            economic cause.
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "ARIMA-Based Forecasting System for Selected Staple Food Prices in Nigeria"
)
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_data
from preprocessing import (
    SELECTED_COMMODITIES,
    prepare_data,
    create_monthly_series,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"

FORECAST_DIR = (
    PROJECT_ROOT / "reports" / "forecasts"
)

FORECAST_FIGURES_DIR = (
    PROJECT_ROOT / "reports"
    / "figures"
    / "forecasts"
)

FORECAST_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FORECAST_FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# COMMODITIES
# ============================================================

COMMODITIES = [
    "Rice (local)",
    "Beans (red)",
    "Yam",
    "Gari (white)",
    "Oil (palm)",
    "Tomatoes",
]


# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILES = {
    "Rice (local)": "rice_local_arima.pkl",
    "Beans (red)": "beans_red_arima.pkl",
    "Yam": "yam_arima.pkl",
    "Gari (white)": "gari_white_arima.pkl",
    "Oil (palm)": "oil_palm_arima.pkl",
    "Tomatoes": "tomatoes_arima.pkl",
}


# ============================================================
# FORECAST HORIZON
# ============================================================

FORECAST_HORIZON = 12


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(commodity):

    model_file = MODEL_FILES[commodity]

    model_path = MODELS_DIR / model_file

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)


# ============================================================
# LOAD MONTHLY DATA
# ============================================================

def prepare_monthly_data():

    df = load_data()

    df = df[
        (df["commodity"].isin(
            SELECTED_COMMODITIES
        ))
        &
        (df["pricetype"] == "Retail")
    ].copy()

    df = prepare_data(df)

    monthly = create_monthly_series(df)

    return monthly


# ============================================================
# GET HISTORICAL SERIES
# ============================================================

def get_historical_series(
    monthly,
    commodity,
):

    commodity_data = monthly[
        monthly["commodity"] == commodity
    ].copy()

    commodity_data = commodity_data.sort_values(
        "month"
    )

    return commodity_data


# ============================================================
# GENERATE FORECAST
# ============================================================

def generate_forecast(
    model,
    commodity,
    horizon=FORECAST_HORIZON,
):

    forecast_result = model.get_forecast(
        steps=horizon
    )

    forecast_values = (
        forecast_result.predicted_mean
    )

    confidence_intervals = (
        forecast_result.conf_int()
    )

    forecast_df = pd.DataFrame(
        {
            "date": forecast_values.index,
            "forecast": forecast_values.values,
            "lower_ci": confidence_intervals.iloc[:, 0].values,
            "upper_ci": confidence_intervals.iloc[:, 1].values,
        }
    )

    # Food prices cannot be negative.
    # We keep the original forecast but prevent
    # the displayed lower confidence bound from
    # going below zero.
    forecast_df["lower_ci"] = (
        forecast_df["lower_ci"].clip(lower=0)
    )

    forecast_df["commodity"] = commodity

    forecast_df = forecast_df[
        [
            "date",
            "commodity",
            "forecast",
            "lower_ci",
            "upper_ci",
        ]
    ]

    return forecast_df


# ============================================================
# SAVE FORECAST
# ============================================================

def save_forecast(
    forecast_df,
    commodity,
):

    safe_name = (
        commodity
        .lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    output_path = (
        FORECAST_DIR
        / f"{safe_name}_forecast.csv"
    )

    forecast_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved forecast: {output_path}"
    )


# ============================================================
# PLOT HISTORICAL + FORECAST
# ============================================================

def plot_forecast(
    historical_data,
    forecast_df,
    commodity,
):

    safe_name = (
        commodity
        .lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    plt.figure(figsize=(13, 7))

    # --------------------------------------------------------
    # Historical prices
    # --------------------------------------------------------

    plt.plot(
        historical_data["month"],
        historical_data["price"],
        label="Historical Price",
    )

    # --------------------------------------------------------
    # Forecast
    # --------------------------------------------------------

    plt.plot(
        forecast_df["date"],
        forecast_df["forecast"],
        marker="o",
        label="ARIMA Forecast",
    )

    # --------------------------------------------------------
    # Confidence interval
    # --------------------------------------------------------

    plt.fill_between(
        forecast_df["date"],
        forecast_df["lower_ci"],
        forecast_df["upper_ci"],
        alpha=0.2,
        label="95% Confidence Interval",
    )

    # --------------------------------------------------------
    # Forecast boundary
    # --------------------------------------------------------

    forecast_start = forecast_df["date"].iloc[0]

    plt.axvline(
        forecast_start,
        linestyle="--",
        label="Forecast Start",
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    plt.title(
        f"{commodity} - Historical Price and "
        f"12-Month ARIMA Forecast"
    )

    plt.xlabel("Date")

    plt.ylabel("Price (NGN)")

    plt.xticks(rotation=45)

    plt.legend()

    plt.tight_layout()

    output_path = (
        FORECAST_FIGURES_DIR
        / f"{safe_name}_forecast.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved forecast plot: {output_path}"
    )


# ============================================================
# CHECK FORECAST RECENCY
# ============================================================

def check_forecast_recency(
    historical_data,
    forecast_df,
    commodity,
):

    latest_historical_date = (
        historical_data["month"].max()
    )

    forecast_start = (
        forecast_df["date"].min()
    )

    gap_months = (
        (forecast_start.year - latest_historical_date.year)
        * 12
        +
        (
            forecast_start.month
            - latest_historical_date.month
        )
    )

    print(
        f"Latest historical observation: "
        f"{latest_historical_date.strftime('%Y-%m')}"
    )

    print(
        f"Forecast starts: "
        f"{forecast_start.strftime('%Y-%m')}"
    )

    if gap_months > 1:

        print(
            "WARNING: Forecast is not immediately "
            "after the latest historical observation."
        )

        print(
            f"Gap between historical data and forecast: "
            f"{gap_months - 1} month(s)"
        )

    else:

        print(
            "Forecast begins immediately after "
            "the latest historical observation."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("STARTING CLEAN FUTURE FORECASTING")
    print("=" * 70)

    print(
        f"\nForecast horizon: "
        f"{FORECAST_HORIZON} months"
    )

    monthly = prepare_monthly_data()

    print(
        f"Monthly observations available: "
        f"{len(monthly)}"
    )

    for commodity in COMMODITIES:

        print("\n" + "-" * 70)
        print(f"FORECASTING: {commodity}")
        print("-" * 70)

        model = load_model(
            commodity
        )

        historical_data = get_historical_series(
            monthly,
            commodity,
        )

        forecast_df = generate_forecast(
            model=model,
            commodity=commodity,
            horizon=FORECAST_HORIZON,
        )

        print("\nForecast period:")

        print(
            f"{forecast_df['date'].min().strftime('%Y-%m')} "
            f"to "
            f"{forecast_df['date'].max().strftime('%Y-%m')}"
        )

        check_forecast_recency(
            historical_data,
            forecast_df,
            commodity,
        )

        print("\nForecast values:")

        print(
            forecast_df[
                [
                    "date",
                    "forecast",
                    "lower_ci",
                    "upper_ci",
                ]
            ].to_string(index=False)
        )

        save_forecast(
            forecast_df,
            commodity,
        )

        plot_forecast(
            historical_data,
            forecast_df,
            commodity,
        )

    print("\n" + "=" * 70)
    print("CLEAN FUTURE FORECASTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
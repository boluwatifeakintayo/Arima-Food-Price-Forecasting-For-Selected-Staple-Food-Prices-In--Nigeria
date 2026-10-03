from pathlib import Path

import joblib
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

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
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FINAL ARIMA ORDERS
# ============================================================

FINAL_ARIMA_ORDERS = {
    "Rice (local)": (1, 1, 0),
    "Beans (red)": (1, 1, 1),
    "Yam": (0, 2, 1),
    "Gari (white)": (0, 2, 1),
    "Oil (palm)": (1, 1, 1),
    "Tomatoes": (1, 1, 0),
}


# ============================================================
# TRAINING SEGMENT STRATEGY
# ============================================================

TRAINING_SEGMENT = {
    "Rice (local)": "full",
    "Beans (red)": "latest",
    "Yam": "full",
    "Gari (white)": "latest",
    "Oil (palm)": "full",
    "Tomatoes": "latest",
}


# ============================================================
# LOAD MONTHLY DATA
# ============================================================

def prepare_monthly_data():

    df = load_data()

    df = df[
        (df["commodity"].isin(SELECTED_COMMODITIES))
        & (df["pricetype"] == "Retail")
    ].copy()

    df = prepare_data(df)

    monthly = create_monthly_series(df)

    return monthly


# ============================================================
# FIND CONTINUOUS SEGMENTS
# ============================================================

def find_continuous_segments(series):

    series = series.sort_values("month").copy()

    month_number = (
        series["month"].dt.year * 12
        + series["month"].dt.month
    )

    previous_month_number = month_number.shift(1)

    new_segment = (
        month_number - previous_month_number != 1
    )

    segment_id = new_segment.cumsum()

    segments = []

    for _, segment in series.groupby(segment_id):

        segments.append(segment.copy())

    return segments


# ============================================================
# SELECT FINAL TRAINING SEGMENT
# ============================================================

def select_training_segment(
    commodity_data,
    strategy,
):

    segments = find_continuous_segments(
        commodity_data
    )

    if not segments:
        raise ValueError(
            "No continuous segment found."
        )

    if strategy == "full":

        # For commodities with one long continuous
        # historical series, use that series.
        selected = max(
            segments,
            key=len,
        )

    elif strategy == "latest":

        # Use the most recent continuous segment.
        selected = max(
            segments,
            key=lambda x: x["month"].max(),
        )

    else:

        raise ValueError(
            f"Unknown training strategy: {strategy}"
        )

    selected = selected.sort_values(
        "month"
    ).copy()

    selected = selected.set_index(
        "month"
    )

    return selected


# ============================================================
# TRAIN FINAL ARIMA MODEL
# ============================================================

def train_final_model(
    commodity,
    training_data,
    order,
):

    series = training_data["price"].astype(float)

    print("\n" + "-" * 70)
    print(f"TRAINING FINAL MODEL: {commodity}")
    print("-" * 70)

    print(
        f"Training period: "
        f"{series.index.min().strftime('%Y-%m')} "
        f"to "
        f"{series.index.max().strftime('%Y-%m')}"
    )

    print(
        f"Training observations: {len(series)}"
    )

    print(
        f"ARIMA order: {order}"
    )

    model = ARIMA(
        series,
        order=order,
    )

    fitted_model = model.fit()

    print(
        f"AIC: {fitted_model.aic:.2f}"
    )

    print(
        f"BIC: {fitted_model.bic:.2f}"
    )

    return fitted_model


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    fitted_model,
    commodity,
):

    safe_name = (
        commodity
        .lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    model_path = (
        MODELS_DIR
        / f"{safe_name}_arima.pkl"
    )

    joblib.dump(
        fitted_model,
        model_path,
    )

    print(
        f"Saved model: {model_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("STARTING FINAL ARIMA MODEL TRAINING")
    print("=" * 70)

    monthly = prepare_monthly_data()

    print(
        f"\nMonthly observations available: "
        f"{len(monthly)}"
    )

    for commodity in SELECTED_COMMODITIES:

        commodity_data = monthly[
            monthly["commodity"] == commodity
        ].copy()

        strategy = TRAINING_SEGMENT[
            commodity
        ]

        order = FINAL_ARIMA_ORDERS[
            commodity
        ]

        training_data = select_training_segment(
            commodity_data,
            strategy,
        )

        fitted_model = train_final_model(
            commodity,
            training_data,
            order,
        )

        save_model(
            fitted_model,
            commodity,
        )

    print("\n" + "=" * 70)
    print("FINAL ARIMA MODEL TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
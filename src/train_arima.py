from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_data
from preprocessing import (
    SELECTED_COMMODITIES,
    load_and_filter_data,
    prepare_data,
    create_monthly_series,
)
from stationarity import get_longest_continuous_segment

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# DIFFERENCING ORDERS
# Based on our stationarity analysis
# ============================================================

DIFFERENCING_ORDERS = {
    "Beans (red)": 1,
    "Gari (white)": 2,
    "Oil (palm)": 1,
    "Rice (local)": 1,
    "Tomatoes": 1,
    "Yam": 2,
}


# ============================================================
# ARIMA CANDIDATE MODELS
# ============================================================

ARIMA_CANDIDATES = {
    "Rice (local)": [
        (1, 1, 0),
        (0, 1, 1),
        (1, 1, 1),
    ],

    "Beans (red)": [
        (1, 1, 0),
        (0, 1, 1),
        (1, 1, 1),
        (2, 1, 0),
    ],

    "Yam": [
        (1, 2, 0),
        (0, 2, 1),
        (1, 2, 1),
        (2, 2, 0),
    ],

    "Gari (white)": [
        (1, 2, 0),
        (0, 2, 1),
        (1, 2, 1),
    ],

    "Oil (palm)": [
        (1, 1, 0),
        (0, 1, 1),
        (1, 1, 1),
    ],

    "Tomatoes": [
        (1, 1, 0),
        (0, 1, 1),
        (1, 1, 1),
    ],
}


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_monthly_data():
    """
    Load WFP Nigeria food price data,
    filter selected commodities,
    and create monthly price series.
    """

    df = load_and_filter_data()

    df = prepare_data(df)

    monthly = create_monthly_series(df)

    return monthly


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def create_train_test_split(
    monthly,
    commodity,
    test_size=0.20
):
    """
    Create a chronological train/test split.

    The longest continuous monthly segment is used so that
    missing months do not create artificial time gaps.
    """

    series = (
        monthly[
            monthly["commodity"] == commodity
        ][["month", "price"]]
        .sort_values("month")
        .copy()
    )

    if series.empty:
        raise ValueError(
            f"No monthly data found for {commodity}."
        )

    # --------------------------------------------------------
    # Keep longest continuous monthly segment
    # --------------------------------------------------------

    continuous = get_longest_continuous_segment(series)

    if len(continuous) < 10:
        raise ValueError(
            f"Not enough continuous observations for {commodity}."
        )

    # --------------------------------------------------------
    # Calculate test size
    # --------------------------------------------------------

    test_count = max(
        1,
        int(round(len(continuous) * test_size))
    )

    train = continuous.iloc[:-test_count].copy()

    test = continuous.iloc[-test_count:].copy()

    if train.empty or test.empty:
        raise ValueError(
            f"Invalid train/test split for {commodity}."
        )

    return train, test


# ============================================================
# TRAIN ARIMA MODEL
# ============================================================

def train_arima_model(
    train_data,
    order
):
    """
    Train an ARIMA model.

    Parameters
    ----------
    train_data : DataFrame
        Training data containing 'month' and 'price'.

    order : tuple
        ARIMA(p, d, q) order.

    Returns
    -------
    fitted_model
        Fitted statsmodels ARIMA model.
    """

    series = (
        train_data
        .set_index("month")["price"]
        .copy()
    )

    # Make sure index is datetime
    series.index = pd.DatetimeIndex(
        series.index
    )

    # Explicit monthly frequency
    series = series.asfreq("MS")

    # Check for missing values after frequency assignment
    if series.isna().any():
        raise ValueError(
            "Training series contains missing monthly values."
        )

    # --------------------------------------------------------
    # Train ARIMA
    # --------------------------------------------------------

    model = ARIMA(
        series,
        order=order
    )

    fitted_model = model.fit()

    return fitted_model


# ============================================================
# FORECAST
# ============================================================

def forecast_model(
    model,
    test_data
):
    """
    Forecast prices for the test period.
    """

    forecast = model.forecast(
        steps=len(test_data)
    )

    return forecast


# ============================================================
# EVALUATION
# ============================================================

def evaluate_forecast(
    test_data,
    forecast
):
    """
    Calculate MAE, RMSE, and MAPE.
    """

    actual = test_data["price"].to_numpy()

    predicted = np.asarray(forecast)

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = mean_absolute_error(
        actual,
        predicted
    )

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    # --------------------------------------------------------
    # MAPE
    # Avoid division by zero
    # --------------------------------------------------------

    non_zero = actual != 0

    if np.any(non_zero):

        mape = np.mean(
            np.abs(
                (
                    actual[non_zero]
                    - predicted[non_zero]
                )
                / actual[non_zero]
            )
        ) * 100

    else:
        mape = np.nan

    return mae, rmse, mape


# ============================================================
# GENERIC FORECAST PLOT
# ============================================================

def plot_forecast(
    train,
    test,
    forecast,
    commodity,
    order
):
    """
    Plot training data, actual test prices,
    and ARIMA forecasts.
    """

    plt.figure(figsize=(12, 6))

    # Training data
    plt.plot(
        train["month"],
        train["price"],
        label="Training Data"
    )

    # Actual test data
    plt.plot(
        test["month"],
        test["price"],
        label="Actual Test Prices"
    )

    # Forecast
    plt.plot(
        test["month"],
        forecast,
        label="ARIMA Forecast",
        linestyle="--"
    )

    plt.title(
        f"{commodity} - ARIMA{order} Forecast"
    )

    plt.xlabel("Date")
    plt.ylabel("Price (NGN)")

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output_dir = (
        PROJECT_ROOT
        / "reports"
        / "figures"
        / "forecasts"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    safe_name = (
        commodity
        .lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    output_path = (
        output_dir
        / f"{safe_name}_arima_{order[0]}_{order[1]}_{order[2]}_forecast.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(
        f"\nSaved forecast plot: {output_path}"
    )


# ============================================================
# RESIDUAL DIAGNOSTICS
# ============================================================

def diagnose_model(
    model,
    commodity,
    order
):
    """
    Perform residual diagnostics for a fitted ARIMA model.

    Diagnostics include:
    1. Residual summary
    2. Ljung-Box test
    3. Residual plot
    4. Residual ACF plot
    """

    residuals = model.resid.dropna()

    if residuals.empty:
        raise ValueError(
            f"No residuals available for {commodity}."
        )

    print("\n" + "=" * 70)

    print(
        f"{commodity.upper()} "
        f"ARIMA{order} RESIDUAL DIAGNOSTICS"
    )

    print("=" * 70)

    print(
        f"\nNumber of residuals: "
        f"{len(residuals)}"
    )

    # --------------------------------------------------------
    # Residual summary
    # --------------------------------------------------------

    print("\nResidual summary:")

    print(
        f"Mean: "
        f"{residuals.mean():.4f}"
    )

    print(
        f"Standard deviation: "
        f"{residuals.std():.4f}"
    )

    # --------------------------------------------------------
    # Ljung-Box test
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print("LJUNG-BOX TEST")

    print("-" * 70)

    # Use a reasonable lag based on sample size
    lag = min(
        12,
        max(1, len(residuals) // 4)
    )

    ljung_box = acorr_ljungbox(
        residuals,
        lags=[lag],
        return_df=True
    )

    print(ljung_box)

    p_value = (
        ljung_box["lb_pvalue"]
        .iloc[0]
    )

    print(
        f"\nLjung-Box p-value: "
        f"{p_value:.4f}"
    )

    if p_value > 0.05:

        print("\nInterpretation:")

        print(
            "There is no strong evidence "
            "of residual autocorrelation "
            f"at lag {lag}."
        )

    else:

        print("\nInterpretation:")

        print(
            "The residuals may still contain "
            "autocorrelation. The model should "
            "be investigated further."
        )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = (
        PROJECT_ROOT
        / "reports"
        / "figures"
        / "diagnostics"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    safe_name = (
        commodity
        .lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    # --------------------------------------------------------
    # Residual plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        residuals.index,
        residuals.values
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.title(
        f"{commodity} "
        f"ARIMA{order} Residuals"
    )

    plt.xlabel("Date")

    plt.ylabel("Residual")

    plt.tight_layout()

    residual_plot_path = (
        output_dir
        / f"{safe_name}_arima_residuals.png"
    )

    plt.savefig(
        residual_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(
        f"\nSaved residual plot: "
        f"{residual_plot_path}"
    )

    # --------------------------------------------------------
    # Residual ACF
    # --------------------------------------------------------

    acf_lags = min(
        24,
        max(1, len(residuals) // 3)
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    plot_acf(
        residuals,
        lags=acf_lags,
        ax=ax
    )

    ax.set_title(
        f"{commodity} "
        f"ARIMA{order} Residual ACF"
    )

    plt.tight_layout()

    acf_plot_path = (
        output_dir
        / f"{safe_name}_arima_residual_acf.png"
    )

    plt.savefig(
        acf_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(
        f"Saved residual ACF plot: "
        f"{acf_plot_path}"
    )

    return {
        "commodity": commodity,
        "order": order,
        "ljung_box_lag": lag,
        "ljung_box_pvalue": p_value,
    }


# ============================================================
# EVALUATE ARIMA CANDIDATES
# ============================================================

def evaluate_arima_candidates(
    monthly,
    commodity,
    candidate_orders
):
    """
    Train and evaluate multiple ARIMA candidate models
    for one commodity.
    """

    train, test = create_train_test_split(
        monthly,
        commodity
    )

    print("\n" + "=" * 70)

    print(
        f"{commodity.upper()} "
        f"ARIMA MODEL EVALUATION"
    )

    print("=" * 70)

    print(
        f"\nTraining observations: "
        f"{len(train)}"
    )

    print(
        f"Testing observations: "
        f"{len(test)}"
    )

    print(
        f"\nTraining period: "
        f"{train['month'].min().strftime('%Y-%m')} "
        f"to "
        f"{train['month'].max().strftime('%Y-%m')}"
    )

    print(
        f"Testing period: "
        f"{test['month'].min().strftime('%Y-%m')} "
        f"to "
        f"{test['month'].max().strftime('%Y-%m')}"
    )

    results = []

    for order in candidate_orders:

        print("\n" + "-" * 70)

        print(
            f"ARIMA{order}"
        )

        print("-" * 70)

        try:

            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            model = train_arima_model(
                train,
                order
            )

            # ------------------------------------------------
            # Forecast
            # ------------------------------------------------

            forecast = forecast_model(
                model,
                test
            )

            # ------------------------------------------------
            # Evaluate
            # ------------------------------------------------

            mae, rmse, mape = (
                evaluate_forecast(
                    test,
                    forecast
                )
            )

            result = {
                "commodity": commodity,
                "order": str(order),
                "aic": model.aic,
                "bic": model.bic,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
            }

            results.append(result)

            print(
                f"AIC:  {model.aic:.2f}"
            )

            print(
                f"BIC:  {model.bic:.2f}"
            )

            print(
                f"MAE:  ₦{mae:,.2f}"
            )

            print(
                f"RMSE: ₦{rmse:,.2f}"
            )

            print(
                f"MAPE: {mape:.2f}%"
            )

        except Exception as error:

            print(
                f"Model ARIMA{order} failed:"
                f" {error}"
            )

    return pd.DataFrame(results)


# ============================================================
# DISPLAY TRAIN / TEST INFORMATION
# ============================================================

def display_split_information(
    monthly
):
    """
    Display training and testing periods
    for every selected commodity.
    """

    print("\n" + "=" * 70)

    print(
        "TIME-SERIES TRAIN / TEST SPLIT"
    )

    print("=" * 70)

    for commodity in SELECTED_COMMODITIES:

        try:

            train, test = (
                create_train_test_split(
                    monthly,
                    commodity
                )
            )

            d = DIFFERENCING_ORDERS[
                commodity
            ]

            print(
                f"\n{'-' * 70}"
            )

            print(
                f"{commodity}"
            )

            print(
                f"{'-' * 70}"
            )

            print(
                f"Differencing order (d): {d}"
            )

            print(
                f"\nTraining observations: "
                f"{len(train)}"
            )

            print(
                f"Training period: "
                f"{train['month'].min().strftime('%Y-%m')} "
                f"to "
                f"{train['month'].max().strftime('%Y-%m')}"
            )

            print(
                f"\nTesting observations: "
                f"{len(test)}"
            )

            print(
                f"Testing period: "
                f"{test['month'].min().strftime('%Y-%m')} "
                f"to "
                f"{test['month'].max().strftime('%Y-%m')}"
            )

        except Exception as error:

            print(
                f"\n{commodity}: "
                f"Could not create split."
            )

            print(
                f"Reason: {error}"
            )


# ============================================================
# INSPECT RICE RESIDUALS
# Kept because it is useful for our Rice investigation.
# ============================================================

def inspect_rice_residuals(
    model
):
    """
    Identify the largest Rice ARIMA residuals.
    """

    residuals = (
        model.resid
        .dropna()
    )

    residual_table = pd.DataFrame(
        {
            "date": residuals.index,
            "residual": residuals.values,
            "absolute_residual": (
                residuals.abs().values
            ),
        }
    )

    residual_table = (
        residual_table
        .sort_values(
            "absolute_residual",
            ascending=False
        )
    )

    print("\n" + "=" * 70)

    print(
        "LARGEST RICE ARIMA RESIDUALS"
    )

    print("=" * 70)

    print(
        residual_table
        .head(10)
        .to_string(index=False)
    )


# ============================================================
# INSPECT RICE PRICE PERIODS
# ============================================================

def inspect_rice_price_periods(
    monthly
):
    """
    Inspect Rice prices around dates
    with large residuals.
    """

    rice = (
        monthly[
            monthly["commodity"]
            == "Rice (local)"
        ]
        .copy()
        .sort_values("month")
    )

    dates_to_check = [
        "2016-01-01",
        "2016-02-01",
        "2016-03-01",
        "2017-07-01",
        "2018-01-01",
        "2018-06-01",
        "2024-01-01",
        "2024-02-01",
    ]

    print("\n" + "=" * 70)

    print(
        "RICE PRICES AROUND LARGE RESIDUALS"
    )

    print("=" * 70)

    for date_string in dates_to_check:

        date = pd.Timestamp(
            date_string
        )

        start = (
            date
            - pd.DateOffset(months=1)
        )

        end = (
            date
            + pd.DateOffset(months=1)
        )

        period = rice[
            (rice["month"] >= start)
            & (rice["month"] <= end)
        ]

        print("\n" + "-" * 70)

        print(
            f"Around {date.strftime('%Y-%m')}"
        )

        print("-" * 70)

        print(
            period[
                ["month", "price", "unit"]
            ].to_string(index=False)
        )


# ============================================================
# INSPECT RAW RICE DATA FOR ONE MONTH
# ============================================================

def inspect_rice_raw_period(
    date_string
):
    """
    Inspect original WFP Rice observations
    for a specific month.
    """

    df = load_data()

    df["date"] = pd.to_datetime(
        df["date"]
    )

    target_date = pd.Timestamp(
        date_string
    )

    rice = df[
        (df["commodity"] == "Rice (local)")
        & (df["pricetype"] == "Retail")
        & (
            df["date"]
            .dt
            .to_period("M")
            == target_date.to_period("M")
        )
    ].copy()

    print("\n" + "=" * 70)

    print(
        f"RAW RICE OBSERVATIONS: "
        f"{date_string}"
    )

    print("=" * 70)

    print(
        f"\nNumber of observations: "
        f"{len(rice)}"
    )

    if rice.empty:

        print(
            "No Rice observations found "
            "for this month."
        )

        return

    print("\nPrice statistics:")

    print(
        rice["price"].describe()
    )

    print("\nMarket prices:")

    print(
        rice[
            [
                "date",
                "admin1",
                "market",
                "unit",
                "price",
            ]
        ]
        .sort_values("price")
        .to_string(index=False)
    )


# ============================================================
# INSPECT RAW RICE DATA ACROSS PERIOD
# ============================================================

def inspect_rice_raw_periods(
    start_date,
    end_date
):
    """
    Inspect original WFP Rice observations
    across a date range.
    """

    df = load_data()

    df["date"] = pd.to_datetime(
        df["date"]
    )

    start = pd.Timestamp(
        start_date
    )

    end = pd.Timestamp(
        end_date
    )

    rice = df[
        (df["commodity"] == "Rice (local)")
        & (df["pricetype"] == "Retail")
        & (df["date"] >= start)
        & (df["date"] <= end)
    ].copy()

    print("\n" + "=" * 70)

    print(
        "RAW RICE OBSERVATIONS"
    )

    print(
        f"{start_date} to {end_date}"
    )

    print("=" * 70)

    if rice.empty:

        print(
            "No observations found."
        )

        return

    for month, group in rice.groupby(
        rice["date"].dt.to_period("M")
    ):

        print("\n" + "-" * 70)

        print(
            f"{month}"
        )

        print("-" * 70)

        print(
            f"Number of observations: "
            f"{len(group)}"
        )

        print(
            group[
                [
                    "date",
                    "admin1",
                    "market",
                    "unit",
                    "price",
                ]
            ]
            .sort_values("price")
            .to_string(index=False)
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Prepare monthly data
    # --------------------------------------------------------

    monthly = prepare_monthly_data()

    print("\n" + "=" * 70)

    print(
        "STARTING ARIMA TRAINING PREPARATION"
    )

    print("=" * 70)

    print(
        f"\nMonthly observations: "
        f"{len(monthly)}"
    )

    # --------------------------------------------------------
    # Show train/test split
    # --------------------------------------------------------

    display_split_information(
        monthly
    )

    # ========================================================
    # TOMATOES MODEL EVALUATION
    # ========================================================

    results = evaluate_arima_candidates(
    monthly,
    "Tomatoes",
    ARIMA_CANDIDATES["Tomatoes"]
    )

    print("\n" + "=" * 70)
    print("TOMATOES MODEL COMPARISON")
    print("=" * 70)

    if results.empty:

        print(
            "No Tomatoes models were successfully evaluated."
        )

        return

    print(
        results.to_string(
            index=False
        )
    )

    # ========================================================
    # SELECT CURRENT BEST TOMATOES CANDIDATE
    # ========================================================
    #
    # Based on the current holdout results:
    # ARIMA(1,1,0) has the lowest MAE/RMSE/MAPE.
    #
    # We still perform residual diagnostics before treating
    # it as the final selected model.
    # ========================================================

    best_order = (1, 1, 0)

    train, test = create_train_test_split(
    monthly,
    "Tomatoes"
    )

    best_model = train_arima_model(
        train,
        best_order
    )

    # --------------------------------------------------------
    # Tomatoes residual diagnostics
    # --------------------------------------------------------

    diagnose_model(
    best_model,
    "Tomatoes",
    best_order
   )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
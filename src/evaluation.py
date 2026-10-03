from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox

from preprocessing import (
    load_and_filter_data,
    prepare_data,
    create_monthly_series,
)
from stationarity import get_longest_continuous_segment

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

EVALUATION_FILE = REPORTS_DIR / "model_evaluation.csv"


SELECTED_COMMODITIES = [
    "Rice (local)",
    "Beans (red)",
    "Yam",
    "Gari (white)",
    "Oil (palm)",
    "Tomatoes",
]


FINAL_ARIMA_ORDERS = {
    "Rice (local)": (1, 1, 0),
    "Beans (red)": (1, 1, 1),
    "Yam": (0, 2, 1),
    "Gari (white)": (0, 2, 1),
    "Oil (palm)": (1, 1, 1),
    "Tomatoes": (1, 1, 0),
}


# The same training-segment strategy used during
# final model training.

TRAINING_SEGMENT = {
    "Rice (local)": "full",
    "Beans (red)": "latest",
    "Yam": "full",
    "Gari (white)": "latest",
    "Oil (palm)": "full",
    "Tomatoes": "latest",
}


TEST_RATIO = 0.20


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_train_test_split(
    series,
    test_size=0.20
):
    """
    Create the same chronological train/test split
    used during ARIMA model selection.

    The longest continuous monthly segment is selected
    first. The final 20% of that segment is used as the
    test set.
    """

    continuous = get_longest_continuous_segment(
        series
    )

    if continuous.empty:
        return None, None

    test_count = max(
        1,
        int(round(len(continuous) * test_size))
    )

    train = continuous.iloc[:-test_count].copy()

    test = continuous.iloc[-test_count:].copy()

    if train.empty or test.empty:
        return None, None

    return train, test

def calculate_mape(actual, predicted):
    """
    Calculate Mean Absolute Percentage Error.

    Zero actual values are excluded because percentage
    error cannot be calculated for zero.
    """

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mask = actual != 0

    if not mask.any():
        return np.nan

    return np.mean(
        np.abs(
            (actual[mask] - predicted[mask])
            / actual[mask]
        )
    ) * 100


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_models(monthly):
    """
    Evaluate the final ARIMA models using a chronological
    80/20 train-test split.

    The test set is never used when fitting the model.
    """

    results = []

    print("\n" + "=" * 70)
    print("FINAL ARIMA MODEL EVALUATION")
    print("=" * 70)

    for commodity in SELECTED_COMMODITIES:

        print(f"\nEvaluating: {commodity}")

        series = monthly[
            monthly["commodity"] == commodity
        ].copy()

        series = series.sort_values("month")

        train_data, test_data = create_train_test_split(
            series,
            test_size=TEST_RATIO
        )

        if train_data is None or test_data is None:
            print(
            f"Could not create train/test split: "
            f"{commodity}"
        )
            continue

        train_data = train_data.sort_values(
        "month"
        ).copy()

        test_data = test_data.sort_values(
        "month"
        ).copy()

        train = train_data["price"].to_numpy()
        test = test_data["price"].to_numpy()

        train_months = train_data["month"]
        test_months = test_data["month"]

        if len(train) < 10 or len(test) < 1:
            print(
                f"Insufficient train/test data: "
                f"{commodity}"
            )
            continue

        order = FINAL_ARIMA_ORDERS[commodity]

        print(f"ARIMA order: {order}")
        print(f"Training observations: {len(train)}")
        print(f"Testing observations: {len(test)}")

        # ----------------------------------------------------
        # FIT MODEL ONLY ON TRAINING DATA
        # ----------------------------------------------------

        train_index = pd.date_range(
            start=train_months.iloc[0],
            periods=len(train),
            freq="MS"
        )

        train_series = pd.Series(
            train,
            index=train_index
        )

        model = ARIMA(
            train_series,
            order=order
        )

        fitted_model = model.fit()

        # ----------------------------------------------------
        # FORECAST THE TEST PERIOD
        # ----------------------------------------------------

        forecast = fitted_model.forecast(
            steps=len(test)
        )

        forecast = np.asarray(forecast)

        # ----------------------------------------------------
        # CALCULATE METRICS
        # ----------------------------------------------------

        mae = mean_absolute_error(
            test,
            forecast
        )

        rmse = np.sqrt(
            mean_squared_error(
                test,
                forecast
            )
        )

        mape = calculate_mape(
            test,
            forecast
        )

        # ----------------------------------------------------
        # LJUNG-BOX RESIDUAL TEST
        # ----------------------------------------------------

        residuals = fitted_model.resid

        # Use a conservative lag for short series.
        lag = min(
            12,
            max(1, len(residuals) // 4)
            )

        try:
            ljung_box = acorr_ljungbox(
                residuals,
                lags=[lag],
                return_df=True
            )

            ljung_box_pvalue = float(
                ljung_box["lb_pvalue"].iloc[0]
            )

        except Exception:
            ljung_box_pvalue = np.nan

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        results.append(
            {
                "commodity": commodity,
                "arima_order": str(order),
                "differencing_order": order[1],
                "training_observations": len(train),
                "testing_observations": len(test),
                "training_start": train_months.iloc[0],
                "training_end": train_months.iloc[-1],
                "testing_start": test_months.iloc[0],
                "testing_end": test_months.iloc[-1],
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "ljung_box_pvalue": ljung_box_pvalue,
            }
        )

        print(
            f"MAE: ₦{mae:,.2f}"
        )

        print(
            f"RMSE: ₦{rmse:,.2f}"
        )

        print(
            f"MAPE: {mape:.2f}%"
        )

        print(
            f"Ljung-Box p-value: "
            f"{ljung_box_pvalue:.4f}"
        )

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nLoading and filtering data...")

    df = load_and_filter_data()

    print("Preparing data...")

    df = prepare_data(df)

    print("Creating monthly series...")

    monthly = create_monthly_series(df)

    evaluation_results = evaluate_models(
        monthly
    )

    if evaluation_results.empty:
        print("\nNo evaluation results generated.")
        return

    evaluation_results.to_csv(
        EVALUATION_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("FINAL MODEL EVALUATION SUMMARY")
    print("=" * 70)

    print(
        evaluation_results[
            [
                "commodity",
                "arima_order",
                "mae",
                "rmse",
                "mape",
                "ljung_box_pvalue",
            ]
        ].to_string(index=False)
    )

    print("\nEvaluation report saved to:")

    print(EVALUATION_FILE)

    print("\n" + "=" * 70)
    print("MODEL EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
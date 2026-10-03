from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from statsmodels.stats.diagnostic import acorr_ljungbox

from preprocessing import (
    load_and_filter_data,
    prepare_data,
    create_monthly_series,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "reports" / "spikes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


SELECTED_COMMODITIES = [
    "Rice (local)",
    "Beans (red)",
    "Yam",
    "Gari (white)",
    "Oil (palm)",
    "Tomatoes",
]


ROBUST_Z_THRESHOLD = 3.5

# Number of previous observations used to understand
# the recent behavior of each commodity.
ROLLING_WINDOW = 12
MIN_ABSOLUTE_PERCENT_CHANGE = 10.0

# Final ARIMA model configurations
FINAL_ARIMA_ORDERS = {
    "Rice (local)": (1, 1, 0),
    "Beans (red)": (1, 1, 1),
    "Yam": (0, 2, 1),
    "Gari (white)": (0, 2, 1),
    "Oil (palm)": (1, 1, 1),
    "Tomatoes": (1, 1, 0),
}

# Training segment used for final forecasting models
TRAINING_SEGMENT = {
    "Rice (local)": "full",
    "Beans (red)": "latest",
    "Yam": "full",
    "Gari (white)": "latest",
    "Oil (palm)": "full",
    "Tomatoes": "latest",
}

def calculate_monthly_changes(monthly):
    """
    Calculate month-to-month percentage changes.

    A change is calculated only when the two observations
    belong to consecutive calendar months.
    """

    results = []

    for commodity in SELECTED_COMMODITIES:

        series = monthly[
            monthly["commodity"] == commodity
        ].copy()

        series = series.sort_values("month").reset_index(drop=True)

        month_number = (
            series["month"].dt.year * 12
            + series["month"].dt.month
        )

        previous_month_number = month_number.shift(1)

        consecutive = (
            month_number - previous_month_number == 1
        )

        series["price_change"] = series["price"].diff()

        series["pct_change"] = (
            series["price"].pct_change() * 100
        )

        # Do not calculate a change across missing months.
        series.loc[~consecutive, "price_change"] = np.nan
        series.loc[~consecutive, "pct_change"] = np.nan

        results.append(series)

    return pd.concat(results, ignore_index=True)


def calculate_rolling_robust_scores(changes):
    """
    Calculate rolling robust z-scores using:

        Median
        Median Absolute Deviation (MAD)

    The current observation is excluded from the rolling
    reference window so that the observation does not
    influence its own anomaly score.
    """

    results = []

    for commodity in SELECTED_COMMODITIES:

        series = changes[
            changes["commodity"] == commodity
        ].copy()

        series = series.sort_values("month").reset_index(drop=True)

        pct_change = series["pct_change"]

        # Rolling median of previous observations.
        rolling_median = (
            pct_change
            .shift(1)
            .rolling(
                window=ROLLING_WINDOW,
                min_periods=6,
            )
            .median()
        )

        # Calculate absolute deviation from the rolling median.
        absolute_deviation = (
            pct_change - rolling_median
        ).abs()

        # Rolling MAD.
        rolling_mad = (
            absolute_deviation
            .shift(1)
            .rolling(
                window=ROLLING_WINDOW,
                min_periods=6,
            )
            .median()
        )

        series["rolling_median"] = rolling_median
        series["rolling_mad"] = rolling_mad

        # Calculate robust z-score.
        series["robust_z"] = np.where(
            rolling_mad > 0,
            (
                0.6745
                * (
                    pct_change
                    - rolling_median
                )
                / rolling_mad
            ),
            np.nan,
        )

        series["spike"] = (
    (series["robust_z"].abs() >= ROBUST_Z_THRESHOLD)
    & (series["pct_change"].abs() >= MIN_ABSOLUTE_PERCENT_CHANGE)
)

        series["data_coverage"] = (
            series["observation_count"]
            .apply(classify_data_coverage)
        )

        series["direction"] = np.where(
            series["robust_z"] > 0,
            "positive",
            "negative",
        )

        results.append(series)

    return pd.concat(results, ignore_index=True)


def get_spike_events(scored_data):
    """
    Return observations identified as abnormal
    based on the rolling robust score.
    """

    spikes = scored_data[
        scored_data["spike"] == True
    ].copy()

    spikes = spikes.sort_values(
        ["commodity", "month"]
    )

    return spikes


def save_results(scored_data, spikes):
    """
    Save all calculations and detected spike events.
    """

    all_changes_path = (
        OUTPUT_DIR / "rolling_price_changes.csv"
    )

    spikes_path = (
        OUTPUT_DIR / "rolling_detected_spikes.csv"
    )

    scored_data.to_csv(
        all_changes_path,
        index=False,
    )

    spikes.to_csv(
        spikes_path,
        index=False,
    )

    print(
        f"\nSaved: {all_changes_path}"
    )

    print(
        f"Saved: {spikes_path}"
    )

def classify_data_coverage(observation_count):
    """
    Classify the amount of source data contributing
    to a monthly price.

    This is a simple data-coverage indicator,
    not a statistical confidence measure.
    """

    if observation_count >= 30:
        return "Good"

    elif observation_count >= 10:
        return "Moderate"

    else:
        return "Limited"

def display_spike_summary(spikes):

    print("\n" + "=" * 70)
    print("ROLLING SPIKE DETECTION SUMMARY")
    print("=" * 70)

    if spikes.empty:

        print(
            "\nNo abnormal price movements detected."
        )

        return

    for commodity in SELECTED_COMMODITIES:

        commodity_spikes = spikes[
            spikes["commodity"] == commodity
        ]

        print(f"\n{commodity}")
        print("-" * 50)

        if commodity_spikes.empty:

            print("No spikes detected.")

            continue

        print(
            f"Spikes detected: "
            f"{len(commodity_spikes)}"
        )

        for _, row in commodity_spikes.iterrows():

            print(
            f"{row['month'].strftime('%Y-%m')} | "
            f"Price: ₦{row['price']:,.2f} | "
            f"Change: {row['pct_change']:+.2f}% | "
            f"Robust Z: {row['robust_z']:+.2f} | "
            f"Observations: {int(row['observation_count'])} | "
            f"Coverage: {row['data_coverage']} | "
            f"{row['direction']}"
            )

def calculate_residual_anomalies(monthly):
    """
    Detect unusual observations using ARIMA model residuals.

    The residual analysis uses the same continuous data segment
    and ARIMA order used during final model training.

    Residual = actual price - fitted price.

    This function is diagnostic and does not modify the data.
    """

    print("\nCalculating ARIMA residual anomalies...")

    results = []

    for commodity in SELECTED_COMMODITIES:

        print(f"Processing: {commodity}")

        # --------------------------------------------------
        # GET COMMODITY DATA
        # --------------------------------------------------

        series = monthly[
            monthly["commodity"] == commodity
        ].copy()

        series = series.sort_values("month")

        # --------------------------------------------------
        # KEEP ONLY CONSECUTIVE MONTHS
        # --------------------------------------------------

        series["month_diff"] = (
            series["month"]
            .diff()
            .dt.days
        )

        series["segment_id"] = (
            series["month_diff"]
            .gt(45)
            .cumsum()
        )

        segments = [
            group.copy()
            for _, group in series.groupby("segment_id")
        ]

        # Remove very short segments
        segments = [
            segment
            for segment in segments
            if len(segment) >= 10
        ]

        if not segments:
            print(
                f"No suitable continuous segment: "
                f"{commodity}"
            )
            continue

        # --------------------------------------------------
        # SELECT FINAL TRAINING SEGMENT
        # --------------------------------------------------

        if TRAINING_SEGMENT[commodity] == "latest":

            modeling_data = max(
                segments,
                key=lambda x: x["month"].max()
            )

        else:

            modeling_data = max(
                segments,
                key=len
            )

        modeling_data = modeling_data.copy()

        modeling_data = modeling_data.sort_values(
            "month"
        )

        # --------------------------------------------------
        # GET FINAL ARIMA ORDER
        # --------------------------------------------------

        order = FINAL_ARIMA_ORDERS[commodity]

        # --------------------------------------------------
# FIT ARIMA MODEL
# --------------------------------------------------

        from statsmodels.tsa.arima.model import ARIMA

# Use a proper monthly DatetimeIndex
        modeling_data = modeling_data.copy()

        modeling_data["month"] = pd.to_datetime(
    modeling_data["month"]
        )

        modeling_data = modeling_data.set_index(
    "month"
        )

        modeling_data = modeling_data.asfreq("MS")

        model = ARIMA(
        modeling_data["price"],
        order=order
        )

        fitted_model = model.fit()

        # --------------------------------------------------
        # GET FITTED VALUES
        # --------------------------------------------------

        fitted_values = fitted_model.fittedvalues

        modeling_data["fitted_price"] = fitted_values

# Put month back as a normal column
        modeling_data = modeling_data.reset_index()

        # --------------------------------------------------
        # REMOVE INVALID ARIMA INITIALIZATION VALUES
        # --------------------------------------------------

        modeling_data.loc[
            ~np.isfinite(
                modeling_data["fitted_price"]
            ),
            "fitted_price"
        ] = np.nan

        modeling_data.loc[
            modeling_data["fitted_price"] <= 0,
            "fitted_price"
        ] = np.nan

        # --------------------------------------------------
        # CALCULATE RESIDUALS
        # --------------------------------------------------

        modeling_data["residual"] = (
            modeling_data["price"]
            - modeling_data["fitted_price"]
        )

        # Remove rows where fitted value/residual is invalid
        modeling_data = modeling_data.dropna(
            subset=[
                "fitted_price",
                "residual"
            ]
        )

        if modeling_data.empty:
            continue

        # --------------------------------------------------
        # ROBUST RESIDUAL SCORE
        # --------------------------------------------------

        median_residual = (
            modeling_data["residual"].median()
        )

        absolute_deviation = (
            modeling_data["residual"]
            - median_residual
        ).abs()

        mad = absolute_deviation.median()

        if mad == 0 or pd.isna(mad):

            modeling_data["robust_z"] = np.nan

        else:

            modeling_data["robust_z"] = (
                0.6745
                * (
                    modeling_data["residual"]
                    - median_residual
                )
                / mad
            )

        # --------------------------------------------------
        # FLAG RESIDUAL ANOMALIES
        # --------------------------------------------------

        modeling_data["residual_anomaly"] = (
            modeling_data["robust_z"].abs()
            >= ROBUST_Z_THRESHOLD
        )

        # --------------------------------------------------
        # DIRECTION
        # --------------------------------------------------

        modeling_data["direction"] = np.where(
            modeling_data["residual"] > 0,
            "above model expectation",
            "below model expectation"
        )

        # --------------------------------------------------
        # STORE RESULTS
        # --------------------------------------------------

        results.append(
            modeling_data[
                [
                    "month",
                    "commodity",
                    "unit",
                    "price",
                    "observation_count",
                    "fitted_price",
                    "residual",
                    "robust_z",
                    "residual_anomaly",
                    "direction",
                ]
            ]
        )

    # ------------------------------------------------------
    # COMBINE RESULTS
    # ------------------------------------------------------

    if not results:
        return pd.DataFrame()

    return pd.concat(
        results,
        ignore_index=True
    )
def display_residual_anomalies(results):
    """
    Display ARIMA residual anomalies.
    """

    print("\n")
    print("=" * 70)
    print("ARIMA RESIDUAL ANOMALY SUMMARY")
    print("=" * 70)

    for commodity in SELECTED_COMMODITIES:

        commodity_results = results[
            results["commodity"] == commodity
        ]

        anomalies = commodity_results[
            commodity_results["residual_anomaly"]
        ]

        print(f"\n{commodity}")
        print("-" * 50)
        print(
            f"Residual anomalies detected: "
            f"{len(anomalies)}"
        )

        for _, row in anomalies.iterrows():

            print(
                f"{row['month'].strftime('%Y-%m')} | "
                f"Actual: ₦{row['price']:,.2f} | "
                f"Fitted: ₦{row['fitted_price']:,.2f} | "
                f"Residual: ₦{row['residual']:,.2f} | "
                f"Robust Z: {row['robust_z']:+.2f} | "
                f"Observations: "
                f"{int(row['observation_count'])} | "
                f"{row['direction']}"
            )

def display_largest_residuals(results, top_n=10):
    """
    Display the observations with the largest absolute
    ARIMA residuals for each commodity.

    This is diagnostic only. It does not classify them
    as anomalies.
    """

    print("\n")
    print("=" * 70)
    print("LARGEST ARIMA RESIDUALS")
    print("=" * 70)

    for commodity in SELECTED_COMMODITIES:

        commodity_results = results[
            results["commodity"] == commodity
        ].copy()

        commodity_results["abs_residual"] = (
            commodity_results["residual"].abs()
        )

        largest = (
            commodity_results
            .sort_values(
                "abs_residual",
                ascending=False
            )
            .head(top_n)
        )

        print(f"\n{commodity}")
        print("-" * 50)

        for _, row in largest.iterrows():

            print(
                f"{row['month'].strftime('%Y-%m')} | "
                f"Actual: ₦{row['price']:,.2f} | "
                f"Fitted: ₦{row['fitted_price']:,.2f} | "
                f"Residual: ₦{row['residual']:+,.2f} | "
                f"Robust Z: {row['robust_z']:+.2f} | "
                f"Observations: "
                f"{int(row['observation_count'])}"
            )


def save_residual_results(results):
    """
    Save ARIMA residual analysis results.
    """

    output_path = (
        OUTPUT_DIR
        / "arima_residual_anomalies.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved residual results: {output_path}"
    )

def compare_detection_methods():
    """
    Compare anomalies detected by:
    1. Rolling robust spike detection
    2. ARIMA residual detection

    Only observations actually flagged as anomalies
    are included in the comparison.
    """

    print("\n" + "=" * 70)
    print("COMPARING ANOMALY DETECTION METHODS")
    print("=" * 70)

    # --------------------------------------------------
    # FILE PATHS
    # --------------------------------------------------

    rolling_path = (
        PROJECT_ROOT
        / "reports"
        / "spikes"
        / "rolling_detected_spikes.csv"
    )

    residual_path = (
        PROJECT_ROOT
        / "reports"
        / "spikes"
        / "arima_residual_anomalies.csv"
    )

    # --------------------------------------------------
    # LOAD ROLLING SPIKE RESULTS
    # --------------------------------------------------

    rolling = pd.read_csv(
        rolling_path,
        parse_dates=["month"]
    )

    # The rolling file should already contain
    # only detected spikes.

    rolling = rolling[
        [
            "month",
            "commodity",
            "unit",
            "price",
            "observation_count",
            "data_coverage",
            "pct_change",
            "robust_z",
            "direction",
        ]
    ].copy()

    rolling = rolling.rename(
        columns={
            "price": "rolling_price",
            "observation_count": "rolling_observations",
            "data_coverage": "rolling_coverage",
            "robust_z": "rolling_robust_z",
            "direction": "rolling_direction",
        }
    )

    rolling["rolling_detected"] = True

    # --------------------------------------------------
    # LOAD ARIMA RESIDUAL RESULTS
    # --------------------------------------------------

    residual = pd.read_csv(
        residual_path,
        parse_dates=["month"]
    )

    # IMPORTANT:
    # The residual CSV contains all modeled observations.
    # Keep ONLY observations actually flagged as anomalies.

    residual = residual[
        residual["residual_anomaly"] == True
    ].copy()

    residual = residual[
        [
            "month",
            "commodity",
            "unit",
            "price",
            "observation_count",
            "fitted_price",
            "residual",
            "robust_z",
            "direction",
        ]
    ].copy()

    residual = residual.rename(
        columns={
            "price": "residual_price",
            "observation_count": "residual_observations",
            "robust_z": "residual_robust_z",
            "direction": "residual_direction",
        }
    )

    residual["residual_detected"] = True

    # --------------------------------------------------
    # MERGE THE TWO DETECTORS
    # --------------------------------------------------

    comparison = pd.merge(
        rolling,
        residual,
        on=[
            "month",
            "commodity",
            "unit",
        ],
        how="outer",
    )

    # --------------------------------------------------
    # COMBINE COMMON INFORMATION
    # --------------------------------------------------

    comparison["actual_price"] = (
        comparison["rolling_price"]
        .combine_first(
            comparison["residual_price"]
        )
    )

    comparison["observation_count"] = (
        comparison["rolling_observations"]
        .combine_first(
            comparison["residual_observations"]
        )
    )

    comparison["data_coverage"] = (
        comparison["rolling_coverage"]
        .combine_first(
            comparison["residual_observations"]
            .apply(classify_data_coverage)
        )
    )

    # --------------------------------------------------
    # CLEAN DETECTION FLAGS
    # --------------------------------------------------

    comparison["rolling_detected"] = (
        comparison["rolling_detected"]
        .fillna(False)
        .astype(bool)
    )

    comparison["residual_detected"] = (
        comparison["residual_detected"]
        .fillna(False)
        .astype(bool)
    )

    # --------------------------------------------------
    # CLASSIFY EACH EVENT
    # --------------------------------------------------

    both_detected = (
        comparison["rolling_detected"].to_numpy(dtype=bool)
        &
        comparison["residual_detected"].to_numpy(dtype=bool)
    )

    rolling_only = (
        comparison["rolling_detected"].to_numpy(dtype=bool)
        &
        ~comparison["residual_detected"].to_numpy(dtype=bool)
    )

    residual_only = (
        ~comparison["rolling_detected"].to_numpy(dtype=bool)
        &
        comparison["residual_detected"].to_numpy(dtype=bool)
    )

    comparison["detection_classification"] = np.select(
        [
            both_detected,
            rolling_only,
            residual_only,
        ],
        [
            "Both methods",
            "Rolling method only",
            "ARIMA residual method only",
        ],
        default="Unknown",
    )

    # --------------------------------------------------
    # SORT
    # --------------------------------------------------

    comparison = comparison.sort_values(
        [
            "commodity",
            "month",
        ]
    )

    # --------------------------------------------------
    # FINAL COLUMNS
    # --------------------------------------------------

    comparison = comparison[
        [
            "month",
            "commodity",
            "unit",
            "actual_price",
            "observation_count",
            "data_coverage",

            "rolling_detected",
            "pct_change",
            "rolling_robust_z",
            "rolling_direction",

            "residual_detected",
            "fitted_price",
            "residual",
            "residual_robust_z",
            "residual_direction",

            "detection_classification",
        ]
    ]

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "spikes"
        / "anomaly_detection_comparison.csv"
    )

    comparison.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("ANOMALY DETECTION COMPARISON SUMMARY")
    print("=" * 70)

    print(
        f"\nTotal unique detected events: "
        f"{len(comparison)}"
    )

    print("\nDetection classification:")

    print(
        comparison[
            "detection_classification"
        ].value_counts()
    )

    print("\nEvents by commodity:")

    print(
        comparison[
            "commodity"
        ].value_counts()
    )

    print(
        f"\nSaved comparison file:\n"
        f"{output_path}"
    )

    return comparison

def main():

    print("\n" + "=" * 70)
    print("STARTING ROLLING SPIKE DETECTION")
    print("=" * 70)

    print("\nLoading data...")

    df = load_and_filter_data()

    df = prepare_data(df)

    monthly = create_monthly_series(df)

    print(
        f"Monthly observations: "
        f"{len(monthly):,}"
    )

    print(
        "\nCalculating month-to-month price changes..."
    )

    changes = calculate_monthly_changes(monthly)

    print(
        "\nCalculating rolling robust anomaly scores..."
    )

    scored_data = calculate_rolling_robust_scores(
        changes
    )

    spikes = get_spike_events(
        scored_data
    )

    display_spike_summary(
        spikes
    )

    save_results(
        scored_data,
        spikes
    )

        # --------------------------------------------------
    # ARIMA RESIDUAL ANOMALY DETECTION
    # --------------------------------------------------

    residual_results = calculate_residual_anomalies(
        monthly
    )

    display_residual_anomalies(
        residual_results
    )

    display_largest_residuals(
        residual_results
    )
    save_residual_results(
        residual_results
    )

    compare_detection_methods()

    print("\n" + "=" * 70)
    print("ROLLING SPIKE DETECTION COMPLETE")
    print("=" * 70)

    

if __name__ == "__main__":

    main()
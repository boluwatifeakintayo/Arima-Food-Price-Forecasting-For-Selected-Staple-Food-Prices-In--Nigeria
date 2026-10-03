from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

from preprocessing import (
    load_and_filter_data,
    prepare_data,
    create_monthly_series,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STATIONARITY_FIGURES_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "eda"
    / "stationarity"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

def create_output_directory():
    """Create the stationarity output directory."""

    STATIONARITY_FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Stationarity figures directory ready: "
        f"{STATIONARITY_FIGURES_DIR}"
    )


# ============================================================
# LOAD MONTHLY DATA
# ============================================================

def load_monthly_data():
    """Load and prepare the monthly commodity data."""

    print("\n=== LOADING DATA FOR STATIONARITY ANALYSIS ===")

    df = load_and_filter_data()
    df = prepare_data(df)
    monthly = create_monthly_series(df)

    return monthly


# ============================================================
# FIND LONGEST CONTINUOUS SEGMENT
# ============================================================

def get_longest_continuous_segment(series):
    """
    Find the longest continuous monthly segment.

    A continuous segment means every observation is exactly
    one month after the previous observation.

    This prevents missing months from being treated as if
    they were normal consecutive observations.
    """

    series = series.sort_values("month").copy()

    # Calculate a numeric month index.
    # This makes it easy to identify consecutive months.
    month_number = (
        series["month"].dt.year * 12
        + series["month"].dt.month
    )

    previous_month_number = month_number.shift(1)

    # True when the current observation follows
    # the previous observation by exactly one month.
    consecutive = (
        month_number - previous_month_number == 1
    )

    # Create a new segment whenever there is a gap.
    segment_id = (~consecutive).cumsum()

    series["segment_id"] = segment_id

    # Find the segment containing the most observations.
    segment_sizes = (
        series.groupby("segment_id")
        .size()
    )

    longest_segment_id = segment_sizes.idxmax()

    longest_segment = (
        series[
            series["segment_id"] == longest_segment_id
        ]
        .copy()
        .sort_values("month")
    )

    longest_segment = longest_segment.drop(
        columns=["segment_id"]
    )

    return longest_segment


# ============================================================
# ADF TEST
# ============================================================

def run_adf_test(series):
    """
    Run the Augmented Dickey-Fuller test.

    H0:
        The series has a unit root and is non-stationary.

    If p-value < 0.05:
        Reject H0.
        Evidence suggests the series is stationary.
    """

    series = series.dropna()

    result = adfuller(
        series,
        autolag="AIC"
    )

    return {
        "test_statistic": result[0],
        "p_value": result[1],
        "lags_used": result[2],
        "observations": result[3],
    }


# ============================================================
# KPSS TEST
# ============================================================

def run_kpss_test(series):
    """
    Run the KPSS test.

    H0:
        The series is stationary.

    If p-value < 0.05:
        Reject H0.
        Evidence suggests the series is non-stationary.
    """

    series = series.dropna()

    try:

        result = kpss(
            series,
            regression="c",
            nlags="auto"
        )

        return {
            "test_statistic": result[0],
            "p_value": result[1],
            "lags_used": result[2],
        }

    except ValueError:

        return {
            "test_statistic": None,
            "p_value": None,
            "lags_used": None,
        }


# ============================================================
# INTERPRET STATIONARITY
# ============================================================

def interpret_stationarity(
    adf_p_value,
    kpss_p_value
):
    """
    Interpret ADF and KPSS results together.
    """

    if (
        adf_p_value < 0.05
        and kpss_p_value >= 0.05
    ):
        return "Likely stationary"

    elif (
        adf_p_value >= 0.05
        and kpss_p_value < 0.05
    ):
        return "Likely non-stationary"

    elif (
        adf_p_value < 0.05
        and kpss_p_value < 0.05
    ):
        return "Mixed evidence - investigate further"

    else:
        return "Mixed evidence - investigate further"


# ============================================================
# ANALYZE ORIGINAL SERIES
# ============================================================

def analyze_original_series(monthly):
    """
    Run ADF and KPSS on the longest continuous original
    monthly price series for each commodity.
    """

    print(
        "\n=== ANALYZING ORIGINAL SERIES ==="
    )

    results = []

    for commodity in monthly["commodity"].unique():

        series = monthly[
            monthly["commodity"] == commodity
        ].copy()

        series = series.sort_values("month")

        # Use the longest continuous segment.
        continuous_series = get_longest_continuous_segment(
            series
        )

        price_series = continuous_series[
            "price"
        ].dropna()

        adf_result = run_adf_test(
            price_series
        )

        kpss_result = run_kpss_test(
            price_series
        )

        interpretation = interpret_stationarity(
            adf_result["p_value"],
            kpss_result["p_value"]
        )

        results.append(
            {
                "commodity": commodity,

                "segment_start":
                    continuous_series["month"].min(),

                "segment_end":
                    continuous_series["month"].max(),

                "segment_observations":
                    len(continuous_series),

                "adf_statistic":
                    adf_result["test_statistic"],

                "adf_p_value":
                    adf_result["p_value"],

                "adf_lags":
                    adf_result["lags_used"],

                "adf_observations":
                    adf_result["observations"],

                "kpss_statistic":
                    kpss_result["test_statistic"],

                "kpss_p_value":
                    kpss_result["p_value"],

                "kpss_lags":
                    kpss_result["lags_used"],

                "interpretation":
                    interpretation,
            }
        )

        print(
            f"\n{commodity}"
        )

        print(
            f"Continuous segment: "
            f"{continuous_series['month'].min().strftime('%Y-%m')} "
            f"to "
            f"{continuous_series['month'].max().strftime('%Y-%m')}"
        )

        print(
            f"Observations: "
            f"{len(continuous_series)}"
        )

        print(
            f"ADF statistic: "
            f"{adf_result['test_statistic']:.4f}"
        )

        print(
            f"ADF p-value: "
            f"{adf_result['p_value']:.4f}"
        )

        print(
            f"KPSS statistic: "
            f"{kpss_result['test_statistic']:.4f}"
        )

        if kpss_result["p_value"] is not None:

            print(
                f"KPSS p-value: "
                f"{kpss_result['p_value']:.4f}"
            )

        else:

            print(
                "KPSS p-value: unavailable"
            )

        print(
            f"Interpretation: "
            f"{interpretation}"
        )

    results_df = pd.DataFrame(results)

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "stationarity_original_series.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved original stationarity results: "
        f"{output_path}"
    )

    return results_df


# ============================================================
# FIRST DIFFERENCE
# ============================================================

def create_first_difference(monthly):
    """
    Create first differences only within continuous monthly
    segments.

    A difference is calculated only when the current month
    immediately follows the previous month.

    This prevents missing-month gaps from being treated as
    one-period changes.
    """

    print(
        "\n=== CREATING FIRST DIFFERENCES ==="
    )

    differenced_data = []

    for commodity in monthly["commodity"].unique():

        series = monthly[
            monthly["commodity"] == commodity
        ].copy()

        series = series.sort_values("month")

        # Identify consecutive months.
        month_number = (
            series["month"].dt.year * 12
            + series["month"].dt.month
        )

        previous_month_number = (
            month_number.shift(1)
        )

        consecutive = (
            month_number - previous_month_number == 1
        )

        # Calculate the normal difference.
        series["first_difference"] = (
            series["price"].diff()
        )

        # Remove differences across missing months.
        series.loc[
            ~consecutive,
            "first_difference"
        ] = pd.NA

        differenced_data.append(
            series
        )

    differenced = pd.concat(
        differenced_data,
        ignore_index=True
    )

    return differenced


# ============================================================
# ANALYZE FIRST DIFFERENCE
# ============================================================

def analyze_first_difference(differenced):
    """
    Run ADF and KPSS on the first-differenced values.

    The tests are performed on the first differences belonging
    to the longest continuous segment of each commodity.
    """

    print(
        "\n=== ANALYZING FIRST-DIFFERENCED SERIES ==="
    )

    results = []

    for commodity in differenced[
        "commodity"
    ].unique():

        series = differenced[
            differenced["commodity"] == commodity
        ].copy()

        series = series.sort_values("month")

        # Find the longest continuous segment
        # from the original monthly data.
        continuous_series = (
            get_longest_continuous_segment(series)
        )

        difference_series = continuous_series[
            "first_difference"
        ].dropna()

        adf_result = run_adf_test(
            difference_series
        )

        kpss_result = run_kpss_test(
            difference_series
        )

        interpretation = interpret_stationarity(
            adf_result["p_value"],
            kpss_result["p_value"]
        )

        results.append(
            {
                "commodity": commodity,

                "segment_start":
                    continuous_series["month"].min(),

                "segment_end":
                    continuous_series["month"].max(),

                "segment_observations":
                    len(continuous_series),

                "difference_observations":
                    len(difference_series),

                "adf_statistic":
                    adf_result["test_statistic"],

                "adf_p_value":
                    adf_result["p_value"],

                "adf_lags":
                    adf_result["lags_used"],

                "adf_observations":
                    adf_result["observations"],

                "kpss_statistic":
                    kpss_result["test_statistic"],

                "kpss_p_value":
                    kpss_result["p_value"],

                "kpss_lags":
                    kpss_result["lags_used"],

                "interpretation":
                    interpretation,
            }
        )

        print(
            f"\n{commodity}"
        )

        print(
            f"Continuous segment: "
            f"{continuous_series['month'].min().strftime('%Y-%m')} "
            f"to "
            f"{continuous_series['month'].max().strftime('%Y-%m')}"
        )

        print(
            f"Difference observations: "
            f"{len(difference_series)}"
        )

        print(
            f"ADF statistic: "
            f"{adf_result['test_statistic']:.4f}"
        )

        print(
            f"ADF p-value: "
            f"{adf_result['p_value']:.4f}"
        )

        print(
            f"KPSS statistic: "
            f"{kpss_result['test_statistic']:.4f}"
        )

        if kpss_result["p_value"] is not None:

            print(
                f"KPSS p-value: "
                f"{kpss_result['p_value']:.4f}"
            )

        else:

            print(
                "KPSS p-value: unavailable"
            )

        print(
            f"Interpretation: "
            f"{interpretation}"
        )

    results_df = pd.DataFrame(results)

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "stationarity_first_difference.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved first-difference stationarity results: "
        f"{output_path}"
    )

    return results_df

def analyze_second_difference(monthly):
    """
    Test the second difference of Gari and Yam.

    Only consecutive months from the longest continuous segment
    are used, so missing months do not contaminate the differences.
    """

    target_commodities = ["Gari (white)", "Yam"]

    print("\n" + "=" * 70)
    print("SECOND DIFFERENCE ANALYSIS")
    print("=" * 70)

    results = []

    for commodity in target_commodities:

        series = (
            monthly[monthly["commodity"] == commodity]
            [["month", "price"]]
            .sort_values("month")
            .copy()
        )

        # Get the longest continuous monthly segment
        continuous = get_longest_continuous_segment(series)

        if len(continuous) < 5:
            print(f"\n{commodity}: Not enough observations.")
            continue

        # First difference
        first_diff = continuous["price"].diff()

        # Second difference
        second_diff = first_diff.diff().dropna()

        print(f"\n--- {commodity} ---")
        print(
            f"Continuous segment: "
            f"{continuous['month'].min().strftime('%Y-%m')} "
            f"to "
            f"{continuous['month'].max().strftime('%Y-%m')}"
        )

        print(f"Original observations: {len(continuous)}")
        print(f"First differences: {len(first_diff.dropna())}")
        print(f"Second differences: {len(second_diff)}")

        # ADF test
        adf_stat, adf_pvalue, *_ = adfuller(
            second_diff,
            autolag="AIC"
        )

        # KPSS test
        kpss_stat, kpss_pvalue, *_ = kpss(
            second_diff,
            regression="c",
            nlags="auto"
        )

        print("\nADF Test:")
        print(f"Statistic: {adf_stat:.4f}")
        print(f"p-value: {adf_pvalue:.4f}")

        print("\nKPSS Test:")
        print(f"Statistic: {kpss_stat:.4f}")
        print(f"p-value: {kpss_pvalue:.4f}")

        # Interpretation
        adf_stationary = adf_pvalue < 0.05
        kpss_stationary = kpss_pvalue > 0.05

        if adf_stationary and kpss_stationary:
            interpretation = "Likely stationary"
        elif not adf_stationary and not kpss_stationary:
            interpretation = "Likely non-stationary"
        else:
            interpretation = "Mixed evidence"

        print(f"\nInterpretation: {interpretation}")

        results.append({
            "commodity": commodity,
            "observations": len(second_diff),
            "adf_statistic": adf_stat,
            "adf_pvalue": adf_pvalue,
            "kpss_statistic": kpss_stat,
            "kpss_pvalue": kpss_pvalue,
            "interpretation": interpretation
        })

    return pd.DataFrame(results)


# ============================================================
# PLOT ORIGINAL VS DIFFERENCED SERIES
# ============================================================

def plot_stationarity_comparison(
    monthly,
    differenced
):
    """
    Plot the original price series and its first difference
    for each commodity.
    """

    print(
        "\n=== CREATING STATIONARITY COMPARISON PLOTS ==="
    )

    for commodity in monthly["commodity"].unique():

        original = monthly[
            monthly["commodity"] == commodity
        ].copy()

        original = original.sort_values("month")

        difference = differenced[
            differenced["commodity"] == commodity
        ].copy()

        difference = difference.sort_values("month")

        # ====================================================
        # ORIGINAL SERIES
        # ====================================================

        plt.figure(figsize=(12, 5))

        plt.plot(
            original["month"],
            original["price"],
            marker="o",
            linewidth=1
        )

        plt.title(
            f"Original Price Series - {commodity}"
        )

        plt.xlabel("Month")

        plt.ylabel(
            "Price (NGN)"
        )

        plt.xticks(
            rotation=45
        )

        plt.tight_layout()

        filename = (
            commodity
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )

        output_path = (
            STATIONARITY_FIGURES_DIR
            / f"{filename}_original_series.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )

        # ====================================================
        # FIRST DIFFERENCE
        # ====================================================

        plt.figure(figsize=(12, 5))

        plt.plot(
            difference["month"],
            difference["first_difference"],
            marker="o",
            linewidth=1
        )

        plt.axhline(
            0,
            linewidth=1
        )

        plt.title(
            f"First-Differenced Price Series - {commodity}"
        )

        plt.xlabel("Month")

        plt.ylabel(
            "Price Difference (NGN)"
        )

        plt.xticks(
            rotation=45
        )

        plt.tight_layout()

        output_path = (
            STATIONARITY_FIGURES_DIR
            / f"{filename}_first_difference.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n=== STARTING EDA STAGE 4: STATIONARITY ==="
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    create_output_directory()

    # --------------------------------------------------------
    # Load monthly data
    # --------------------------------------------------------

    monthly = load_monthly_data()

    print(
        f"Monthly observations: {len(monthly):,}"
    )

    # --------------------------------------------------------
    # Analyze original series
    # --------------------------------------------------------

    original_results = analyze_original_series(
        monthly
    )

    # --------------------------------------------------------
    # Create first difference
    # --------------------------------------------------------

    differenced = create_first_difference(
        monthly
    )

    # --------------------------------------------------------
    # Analyze first difference
    # --------------------------------------------------------

    difference_results = analyze_first_difference(
        differenced
    )

    second_difference_results = analyze_second_difference(monthly)

    print("\n=== SECOND DIFFERENCE RESULTS ===")
    print(second_difference_results)
    # --------------------------------------------------------
    # Create comparison plots
    # --------------------------------------------------------

    plot_stationarity_comparison(
        monthly,
        differenced
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print(
        "\n=== EDA STAGE 4 COMPLETE ==="
    )


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()
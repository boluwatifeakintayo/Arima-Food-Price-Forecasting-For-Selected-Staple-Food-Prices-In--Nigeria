from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import (
    load_and_filter_data,
    prepare_data,
    create_monthly_series,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EDA_FIGURES_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "eda"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

def create_output_directory():
    """Create the EDA output directory if it does not exist."""

    EDA_FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"EDA figures directory ready: {EDA_FIGURES_DIR}"
    )


# ============================================================
# LOAD MONTHLY DATA
# ============================================================

def load_monthly_data():
    """Load, filter, prepare, and aggregate the data monthly."""

    print("\n=== LOADING DATA FOR EDA ===")

    df = load_and_filter_data()

    print(
        f"Filtered observations: {len(df):,}"
    )

    df = prepare_data(df)

    monthly = create_monthly_series(df)

    print(
        f"Monthly observations: {len(monthly):,}"
    )

    return monthly


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def create_summary_statistics(monthly):
    """Create descriptive statistics for each commodity."""

    print("\n=== CREATING SUMMARY STATISTICS ===")

    summary = (
        monthly
        .groupby("commodity")["price"]
        .describe()
    )

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "commodity_summary_statistics.csv"
    )

    summary.to_csv(output_path)

    print(
        f"Saved summary statistics: {output_path}"
    )

    print("\n=== SUMMARY STATISTICS ===")
    print(summary)

    return summary


# ============================================================
# PRICE DISTRIBUTIONS
# ============================================================

def plot_price_distributions(monthly):
    """Create price distribution plots for each commodity."""

    print("\n=== CREATING PRICE DISTRIBUTION PLOTS ===")

    commodities = monthly["commodity"].unique()

    for commodity in commodities:

        series = monthly[
            monthly["commodity"] == commodity
        ]

        plt.figure(figsize=(10, 6))

        sns.histplot(
            data=series,
            x="price",
            kde=True
        )

        plt.title(
            f"Price Distribution - {commodity}"
        )

        plt.xlabel("Price (NGN)")
        plt.ylabel("Frequency")

        plt.tight_layout()

        filename = (
            commodity
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )

        output_path = (
            EDA_FIGURES_DIR
            / f"{filename}_price_distribution.png"
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
# BOX PLOTS
# ============================================================

def plot_price_boxplots(monthly):
    """Create a boxplot comparing commodity price distributions."""

    print("\n=== CREATING PRICE BOXPLOT ===")

    plt.figure(figsize=(12, 7))

    sns.boxplot(
        data=monthly,
        x="commodity",
        y="price"
    )

    plt.title(
        "Price Distribution by Commodity"
    )

    plt.xlabel("Commodity")
    plt.ylabel("Price (NGN)")

    plt.xticks(rotation=30)

    plt.tight_layout()

    output_path = (
        EDA_FIGURES_DIR
        / "commodity_boxplots.png"
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
# POTENTIAL OUTLIERS
# ============================================================

def identify_potential_outliers(monthly):
    """
    Identify potential statistical outliers using the IQR method.

    Important:
    These are statistical outliers, not automatically data errors.
    """

    print("\n=== IDENTIFYING POTENTIAL OUTLIERS ===")

    all_outliers = []

    for commodity in monthly["commodity"].unique():

        series = monthly[
            monthly["commodity"] == commodity
        ].copy()

        q1 = series["price"].quantile(0.25)
        q3 = series["price"].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = series[
            (series["price"] < lower_bound)
            | (series["price"] > upper_bound)
        ].copy()

        outliers["lower_bound"] = lower_bound
        outliers["upper_bound"] = upper_bound

        all_outliers.append(outliers)

        print(
            f"{commodity}: "
            f"{len(outliers)} potential outliers"
        )

    if all_outliers:

        outlier_data = pd.concat(
            all_outliers,
            ignore_index=True
        )

    else:

        outlier_data = pd.DataFrame()

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "potential_outliers.csv"
    )

    outlier_data.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved potential outliers: {output_path}"
    )

    return outlier_data


# ============================================================
# OUTLIERS OVER TIME
# ============================================================

def plot_outliers_over_time(monthly):
    """Plot commodity prices over time and highlight IQR outliers."""

    print("\n=== PLOTTING OUTLIERS OVER TIME ===")

    for commodity in monthly["commodity"].unique():

        series = monthly[
            monthly["commodity"] == commodity
        ].copy()

        q1 = series["price"].quantile(0.25)
        q3 = series["price"].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = series[
            (series["price"] < lower_bound)
            | (series["price"] > upper_bound)
        ]

        plt.figure(figsize=(12, 6))

        plt.plot(
            series["month"],
            series["price"],
            marker="o",
            linewidth=1,
            label="Monthly Price"
        )

        if not outliers.empty:

            plt.scatter(
                outliers["month"],
                outliers["price"],
                s=50,
                label="Potential Outlier"
            )

        plt.title(
            f"Price Trend and Potential Outliers - {commodity}"
        )

        plt.xlabel("Month")
        plt.ylabel("Price (NGN)")

        plt.xticks(rotation=45)

        plt.legend()

        plt.tight_layout()

        filename = (
            commodity
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )

        output_path = (
            EDA_FIGURES_DIR
            / f"{filename}_outliers_over_time.png"
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
# PRICE CHANGE AND VOLATILITY ANALYSIS
# ============================================================

def analyze_price_changes(monthly):
    """Analyze month-to-month price changes and volatility."""

    print(
        "\n=== ANALYZING PRICE CHANGES AND VOLATILITY ==="
    )

    all_changes = []

    for commodity in monthly["commodity"].unique():

        series = monthly[
            monthly["commodity"] == commodity
        ].copy()

        series = series.sort_values("month")

        # ----------------------------------------------------
        # Month-to-month price change
        # ----------------------------------------------------

        series["price_change"] = (
            series["price"].diff()
        )

        # ----------------------------------------------------
        # Month-to-month percentage change
        # ----------------------------------------------------

        series["pct_change"] = (
            series["price"].pct_change() * 100
        )

        # ----------------------------------------------------
        # Create continuous calendar-month number
        # ----------------------------------------------------
        # This allows us to identify whether two observations
        # are actually consecutive months.
        #
        # Example:
        #
        # 2022-05 -> 2022-06 = consecutive
        #
        # 2022-05 -> 2023-02 = NOT consecutive
        #
        # Therefore, we do not calculate a "month-to-month"
        # change across a long missing-data gap.
        # ----------------------------------------------------

        month_number = (
            series["month"].dt.year * 12
            + series["month"].dt.month
        )

        previous_month_number = (
            month_number.shift(1)
        )

        consecutive = (
            month_number
            - previous_month_number
            == 1
        )

        # Remove changes calculated across missing periods.

        series.loc[
            ~consecutive,
            "price_change"
        ] = pd.NA

        series.loc[
            ~consecutive,
            "pct_change"
        ] = pd.NA

        # ----------------------------------------------------
        # Rolling statistics
        # ----------------------------------------------------

        series["rolling_mean"] = (
            series["price"]
            .rolling(
                window=12,
                min_periods=6
            )
            .mean()
        )

        series["rolling_std"] = (
            series["price"]
            .rolling(
                window=12,
                min_periods=6
            )
            .std()
        )

        all_changes.append(series)

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(f"\n{commodity}")

        print(
            f"Largest absolute increase: "
            f"₦{series['price_change'].max():,.2f}"
        )

        print(
            f"Largest absolute decrease: "
            f"₦{series['price_change'].min():,.2f}"
        )

        print(
            f"Largest percentage increase: "
            f"{series['pct_change'].max():.2f}%"
        )

        print(
            f"Largest percentage decrease: "
            f"{series['pct_change'].min():.2f}%"
        )

    # --------------------------------------------------------
    # Combine all commodities
    # --------------------------------------------------------

    changes = pd.concat(
        all_changes,
        ignore_index=True
    )

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "price_change_analysis.csv"
    )

    changes.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved price change analysis: {output_path}"
    )

    return changes


# ============================================================
# VOLATILITY VISUALIZATIONS
# ============================================================

def plot_volatility(monthly):
    """Create price-change and rolling-volatility visualizations."""

    print(
        "\n=== CREATING VOLATILITY VISUALIZATIONS ==="
    )

    for commodity in monthly["commodity"].unique():

        series = monthly[
            monthly["commodity"] == commodity
        ].copy()

        series = series.sort_values("month")

        # ----------------------------------------------------
        # Percentage price change
        # ----------------------------------------------------

        series["pct_change"] = (
            series["price"].pct_change() * 100
        )

        # ----------------------------------------------------
        # Calendar month number
        # ----------------------------------------------------

        month_number = (
            series["month"].dt.year * 12
            + series["month"].dt.month
        )

        previous_month_number = (
            month_number.shift(1)
        )

        consecutive = (
            month_number
            - previous_month_number
            == 1
        )

        # Remove changes across missing periods.

        series.loc[
            ~consecutive,
            "pct_change"
        ] = pd.NA

        # ----------------------------------------------------
        # 12-month rolling standard deviation
        # ----------------------------------------------------

        series["rolling_std"] = (
            series["price"]
            .rolling(
                window=12,
                min_periods=6
            )
            .std()
        )

        # ----------------------------------------------------
        # Create filename
        # ----------------------------------------------------

        filename = (
            commodity
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )

        # ====================================================
        # PLOT 1: MONTH-TO-MONTH PERCENTAGE CHANGE
        # ====================================================

        plt.figure(figsize=(12, 5))

        plt.plot(
            series["month"],
            series["pct_change"],
            marker="o",
            linewidth=1
        )

        plt.axhline(
            0,
            linewidth=1
        )

        plt.title(
            f"Month-to-Month Percentage Price Change - {commodity}"
        )

        plt.xlabel("Month")

        plt.ylabel(
            "Percentage Change (%)"
        )

        plt.xticks(
            rotation=45
        )

        plt.tight_layout()

        output_path = (
            EDA_FIGURES_DIR
            / f"{filename}_percentage_change.png"
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
        # PLOT 2: ROLLING VOLATILITY
        # ====================================================

        plt.figure(figsize=(12, 5))

        plt.plot(
            series["month"],
            series["rolling_std"],
            linewidth=2
        )

        plt.title(
            f"12-Month Rolling Price Volatility - {commodity}"
        )

        plt.xlabel("Month")

        plt.ylabel(
            "Rolling Standard Deviation"
        )

        plt.xticks(
            rotation=45
        )

        plt.tight_layout()

        output_path = (
            EDA_FIGURES_DIR
            / f"{filename}_rolling_volatility.png"
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
# MAIN EDA WORKFLOW
# ============================================================

def main():

    print("\n=== STARTING EDA ===")

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

    # ========================================================
    # EDA STAGE 1
    # ========================================================

    print("\n=== EDA STAGE 1 ===")

    # Descriptive statistics
    create_summary_statistics(monthly)

    # Price distributions
    plot_price_distributions(monthly)

    # Boxplots
    plot_price_boxplots(monthly)

    # Statistical outliers
    identify_potential_outliers(monthly)

    # Outliers over time
    plot_outliers_over_time(monthly)

    # ========================================================
    # EDA STAGE 2
    # ========================================================

    print("\n=== EDA STAGE 2 ===")

    # Price changes and volatility analysis
    analyze_price_changes(monthly)

    # Volatility visualizations
    plot_volatility(monthly)

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n=== EDA STAGE 2 COMPLETE ===")


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()
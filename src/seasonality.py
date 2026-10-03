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

SEASONALITY_FIGURES_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "eda"
    / "seasonality"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

def create_output_directory():
    """Create the seasonality output directory."""

    SEASONALITY_FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Seasonality figures directory ready: "
        f"{SEASONALITY_FIGURES_DIR}"
    )


# ============================================================
# LOAD MONTHLY DATA
# ============================================================

def load_monthly_data():
    """Load and prepare the monthly commodity data."""

    print("\n=== LOADING DATA FOR SEASONALITY ANALYSIS ===")

    df = load_and_filter_data()

    df = prepare_data(df)

    monthly = create_monthly_series(df)

    return monthly


# ============================================================
# MONTHLY SEASONAL PATTERN
# ============================================================

def analyze_monthly_seasonality(monthly):
    """
    Calculate average price for each calendar month
    for every commodity.
    """

    print("\n=== ANALYZING MONTHLY SEASONALITY ===")

    data = monthly.copy()

    data["month_number"] = data["month"].dt.month

    data["month_name"] = data["month"].dt.month_name()

    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    seasonal_summary = (
        data
        .groupby(
            [
                "commodity",
                "month_number",
                "month_name",
            ],
            as_index=False
        )["price"]
        .mean()
    )

    seasonal_summary["month_name"] = pd.Categorical(
        seasonal_summary["month_name"],
        categories=month_order,
        ordered=True
    )

    seasonal_summary = seasonal_summary.sort_values(
        [
            "commodity",
            "month_number"
        ]
    )

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "monthly_seasonality_summary.csv"
    )

    seasonal_summary.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved monthly seasonality summary: "
        f"{output_path}"
    )

    return seasonal_summary


# ============================================================
# PLOT MONTHLY SEASONAL PATTERNS
# ============================================================

def plot_monthly_seasonality(
    seasonal_summary
):
    """Plot average price by calendar month."""

    print(
        "\n=== CREATING MONTHLY SEASONALITY PLOTS ==="
    )

    commodities = seasonal_summary[
        "commodity"
    ].unique()

    for commodity in commodities:

        series = seasonal_summary[
            seasonal_summary["commodity"] == commodity
        ].copy()

        plt.figure(figsize=(12, 6))

        sns.lineplot(
            data=series,
            x="month_name",
            y="price",
            marker="o"
        )

        plt.title(
            f"Average Monthly Price Pattern - {commodity}"
        )

        plt.xlabel("Month")

        plt.ylabel(
            "Average Price (NGN)"
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
            SEASONALITY_FIGURES_DIR
            / f"{filename}_monthly_seasonality.png"
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
# SEASONAL BOXPLOTS
# ============================================================

def plot_seasonal_boxplots(monthly):
    """
    Create boxplots showing the distribution of prices
    for each calendar month.
    """

    print(
        "\n=== CREATING SEASONAL BOXPLOTS ==="
    )

    data = monthly.copy()

    month_order = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    data["month"] = data["month"].dt.strftime("%b")

    for commodity in data["commodity"].unique():

        series = data[
            data["commodity"] == commodity
        ].copy()

        plt.figure(figsize=(12, 6))

        sns.boxplot(
            data=series,
            x="month",
            y="price",
            order=month_order
        )

        plt.title(
            f"Monthly Price Distribution - {commodity}"
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
            SEASONALITY_FIGURES_DIR
            / f"{filename}_seasonal_boxplot.png"
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
# YEAR-OVER-YEAR SEASONAL PROFILE
# ============================================================

def plot_yearly_seasonal_profiles(monthly):
    """
    Plot monthly prices for each year.

    This helps us determine whether similar monthly
    patterns repeat across different years.
    """

    print(
        "\n=== CREATING YEAR-OVER-YEAR SEASONAL PROFILES ==="
    )

    data = monthly.copy()

    data["year"] = data["month"].dt.year

    data["month_number"] = data["month"].dt.month

    for commodity in data["commodity"].unique():

        series = data[
            data["commodity"] == commodity
        ].copy()

        plt.figure(figsize=(12, 6))

        sns.lineplot(
            data=series,
            x="month_number",
            y="price",
            hue="year",
            marker="o",
            legend=False
        )

        plt.title(
            f"Year-over-Year Monthly Price Profiles - {commodity}"
        )

        plt.xlabel("Month")

        plt.ylabel(
            "Price (NGN)"
        )

        plt.xticks(
            range(1, 13)
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
            SEASONALITY_FIGURES_DIR
            / f"{filename}_yearly_seasonal_profiles.png"
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
        "\n=== STARTING EDA STAGE 3: SEASONALITY ==="
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
    # Monthly seasonal averages
    # --------------------------------------------------------

    seasonal_summary = analyze_monthly_seasonality(
        monthly
    )

    # --------------------------------------------------------
    # Monthly seasonal line charts
    # --------------------------------------------------------

    plot_monthly_seasonality(
        seasonal_summary
    )

    # --------------------------------------------------------
    # Seasonal boxplots
    # --------------------------------------------------------

    plot_seasonal_boxplots(
        monthly
    )

    # --------------------------------------------------------
    # Year-over-year seasonal profiles
    # --------------------------------------------------------

    plot_yearly_seasonal_profiles(
        monthly
    )

    print(
        "\n=== EDA STAGE 3 COMPLETE ==="
    )


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_data


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Selected commodities for the project
SELECTED_COMMODITIES = [
    "Rice (local)",
    "Beans (red)",
    "Yam",
    "Gari (white)",
    "Oil (palm)",
    "Tomatoes",
]


def load_and_filter_data():
    """Load the raw dataset and keep selected retail commodities."""

    df = load_data()

    filtered = df[
        (df["commodity"].isin(SELECTED_COMMODITIES))
        & (df["pricetype"] == "Retail")
    ].copy()

    return filtered


def prepare_data(df):
    """Prepare the raw data for time-series analysis."""

    df["date"] = pd.to_datetime(df["date"])

    columns = [
        "date",
        "admin1",
        "market",
        "commodity",
        "unit",
        "price",
        "currency",
    ]

    df = df[columns].copy()

    df = df.sort_values(["commodity", "date"])

    return df


def create_monthly_series(df):
    """
    Create monthly commodity price series.

    For each commodity and month:
    - price = median of available retail observations
    - observation_count = number of retail observations
      used to calculate that median
    """

    df["month"] = df["date"].dt.to_period("M")

    monthly = (
        df.groupby(
            ["month", "commodity", "unit"]
        )
        .agg(
            price=("price", "median"),
            observation_count=("price", "count"),
        )
        .reset_index()
    )

    monthly["month"] = (
        monthly["month"]
        .dt.to_timestamp()
    )

    monthly = monthly.sort_values(
        ["commodity", "month"]
    )

    return monthly


def check_missing_months(monthly):
    """Check for missing months in each commodity time series."""

    print("\n=== MISSING MONTHS CHECK ===")

    for commodity in monthly["commodity"].unique():

        series = monthly[
            monthly["commodity"] == commodity
        ].copy()

        series = series.set_index("month")

        full_range = pd.date_range(
            start=series.index.min(),
            end=series.index.max(),
            freq="MS",
        )

        missing_months = full_range.difference(series.index)

        print(f"\n{commodity}")
        print(f"Expected months: {len(full_range)}")
        print(f"Available months: {len(series)}")
        print(f"Missing months: {len(missing_months)}")

        if len(missing_months) > 0:
            print(
                "Missing:",
                missing_months.strftime("%Y-%m").tolist()
            )


def analyze_missing_data(monthly):
    """Analyze the amount and pattern of missing monthly observations."""

    print("\n=== MISSING DATA ANALYSIS ===")

    results = []

    for commodity in monthly["commodity"].unique():

        series = monthly[
            monthly["commodity"] == commodity
        ].copy()

        series = series.sort_values("month")
        series = series.set_index("month")

        start_date = series.index.min()
        end_date = series.index.max()

        full_range = pd.date_range(
            start=start_date,
            end=end_date,
            freq="MS",
        )

        missing_months = full_range.difference(series.index)

        expected_months = len(full_range)
        available_months = len(series)
        missing_count = len(missing_months)

        missing_percentage = (
            missing_count / expected_months
        ) * 100

        # Find longest consecutive missing period
        longest_gap = 0
        current_gap = 0

        for month in full_range:

            if month in missing_months:
                current_gap += 1
                longest_gap = max(
                    longest_gap,
                    current_gap,
                )
            else:
                current_gap = 0

        results.append(
            {
                "commodity": commodity,
                "start_date": start_date.strftime("%Y-%m"),
                "end_date": end_date.strftime("%Y-%m"),
                "expected_months": expected_months,
                "available_months": available_months,
                "missing_months": missing_count,
                "missing_percentage": round(
                    missing_percentage,
                    2,
                ),
                "longest_gap_months": longest_gap,
            }
        )

    report = pd.DataFrame(results)

    print(report.to_string(index=False))

    return report

def plot_missing_data(monthly):
    """Visualize observed and missing months for each commodity."""

    print("\n=== CREATING MISSING DATA VISUALIZATION ===")

    figures_dir = PROJECT_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for commodity in monthly["commodity"].unique():

        series = monthly[
            monthly["commodity"] == commodity
        ].copy()

        series = series.sort_values("month")

        full_range = pd.date_range(
            start=series["month"].min(),
            end=series["month"].max(),
            freq="MS",
        )

        available_months = set(series["month"])

        for month in full_range:

            rows.append(
                {
                    "commodity": commodity,
                    "month": month,
                    "status": (
                        "Available"
                        if month in available_months
                        else "Missing"
                    ),
                }
            )

    missing_df = pd.DataFrame(rows)

    plt.figure(figsize=(14, 7))

    sns.scatterplot(
        data=missing_df,
        x="month",
        y="commodity",
        hue="status",
        style="status",
        s=80,
    )

    plt.title("Monthly Data Availability by Commodity")
    plt.xlabel("Month")
    plt.ylabel("Commodity")
    plt.xticks(rotation=45)

    plt.tight_layout()

    output_path = (
        figures_dir / "monthly_data_availability.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")

def plot_monthly_series(monthly):
    """Create and save monthly price trend charts."""

    print("\n=== CREATING TIME-SERIES VISUALIZATIONS ===")

    figures_dir = PROJECT_ROOT / "reports" / "figures"

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    commodities = monthly["commodity"].unique()

    for commodity in commodities:

        series = monthly[
            monthly["commodity"] == commodity
        ].copy()

        series = series.sort_values("month")

        plt.figure(figsize=(12, 5))

        sns.lineplot(
            data=series,
            x="month",
            y="price",
            marker="o",
        )

        plt.title(
            f"Monthly Price Trend - {commodity}"
        )

        plt.xlabel("Month")
        plt.ylabel("Price (NGN)")

        plt.xticks(rotation=45)

        plt.tight_layout()

        filename = (
            commodity
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )

        output_path = (
            figures_dir
            / f"{filename}_monthly_trend.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Saved: {output_path}")

def analyze_missing_periods(monthly):
    """Identify consecutive periods of missing monthly observations."""

    print("\n=== MISSING PERIODS ===")

    for commodity in monthly["commodity"].unique():

        series = monthly[
            monthly["commodity"] == commodity
        ].copy()

        series = series.sort_values("month")
        series = series.set_index("month")

        full_range = pd.date_range(
            start=series.index.min(),
            end=series.index.max(),
            freq="MS",
        )

        missing_months = full_range.difference(series.index)

        if len(missing_months) == 0:
            print(f"\n{commodity}")
            print("No missing periods.")
            continue

        missing_set = set(missing_months)

        periods = []
        period_start = None
        previous_month = None

        for month in missing_months:

            if period_start is None:
                period_start = month

            elif (
                month != previous_month + pd.DateOffset(months=1)
            ):
                periods.append(
                    (period_start, previous_month)
                )
                period_start = month

            previous_month = month

        periods.append(
            (period_start, previous_month)
        )

        print(f"\n{commodity}")

        for start, end in periods:

            months = (
                (end.year - start.year) * 12
                + end.month
                - start.month
                + 1
            )

            print(
                f"{start.strftime('%Y-%m')} "
                f"to {end.strftime('%Y-%m')} "
                f"({months} month(s))"
            )

def check_continuous_periods(monthly):
    """Identify continuous observed periods for each commodity."""

    print("\n=== CONTINUOUS DATA PERIODS ===")

    for commodity in monthly["commodity"].unique():

        series = monthly[
            monthly["commodity"] == commodity
        ].copy()

        series = series.sort_values("month")

        dates = series["month"].tolist()

        periods = []

        start = dates[0]
        previous = dates[0]

        for current in dates[1:]:

            expected_next = previous + pd.DateOffset(months=1)

            if current != expected_next:

                periods.append(
                    (start, previous)
                )

                start = current

            previous = current

        periods.append(
            (start, previous)
        )

        print(f"\n{commodity}")

        for start_date, end_date in periods:

            months = (
                (end_date.year - start_date.year) * 12
                + end_date.month
                - start_date.month
                + 1
            )

            print(
                f"{start_date.strftime('%Y-%m')} "
                f"→ "
                f"{end_date.strftime('%Y-%m')} "
                f"| {months} month(s)"
            )
def check_commodity_units(df):
    """Check the measurement units used by each selected commodity."""

    print("\n=== COMMODITY UNIT CHECK ===")

    unit_check = (
        df.groupby("commodity")["unit"]
        .unique()
        .reset_index()
    )

    for _, row in unit_check.iterrows():

        commodity = row["commodity"]
        units = row["unit"]

        print(f"\n{commodity}")
        print(f"Units: {list(units)}")
        print(f"Number of units: {len(units)}")

    return unit_check
def main():

    print("\n=== LOADING DATA ===")

    df = load_and_filter_data()

    print(
        f"Rows after filtering: {len(df):,}"
    )

    print("\n=== PREPARING DATA ===")

    df = prepare_data(df)

    print(
        f"Rows after preparation: {len(df):,}"
    )

    print("\n=== CREATING MONTHLY TIME SERIES ===")

    monthly = create_monthly_series(df)

    print(
        f"Monthly rows: {len(monthly):,}"
    )

    print("\n=== MONTHLY DATA SAMPLE ===")

    print(monthly.head(15))

    print("\n=== MONTHLY DATA TYPES ===")

    print(monthly.dtypes)

    print(
        "\n=== MONTHLY OBSERVATIONS BY COMMODITY ==="
    )

    print(
        monthly.groupby("commodity").size()
    )

    # Missing-month analysis
    check_missing_months(monthly)

    # Missing-data summary
    analyze_missing_data(monthly)
     #CREATING MISSING DATA VISUALIZATIONp
    plot_missing_data(monthly)
    # Create visualizations
    plot_monthly_series(monthly)

    analyze_missing_periods(monthly)
    
    check_continuous_periods(monthly)

    check_commodity_units(df)


if __name__ == "__main__":
    main()
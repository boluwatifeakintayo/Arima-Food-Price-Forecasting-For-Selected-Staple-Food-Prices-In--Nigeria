from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from data_loader import load_data
from preprocessing import (
    SELECTED_COMMODITIES,
    load_and_filter_data,
    prepare_data,
    create_monthly_series,
)
from stationarity import get_longest_continuous_segment


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "acf_pacf"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Working differencing orders from stationarity analysis
DIFFERENCING_ORDERS = {
    "Beans (red)": 1,
    "Gari (white)": 2,
    "Oil (palm)": 1,
    "Rice (local)": 1,
    "Tomatoes": 1,
    "Yam": 2,
}


def prepare_monthly_data():
    """
    Load, filter, prepare, and aggregate the WFP data
    into monthly commodity price series.
    """

    df = load_and_filter_data()
    df = prepare_data(df)
    monthly = create_monthly_series(df)

    return monthly


def create_differenced_series(series, d):
    """
    Apply differencing d times.

    Differencing is performed only on the longest
    continuous monthly segment.
    """

    continuous = get_longest_continuous_segment(series)

    price_series = continuous["price"].copy()

    for _ in range(d):
        price_series = price_series.diff().dropna()

    return price_series


def plot_acf_pacf_for_commodity(monthly, commodity, d):
    """
    Create ACF and PACF plots for one commodity.
    """

    series = (
        monthly[monthly["commodity"] == commodity]
        [["month", "price"]]
        .sort_values("month")
        .copy()
    )

    differenced = create_differenced_series(series, d)

    print(f"\n{'=' * 70}")
    print(f"{commodity}")
    print(f"{'=' * 70}")

    print(f"Differencing order (d): {d}")
    print(f"Observations after differencing: {len(differenced)}")

    if len(differenced) < 10:
        print("Not enough observations for reliable ACF/PACF analysis.")
        return

    # Choose a reasonable number of lags.
    # We don't want the lag count to be too large
    # relative to the sample size.
    max_lags = min(24, len(differenced) // 3)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 9)
    )

    plot_acf(
        differenced,
        lags=max_lags,
        ax=axes[0],
        zero=False
    )

    axes[0].set_title(
        f"{commodity} - ACF (d={d})"
    )

    plot_pacf(
        differenced,
        lags=max_lags,
        ax=axes[1],
        zero=False,
        method="ywm"
    )

    axes[1].set_title(
        f"{commodity} - PACF (d={d})"
    )

    plt.tight_layout()

    safe_name = (
        commodity
        .lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    output_path = FIGURES_DIR / f"{safe_name}_acf_pacf.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


def main():

    print("\n" + "=" * 70)
    print("STARTING ACF/PACF ANALYSIS")
    print("=" * 70)

    monthly = prepare_monthly_data()

    print(f"\nMonthly observations: {len(monthly):,}")

    for commodity in SELECTED_COMMODITIES:

        d = DIFFERENCING_ORDERS[commodity]

        plot_acf_pacf_for_commodity(
            monthly,
            commodity,
            d
        )

    print("\n" + "=" * 70)
    print("ACF/PACF ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
from data_loader import load_data
import pandas as pd


# def main():
#     df = load_data()

#     selected = [
#         "Rice",
#         "Beans",
#         "Yam",
#         "Gari",
#         "Palm oil",
#         "Tomatoes",
#     ]

#     pattern = "|".join(selected)

#     filtered = df[
#         df["commodity"].str.contains(
#             pattern,
#             case=False,
#             na=False
#         )
#     ]

#     result = (
#         filtered
#         .groupby(["commodity", "unit", "pricetype"])
#         .size()
#         .sort_values(ascending=False)
#     )

#     print("\n=== SELECTED COMMODITIES ===")
#     print(result)


# if __name__ == "__main__":
#     main()



# from data_loader import load_data


# def main():
#     df = load_data()

#     oil = df[
#         df["commodity"].str.contains(
#             "Oil",
#             case=False,
#             na=False
#         )
#     ]

#     result = (
#         oil
#         .groupby(["commodity", "unit", "pricetype"])
#         .size()
#         .sort_values(ascending=False)
#     )

#     print("\n=== PALM OIL DATA ===")
#     print(result)


# if __name__ == "__main__":
#     main()

# def main():
#     df = load_data()

#     selected = [
#         "Rice (local)",
#         "Rice (imported)",
#         "Beans (red)",
#         "Beans (white)",
#         "Yam",
#         "Gari (white)",
#         "Tomatoes",
#         "Oil (palm)",
#     ]

#     filtered = df[
#         (df["commodity"].isin(selected))
#         & (df["pricetype"] == "Retail")
#     ].copy()

#     filtered["date"] = pd.to_datetime(filtered["date"])

#     result = (
#         filtered
#         .groupby("commodity")["date"]
#         .agg(["min", "max", "count"])
#         .sort_values("count", ascending=False)
#     )

#     print("\n=== DATE COVERAGE OF SELECTED RETAIL COMMODITIES ===")
#     print(result)


# if __name__ == "__main__":
#     main()



def main():
    df = load_data()

    selected = [
        "Rice (local)",
        "Rice (imported)",
        "Beans (red)",
        "Beans (white)",
        "Yam",
        "Gari (white)",
        "Tomatoes",
        "Oil (palm)",
    ]

    filtered = df[
        (df["commodity"].isin(selected))
        & (df["pricetype"] == "Retail")
    ].copy()

    result = (
        filtered
        .groupby("commodity")
        .agg(
            states=("admin1", "nunique"),
            markets=("market", "nunique"),
            observations=("price", "count"),
        )
        .sort_values("observations", ascending=False)
    )

    print("\n=== MARKET COVERAGE ===")
    print(result)


if __name__ == "__main__":
    main()
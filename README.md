
My Final Year HND Project

# ARIMA-Based Forecasting System for Selected Staple Food Prices in Nigeria

An end-to-end time-series forecasting system that uses **ARIMA models** to analyse historical retail prices and generate forecasts for selected staple foods in Nigeria.

The project combines data preprocessing, exploratory data analysis, stationarity testing, ARIMA modelling, model evaluation, forecast generation, abnormal price-movement detection, and an interactive **Streamlit dashboard**.

> **Important:** The system uses retail-market observations available in the WFP Nigeria food-price dataset. The results should not be interpreted as the price at every shop or market across Nigeria.

---

## Project Overview

Food prices can change over time due to seasonal patterns, supply conditions, market dynamics, and other factors. Having a systematic way to analyse historical prices and estimate future prices can support data-driven monitoring and planning.

This project develops a time-series forecasting system for six selected staple food commodities using the **Autoregressive Integrated Moving Average (ARIMA)** model.

The system:

* Processes historical WFP Nigeria food-price data
* Filters the analysis to retail observations
* Selects six staple food commodities
* Converts observations into monthly price series
* Performs exploratory and statistical time-series analysis
* Tests for stationarity and determines differencing requirements
* Evaluates candidate ARIMA models
* Trains final ARIMA models
* Generates 12-month forecasts
* Detects statistically unusual price movements
* Provides model-performance metrics
* Presents the results through an interactive Streamlit dashboard

---

## Selected Commodities

The system currently analyses:

| Commodity    | Unit   |
| ------------ | ------ |
| Rice (local) | 2.7 KG |
| Beans (red)  | 2.5 KG |
| Yam          | 2.5 KG |
| Gari (white) | KG     |
| Oil (palm)   | L      |
| Tomatoes     | 0.5 KG |

The commodity units correspond to the units recorded in the source dataset.

---

## Data Source

The project uses the **World Food Programme (WFP) Nigeria food-price dataset**.

The raw dataset contains fields including:

* Date
* State
* Market
* Commodity
* Unit
* Price
* Currency
* Price type
* Market identifier
* Geographic information

The analysis specifically uses observations where:

```text
pricetype = Retail
```

Therefore, the forecasts represent estimates based on historical **retail-market price observations**, rather than wholesale, importer, or direct-source prices.

### Monthly Aggregation

The available retail observations are aggregated into monthly commodity price series.

For each commodity and month:

```text
Monthly price = median of available retail observations
```

The number of observations contributing to each monthly price is also retained.

This allows the dashboard to communicate the availability of source observations alongside the price.

---

## Data Coverage

The source data does not contain the same amount of historical information for every commodity.

For example:

* Yam has a long continuous history.
* Rice (local) has a small number of missing months.
* Beans (red) contains a longer gap in its historical series.
* Tomatoes has several missing months.
* Gari (white) has substantially shorter and more fragmented historical coverage.

Because of these differences, the system does not treat every commodity as having identical historical coverage.

The dashboard therefore displays the number of source observations and classifies monthly data availability as:

* **Good:** 30 or more observations
* **Moderate:** 10–29 observations
* **Limited:** fewer than 10 observations

This classification describes **data availability**, not statistical confidence.

---

## Time-Series Preparation

Before modelling, the data passes through the following pipeline:

```text
Raw WFP Data
      ↓
Select Retail Observations
      ↓
Select Six Commodities
      ↓
Convert Dates
      ↓
Monthly Aggregation
      ↓
Check Data Continuity
      ↓
Stationarity Analysis
      ↓
ARIMA Modelling
      ↓
Forecast Generation
```

### Handling Missing Months

Missing months are not automatically treated as normal consecutive observations.

For historical model evaluation, the system identifies the **longest continuous monthly segment** for each commodity.

A continuous segment means that each observation follows the previous observation by exactly one calendar month.

This prevents gaps in the source data from being incorrectly interpreted as regular consecutive time-series observations.

---

# Exploratory Data Analysis

The project includes exploratory analysis of the selected commodities.

Analysis includes:

* Summary statistics
* Price distributions
* Boxplots
* Potential outlier identification
* Price-change analysis
* Rolling volatility
* Monthly trends
* Seasonal patterns
* Monthly data availability

The resulting figures are stored in:

```text
reports/figures/
```

---

# Stationarity Analysis

ARIMA models require appropriate treatment of non-stationary time series.

The project evaluates stationarity using statistical tests and examines the series before and after differencing.

The differencing orders used by the final models are:

| Commodity    | Differencing Order (d) |
| ------------ | ---------------------: |
| Rice (local) |                      1 |
| Beans (red)  |                      1 |
| Yam          |                      2 |
| Gari (white) |                      2 |
| Oil (palm)   |                      1 |
| Tomatoes     |                      1 |

ACF and PACF analysis was also performed to help identify suitable ARIMA candidates.

---

# ARIMA Models

Separate ARIMA models are trained for each commodity.

The final model configurations are:

| Commodity    | ARIMA Order  |
| ------------ | ------------ |
| Rice (local) | ARIMA(1,1,0) |
| Beans (red)  | ARIMA(1,1,1) |
| Yam          | ARIMA(0,2,1) |
| Gari (white) | ARIMA(0,2,1) |
| Oil (palm)   | ARIMA(1,1,1) |
| Tomatoes     | ARIMA(1,1,0) |

The models were selected after comparing candidate configurations using historical holdout performance and residual diagnostics.

Saved trained models are located in:

```text
models/
```

---

# Model Evaluation

The forecasting models were evaluated using historical data that was kept separate from the training data.

The evaluation uses:

* **MAE** — Mean Absolute Error
* **RMSE** — Root Mean Squared Error
* **MAPE** — Mean Absolute Percentage Error
* **Ljung-Box test** — residual autocorrelation diagnostic

### Evaluation Results

| Commodity    | ARIMA   |     MAE |    RMSE |   MAPE | Test Observations |
| ------------ | ------- | ------: | ------: | -----: | ----------------: |
| Rice (local) | (1,1,0) | ₦714.58 | ₦854.82 | 20.47% |                29 |
| Beans (red)  | (1,1,1) | ₦149.51 | ₦199.81 |  9.74% |                14 |
| Yam          | (0,2,1) | ₦601.32 | ₦741.52 | 18.88% |                29 |
| Gari (white) | (0,2,1) |  ₦14.38 |  ₦16.32 |  4.13% |                 7 |
| Oil (palm)   | (1,1,1) | ₦783.30 | ₦892.91 | 32.32% |                29 |
| Tomatoes     | (1,1,0) |  ₦15.54 |  ₦19.55 | 17.32% |                13 |

The results describe historical test performance and do not guarantee the same level of accuracy for future forecasts.

### Residual Diagnostics

The Ljung-Box p-values obtained during evaluation were above 0.05 for all six final models.

This means the test did not provide strong statistical evidence of residual autocorrelation at the tested lag.

This should not be interpreted as proof that the models are perfect or that future forecasts are guaranteed to be accurate.

### Important Data-Coverage Note

The Gari model was evaluated using only **7 test observations**, and the available Gari data ends in **January 2023**.

Consequently, its evaluation and forecast should be interpreted with particular caution.

---

# Forecasting

Each final ARIMA model generates a **12-month forecast**.

For each forecast month, the system provides:

* Forecasted price
* Lower expected boundary
* Upper expected boundary

The dashboard presents the interval as a **95% expected range**.

The range represents uncertainty around the model's forecast and generally becomes wider farther into the future.

It is not a guarantee that the actual market price will fall within the range.

Forecast files are stored in:

```text
reports/forecasts/
```

---

# Abnormal Price-Movement Detection

The system also identifies price movements that are statistically unusual compared with historical patterns.

Two approaches are used:

### 1. Rolling Robust Detection

The rolling method examines recent historical price changes and uses a robust statistical measure based on:

* Rolling median
* Median Absolute Deviation (MAD)
* Robust z-score

A minimum absolute price-change threshold is also applied.

### 2. ARIMA Residual Detection

The second method examines the difference between the actual historical price and the price fitted by the ARIMA model.

Large robust residual deviations are flagged as potentially unusual observations.

### Combined Detection

The two approaches are compared to identify:

* Events detected by the rolling method only
* Events detected by the ARIMA residual method only
* Events detected by both methods

The comparison results are stored in:

```text
reports/spikes/anomaly_detection_comparison.csv
```

An unusual movement does **not** automatically mean that the source data is incorrect, nor does the system determine the cause of the movement.

---

# Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

The dashboard allows users to:

* Select a commodity
* View the latest available actual retail price
* View the latest actual month in the source data
* See source observation coverage
* Explore historical price trends
* Select a forecast month
* View the forecasted price
* View the expected forecast range
* Review model performance
* Explore unusual price movements
* View technical model information
* Switch between light and dark themes

The dashboard is designed with two levels of interpretation:

### General-user layer

Provides plain-language explanations of:

* Forecasts
* Expected ranges
* Data coverage
* Unusual price movements
* Model performance

### Technical layer

Provides additional information such as:

* ARIMA order
* Differencing order
* MAE
* RMSE
* MAPE
* Ljung-Box p-value
* Detection methods

---

# Dashboard Data Recency

The dashboard distinguishes between:

**Latest actual data**

and

**Current/future forecast periods**

For example, if the latest available WFP observation is July 2026, the dashboard continues to display July 2026 as the latest actual month.

It does not replace the actual data with the current calendar month.

The dashboard automatically obtains the current calendar month from the system date and uses it when determining which forecast months are available for selection.

---

# Project Structure

```text
Arima-Food-Price-Forecasting-For-Selected-Staple-Food-Prices-In-Nigeria/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── models/
│   ├── beans_red_arima.pkl
│   ├── gari_white_arima.pkl
│   ├── oil_palm_arima.pkl
│   ├── rice_local_arima.pkl
│   ├── tomatoes_arima.pkl
│   └── yam_arima.pkl
│
├── reports/
│   ├── figures/
│   ├── forecasts/
│   ├── spikes/
│   ├── model_evaluation.csv
│   ├── monthly_seasonality_summary.csv
│   ├── price_change_analysis.csv
│   └── stationarity_*.csv
│
├── src/
│   ├── acf_pacf.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── evaluation.py
│   ├── final_training.py
│   ├── forecasting.py
│   ├── preprocessing.py
│   ├── seasonality.py
│   ├── spike_detection.py
│   ├── stationarity.py
│   └── train_arima.py
│
├── streamlit_app/
│   └── app.py
│
├── .streamlit/
│   └── config.toml
│
├── .gitignore
├── main.py
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/boluwatifeakintayo/Arima-Food-Price-Forecasting-For-Selected-Staple-Food-Prices-In--Nigeria.git
```

Move into the project directory:

```bash
cd Arima-Food-Price-Forecasting-For-Selected-Staple-Food-Prices-In--Nigeria
```

## 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
```

Activate it using Git Bash:

```bash
source .venv/Scripts/activate
```

## 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

# Running the Dashboard

From the project root:

```bash
python -m streamlit run streamlit_app/app.py
```

Streamlit will provide a local URL where the dashboard can be opened in a browser.

---

# Reproducibility

The project separates the major stages of the workflow into Python modules.

Examples include:

```text
src/data_loader.py
src/preprocessing.py
src/eda.py
src/stationarity.py
src/acf_pacf.py
src/train_arima.py
src/final_training.py
src/evaluation.py
src/forecasting.py
src/spike_detection.py
```

This structure makes it possible to inspect and reproduce individual stages of the analysis rather than relying entirely on a single notebook.

---

# Limitations

Several limitations should be considered when interpreting the results.

### 1. Market coverage

The source dataset contains observations from available WFP-covered markets and does not represent every market or shop in Nigeria.

### 2. Unequal historical coverage

Different commodities have different amounts of historical data and different numbers of missing months.

### 3. Gari data recency

The available Gari series ends in January 2023, making its current forecast particularly limited by source-data recency.

### 4. Monthly aggregation

The system uses the median of available monthly retail observations. This provides a summary of the available observations but should not be interpreted as an official national average price.

### 5. Forecast uncertainty

ARIMA forecasts are estimates based on historical patterns. Unexpected economic, climatic, policy, supply-chain, or market events can cause actual prices to differ from forecasts.

### 6. Anomaly interpretation

The anomaly detector identifies statistically unusual movements but does not determine their underlying causes.

---

# Future Improvements

Potential future improvements include:

* Adding more current WFP observations as they become available
* Improving handling of commodities with fragmented historical coverage
* Comparing ARIMA with alternative forecasting models
* Investigating seasonal ARIMA/SARIMA where justified by diagnostics
* Adding additional market-level analysis
* Incorporating external explanatory variables
* Adding automated model retraining
* Adding forecast monitoring and model drift detection
* Deploying the dashboard to a public hosting platform
* Adding automated testing and continuous integration

---

# Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Statsmodels**
* **Scikit-learn**
* **SciPy**
* **Matplotlib**
* **Seaborn**
* **Plotly**
* **Streamlit**
* **Joblib**
* **Git & GitHub**

---

# Project Status

**Current status: Working end-to-end prototype**

The current implementation includes:

* Data preprocessing
* Exploratory data analysis
* Stationarity analysis
* ACF/PACF analysis
* ARIMA model training
* Historical model evaluation
* 12-month forecasting
* Abnormal price-movement detection
* Interactive Streamlit dashboard
* Light and dark dashboard themes

---

## Author

**Boluwatife Akintayo**

Computer Science / Data Science Student
Interested in Machine Learning, Data Science, and AI Engineering.

My Final Year HND Project

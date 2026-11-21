# Forecasting Sri Lanka Telecom Closing Prices with Ensemble Learning

This project uses daily Sri Lanka Telecom (SLT) share prices from 2009 to 2023 to forecast the next trading day’s closing price in Sri Lankan rupees (LKR). Two Python notebooks train and evaluate the models. A Streamlit dashboard displays their forecasts so users can explore individual trading days and compare results across the full test period.

[Open the Streamlit dashboard](https://ensemble-learning-dashboard-assign1.streamlit.app/) · [View the GitHub repository](https://github.com/Navo-Ama-Perera/ML_Assignment1)

## Dataset

[Download the original daily SLT stock-price dataset](./TeleCom_2009_2023_Daily.csv?raw=true)

The dataset contains 3,239 daily records with the date, closing price (`Price`), opening price, daily high and low, trading volume, and percentage change. Both notebooks use the date and closing price to prepare inputs such as earlier closing prices and moving averages. The target is the closing price on the next recorded trading day.

The earliest 80% of the prepared rows are used for model development. All models are evaluated on the same 646 trading days, from 28 July 2020 to 12 April 2023. The **previous-close baseline** uses the latest available closing price as the forecast for the next trading day.

## What each notebook contains

### [Notebook 1 — Random Forest, XGBoost and stacking](./01_RF_vs_XGBoost.ipynb)

- **Experiment 1 compares bagging and boosting.** Random Forest averages forecasts from independently trained trees. XGBoost builds trees in sequence, with each new tree helping to reduce the errors of the combined model.
- **Experiment 2 tests stacking.** Linear Regression learns how to combine Random Forest and XGBoost forecasts. The combiner is trained on forecasts for a chronological validation period, using base models fitted on earlier dates.
- The notebook compares all three methods with the previous-close baseline using error tables and actual-versus-predicted charts. Its research section compares Random Forest with XGBoost, LightGBM, and CatBoost and discusses tree ensembles for tabular, image, and text data.

### [Notebook 2 — Hybrid forecasting models](./02_SLT_Hybrid_Models.ipynb)

- **Experiment 1 tests ARIMA + Random Forest for time-series forecasting.** ARIMA forecasts the next closing price. Random Forest uses the five most recent ARIMA forecast errors to estimate a correction.
- **Experiment 2 tests Random Forest + ANN for tabular regression.** Random Forest and an artificial neural network use the same price features. Their forecasts are averaged with a 50% weight for each model.
- Each hybrid is compared with its individual models and the previous-close baseline on the same test dates.

## Files used by the dashboard

The notebooks export three forecast CSVs. Each contains the test date, actual closing price, baseline forecast, and model forecasts for one dashboard tab.

| File | Purpose |
|---|---|
| [`01_ensemble_predictions.csv`](./01_ensemble_predictions.csv) | Notebook 1 forecasts for Random Forest, XGBoost, and stacking |
| [`02_arima_rf_predictions.csv`](./02_arima_rf_predictions.csv) | Notebook 2 forecasts for ARIMA, Random Forest, and ARIMA + RF |
| [`03_rf_ann_predictions.csv`](./03_rf_ann_predictions.csv) | Notebook 2 forecasts for Random Forest, ANN, and their 50/50 average |
| [`app.py`](./app.py) | Streamlit application that reads the forecast files and displays the three experiments |
| [`requirements.txt`](./requirements.txt) | Python packages needed to run the dashboard |

## How to use the dashboard

1. Open the dashboard and choose one of the three experiment tabs.
2. Choose which forecasts to display on the chart.
3. Select a test trading day. **Selected trading day** shows the actual closing price, each forecast, and its absolute error in LKR.
4. Read **Full test period** to compare performance across all 646 days. MAE is the average absolute error per share. RMSE gives greater weight to larger errors. Lower values indicate smaller errors.

## Main findings

In Notebook 1, the previous-close baseline had the lowest MAE at LKR 1.166. Among the ensemble methods, Random Forest had an MAE of LKR 3.346, followed by XGBoost at LKR 3.390 and stacking at LKR 7.869.

Across both notebooks, ARIMA had the lowest MAE at LKR 1.156, about one cent per share below the baseline. The previous-close baseline had the lowest RMSE at LKR 2.173. Averaging Random Forest and ANN improved on Random Forest alone, while ANN alone remained more accurate than their average.

## Run the notebooks in Google Colab

Open either notebook in Google Colab and run the cells in order. When prompted, upload `TeleCom_2009_2023_Daily.csv`. Each notebook generates its own forecast CSVs for the dashboard.

# Forecasting Sri Lanka Telecom Share Prices with Ensemble Learning

This project uses historical Sri Lanka Telecom (SLT) share prices to forecast the next trading day’s closing price in Sri Lankan rupees (LKR). Two notebooks compare ensemble and hybrid models. An interactive dashboard lets users explore the forecasts for individual trading days and compare model performance across the full test period.

[Open the Streamlit dashboard](https://ensemble-learning-dashboard-assign1.streamlit.app/) · [View the GitHub repository](https://github.com/Navo-Ama-Perera/ML_Assignment1)

## Dataset

[Download the SLT daily share-price dataset](https://raw.githubusercontent.com/Navo-Ama-Perera/ML_Assignment1/main/TeleCom_2009_2023_Daily.csv)

The dataset contains 3,239 daily records from July 2009 to April 2023. It includes the trading date, closing price, opening price, daily high and low, volume, and percentage change.

The notebooks use the date and closing price to create features such as earlier closing prices and moving averages. The target is the closing price on the next recorded trading day. The data is kept in date order so the models train on earlier observations and are tested on later ones.

All models are evaluated on the same **646 trading days, from 28 July 2020 to 12 April 2023**. The **previous-close baseline** uses the latest available closing price as the forecast for the next trading day.

## What each notebook contains

### [Notebook 1 — Random Forest, XGBoost and Stacking](./01_RF_vs_XGBoost.ipynb)

- **Bagging with Random Forest.** Several decision trees make forecasts independently, and their forecasts are averaged.
- **Boosting with XGBoost.** Trees are added one at a time, with each new tree helping to reduce the errors of the combined model.
- **Stacking with Linear Regression.** A Linear Regression model learns how to combine the Random Forest and XGBoost forecasts using a separate validation period. A regression model is suitable because the output is a numerical share price.
- **Research comparison.** The notebook compares Random Forest with XGBoost, LightGBM, and CatBoost and discusses the strengths and limitations of tree ensembles for tabular, image, and text data.

### [Notebook 2 — Hybrid Forecasting Models](./02_SLT_Hybrid_Models.ipynb)

- **ARIMA + Random Forest for time-series forecasting.** ARIMA produces the initial price forecast. Random Forest uses the five most recent ARIMA forecast errors to estimate a correction.
- **Random Forest + ANN for tabular regression.** Random Forest and an artificial neural network (ANN) each predict the closing price using the same price features. The final forecast is the average of the two predictions, giving each model a 50% weight.

Each hybrid is compared with its individual models and the previous-close baseline.

## Dashboard files

| File | What it contains |
|---|---|
| [TeleCom_2009_2023_Daily.csv](./TeleCom_2009_2023_Daily.csv) | Original daily SLT share-price records |
| [01_ensemble_predictions.csv](./01_ensemble_predictions.csv) | Actual prices and forecasts for the bagging, boosting, and stacking tab |
| [02_arima_rf_predictions.csv](./02_arima_rf_predictions.csv) | Actual prices and forecasts for the ARIMA + Random Forest tab |
| [03_rf_ann_predictions.csv](./03_rf_ann_predictions.csv) | Actual prices and forecasts for the Random Forest + ANN tab |
| [app.py](./app.py) | Streamlit dashboard that reads the saved forecasts |
| [requirements.txt](./requirements.txt) | Python packages needed to run the dashboard |

## How to use the dashboard

1. Open the dashboard and choose an experiment tab.
2. Choose which forecasts to show on the price chart. Select **Previous close** to see how the models compare with using the previous trading day’s closing price.
3. Select a test trading day. The **Selected trading day** table shows the actual close, each model’s forecast, and how far it was from the actual price.
4. Use the **Full test period** table and the MAE and RMSE bar chart to compare models across all 646 days. Shorter bars mean smaller errors.
5. Select a model in the **actual-versus-predicted scatter plot**. Each dot represents one test trading day. A dot on the dashed line means the forecast matched the actual close. Dots above the line are overestimates, and dots below it are underestimates.

### How to read the scores

| Metric | What it means |
|---|---|
| **MAE** | The average size of a forecast error in LKR per share. Lower is better. |
| **RMSE** | An error measure in LKR per share that gives more weight to large misses. Lower is better. |
| **R²** | Compares the model’s squared errors with using the mean actual price across the test period. A score of 1 is a perfect fit, 0 matches that reference, and a negative score is worse. |

## Main findings

- **Ensemble comparison.** Random Forest had a slightly lower MAE than XGBoost (LKR 3.346 versus LKR 3.390). Stacking had the highest MAE at LKR 7.869. The previous-close baseline performed best in this experiment, with an MAE of LKR 1.166.
- **ARIMA + Random Forest.** ARIMA alone had the lowest MAE across both notebooks at LKR 1.156, about one cent per share below the baseline. Adding the Random Forest correction increased MAE to LKR 1.190. The baseline still had the lowest RMSE at LKR 2.173.
- **Random Forest + ANN.** The 50/50 average had an MAE of LKR 2.244. It improved on Random Forest alone (LKR 3.346), while ANN alone was more accurate (LKR 1.547).

## Run the notebooks

Open either notebook in Google Colab and run the cells from top to bottom. The updated dataset-loading cell reads the CSV directly from GitHub. If using a notebook version that asks for an upload, select the original dataset linked above.

## Run the dashboard locally

Open a terminal in the project folder containing `app.py` and run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard reads the three saved prediction CSVs, so it can run without rerunning the notebooks.
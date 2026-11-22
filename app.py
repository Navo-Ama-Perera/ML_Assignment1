from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


APP_FOLDER = Path(__file__).resolve().parent
DATASET_NAME = "TeleCom_2009_2023_Daily.csv"

# Shared labels and formats used across all three tabs.
DATE_FORMAT = "%d %b %Y"
DATE_FIELD = "Date:T"
MAE_LABEL = "MAE (LKR)"
RMSE_LABEL = "RMSE (LKR)"
METRIC_FIELD = "Metric:N"

ACTUAL = "Actual"
BASELINE = "Previous close"
RANDOM_FOREST = "Random Forest"

EXPERIMENTS = {
    "RF vs XGBoost & Stacking": "01_ensemble_predictions.csv",
    "ARIMA + Random Forest": "02_arima_rf_predictions.csv",
    "Random Forest + ANN": "03_rf_ann_predictions.csv",
}

MODEL_COLORS = {
    ACTUAL: "#4D8DFF",
    BASELINE: "#E6EC2B",
    RANDOM_FOREST: "#23C552",
    "XGBoost": "#F05A52",
    "Stacking (Linear Regression combiner)": "#B07AF5",
    "ARIMA": "#37C9BD",
    "ARIMA + RF": "#F79945",
    "ANN": "#F078B4",
    "RF + ANN (50/50)": "#B07AF5",
}


def show_project_intro():
    st.title("SLT Ensemble Learning Dashboard")
    st.caption("Sri Lanka Telecom · Next-trading-day closing price · LKR")

    st.write(
        "Can a model forecast Sri Lanka Telecom’s next closing share price "
        "more accurately than using the latest close? This dashboard compares "
        "ensemble and hybrid models on the same 646 historical test trading days."
    )

    st.subheader("About the dataset")
    st.write(
        "Sri Lanka Telecom PLC provides telecommunications, broadband and ICT "
        "services in Sri Lanka. Its daily stock-market dataset contains 3,239 "
        "records from 8 July 2009 to 12 April 2023. Each row represents one "
        "trading day. Share prices are in Sri Lankan rupees per share."
    )

    with st.expander("Dataset columns, model inputs and forecast target"):
        columns = pd.DataFrame(
            {
                "CSV column": [
                    "Date", "Price", "Open", "High", "Low", "Vol.", "Change %"
                ],
                "Meaning": [
                    "Trading date",
                    "Closing share price",
                    "Opening share price",
                    "Highest share price during the day",
                    "Lowest share price during the day",
                    "Reported trading volume",
                    "Reported percentage change in price",
                ],
            }
        )
        st.table(columns.set_index("CSV column"))

        st.markdown(
            "**Model inputs.** The tabular models use five values: today’s "
            "close, the close one trading day earlier, the close five trading "
            "days earlier, and the 5-day and 10-day average closing prices. "
            "Both averages include today’s close."
        )
        st.markdown(
            "**Forecast target.** The target is the closing price on the next "
            "recorded trading day. For example, SLT closed at LKR 101.50 on "
            "11 April 2023. The target for that row was the 12 April closing "
            "price of LKR 99.40."
        )
        st.markdown(
            "**Previous-close baseline.** The baseline uses the latest close "
            "as its forecast. In the example above, it predicted LKR 101.50 "
            "and missed the actual close by LKR 2.10."
        )
        st.markdown(
            "**Training and testing.** The notebooks use the earliest 80% "
            "of prepared rows for model development. The final 646 target dates, "
            "from 28 July 2020 to 12 April 2023, form the test period."
        )

    with st.expander("Models and experiments"):
        st.markdown(
            "- **Bagging, boosting and stacking.** Random Forest averages "
            "predictions from decision trees. XGBoost builds trees in sequence. "
            "For stacking, Linear Regression learns how to combine the two "
            "forecasts using earlier validation dates.\n"
            "- **ARIMA + Random Forest.** ARIMA forecasts the next closing "
            "price. Random Forest uses the five most recent ARIMA forecast "
            "errors to estimate a correction.\n"
            "- **Random Forest + artificial neural network (ANN).** Both "
            "models predict the next close from the same five price features. "
            "The hybrid gives each forecast a 50% weight."
        )

    dataset_path = APP_FOLDER / DATASET_NAME
    if dataset_path.is_file():
        st.download_button(
            "Download SLT daily prices (CSV)",
            data=dataset_path.read_bytes(),
            file_name=DATASET_NAME,
            mime="text/csv",
        )


def load_forecasts(source):
    """Read one notebook export and put its trading dates in order."""
    frame = pd.read_csv(source)

    if not {"Date", ACTUAL}.issubset(frame.columns):
        raise ValueError("The forecast CSV needs Date and Actual columns.")

    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    values = frame.set_index("Date").sort_index()
    values = values.apply(pd.to_numeric, errors="raise")

    if values.empty or values.index.has_duplicates or values.index.hasnans:
        raise ValueError("Check the forecast CSV for missing or duplicate dates.")

    if values.isna().any().any() or values.isin([float("inf"), -float("inf")]).any().any():
        raise ValueError("Each actual price and forecast must be a finite number.")

    if len(values.columns) < 2:
        raise ValueError("Include at least one forecast column alongside Actual.")

    return values


def show_price_chart(values, visible_models, day):
    """Plot prices over time and mark the selected trading day."""
    series = [ACTUAL, *visible_models]
    chart_data = (
        values[series]
        .reset_index()
        .melt(id_vars="Date", var_name="Series", value_name="Price")
    )

    date_axis = alt.X(
        DATE_FIELD,
        title="Date",
        axis=alt.Axis(format="%b %Y"),
    )
    colors = alt.Scale(
        domain=series,
        range=[MODEL_COLORS.get(name, "#909090") for name in series],
    )

    lines = alt.Chart(chart_data).mark_line().encode(
        x=date_axis,
        y=alt.Y(
            "Price:Q",
            title="Closing price (LKR)",
            scale=alt.Scale(zero=False),
        ),
        color=alt.Color("Series:N", scale=colors),
        tooltip=[
            alt.Tooltip(DATE_FIELD, format=DATE_FORMAT),
            "Series:N",
            alt.Tooltip("Price:Q", format=".3f"),
        ],
    )

    marker = (
        alt.Chart(pd.DataFrame({"Date": [day]}))
        .mark_rule(color="gray")
        .encode(x=date_axis)
    )

    st.altair_chart((lines + marker).interactive(), use_container_width=True)


def calculate_scores(actual, errors):
    """Calculate full-period scores using the same dates for each model."""
    squared_errors = errors.pow(2)
    scores = pd.DataFrame(
        {
            MAE_LABEL: errors.abs().mean(),
            RMSE_LABEL: squared_errors.mean().pow(0.5),
        }
    )

    # R² compares squared forecast errors with variation around the actual mean.
    total_variation = (actual - actual.mean()).pow(2).sum()
    if len(actual) < 2:
        scores["R²"] = float("nan")
    elif total_variation == 0:
        # Match sklearn's default for a constant target.
        scores["R²"] = squared_errors.sum().eq(0).astype(float)
    else:
        scores["R²"] = 1 - squared_errors.sum() / total_variation

    return scores.sort_values(MAE_LABEL)


def show_error_bars(scores):
    st.subheader("MAE and RMSE by model")
    st.caption("Shorter bars mean smaller errors across the full test period.")

    # Both charted metrics use LKR. R² is shown in the table.
    metric_data = (
        scores[[MAE_LABEL, RMSE_LABEL]]
        .rename_axis("Model")
        .reset_index()
        .melt(id_vars="Model", var_name="Metric", value_name="Error")
    )

    bars = (
        alt.Chart(metric_data)
        .mark_bar()
        .encode(
            y=alt.Y("Model:N", sort=scores.index.tolist()),
            yOffset=METRIC_FIELD,
            x=alt.X("Error:Q", title="Error (LKR per share)"),
            color=alt.Color(
                METRIC_FIELD,
                scale=alt.Scale(
                    domain=[MAE_LABEL, RMSE_LABEL],
                    range=["#4D8DFF", "#E68A35"],
                ),
            ),
            tooltip=[
                "Model:N",
                METRIC_FIELD,
                alt.Tooltip("Error:Q", format=".3f"),
            ],
        )
        .properties(height=54 * len(scores))
    )

    st.altair_chart(bars, use_container_width=True)


def show_scatter_plot(values, models, filename):
    st.subheader("Actual vs predicted closing price")
    st.caption(
        "Each dot represents one test day. A dot on the dashed line is a "
        "perfect prediction. Above the line, the forecast was too high. "
        "Below the line, it was too low."
    )

    default_index = next(
        (index for index, model in enumerate(models) if model != BASELINE),
        0,
    )
    selected_model = st.selectbox(
        "Scatter plot model",
        options=models,
        index=default_index,
        key="scatter_" + filename,
    )

    scatter_data = pd.DataFrame(
        {
            "Date": values.index,
            ACTUAL: values[ACTUAL].to_numpy(),
            "Forecast": values[selected_model].to_numpy(),
        }
    )

    # Keep the axes consistent when the user switches between models.
    limits = [float(values.min().min()) - 1, float(values.max().max()) + 1]
    x_axis = alt.X(
        "Actual:Q",
        title="Actual close (LKR)",
        scale=alt.Scale(domain=limits, nice=False),
    )
    y_axis = alt.Y(
        "Forecast:Q",
        title="Predicted close (LKR)",
        scale=alt.Scale(domain=limits, nice=False),
    )

    diagonal = (
        alt.Chart(pd.DataFrame({ACTUAL: limits, "Forecast": limits}))
        .mark_line(strokeDash=[6, 4], color="#888888")
        .encode(x=x_axis, y=y_axis)
    )
    dots = (
        alt.Chart(scatter_data)
        .mark_circle(size=42, opacity=0.55, color="#4D8DFF")
        .encode(
            x=x_axis,
            y=y_axis,
            tooltip=[
                alt.Tooltip(DATE_FIELD, format=DATE_FORMAT),
                alt.Tooltip("Actual:Q", title="Actual close (LKR)", format=".3f"),
                alt.Tooltip("Forecast:Q", title="Forecast (LKR)", format=".3f"),
            ],
        )
    )

    st.altair_chart(
        (diagonal + dots).properties(height=390),
        use_container_width=True,
    )


def show_experiment(filename):
    # Use the exported forecasts already stored in the repository.
    path = APP_FOLDER / filename
    if path.is_file():
        source = path
    else:
        source = st.file_uploader(
            f"Upload {filename}",
            type="csv",
            key="upload_" + filename,
        )
        if source is None:
            return

    try:
        values = load_forecasts(source)
    except (ValueError, OSError) as error:
        st.error(str(error))
        return

    models = values.columns.drop(ACTUAL).tolist()
    dates = values.index.tolist()

    visible_models = st.multiselect(
        "Show forecasts",
        options=models,
        default=models,
        key="models_" + filename,
    )
    day = st.selectbox(
        "Explore a test trading day",
        options=dates,
        index=len(dates) - 1,
        format_func=lambda date: date.strftime(DATE_FORMAT),
        key="day_" + filename,
    )

    with st.expander("How to use this tab"):
        st.markdown(
            "- **Show forecasts** selects the model lines to display alongside "
            "the actual closing price.\n"
            "- **Explore a test trading day** marks the date on the price "
            "chart and updates the selected-day table.\n"
            "- **MAE** measures the average size of a forecast error. **RMSE** "
            "gives larger errors more weight. Lower values are better for both.\n"
            "- **R²** compares a model’s squared errors with those of always "
            "predicting the mean actual test price. Higher is better, and 1 "
            "means perfect predictions.\n"
            "- **Scatter plot model** chooses which model to inspect across "
            "the full test period."
        )

    show_price_chart(values, visible_models, day)

    errors = values[models].sub(values[ACTUAL], axis=0)
    scores = calculate_scores(values[ACTUAL], errors)

    left, right = st.columns(2)

    with left:
        st.subheader("Selected trading day")
        st.write(f"Actual close · LKR {values.at[day, ACTUAL]:.2f}")

        daily = pd.DataFrame(
            {
                "Forecast (LKR)": values.loc[day, models],
                "Absolute error (LKR)": errors.loc[day].abs(),
            }
        )
        st.dataframe(daily.sort_values("Absolute error (LKR)").round(3))

    with right:
        st.subheader("Full test period")
        st.write(
            f"{len(dates)} trading days · "
            f"{dates[0].strftime(DATE_FORMAT)} – "
            f"{dates[-1].strftime(DATE_FORMAT)}"
        )
        st.dataframe(scores.round(3))

    show_error_bars(scores)
    show_scatter_plot(values, models, filename)


st.set_page_config(page_title="SLT forecasts", layout="wide")
show_project_intro()

for tab, filename in zip(st.tabs(list(EXPERIMENTS)), EXPERIMENTS.values()):
    with tab:
        show_experiment(filename)
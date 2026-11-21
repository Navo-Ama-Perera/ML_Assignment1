from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATE_FIELD = "Date:T"
SERIES_COLORS = ("#4D8DFF", "#E6EC2B", "#23E033", "#E71111", "#9333EA")

st.set_page_config(page_title="SLT forecasts", layout="wide")
st.title("SLT ensemble learning dashboard")
st.caption("Sri Lanka Telecom · Next-trading-day closing price · LKR")

dataset_path = Path(__file__).with_name("TeleCom_2009_2023_Daily.csv")
if dataset_path.is_file():
    st.download_button(
        "Dataset: Download SLT daily prices (CSV)",
        data=dataset_path.read_bytes(),
        file_name=dataset_path.name,
        mime="text/csv",
    )

experiments = {
    "RF vs XGBoost & Stacking": "01_ensemble_predictions.csv",
    "ARIMA + Random Forest": "02_arima_rf_predictions.csv",
    "Random Forest + ANN": "03_rf_ann_predictions.csv",
}

def show_experiment(filename):
    path = Path(__file__).parent / filename
    source = path if path.is_file() else st.file_uploader(
        f"Select {filename}", type="csv", key=filename
    )
    if source is None:
        return
    try:
        frame = pd.read_csv(source)
        if not {"Date", "Actual"}.issubset(frame):
            raise ValueError("The export needs Date and Actual columns.")
        frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
        values = frame.set_index("Date").sort_index().apply(pd.to_numeric, errors="raise")
        if values.empty or values.index.has_duplicates or values.index.hasnans or values.isna().any().any():
            raise ValueError("Check the export for missing values, empty rows or duplicate dates.")
        models = values.columns.drop("Actual").tolist()
        if not models:
            raise ValueError("Include at least one forecast column.")
    except (ValueError, OSError) as error:
        st.error(str(error))
        return
    visible = st.multiselect("Show forecasts", models, default=models, key="models"+filename)
    dates = values.index.tolist()
    day = st.selectbox("Explore a test trading day", dates, index=len(dates)-1,
                       format_func=lambda d: d.strftime("%d %b %Y"), key="day"+filename)
    st.markdown(
        """
**How to use**

- **Show forecasts** lets you choose which model lines appear. The **Actual** line shows the observed closing price.
- **Previous close** is the baseline. Select it to compare the models with a simple forecast that uses the prior trading day's close.
- **Explore a test trading day** lists the trading dates used to evaluate the models. Pick a date to mark it on the chart and see each forecast and its error for that day.
- **Absolute error** is how far a forecast was from the actual close on the selected day. **MAE** is the average absolute error over all test dates. **RMSE** gives larger errors more weight. Lower values are better.
"""
    )
    chart_data = values[["Actual", *visible]].reset_index().melt(
        id_vars="Date", var_name="Series", value_name="Price"
    )
    date_axis = alt.X(DATE_FIELD, axis=alt.Axis(format="%b %Y"))
    palette = dict(zip(["Actual", *models], SERIES_COLORS))
    displayed_series = ["Actual", *visible]
    if len(models) < len(SERIES_COLORS):
        color_scale = alt.Scale(
            domain=displayed_series,
            range=[palette[name] for name in displayed_series],
        )
    else:
        color_scale = alt.Scale(scheme="tableau20")

    chart = alt.Chart(chart_data).mark_line().encode(
        x=date_axis,
        y=alt.Y("Price:Q", title="Closing price (LKR)", scale=alt.Scale(zero=False)),
        color=alt.Color("Series:N", scale=color_scale),
        tooltip=[DATE_FIELD, "Series:N", "Price:Q"],
    )
    marker = alt.Chart(pd.DataFrame({"Date": [day]})).mark_rule(color="gray").encode(x=date_axis)
    st.altair_chart((chart + marker).interactive(), use_container_width=True)
    errors = values[models].sub(values["Actual"], axis=0)
    left, right = st.columns(2)
    with left:
        st.subheader("Selected trading day")
        st.write(f"Actual close · LKR {values.at[day, 'Actual']:.2f}")
        daily = pd.DataFrame({"Forecast (LKR)": values.loc[day, models],
                              "Absolute error (LKR)": errors.loc[day].abs()})
        st.dataframe(daily.sort_values("Absolute error (LKR)").round(3))
    with right:
        st.subheader("Full test period")
        st.write(f"{len(dates)} trading days · {dates[0]:%d %b %Y} – {dates[-1]:%d %b %Y}")
        scores = pd.DataFrame({"MAE (LKR)": errors.abs().mean(),
                               "RMSE (LKR)": errors.pow(2).mean().pow(0.5)})
        st.dataframe(scores.sort_values("MAE (LKR)").round(3))

for tab, filename in zip(st.tabs(list(experiments)), experiments.values()):
    with tab:
        show_experiment(filename)

# =====================================================
# AI-BASED SALES FORECASTING SYSTEM
# FINAL CORRECT STREAMLIT APP
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from statsmodels.tsa.arima.model import ARIMA

import warnings
warnings.filterwarnings('ignore')

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title='Sales Forecasting System',
    layout='centered'
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 4rem;
        padding-right: 4rem;
        max-width: 900px;
    }

    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# TITLE
# =====================================================

st.title('AI-Based Sales Forecasting System')

st.write(
    'Machine Learning and Time Series Forecasting Project'
)

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    'Upload Any Sales Dataset (CSV)',
    type=['csv']
)

# =====================================================
# MAIN APPLICATION
# =====================================================

if uploaded_file is not None:

    # =====================================================
    # LOAD DATASET
    # =====================================================

    df = pd.read_csv(uploaded_file)

    st.subheader('Dataset Preview')

    st.dataframe(
        df.head(),
        height=220
    )

    # =====================================================
    # COLUMN SELECTION
    # =====================================================

    all_columns = df.columns.tolist()

    st.subheader('Select Important Columns')

    # DATE COLUMN

    date_column = st.selectbox(
        'Select Date Column',
        all_columns
    )

    # =====================================================
    # DATE CONVERSION
    # =====================================================

    df[date_column] = pd.to_datetime(
        df[date_column],
        errors='coerce'
    )

    # Remove invalid dates

    df = df.dropna(
        subset=[date_column]
    )

    # =====================================================
    # NUMERIC COLUMNS
    # =====================================================

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # SALES COLUMN

    sales_column = st.selectbox(
        'Select Sales Column',
        numeric_columns
    )

    # =====================================================
    # SORT DATA
    # =====================================================

    df = df.sort_values(
        date_column
    )

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    df['Year'] = df[date_column].dt.year

    df['Month'] = df[date_column].dt.month

    df['Day'] = df[date_column].dt.day

    df['Week'] = (
        df[date_column]
        .dt
        .isocalendar()
        .week
        .astype(int)
    )

    # =====================================================
    # SALES TREND ANALYSIS
    # =====================================================

    st.subheader('Sales Trend Analysis')

    fig, ax = plt.subplots(
        figsize=(10,4)
    )

    ax.plot(
        df[date_column],
        df[sales_column]
    )

    ax.set_title(
        'Sales Trend Over Time'
    )

    ax.set_xlabel('Date')

    ax.set_ylabel('Sales')

    st.pyplot(fig)

    # =====================================================
    # FEATURE SELECTION
    # =====================================================

    feature_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # Remove target column

    if sales_column in feature_columns:

        feature_columns.remove(
            sales_column
        )

    # Remove datetime columns accidentally converted

    clean_features = []

    for col in feature_columns:

        if not pd.api.types.is_datetime64_any_dtype(df[col]):

            clean_features.append(col)

    X = df[clean_features]

    y = df[sales_column]

    # =====================================================
    # HANDLE MISSING VALUES
    # =====================================================

    X = X.fillna(0)

    y = y.fillna(0)

    # =====================================================
    # TRAIN TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # =====================================================
    # SAVE TRAIN AND TEST DATASETS
    # =====================================================

    train_data = X_train.copy()

    train_data['Target'] = y_train.values

    train_data.to_csv(
        'train_dataset.csv',
        index=False
    )

    test_data = X_test.copy()

    test_data['Target'] = y_test.values

    test_data.to_csv(
        'test_dataset.csv',
        index=False
    )

    # =====================================================
    # RANDOM FOREST MODEL
    # =====================================================

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    # =====================================================
    # PREDICTIONS
    # =====================================================

    predictions = model.predict(
        X_test
    )

    # =====================================================
    # EVALUATION METRICS
    # =====================================================

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    # =====================================================
    # DISPLAY METRICS
    # =====================================================

    st.subheader(
        'Model Performance'
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        'MAE',
        round(mae,2)
    )

    col2.metric(
        'MSE',
        round(mse,2)
    )

    col3.metric(
        'RMSE',
        round(rmse,2)
    )

    # =====================================================
    # ACTUAL VS PREDICTED GRAPH
    # =====================================================

    st.subheader(
        'Actual vs Predicted Sales'
    )

    fig2, ax2 = plt.subplots(
        figsize=(10,4)
    )

    ax2.plot(
        y_test.values[:100],
        label='Actual Sales'
    )

    ax2.plot(
        predictions[:100],
        label='Predicted Sales'
    )

    ax2.legend()

    st.pyplot(fig2)

    # =====================================================
    # TIME SERIES FORECASTING
    # =====================================================

    st.subheader(
        'Future Sales Forecast'
    )

    sales_series = df.groupby(
        date_column
    )[sales_column].sum()

    sales_series = pd.to_numeric(
        sales_series,
        errors='coerce'
    )

    sales_series = sales_series.dropna()

    # =====================================================
    # ARIMA MODEL
    # =====================================================

    arima_model = ARIMA(
        sales_series,
        order=(5,1,0)
    ).fit()

    # =====================================================
    # FUTURE FORECAST
    # =====================================================

    future_forecast = arima_model.forecast(
        steps=30
    )

    future_dates = pd.date_range(
        start=sales_series.index[-1],
        periods=30,
        freq='D'
    )

    forecast_df = pd.DataFrame({

        'Date': future_dates,

        'Forecasted Sales': future_forecast

    })

    # =====================================================
    # SAVE FORECAST RESULTS
    # =====================================================

    forecast_df.to_csv(
        'forecast_results.csv',
        index=False
    )

    st.dataframe(
        forecast_df,
        height=250
    )

    # =====================================================
    # FORECAST GRAPH
    # =====================================================

    fig3, ax3 = plt.subplots(
        figsize=(10,4)
    )

    ax3.plot(
        sales_series.index,
        sales_series.values,
        label='Historical Sales'
    )

    ax3.plot(
        future_dates,
        future_forecast,
        label='Forecasted Sales'
    )

    ax3.legend()

    st.pyplot(fig3)

    # =====================================================
    # DOWNLOAD BUTTON
    # =====================================================

    csv = forecast_df.to_csv(
        index=False
    )

    st.download_button(

        label='Download Forecast Results',

        data=csv,

        file_name='forecast_results.csv',

        mime='text/csv'

    )

    # =====================================================
    # SUCCESS MESSAGE
    # =====================================================

    st.success(
        'Sales Forecasting System Running Successfully!'
    )

# =====================================================
# NO FILE UPLOADED
# =====================================================

else:

    st.info(
        'Upload any sales dataset to begin forecasting.'
    )
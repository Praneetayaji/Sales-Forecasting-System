# Sales Forecasting System

The Sales Forecasting System is a Machine Learning and Time Series Analysis based web application that predicts future sales using historical sales data. The project analyzes sales trends, performs forecasting, and generates future sales predictions to help businesses improve planning, inventory management, and decision-making.

The system supports uploading different sales datasets and uses Machine Learning and ARIMA Time Series models for accurate forecasting.

---

# Features

- Sales prediction using Machine Learning
- Future sales forecasting
- Time Series Analysis
- Sales trend visualization
- Data preprocessing and feature engineering
- Train-test dataset splitting
- Random Forest Regression model
- ARIMA forecasting model
- Evaluation metrics and analytics
- Interactive Streamlit frontend
- Forecast result download

---

# Technologies Used

## Programming Language
- Python

## Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels
- Streamlit
- Joblib

---

# Dataset Used

Walmart Sales Dataset (`Walmart.csv`)

### Example Dataset Columns

- Store
- Date
- Weekly_Sales
- Holiday_Flag
- Temperature
- Fuel_Price
- CPI
- Unemployment

The system can also work with other sales datasets dynamically.

---

# Machine Learning Models Used

- Random Forest Regressor
- ARIMA Time Series Model

Random Forest is used for sales prediction and ARIMA is used for future time series forecasting.

---

# Project Structure

```text
Sales-Forecasting-System/
│
├── Walmart.csv
├── Sales_Forecasting.ipynb
├── app.py
├── random_forest_model.pkl
├── arima_model.pkl
├── train_dataset.csv
├── test_dataset.csv
├── forecast_results.csv
├── requirements.txt
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/Sales-Forecasting-System.git
```

## Open Project Folder

```bash
cd Sales-Forecasting-System
```

## Install Required Libraries

```bash
pip install -r requirements.txt
```

---

# Run the Project

## Run Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Sales_Forecasting.ipynb
```

Run all cells to:
- preprocess data
- perform EDA
- train models
- evaluate performance
- save trained models

---

## Run Streamlit Frontend

```bash
streamlit run app.py
```

The application will open in the browser at:

```text
http://localhost:8501
```

---

# Evaluation Metrics

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)

---

# Frontend Features

- CSV dataset upload
- Dataset preview
- Date and sales column selection
- Sales trend analysis
- Forecast visualization
- Actual vs Predicted graph
- Download forecast results

---

# Future Enhancements

- Real-time forecasting
- LSTM deep learning model
- Cloud deployment
- ERP integration
- Multi-store forecasting
- AI-based demand prediction

---

# Conclusion

This project demonstrates how Machine Learning and Time Series Analysis can be used to predict future sales and support data-driven business decisions. The system helps businesses forecast demand, identify trends, reduce losses, and improve planning effectively.

---

# Author

Praneeta Narayan Yaji

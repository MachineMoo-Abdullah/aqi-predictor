# 🌍 AQI Predictor — 3-Day Air Quality Forecasting

An end-to-end **Machine Learning application for forecasting Air Quality Index (AQI) for the next three days** using historical air-quality and meteorological data.

The project combines a **Random Forest multi-output regression model**, **SHAP explainability**, **Hugging Face model hosting**, and a **Streamlit web application** to provide interpretable AQI forecasts.

---

## 🚀 Live Application

🔗 **Streamlit App:**  
https://aqi-predictor-abduullah.streamlit.app/

🔗 **Production Model:**  
https://huggingface.co/abdullahadnan123/Random_Forest

---

## 📌 Project Overview

Air pollution is an important environmental and public-health concern. Accurate short-term AQI forecasting can help individuals, organizations, and decision-makers better understand upcoming air-quality conditions.

This project predicts the **average AQI for the next three days** using the most recent historical observations.

The system uses the previous **72 hourly observations** as input to predict:

- 📅 Day 1 AQI
- 📅 Day 2 AQI
- 📅 Day 3 AQI

The trained Random Forest model is hosted separately on Hugging Face to keep the GitHub repository lightweight and suitable for cloud deployment.

---

## ✨ Key Features

- 🔮 **3-Day AQI Forecasting**
- 🌲 **Random Forest Multi-Output Regression**
- ⏱️ Uses the latest **72 hours of historical data**
- 🌦️ Incorporates meteorological variables
- 🏭 Incorporates major air-pollutant measurements
- 📊 Interactive AQI forecast visualization
- 🧠 **SHAP-based model explainability**
- ❤️ Health recommendations based on predicted AQI
- 📈 Historical AQI trend visualization
- ☁️ Production model hosted on Hugging Face
- 🚀 Streamlit Cloud deployment
- 🔄 Automated model retraining through GitHub Actions
- 🏆 Candidate-vs-production model comparison
- 💾 Training history and evaluation tracking

---

# 🏗️ System Architecture

```text
                         ┌─────────────────────────────┐
                         │       Historical Data       │
                         │                             │
                         │  Weather + Pollutants + AQI │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │      Data Processing        │
                         │                             │
                         │ • Cleaning                  │
                         │ • Feature Engineering       │
                         │ • Daily AQI                 │
                         │ • 72-hour sequences         │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │       Model Training        │
                         │                             │
                         │ Random Forest Regressor     │
                         │ Multi-output regression     │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │      Model Evaluation       │
                         │                             │
                         │ R² • MAE • RMSE             │
                         │ Day 1 • Day 2 • Day 3       │
                         └──────────────┬──────────────┘
                                        │
                              Candidate vs Production
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
                         ▼                             ▼
                  ┌──────────────┐             ┌──────────────┐
                  │ Better Model │             │ Not Better   │
                  │     YES      │             │     NO       │
                  └──────┬───────┘             └──────┬───────┘
                         │                            │
                         ▼                            │
              ┌──────────────────────┐               │
              │ Hugging Face          │◄──────────────┘
              │ Production Model      │
              │                       │
              │ rf_production.pkl     │
              └──────────┬────────────┘
                         │
                         │ Download
                         ▼
              ┌────────────────────────┐
              │    Streamlit Cloud     │
              │                        │
              │       src/app.py       │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │ Latest 72 Hours Data   │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │ Random Forest Model    │
              │                        │
              │ Predict next 3 days    │
              └───────────┬────────────┘
                          │
                ┌─────────┼─────────┐
                ▼         ▼         ▼
             Day 1      Day 2      Day 3
              AQI        AQI        AQI
                │         │         │
                └─────────┼─────────┘
                          ▼
              ┌────────────────────────┐
              │     Streamlit UI       │
              │                        │
              │ • AQI Forecast         │
              │ • AQI Category         │
              │ • Health Guidance      │
              │ • Historical Trend     │
              │ • SHAP Explanation     │
              └────────────────────────┘

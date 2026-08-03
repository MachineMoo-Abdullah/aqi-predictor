**What is the project?**

it tells you,

- Current AQI: 142 (Unhealthy for Sensitive Groups)
- Tomorrow: 135
- Day 2: 118
- Day 3: 104

This project is about building the complete system behind that app.

**What is AQI?**

AQI stands for Air Quality Index.
It measures how polluted the air is.

Example:

| AQI Range | Air Quality Category |
|-----------|----------------------|
| 0–50 | Good |
| 51–100 | Moderate |
| 101–150 | Unhealthy for Sensitive Groups |
| 151–200 | Unhealthy |
| 201–300 | Very Unhealthy |
| 301+ | Hazardous |

Instead of measuring AQI yourself, you'll collect pollution data from APIs.

**Main Goal**

Predict:
AQI for the next 3 days

Using,
- Weather
- Pollution
- Historical AQI
- Machine Learning

**Interesting Features**

Real-Time Data Collection – Collects live weather and air pollution data from external APIs.
3-Day AQI Forecasting – Predicts the Air Quality Index for the next 3 days using machine learning.
Automated MLOps Pipeline – Automatically collects data, retrains the model, and updates predictions.
Interactive Dashboard – Displays real-time AQI, forecasts, weather, and pollutant information.
Explainable AI & Hazard Alerts – Explains model predictions using SHAP/LIME and alerts users when AQI reaches unhealthy levels.
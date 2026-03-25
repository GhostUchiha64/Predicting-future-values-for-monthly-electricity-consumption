# Electricity Consumption Forecasting: Minneapolis & Phoenix

**ML Team Project — Fall 2025** | Vigneswaran & Siddartha Bandi

---

## Overview

This project develops and evaluates machine learning models to forecast monthly **electricity consumption** in two climatically distinct U.S. cities: **Minneapolis, Minnesota** (cold climate) and **Phoenix, Arizona** (hot/arid climate). By incorporating meteorological features such as temperature, solar radiation, humidity, and engineered lag variables, the models aim to capture seasonal and temporal patterns in residential/commercial energy demand.

A key emphasis of this project is **model interpretability** — using SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) to understand which features drive predictions for each city.

---

## Data Sources

| Dataset | Description |
|---------|-------------|
| `minnesota-dataframe.xlsx` | Historical electricity usage + meteorological features for Minneapolis, MN |
| `arizona-dataframe.xlsx` | Historical electricity usage + meteorological features for Phoenix, AZ |

**Key Features:**
- `t2m` — 2-meter temperature
- `d2m` — Dew point temperature
- `solar_rad` — Solar radiation
- `HUMIDEX_U` — Humidex index (Minneapolis only)
- `month`, `month_sin`, `month_cos` — Seasonal encoding
- `lag_elecuse_1/6/12` — Electricity usage lags (1, 6, 12 months)
- `lag_temp_1/6/12`, `temp_roll3`, `temp_diff` — Temperature lags and rolling stats
- `lag_solar_1/6/12`, `solar_roll3` — Solar radiation lags
- **Target:** `elecuse` — Monthly electricity consumption

---

## Methodology

### Models Trained
| Model | Type | Notes |
|-------|------|-------|
| **FFNN** (Feed-Forward Neural Network) | Deep Learning | Primary model; trained separately per city |
| **LSTM** | Deep Learning | Sequential time-series modeling |
| **SVR** | Machine Learning | Support Vector Regression baseline |

### Pipeline
1. **Feature Engineering** — lag variables, rolling means, seasonal sine/cosine encoding
2. **Preprocessing** — MinMaxScaler normalization, train/test split (no shuffle, time-ordered)
3. **Model Training** — TimeSeriesSplit cross-validation to prevent data leakage
4. **Evaluation** — R², MAE, RMSE logged per city
5. **Interpretability**
   - **SHAP** — global feature importance rankings (bar plots)
   - **LIME** — local prediction explanations for individual samples
   - **Correlation Heatmaps** — feature relationship visualization

---

## Repository Contents

```
Final Deliverables ML Project/
├── README.md
├── ML_vig:sid_python_file.ipynb              # Main Jupyter Notebook (Minneapolis + Phoenix)
├── minnesota-dataframe.xlsx                  # Minneapolis feature dataset
├── arizona-dataframe.xlsx                    # Phoenix feature dataset
├── class-final-citations.rtf                 # Project citations/references
├── Feedback(SELFI).png                       # Peer feedback screenshot
├── Peer Evaluation Report.pdf               # Team peer evaluation
├── Vig,Sid ML Project Fall 2025 Report.pdf  # Final written report
└── Vig,Sid-ML Project Slides_For Zoom
    Video Presentation 1.pptx               # Presentation slides
```

> **Note:** The video presentation file (`.mp4`) is excluded from this repository due to file size.

---

## Results Summary

- FFNN models were trained independently for Minneapolis and Phoenix, capturing city-specific climate patterns
- SHAP analysis revealed that **temperature lags** and **seasonal encodings** were the top predictors for both cities
- Phoenix model excluded Humidex (irrelevant for desert climate), demonstrating feature relevance by geography
- TimeSeriesSplit cross-validation ensured temporally-honest performance estimates

---

## Setup & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn tensorflow keras plotly seaborn matplotlib shap lime joblib
```

### Running the Notebook
1. Open `ML_vig:sid_python_file.ipynb` in Google Colab
2. Mount your Google Drive and place model artifacts (`.keras`, `.pkl`) in the expected directory
3. Run cells sequentially — Part 1 covers Minneapolis, Part 2 covers Phoenix

---

## Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| ML/DL | Scikit-learn, TensorFlow/Keras |
| Interpretability | SHAP, LIME |
| Data | Pandas, NumPy, openpyxl |
| Visualization | Matplotlib, Seaborn, Plotly |
| Environment | Google Colab |

---

## Authors

**Vigneswaran & Siddartha Bandi**
Machine Learning — Fall 2025

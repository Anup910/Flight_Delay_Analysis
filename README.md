# Flight Delay Analysis ✈️📉

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)

## 📌 Project Overview

This project analyzes U.S. commercial flight records to uncover **patterns of delays** and build baseline models for predicting whether a flight will be **on-time** or **delayed**. The pipeline includes:

* Data cleaning and preprocessing
* Feature engineering of time-based and operational features
* Handling of class imbalance (SMOTE)
* Exploratory Data Analysis (EDA)
* Benchmarking ML models: Logistic Regression, Gradient Boosting, and KNN

---

## 🛠️ Tools & Technologies

* **Python:** pandas, numpy
* **ML & Modeling:** scikit-learn, imbalanced-learn, feature-engine
* **Visualization:** seaborn, matplotlib
* **Notebook:** Jupyter

---

## 📂 Dataset

* **Source file:** `flights_sample_3m.csv`
* **Size (after cleaning):** \~53,800 rows × \~345 features
* **Target:** `FLIGHT_STATUS` → `0` (on-time, ARR\_DELAY ≤ 10) or `1` (delayed, ARR\_DELAY > 10)
* **Class distribution:** \~80% on-time, 20% delayed

---

## 🔎 Exploratory Data Analysis (EDA)

* Delay class distribution visualization
* Correlation heatmap between operational features and target
* Key correlated features: `DEP_DELAY`, `ELAPSED_TIME_DIFF`, `TAXI_OUT`

---

## 🧪 Models & Results

* **Logistic Regression**: reported 100% accuracy (due to target leakage via `ARR_DELAY`)
* **Gradient Boosting**: \~98% accuracy, macro-F1 ≈ 0.96
* **K-Nearest Neighbors**: \~98% accuracy, macro-F1 ≈ 0.96

⚠️ **Caveat:** These results are inflated due to target leakage (features like `ARR_DELAY` were used to predict `FLIGHT_STATUS`, which is derived from `ARR_DELAY`).

---

## ✅ Recommendations

* Remove leakage features (`ARR_DELAY`, `ARR_TIME`, etc.) for realistic prediction.
* Use **time-based train/test split** instead of random split.
* Explore advanced models (LightGBM, XGBoost) and feature engineering (weather, congestion, historical averages).
* Consider dimensionality reduction for high-cardinality categorical encodings.

---

## 🚀 Impact

This analysis highlights operational factors most strongly tied to delays and sets up a baseline ML pipeline. With leakage removed and enriched features, it can be scaled into a **reliable forecasting system** for airlines, airports, and travel apps.

---

## 🔁 How to Reproduce

```bash
# Clone repo
https://github.com/your-username/flight-delay-analysis

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook Flight_Delay_Analysis.ipynb
```

---

## 📄 Requirements

A `requirements.txt` is provided for reproducibility:

```txt
pandas
numpy
scikit-learn
imbalanced-learn
feature-engine
matplotlib
seaborn
```

---

## 📈 Sample Visuals

* Class distribution plots
* Correlation heatmap
* Confusion matrices & classification reports

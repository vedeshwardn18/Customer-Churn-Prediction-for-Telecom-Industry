# Customer Churn Prediction (Telecom) – Web App

Supervised Learning – **Classification**: predict whether a telecom customer will churn (0 → Not Churn, 1 → Churn).

## Dataset

Use the **Telco Customer Churn** dataset from Kaggle:

- **Download:** [https://www.kaggle.com/blastchar/telco-customer-churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- After downloading, you’ll get a CSV (often named `WA_Fn-UseC_-Telco-Customer-Churn.csv`). You can rename it to `Telco-Customer-Churn.csv` if you like.
- Upload this file in **Step 1** of the web app.

## Setup

1. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Run the website

From the project folder:

```bash
streamlit run app.py
```

Your browser will open at **http://localhost:8501**.

## What the app does

| Step | Description |
|------|-------------|
| **1. Load & Clean Data** | Upload CSV, drop `customerID`, convert `TotalCharges` to numeric, drop missing rows. |
| **2. Exploratory Analysis** | Churn distribution, numeric feature histograms, correlation heatmap. |
| **3. Train Models** | Logistic Regression, Decision Tree, Gradient Boosting; train/test split 80/20. |
| **4. Hyperparameter Tuning** | GridSearchCV on Decision Tree (example). |
| **5. Model Comparison & Evaluation** | Accuracy table, confusion matrix, classification report for best model. |
| **6. Predict Churn** | Use the best model to predict on a sample or compatible single-row data. |

## Pipeline (course mapping)

- **Data** → **Cleaning** → **Feature encoding** → **Train/Test split** → **Training** → **Hyperparameter tuning** → **Evaluation** → **Comparison**
- **Boosting:** Gradient Boosting (often best accuracy ~85–88%).
- **Bias–variance:** LR (higher bias), Tree (higher variance), Boosting (balance).

## Deliverables

1. Dataset (from Kaggle)
2. This app (replace a notebook with the Streamlit UI)
3. Model comparison results (in the app)
4. Graphs and visualizations (in the app)
5. Conclusion: Gradient Boosting typically gives the highest accuracy.

## Requirements

- Python 3.8+
- See `requirements.txt`: pandas, numpy, matplotlib, seaborn, scikit-learn, streamlit

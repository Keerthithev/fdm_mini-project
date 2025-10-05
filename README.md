# Fraud Detection Streamlit App

## Setup

1. Create a virtual environment (recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Run the app
```bash
streamlit run streamlit_app.py
```

- The app will attempt to load default files:
  - Data: `synthetic_fraud_datasets.csv`
  - Model: `fraud_model-3.joblib`

You can change paths in the sidebar.

## Pages
- Overview: business framing and goals
- Data Exploration: preview, stats, missing values, simple charts
- Model Evaluation: metrics (Accuracy, Precision, Recall, F1, ROC AUC) if labels exist
- Batch Prediction: run predictions and download results
- Documentation: methodology pointers for your report

## Notes
- If your model is a scikit-learn Pipeline, feature selection/encoding inside the pipeline is handled automatically. Otherwise, select the feature columns in the UI.

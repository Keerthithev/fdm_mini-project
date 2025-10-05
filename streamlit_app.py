import io
import os
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st


def load_model(model_path: str):
    """Load a trained model from a joblib file.

    Supports scikit-learn estimators or Pipelines. Returns None if not found.
    """
    try:
        if not os.path.exists(model_path):
            return None
        return joblib.load(model_path)
    except Exception as exc:  # pragma: no cover
        st.error(f"Failed to load model from {model_path}: {exc}")
        return None


def safe_read_csv(file_or_path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(file_or_path)
    except Exception as exc:  # pragma: no cover
        st.error(f"Could not read CSV: {exc}")
        return None


def infer_candidate_target_columns(df: pd.DataFrame) -> List[str]:
    lowercase_cols = [c.lower() for c in df.columns]
    candidates = []
    for name in ["label", "target", "is_fraud", "fraud", "y"]:
        for col, low in zip(df.columns, lowercase_cols):
            if low == name:
                candidates.append(col)
    return candidates


def select_features_ui(df: pd.DataFrame, target_col: Optional[str]) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    default_features = [c for c in numeric_cols if c != target_col]
    st.caption("Select feature columns used for the model. Defaults to numeric columns.")
    return st.multiselect("Feature columns", options=df.columns.tolist(), default=default_features)


def ensure_dataframe_has_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    # Basic NA handling: fill numeric with median, others with mode
    prepared = df.copy()
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(prepared[col]):
            prepared[col] = prepared[col].fillna(prepared[col].median())
        else:
            if prepared[col].isna().any():
                mode_values = prepared[col].mode(dropna=True)
                fill_value = mode_values.iloc[0] if not mode_values.empty else ""
                prepared[col] = prepared[col].fillna(fill_value)
    return prepared[feature_cols]


def model_predict(model, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    preds = model.predict(X)
    proba = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            # Convert decision scores to 2-column pseudo-probabilities via min-max scaling
            if scores.ndim == 1:
                s_min, s_max = float(np.min(scores)), float(np.max(scores))
                s_scaled = (scores - s_min) / (s_max - s_min + 1e-9)
                proba = np.vstack([1.0 - s_scaled, s_scaled]).T
    except Exception:  # pragma: no cover
        proba = None
    return preds, proba


def explain_prediction(
    model,
    X_one: pd.DataFrame,
    reference_df: Optional[pd.DataFrame],
    prob1: Optional[float],
) -> Optional[pd.DataFrame]:
    """Return per-feature contributions toward class-1 probability if possible.

    Tries SHAP first; if unavailable, uses a simple perturbation sensitivity.
    Returns a DataFrame with columns: feature, contribution.
    """
    # Prefer SHAP if available and we have some background data
    try:
        import shap  # type: ignore

        background = None
        if reference_df is not None and not reference_df.empty:
            # Use numeric columns aligned with X_one columns
            bg_cols = [c for c in X_one.columns if c in reference_df.columns]
            if bg_cols:
                # Sample up to 200 rows for speed
                background = reference_df[bg_cols].select_dtypes(include=[np.number]).dropna()
                if len(background) > 200:
                    background = background.sample(200, random_state=0)
        if background is None or background.empty:
            background = X_one

        explainer = shap.Explainer(model, background)
        shap_values = explainer(X_one)

        # Handle different SHAP shapes
        vals = None
        if hasattr(shap_values, "values"):
            vals = np.array(shap_values.values)
        else:
            vals = np.array(shap_values)

        # If binary classification may be (n_samples, n_features) or (n_samples, 2, n_features)
        if vals.ndim == 2:
            contrib = vals[0]
        elif vals.ndim == 3 and vals.shape[1] >= 2:
            contrib = vals[0, 1]  # class 1 contributions
        else:
            return None

        return pd.DataFrame({"feature": X_one.columns, "contribution": contrib})
    except Exception:
        pass

    # Fallback: perturb each feature slightly and measure delta probability
    try:
        if prob1 is None:
            # Need probabilities for meaningful explanation
            return None
        base_prob = float(prob1)
        contributions = []
        for feature in X_one.columns:
            X_mod = X_one.copy()
            val = float(X_mod.iloc[0][feature])
            # +10% or +0.01 fallback
            delta = 0.1 * abs(val) if val != 0 else 0.01
            X_mod.at[X_mod.index[0], feature] = val + delta
            _, proba_mod = model_predict(model, X_mod)
            p1_mod = None
            if proba_mod is not None and proba_mod.ndim == 2 and proba_mod.shape[1] >= 2:
                p1_mod = float(proba_mod[0, 1])
            if p1_mod is None:
                continue
            contributions.append((feature, p1_mod - base_prob))
        if not contributions:
            return None
        return pd.DataFrame(contributions, columns=["feature", "contribution"]) 
    except Exception:
        return None


def render_overview(model_loaded: bool):
    st.header("Fraud Detection Dashboard")
    st.markdown(
        """
        This Streamlit app demonstrates a complete mini end-to-end fraud detection
        workflow: business framing, data exploration, model evaluation, and batch prediction.
        """
    )

    st.subheader("1) Problem definition and business goals")
    st.markdown(
        """
        - Identify potentially fraudulent transactions in near-real-time to reduce chargebacks and losses.
        - Balance risk and user experience: minimize false positives to avoid blocking legitimate users.
        - Provide explainable indicators to assist analysts during manual reviews.
        - Operational KPI examples: precision@topK, recall, AUC-ROC, and average handling time reduction.
        """
    )

    st.subheader("Functionality")
    st.markdown(
        """
        - Upload CSVs for exploration and evaluation.
        - Compute common classification metrics when ground-truth labels are available.
        - Run batch predictions and download results.
        - View model readiness status: {status}
        """.format(status=("Model loaded" if model_loaded else "Model not found"))
    )


def render_data_exploration(df: pd.DataFrame):
    st.header("Data Exploration")
    if df is None or df.empty:
        st.info("No data available. Upload a CSV in the sidebar or place one in the project root.")
        return

    st.write("Shape:", df.shape)
    st.dataframe(df.head(50))

    with st.expander("Summary statistics"):
        st.write(df.describe(include="all").transpose())

    with st.expander("Missing values"):
        na_counts = df.isna().sum().sort_values(ascending=False)
        st.bar_chart(na_counts)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.subheader("Distributions")
        col = st.selectbox("Select a numeric column", numeric_cols)
        st.line_chart(df[col].dropna().reset_index(drop=True))


def render_model_evaluation(model, df: pd.DataFrame):
    st.header("Model Evaluation")
    if model is None:
        st.warning("Model was not found. Place your trained model file in the root directory.")
        return
    if df is None or df.empty:
        st.info("Upload a labeled CSV in the sidebar to evaluate.")
        return

    candidate_targets = infer_candidate_target_columns(df)
    target_col = st.selectbox(
        "Select target/label column",
        options=["-- none --"] + df.columns.tolist(),
        index=(candidate_targets and df.columns.tolist().index(candidate_targets[0]) + 1) if candidate_targets else 0,
    )
    if target_col == "-- none --":
        st.info("Select a target column to compute metrics.")
        return

    feature_cols = select_features_ui(df, target_col)
    if not feature_cols:
        st.info("Select at least one feature column.")
        return

    try:
        X = ensure_dataframe_has_features(df, feature_cols)
    except ValueError as exc:
        st.error(str(exc))
        return

    y = df[target_col]
    try:
        preds, proba = model_predict(model, X)
    except Exception as exc:  # pragma: no cover
        st.error(f"Prediction failed: {exc}")
        return

    # Metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    y_clean = y
    try:
        # Try to coerce boolean/strings to ints
        if y_clean.dtype == bool:
            y_clean = y_clean.astype(int)
        elif y_clean.dtype == object:
            y_clean = y_clean.astype(str)
            if set(y_clean.unique()).issubset({"0", "1", "False", "True", "false", "true"}):
                y_clean = y_clean.str.lower().map({"0": 0, "1": 1, "false": 0, "true": 1})
    except Exception:  # pragma: no cover
        pass

    acc = accuracy_score(y_clean, preds)
    prec = precision_score(y_clean, preds, zero_division=0)
    rec = recall_score(y_clean, preds, zero_division=0)
    f1 = f1_score(y_clean, preds, zero_division=0)

    st.subheader("Classification metrics")
    st.metric("Accuracy", f"{acc:.3f}")
    st.metric("Precision", f"{prec:.3f}")
    st.metric("Recall", f"{rec:.3f}")
    st.metric("F1", f"{f1:.3f}")

    if proba is not None:
        try:
            if proba.ndim == 2 and proba.shape[1] >= 2:
                auc = roc_auc_score(y_clean, proba[:, 1])
                st.metric("ROC AUC", f"{auc:.3f}")
        except Exception:  # pragma: no cover
            pass

    # Confusion matrix
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    try:
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_clean, preds)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)
        st.pyplot(fig)
    except Exception:  # pragma: no cover
        st.write("Confusion matrix unavailable (matplotlib not installed?).")


def render_batch_prediction(model, df: Optional[pd.DataFrame]):
    st.header("Batch Prediction")
    if model is None:
        st.warning("Model was not found. Place your trained model file in the root directory.")
        return

    st.markdown("Upload a CSV or use the loaded data for prediction.")
    use_loaded = st.checkbox("Use currently loaded data (if any)", value=bool(df is not None and not df.empty))
    upload = None
    if not use_loaded:
        upload = st.file_uploader("Upload CSV for prediction", type=["csv"])
    pred_df = df if use_loaded else (safe_read_csv(upload) if upload is not None else None)

    if pred_df is None or pred_df.empty:
        st.info("Provide data to predict.")
        return

    feature_cols = select_features_ui(pred_df, target_col=None)
    if not feature_cols:
        st.info("Select at least one feature column.")
        return

    try:
        X = ensure_dataframe_has_features(pred_df, feature_cols)
    except ValueError as exc:
        st.error(str(exc))
        return

    try:
        preds, proba = model_predict(model, X)
    except Exception as exc:  # pragma: no cover
        st.error(f"Prediction failed: {exc}")
        return

    out = pred_df.copy()
    out["prediction"] = preds
    if proba is not None and proba.ndim == 2 and proba.shape[1] >= 2:
        out["probability"] = proba[:, 1]

    st.subheader("Preview")
    st.dataframe(out.head(50))

    csv_buf = io.StringIO()
    out.to_csv(csv_buf, index=False)
    st.download_button("Download predictions CSV", data=csv_buf.getvalue(), file_name="predictions.csv", mime="text/csv")


def render_single_prediction(model, reference_df: Optional[pd.DataFrame]):
    st.header("Single Prediction (Manual Input)")
    if model is None:
        st.warning("Model was not found. Place your trained model file in the root directory.")
        return

    st.markdown("Enter feature values to classify one transaction/event.")

    # Advanced-only input with improved UX
    mode = "Advanced (from dataset)"

    # Prefill buttons
    fraud_example_defaults = {
        "Transaction_Amount": 2499.99,
        "Account_Balance": 500.0,
        "Risk_Score": 0.92,
        "Transaction_Distance": 350.0,
        "Avg_Transaction_Amount_7d": 120.0,
        "Daily_Transaction_Count": 25,
        "Failed_Transaction_Count_7d": 4,
        "Card_Age": 10,
        "Previous_Fraudulent_Activity": True,
        "Is_Weekend": True,
        "Transaction_Type": "Online",
        "Device_Type": "Mobile",
        "Merchant_Category": "Electronics",
        "Card_Type": "Prepaid",
        "Authentication_Method": "OTP",
    }
    nonfraud_example_defaults = {
        "Transaction_Amount": 34.5,
        "Account_Balance": 8200.0,
        "Risk_Score": 0.08,
        "Transaction_Distance": 2.1,
        "Avg_Transaction_Amount_7d": 30.0,
        "Daily_Transaction_Count": 2,
        "Failed_Transaction_Count_7d": 0,
        "Card_Age": 950,
        "Previous_Fraudulent_Activity": False,
        "Is_Weekend": False,
        "Transaction_Type": "POS",
        "Device_Type": "Other",
        "Merchant_Category": "Grocery",
        "Card_Type": "Debit",
        "Authentication_Method": "PIN",
    }

    if "prefill_inputs" not in st.session_state:
        st.session_state["prefill_inputs"] = fraud_example_defaults

    b1, b2 = st.columns(2)
    if b1.button("Prefill: Fraud example"):
        st.session_state["prefill_inputs"] = fraud_example_defaults
    if b2.button("Prefill: Non-fraud example"):
        st.session_state["prefill_inputs"] = nonfraud_example_defaults

    defaults = st.session_state.get("prefill_inputs", fraud_example_defaults)

    with st.form("single_pred_form"):
        # Fixed, lenient threshold to catch borderline cases by default
        threshold = 0.35

        inputs = {}

        # Advanced mode: derive schema from dataset when available, else show domain-specific fields
        feature_cols: List[str] = []
        if reference_df is not None and not reference_df.empty:
            feature_cols = reference_df.columns.tolist()

        # Numeric fields
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            inputs["Transaction_Amount"] = st.number_input("Transaction Amount", value=float(defaults.get("Transaction_Amount", 0.0)), min_value=0.0, step=1.0)
            inputs["Account_Balance"] = st.number_input("Account Balance", value=float(defaults.get("Account_Balance", 0.0)), step=1.0)
            inputs["Risk_Score"] = st.slider("Risk Score", 0.0, 1.0, float(defaults.get("Risk_Score", 0.5)), 0.01)
            inputs["Transaction_Distance"] = st.number_input("Transaction Distance", value=float(defaults.get("Transaction_Distance", 0.0)), step=0.1)
        with col_b:
            inputs["Avg_Transaction_Amount_7d"] = st.number_input("Avg Amount (7d)", value=float(defaults.get("Avg_Transaction_Amount_7d", 0.0)), step=1.0)
            inputs["Daily_Transaction_Count"] = st.number_input("Daily Tx Count", value=int(defaults.get("Daily_Transaction_Count", 0)), min_value=0, step=1)
            inputs["Failed_Transaction_Count_7d"] = st.number_input("Failed Tx Count (7d)", value=int(defaults.get("Failed_Transaction_Count_7d", 0)), min_value=0, step=1)
            inputs["Card_Age"] = st.number_input("Card Age (days)", value=int(defaults.get("Card_Age", 0)), min_value=0, step=1)
        with col_c:
            inputs["Previous_Fraudulent_Activity"] = 1 if st.checkbox("Previous Fraudulent Activity", value=bool(defaults.get("Previous_Fraudulent_Activity", False))) else 0
            inputs["Is_Weekend"] = 1 if st.checkbox("Weekend transaction", value=bool(defaults.get("Is_Weekend", False))) else 0

            # Build dropdowns using dataset categories if columns exist, else defaults
            tx_type_options = ["Other", "Online", "POS"]
            device_options = ["Other", "Mobile", "Tablet"]
            merchant_options = ["Other", "Electronics", "Grocery", "Restaurants", "Travel"]
            card_type_options = ["Other", "Debit", "Prepaid"]
            auth_options = ["Other", "OTP", "PIN"]

            tx_type_default = defaults.get("Transaction_Type", "Other")
            device_default = defaults.get("Device_Type", "Other")
            merchant_default = defaults.get("Merchant_Category", "Other")
            card_type_default = defaults.get("Card_Type", "Other")
            auth_default = defaults.get("Authentication_Method", "Other")

            tx_type = st.selectbox("Transaction Type", tx_type_options, index=max(tx_type_options.index(tx_type_default) if tx_type_default in tx_type_options else 0, 0))
            device = st.selectbox("Device Type", device_options, index=max(device_options.index(device_default) if device_default in device_options else 0, 0))
            merchant = st.selectbox("Merchant Category", merchant_options, index=max(merchant_options.index(merchant_default) if merchant_default in merchant_options else 0, 0))
            card_type = st.selectbox("Card Type", card_type_options, index=max(card_type_options.index(card_type_default) if card_type_default in card_type_options else 0, 0))
            auth = st.selectbox("Authentication Method", auth_options, index=max(auth_options.index(auth_default) if auth_default in auth_options else 0, 0))

        # One-hot encode selections
        one_hot = {
            "Transaction_Type_Online": 1 if tx_type == "Online" else 0,
            "Transaction_Type_POS": 1 if tx_type == "POS" else 0,
            "Device_Type_Mobile": 1 if device == "Mobile" else 0,
            "Device_Type_Tablet": 1 if device == "Tablet" else 0,
            "Merchant_Category_Electronics": 1 if merchant == "Electronics" else 0,
            "Merchant_Category_Grocery": 1 if merchant == "Grocery" else 0,
            "Merchant_Category_Restaurants": 1 if merchant == "Restaurants" else 0,
            "Merchant_Category_Travel": 1 if merchant == "Travel" else 0,
            "Card_Type_Debit": 1 if card_type == "Debit" else 0,
            "Card_Type_Prepaid": 1 if card_type == "Prepaid" else 0,
            "Authentication_Method_OTP": 1 if auth == "OTP" else 0,
            "Authentication_Method_PIN": 1 if auth == "PIN" else 0,
        }
        inputs.update(one_hot)

        # Ensure expected features exist (zeros for missing)
        expected = [
            "Transaction_Amount","Account_Balance","Previous_Fraudulent_Activity","Daily_Transaction_Count",
            "Avg_Transaction_Amount_7d","Failed_Transaction_Count_7d","Card_Age","Transaction_Distance",
            "Risk_Score","Is_Weekend","Transaction_Type_Online","Transaction_Type_POS","Device_Type_Mobile",
            "Device_Type_Tablet","Merchant_Category_Electronics","Merchant_Category_Grocery",
            "Merchant_Category_Restaurants","Merchant_Category_Travel","Card_Type_Debit","Card_Type_Prepaid",
            "Authentication_Method_OTP","Authentication_Method_PIN",
        ]
        for f in expected:
            inputs.setdefault(f, 0)

        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    if not inputs:
        st.error("Provide at least one feature value.")
        return

    X_one = pd.DataFrame([inputs])

    try:
        preds, proba = model_predict(model, X_one)
    except Exception as exc:  # pragma: no cover
        st.error(f"Prediction failed: {exc}")
        return

    pred_class = int(preds[0]) if hasattr(preds, "__iter__") else int(preds)
    prob1 = None
    if proba is not None and proba.ndim == 2 and proba.shape[1] >= 2:
        prob1 = float(proba[0, 1])

    decision = None
    if prob1 is not None:
        decision = "FRAUD" if prob1 >= threshold else "NON-FRAUD"
    else:
        decision = "FRAUD" if pred_class == 1 else "NON-FRAUD"

    st.subheader("Result")
    if decision == "FRAUD":
        st.error(f"Decision: {decision}")
    else:
        st.success(f"Decision: {decision}")
    st.write("Predicted class:", pred_class)
    if prob1 is not None:
        st.write("Probability of fraud (class 1):", f"{prob1:.3f}")
        st.progress(min(max(int(prob1 * 100), 0), 100))

    # Explanations
    try:
        contrib_df = explain_prediction(model, X_one, reference_df, prob1)
    except Exception:
        contrib_df = None

    if contrib_df is not None and not contrib_df.empty:
        # Short summary paragraph first
        try:
            contrib_df["abs_contribution"] = contrib_df["contribution"].abs()
            inc_sorted = contrib_df.sort_values("contribution", ascending=False)
            dec_sorted = contrib_df.sort_values("contribution", ascending=True)
            top_inc = inc_sorted.head(2)[["feature","contribution"]].values.tolist()
            top_dec = dec_sorted.head(2)[["feature","contribution"]].values.tolist()
            drivers = ", ".join([f"{f} (+{c:.3f})" for f, c in top_inc]) if top_inc else ""
            mitigators = ", ".join([f"{f} ({c:.3f})" for f, c in top_dec]) if top_dec else ""
            prob_text = f" at probability {prob1:.3f}" if prob1 is not None else ""
            if decision == "FRAUD":
                summary = f"This transaction is classified as FRAUD{prob_text}, mainly due to {drivers}. Offsetting factors include {mitigators}."
            else:
                summary = f"This transaction is classified as NON-FRAUD{prob_text}. Risk is lowered by {mitigators}, while increasing factors were {drivers}."
            st.markdown(summary)
        except Exception:
            pass

        st.subheader("Why this decision?")
        # Sort by absolute impact and show top 8
        contrib_df["abs_contribution"] = contrib_df["contribution"].abs()
        top = contrib_df.sort_values("abs_contribution", ascending=False).head(8)

        plus = top[top["contribution"] > 0]
        minus = top[top["contribution"] < 0]

        col1, col2 = st.columns(2)
        with col1:
            st.caption("Factors increasing fraud probability")
            if not plus.empty:
                st.bar_chart(plus.set_index("feature")["contribution"]) 
            else:
                st.write("No strong increasing factors found.")
        with col2:
            st.caption("Factors decreasing fraud probability")
            if not minus.empty:
                # Flip sign to show magnitudes in positive direction
                st.bar_chart((-minus.set_index("feature")["contribution"]))
            else:
                st.write("No strong decreasing factors found.")

        # Friendly summary sentence
        top_feature = top.iloc[0]
        direction = "increased" if top_feature["contribution"] > 0 else "decreased"
        st.info(f"The feature '{top_feature['feature']}' most {direction} the fraud probability in this prediction.")

        # Detailed table with current values and (optional) dataset medians
        try:
            details = top.copy()
            details = details[["feature", "contribution"]]
            details["value"] = details["feature"].apply(lambda f: X_one.iloc[0][f] if f in X_one.columns else None)
            if reference_df is not None and not reference_df.empty:
                med_map = reference_df.median(numeric_only=True)
                details["dataset_median"] = details["feature"].apply(lambda f: float(med_map.get(f, np.nan)))
            st.markdown("Top factors with values")
            st.dataframe(details.rename(columns={"feature": "Feature", "contribution": "Impact (+ raises risk)", "value": "This input", "dataset_median": "Dataset median"}))
        except Exception:
            pass

        # Natural-language bullets for top 3 increasing and decreasing factors
        try:
            def _to_msg(row):
                name = row["feature"]
                val = X_one.iloc[0][name] if name in X_one.columns else None
                return f"- {name}: value {val} contributed {row['contribution']:+.3f} to the risk"

            inc_msgs = [ _to_msg(r) for _, r in plus.sort_values("contribution", ascending=False).head(3).iterrows() ]
            dec_msgs = [ _to_msg(r) for _, r in minus.sort_values("contribution").head(3).iterrows() ]
            if inc_msgs:
                st.markdown("Stronger risk drivers:")
                st.markdown("\n".join(inc_msgs))
            if dec_msgs:
                st.markdown("Risk-reducing signals:")
                st.markdown("\n".join(dec_msgs))
        except Exception:
            pass

        # Removed downloadable text report per request
    else:
        st.caption("Explanation not available for this model configuration.")

    # Save to history and offer downloads
    try:
        import json
        record = {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "decision": decision,
            "probability": float(prob1) if prob1 is not None else None,
            "inputs_json": json.dumps({k: (float(v) if isinstance(v, (int,float,np.floating)) else v) for k,v in inputs.items()}),
        }
        st.session_state["prediction_history"] = pd.concat([
            st.session_state["prediction_history"], pd.DataFrame([record])
        ], ignore_index=True)

        # Show simple history table with clear and download
        hist = st.session_state["prediction_history"].copy().sort_values("timestamp", ascending=False)
        with st.expander("Prediction history"):
            st.dataframe(hist)
            if st.button("Clear history"):
                st.session_state["prediction_history"] = st.session_state["prediction_history"].iloc[0:0]
            csv_buf = io.StringIO()
            st.session_state["prediction_history"].to_csv(csv_buf, index=False)
            st.download_button("Download history CSV", data=csv_buf.getvalue(), file_name="prediction_history.csv", mime="text/csv")

        # Removed downloadable explanation CSV per request
    except Exception:
        pass

def render_documentation():
    st.header("Documentation and Methodology")
    st.markdown(
        """
        - Business goals and functionalities identified and explained in detail.
        - Data selection, preparation, and preprocessing should include handling missing values, encoding categoricals,
          scaling or normalization for numeric features, and class imbalance treatment (e.g., class weights, resampling).
        - Modeling: baseline model first (e.g., Logistic Regression), then tuned models (e.g., XGBoost/RandomForest),
          with robust cross-validation and a rationale for chosen metrics.
        - Evaluation: include ROC AUC, PR AUC, confusion matrix, threshold analysis, and cost-sensitive evaluation.
        - Alternatives and justifications: compare multiple models and feature sets, discuss trade-offs.
        - Deployment: this Streamlit app provides interactive analysis and batch prediction for stakeholders.
        """
    )


def main():
    st.set_page_config(page_title="Fraud Detection", layout="wide")

    # Auto-load defaults (no setup UI)
    default_data_path = "/Users/keerthidev/Desktop/fdm_final/synthetic_fraud_datasets.csv"
    default_model_path = "/Users/keerthidev/Desktop/fdm_final/fraud_model-3.joblib"
    model = load_model(default_model_path)
    df = safe_read_csv(default_data_path) if os.path.exists(default_data_path) else None

    # Sidebar: navigation + history filters
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ("Single Prediction",), index=0)

    # Initialize session state for history
    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = pd.DataFrame(
            columns=[
                "timestamp","decision","probability","inputs_json"
            ]
        )

    if page == "Single Prediction":
        render_single_prediction(model, df)


if __name__ == "__main__":
    main()



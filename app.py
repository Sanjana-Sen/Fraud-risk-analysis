# # app.py
# # ------------------------------------------------------------
# # Financial Fraud Detection Model + Interactive Dashboard
# # - Upload your transactions CSV (or use the auto-generated demo dataset)
# # - Choose the target label column (e.g., `is_fraud` as 0/1)
# # - Train ML models, inspect metrics, charts, and feature importance
# # - Score new/unseen data and download flagged transactions
# # ------------------------------------------------------------

# import io
# import json
# import time
# from typing import List, Tuple

# import numpy as np
# import pandas as pd

# import streamlit as st

# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
#     confusion_matrix,
#     RocCurveDisplay,
#     PrecisionRecallDisplay,
# )
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, IsolationForest
# from sklearn.inspection import permutation_importance

# # --------- Page setup ---------
# st.set_page_config(
#     page_title="💳 Financial Fraud Detection",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# st.title("💳 Financial Fraud Detection")
# st.caption(
#     "Detect suspicious transactions, evaluate models, and get real‑time insights. Upload your data or explore the built‑in demo dataset."
# )

# # --------- Helpers ---------

# def generate_demo_data(n_samples: int = 50_000, fraud_rate: float = 0.02, random_state: int = 42) -> pd.DataFrame:
#     """Create a realistic-ish synthetic transaction dataset."""
#     rng = np.random.default_rng(random_state)

#     # Basic fields
#     amounts = np.round(np.exp(rng.normal(5, 0.8, n_samples)), 2)  # skewed amounts
#     channels = rng.choice(["POS", "ECOM", "ATM", "WALLET"], size=n_samples, p=[0.5, 0.25, 0.15, 0.10])
#     countries = rng.choice(["US", "IN", "GB", "DE", "SG", "AE", "BR", "ZA"], size=n_samples)
#     hours = rng.integers(0, 24, size=n_samples)
#     merchant_cat = rng.choice(["grocery", "electronics", "travel", "fashion", "fuel", "food"], size=n_samples)
#     # customer/merchant ids (hashed)
#     cust_id = rng.integers(10_000, 99_999, size=n_samples)
#     merch_id = rng.integers(1_000, 9_999, size=n_samples)

#     # base fraud probability influenced by some patterns
#     base = (
#         0.5 * (amounts > np.percentile(amounts, 90)).astype(int)
#         + 0.4 * (np.isin(hours, [0, 1, 2, 3, 4])).astype(int)   
#         + 0.3 * (channels == "ECOM").astype(int)
#         + 0.2 * (merchant_cat == "electronics").astype(int)
#     )
#     base = (base - base.min()) / (base.max() - base.min() + 1e-9)
#     # calibrate to target fraud rate
#     thresh = np.quantile(base, 1 - fraud_rate)
#     y = (base >= thresh).astype(int)

#     df = pd.DataFrame(
#         {
#             "transaction_id": np.arange(1, n_samples + 1),
#             "amount": amounts,
#             "channel": channels,
#             "country": countries,
#             "hour": hours,
#             "merchant_category": merchant_cat,
#             "customer_id": cust_id,
#             "merchant_id": merch_id,
#             "is_fraud": y,
#         }
#     )
#     return df


# def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
#     if target_col not in df.columns:
#         raise ValueError(f"Target column '{target_col}' not found in data.")
#     y = df[target_col].astype(int)
#     X = df.drop(columns=[target_col])
#     return X, y


# def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
#     cat_cols = [c for c in X.columns if X[c].dtype == 'object']
#     num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", StandardScaler(with_mean=False), num_cols),
#             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
#         ],
#         remainder="drop",
#         verbose_feature_names_out=False,
#     )
#     return preprocessor, num_cols, cat_cols


# def train_supervised_model(X: pd.DataFrame, y: pd.Series, model_name: str = "RandomForest", random_state: int = 42) -> Pipeline:
#     preprocessor, _, _ = build_preprocessor(X)

#     if model_name == "LogisticRegression":
#         model = LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=None)
#     elif model_name == "RandomForest":
#         model = RandomForestClassifier(
#             n_estimators=300,
#             max_depth=None,
#             min_samples_split=2,
#             n_jobs=-1,
#             class_weight="balanced_subsample",
#             random_state=random_state,
#         )
#     else:
#         raise ValueError("Unsupported model")

#     pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
#     pipe.fit(X, y)
#     return pipe


# def evaluate_classifier(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
#     y_prob = pipe.predict_proba(X_test)[:, 1]
#     y_pred = (y_prob >= 0.5).astype(int)
#     metrics = {
#         "accuracy": accuracy_score(y_test, y_pred),
#         "precision": precision_score(y_test, y_pred, zero_division=0),
#         "recall": recall_score(y_test, y_pred, zero_division=0),
#         "f1": f1_score(y_test, y_pred, zero_division=0),
#         "roc_auc": roc_auc_score(y_test, y_prob),
#         "conf_mat": confusion_matrix(y_test, y_pred),
#         "y_prob": y_prob,
#         "y_pred": y_pred,
#     }
#     return metrics


# def compute_feature_importance(pipe: Pipeline, X: pd.DataFrame, n_repeats: int = 3, random_state: int = 42) -> pd.DataFrame:
#     try:
#         r = permutation_importance(pipe, X, pipe.predict(X), n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
#         importances = r.importances_mean
#         feature_names = pipe.named_steps["pre"].get_feature_names_out()
#         imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
#         return imp_df
#     except Exception:
#         return pd.DataFrame()


# def score_unseen(pipe: Pipeline, df_unseen: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
#     proba = pipe.predict_proba(df_unseen)[:, 1]
#     out = df_unseen.copy()
#     out["fraud_probability"] = proba
#     out["flagged"] = (proba >= threshold).astype(int)
#     return out

# # --------- Sidebar: Data & Options ---------
# with st.sidebar:
#     st.header("⚙️ Configuration")

#     data_source = st.radio("Data source", ["Upload CSV", "Use demo dataset"], index=1)

#     uploaded = None
#     if data_source == "Upload CSV":
#         uploaded = st.file_uploader("Upload transactions CSV", type=["csv"]) 

#     target_col = st.text_input("Target column (label)", value="is_fraud", help="Binary 0/1 indicating fraud.")

#     model_choice = st.selectbox("Model", ["RandomForest", "LogisticRegression"], index=0)

#     test_size = st.slider("Test size (%)", 10, 40, 20, help="Portion of data used for testing.") / 100.0
#     threshold = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.01, help="Probability cutoff for classifying as fraud.")

#     st.divider()
#     st.subheader("Unsupervised (optional)")
#     iso_toggle = st.toggle("Run IsolationForest anomaly scores")

# # --------- Load data ---------
# if uploaded is not None:
#     df = pd.read_csv(uploaded)
#     source_note = "Uploaded CSV"
# else:
#     df = generate_demo_data(n_samples=60_000, fraud_rate=0.02)
#     source_note = "Demo dataset (synthetic)"

# st.info(f"Using: **{source_note}** | Rows: **{len(df):,}** | Columns: **{len(df.columns)}**")

# # Basic data preview
# with st.expander("🔎 Peek at data", expanded=False):
#     st.dataframe(df.head(20), use_container_width=True)

# # Ensure target exists
# if target_col not in df.columns:
#     st.error(f"Target column '{target_col}' was not found. Please update the target name in the sidebar or upload a CSV containing it.")
#     st.stop()

# # Train/Test split
# X, y = split_features_target(df, target_col)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=test_size, stratify=y, random_state=42
# )

# # --------- Train supervised model ---------
# with st.spinner("Training model..."):
#     pipe = train_supervised_model(X_train, y_train, model_name=model_choice)

# metrics = evaluate_classifier(pipe, X_test, y_test)

# # --------- Top KPIs ---------
# col1, col2, col3, col4 = st.columns(4)
# with col1:
#     st.metric("Fraud rate (test)", f"{y_test.mean()*100:.2f}%")
# with col2:
#     st.metric("Precision", f"{metrics['precision']:.3f}")
# with col3:
#     st.metric("Recall", f"{metrics['recall']:.3f}")
# with col4:
#     st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")

# st.divider()

# # --------- Charts ---------
# left, right = st.columns(2)
# with left:
#     st.subheader("ROC Curve")
#     RocCurveDisplay.from_predictions(y_test, metrics["y_prob"])  # matplotlib figure
#     st.pyplot(clear_figure=True)st.subheader("ROC Curve")
 

# with right:
#     st.subheader("Precision-Recall Curve")
#     PrecisionRecallDisplay.from_predictions(y_test, metrics["y_prob"])  # matplotlib figure
#     st.pyplot(clear_figure=True)

# # Confusion matrix at chosen threshold
# st.subheader(f"Confusion Matrix @ threshold = {threshold:.2f}")
# y_pred_thresh = (metrics["y_prob"] >= threshold).astype(int)
# cm = confusion_matrix(y_test, y_pred_thresh)
# cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
# st.dataframe(cm_df, use_container_width=True)

# # --------- Feature importance ---------
# st.subheader("Feature Importance (Permutation)")
# imp_df = compute_feature_importance(pipe, X_test)
# if not imp_df.empty:
#     st.dataframe(imp_df.head(30), use_container_width=True)
# else:
#     st.caption("(Permutation importance unavailable for this configuration.)")

# # --------- Unsupervised Anomaly Scores (optional) ---------
# if iso_toggle:
#     st.subheader("IsolationForest Anomaly Scores (unsupervised)")
#     # Simple numeric-only view for anomaly scores
#     numeric_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
#     if numeric_cols:
#         iso = IsolationForest(n_estimators=300, contamination=0.02, random_state=42)
#         iso.fit(X_train[numeric_cols])
#         anomaly_score = -iso.score_samples(X_test[numeric_cols])  # higher = more anomalous
#         iso_df = X_test.copy()
#         iso_df["anomaly_score"] = anomaly_score
#         st.dataframe(iso_df.sort_values("anomaly_score", ascending=False).head(50), use_container_width=True)
#     else:
#         st.caption("No numeric columns found for IsolationForest.")

# st.divider()

# # --------- Score new/unseen data ---------
# st.header("🚨 Score New/Unseen Transactions")
# new_file = st.file_uploader("Upload CSV to score (no target column needed)", type=["csv"], key="score_uploader")
# if new_file is not None:
#     df_new = pd.read_csv(new_file)
#     missing_cols = [c for c in X.columns if c not in df_new.columns]
#     if missing_cols:
#         st.warning(
#             "The uploaded file is missing some training columns. Those will be treated as absent/NaN and may affect predictions.\n" +
#             ", ".join(missing_cols)
#         )
#         # Align columns: add missing as NaN
#         for c in missing_cols:
#             df_new[c] = np.nan
#         # Drop extras not in training
#         df_new = df_new[X.columns]

#     scored = score_unseen(pipe, df_new, threshold=threshold)
#     st.dataframe(scored.head(50), use_container_width=True)

#     # Download flagged
#     flagged = scored[scored["flagged"] == 1].copy()
#     csv_buf = io.StringIO()
#     flagged.to_csv(csv_buf, index=False)
#     st.download_button(
#         label=f"Download flagged transactions (n={len(flagged)})",
#         data=csv_buf.getvalue(),
#         file_name="flagged_transactions.csv",
#         mime="text/csv",
#     )

# # --------- Model persistence (optional quick save) ---------
# st.divider()
# with st.expander("💾 Save / Load Model (advanced)"):
#     st.write("You can export/import the trained pipeline with joblib.")
#     st.code(
#         """
#         import joblib
#         # Save
#         joblib.dump(pipe, "fraud_model.joblib")
#         # Load later
#         pipe = joblib.load("fraud_model.joblib")
#         """,
#         language="python",
#     )

# st.caption("Built with scikit-learn + Streamlit. Tip: tune threshold based on business cost of false positives vs false negatives.")
# app.py
# ------------------------------------------------------------
# Financial Fraud Detection Model + Interactive Dashboard
# ------------------------------------------------------------

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.inspection import permutation_importance

# --------- Page setup ---------
st.set_page_config(
    page_title="💳 Financial Fraud Detection ",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("💳 Financial Fraud Detection ")

# --------- Helpers ---------

def generate_demo_data(n_samples: int = 50_000, fraud_rate: float = 0.02, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    amounts = np.round(np.exp(rng.normal(5, 0.8, n_samples)), 2)
    channels = rng.choice(["POS", "ECOM", "ATM", "WALLET"], size=n_samples, p=[0.5, 0.25, 0.15, 0.10])
    countries = rng.choice(["US", "IN", "GB", "DE", "SG", "AE", "BR", "ZA"], size=n_samples)
    hours = rng.integers(0, 24, size=n_samples)
    merchant_cat = rng.choice(["grocery", "electronics", "travel", "fashion", "fuel", "food"], size=n_samples)
    cust_id = rng.integers(10_000, 99_999, size=n_samples)
    merch_id = rng.integers(1_000, 9_999, size=n_samples)

    base = (
        0.5 * (amounts > np.percentile(amounts, 90)).astype(int)
        + 0.4 * (np.isin(hours, [0, 1, 2, 3, 4])).astype(int)
        + 0.3 * (channels == "ECOM").astype(int)
        + 0.2 * (merchant_cat == "electronics").astype(int)
    )
    base = (base - base.min()) / (base.max() - base.min() + 1e-9)
    thresh = np.quantile(base, 1 - fraud_rate)
    y = (base >= thresh).astype(int)

    df = pd.DataFrame(
        {
            "transaction_id": np.arange(1, n_samples + 1),
            "amount": amounts,
            "channel": channels,
            "country": countries,
            "hour": hours,
            "merchant_category": merchant_cat,
            "customer_id": cust_id,
            "merchant_id": merch_id,
            "is_fraud": y,
        }
    )
    return df


def split_features_target(df: pd.DataFrame, target_col: str):
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    return X, y


def build_preprocessor(X: pd.DataFrame):
    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def train_supervised_model(X, y, model_name="RandomForest", random_state=42):
    preprocessor = build_preprocessor(X)

    if model_name == "LogisticRegression":
        model = LogisticRegression(max_iter=200, class_weight="balanced")
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1
        )

    pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
    pipe.fit(X, y)
    return pipe


def evaluate_classifier(pipe, X_test, y_test):
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "y_prob": y_prob,
    }

# --------- Sidebar ---------
with st.sidebar:
    st.header("⚙️ Configuration")
    data_source = st.radio("Data source", ["Upload CSV", "Use demo dataset"], index=1)
    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"]) if data_source == "Upload CSV" else None
    target_col = st.text_input("Target column (label)", value="is_fraud")
    model_choice = st.selectbox("Model", ["RandomForest", "LogisticRegression"], index=0)
    test_size = st.slider("Test size (%)", 10, 40, 20) / 100
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.01)

# --------- Load data ---------
df = pd.read_csv(uploaded) if uploaded else generate_demo_data()
X, y = split_features_target(df, target_col)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

# --------- Train ---------
pipe = train_supervised_model(X_train, y_train, model_choice)
metrics = evaluate_classifier(pipe, X_test, y_test)

# --------- KPIs ---------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Fraud rate (test)", f"{y_test.mean()*100:.2f}%")
col2.metric("Precision", f"{metrics['precision']:.3f}")
col3.metric("Recall", f"{metrics['recall']:.3f}")
col4.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")

st.divider()

# --------- Charts ---------
# st.subheader("ROC Curve")
# fig, ax = plt.subplots(figsize=(3, 2))
# RocCurveDisplay.from_predictions(y_test, metrics["y_prob"], ax=ax)
# st.pyplot(fig)
st.subheader("ROC Curve")
fig, ax = plt.subplots(figsize=(4, 3))
RocCurveDisplay.from_predictions(y_test, metrics["y_prob"], ax=ax)
st.pyplot(fig, clear_figure=True, use_container_width=False)


st.subheader("Precision-Recall Curve")
fig, ax = plt.subplots(figsize=(3, 2))
PrecisionRecallDisplay.from_predictions(y_test, metrics["y_prob"], ax=ax)
st.pyplot(fig, clear_figure=True, use_container_width=False)
st.markdown("---")
st.header("📊 Fraud Visualization Dashboard")

# Heatmap of fraud by hour vs. transaction amount bins
if "hour" in df.columns and "amount" in df.columns:
    st.subheader("Fraud Heatmap (Hour vs Amount)")
    df['amount_bin'] = pd.cut(df['amount'], bins=10)  # group amounts
    fraud_heatmap = pd.crosstab(
        df['hour'],
        df['amount_bin'],
        values=df['is_fraud'],
        aggfunc="sum"
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(fraud_heatmap, cmap="Reds", ax=ax)
    st.pyplot(fig, clear_figure=True, use_container_width=False)

# Bar chart - Fraud by merchant
if "merchant" in df.columns:
    st.subheader("Fraud Count by Merchant")
    fraud_by_merchant = df[df['is_fraud'] == 1]['merchant'].value_counts().head(10)
    st.bar_chart(fraud_by_merchant)

# Anomaly scores (Top suspicious transactions)
st.subheader("Top Suspicious Transactions (Anomaly Scores)")
suspicious = pd.DataFrame({
    "TransactionID": X_test.index,
    "Fraud Probability": metrics["y_prob"]
})
suspicious = suspicious.sort_values("Fraud Probability", ascending=False).head(10)
st.dataframe(suspicious.style.format({"Fraud Probability": "{:.2f}"}))

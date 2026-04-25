"""
Customer Churn Prediction for Telecom Industry - Web App

This app can work with ANY churn-related CSV as long as you select the correct target column.
It automatically:
- converts numeric-like text columns to numbers
- imputes missing values
- one-hot encodes categorical features
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header { font-size: 5rem !important; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; line-height: 1.05; }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)


TARGET_CANDIDATES = ("churn", "exited", "attrition", "leave", "left", "is_churn", "churned")


def _normalize_col(c: str) -> str:
    return str(c).strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def guess_target_column(columns: list[str]) -> str | None:
    norm_map = {c: _normalize_col(c) for c in columns}
    for c, n in norm_map.items():
        if any(k in n for k in TARGET_CANDIDATES):
            return c
    return None


def guess_id_columns(df: pd.DataFrame) -> list[str]:
    guesses: list[str] = []
    n = len(df)
    for c in df.columns:
        ncol = _normalize_col(c)
        if "id" == ncol or ncol.endswith("id") or ncol.startswith("id") or "customer" in ncol and "id" in ncol:
            guesses.append(c)
            continue
        if n > 0 and df[c].nunique(dropna=False) / n > 0.98:
            guesses.append(c)
    return sorted(set(guesses))


def coerce_numeric_like(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            s = out[c].astype(str).str.strip()
            converted = pd.to_numeric(s, errors="coerce")
            non_na_ratio = converted.notna().mean()
            if non_na_ratio >= 0.90:
                out[c] = converted
            else:
                out[c] = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return out


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def make_label_mapping(y: pd.Series, positive_value) -> tuple[pd.Series, dict]:
    y_raw = y.copy()
    if y_raw.dtype != "object":
        uniques = pd.unique(y_raw.dropna())
        if len(uniques) == 2 and set(uniques) <= {0, 1}:
            return y_raw.astype(int), {0: 0, 1: 1}

    # categorical / mixed: map chosen positive_value -> 1, everything else -> 0
    mapped = (y_raw == positive_value).astype(int)
    return mapped, {"positive_value": positive_value}


def ensure_training_schema(df_features: pd.DataFrame, training_columns: list[str]) -> pd.DataFrame:
    X = df_features.copy()
    for c in training_columns:
        if c not in X.columns:
            X[c] = np.nan
    X = X[training_columns]
    return X


def main():
    st.markdown(
        '<p class="main-header">Customer Churn Prediction for Telecom Industry</p>',
        unsafe_allow_html=True,
    )

    sidebar = st.sidebar
    sidebar.title("Navigation")
    step = sidebar.radio(
        "Go to",
        [
            "1. Upload & Configure Dataset",
            "2. Exploratory Analysis",
            "3. Train Models",
            "4. Hyperparameter Tuning",
            "5. Model Comparison & Evaluation",
            "6. Predict Churn",
        ],
        label_visibility="collapsed",
    )

    # State
    st.session_state.setdefault("df_raw", None)
    st.session_state.setdefault("df_ready", None)
    st.session_state.setdefault("target_col", None)
    st.session_state.setdefault("positive_value", None)
    st.session_state.setdefault("X_train", None)
    st.session_state.setdefault("X_test", None)
    st.session_state.setdefault("y_train", None)
    st.session_state.setdefault("y_test", None)
    st.session_state.setdefault("feature_cols", None)
    st.session_state.setdefault("models", {})

    # --- 1. Upload & Configure Dataset ---
    if step == "1. Upload & Configure Dataset":
        st.header("1. Upload & Configure Dataset")
        st.info("Upload the CSV below.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

        if not uploaded:
            st.warning("Upload a CSV to continue.")
            return

        df_raw = pd.read_csv(uploaded)
        st.session_state.df_raw = df_raw

        st.subheader("Dataset Preview")
        st.dataframe(df_raw.head(10), use_container_width=True)

        # Target selection
        guess = guess_target_column(df_raw.columns.tolist())
        target_col = st.selectbox(
            "Select target column (the churn column)",
            options=df_raw.columns.tolist(),
            index=(df_raw.columns.get_loc(guess) if guess in df_raw.columns else 0),
        )
        st.session_state.target_col = target_col

        # Drop columns (IDs etc.)
        id_guesses = guess_id_columns(df_raw)
        drop_cols = st.multiselect(
            "Columns to drop (IDs / unused columns)",
            options=[c for c in df_raw.columns if c != target_col],
            default=[c for c in id_guesses if c != target_col and c in df_raw.columns],
        )

        # Positive label selection (for non-binary targets)
        y = df_raw[target_col]
        unique_vals = pd.unique(y.dropna())
        positive_value = None
        if len(unique_vals) == 2 and not (set(unique_vals) <= {0, 1}):
            positive_value = st.selectbox(
                "Which value means CHURN? (positive class)",
                options=list(unique_vals),
                index=0,
            )
        elif len(unique_vals) == 2 and set(unique_vals) <= {0, 1}:
            positive_value = 1
        else:
            st.warning("Target column should be binary (2 unique values). Please choose a binary churn column.")

        st.session_state.positive_value = positive_value

        if st.button("Apply configuration & prepare data", type="primary"):
            with st.spinner("Preparing data (numeric conversion, missing handling setup)..."):
                df_work = df_raw.drop(columns=drop_cols, errors="ignore")
                df_work = coerce_numeric_like(df_work)
                df_work = df_work.dropna(subset=[target_col])
                st.session_state.df_ready = df_work
                st.success(f"Prepared dataset: **{len(df_work)}** rows, **{len(df_work.columns)}** columns.")

                st.subheader("Prepared Data Preview")
                st.dataframe(df_work.head(10), use_container_width=True)

    # --- 2. Exploratory Analysis ---
    elif step == "2. Exploratory Analysis":
        st.header("2. Exploratory Analysis")
        if st.session_state.df_ready is None or st.session_state.target_col is None:
            st.warning("Upload and configure data first (Step 1).")
            return

        df = st.session_state.df_ready
        target_col = st.session_state.target_col
        tab1, tab2 = st.tabs(["Target Distribution", "Feature Distributions"])

        with tab1:
            fig, ax = plt.subplots(figsize=(6, 4))
            vc = df[target_col].value_counts(dropna=False)
            vc.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c", "#95a5a6"])
            ax.set_title(f"Target Distribution: {target_col}")
            ax.set_xlabel(target_col)
            ax.set_ylabel("Count")
            st.pyplot(fig)
            plt.close()

        with tab2:
            numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
            if not numeric_cols:
                st.write("No numeric feature columns detected.")
            else:
                sel = st.selectbox("Select numeric feature", numeric_cols)
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                df[sel].hist(ax=ax2, bins=30, color="#3498db", edgecolor="white")
                ax2.set_title(f"Distribution of {sel}")
                st.pyplot(fig2)
                plt.close()

    # --- 3. Train Models ---
    elif step == "3. Train Models":
        st.header("3. Train Models")
        if st.session_state.df_ready is None or st.session_state.target_col is None:
            st.warning("Upload and configure data first (Step 1).")
            return

        df = st.session_state.df_ready
        target_col = st.session_state.target_col
        positive_value = st.session_state.positive_value

        if positive_value is None:
            st.warning("Select which target value means CHURN in Step 1, then prepare data.")
            return

        X = df.drop(columns=[target_col])
        y_raw = df[target_col]
        y, _ = make_label_mapping(y_raw, positive_value=positive_value)

        if y.nunique() != 2:
            st.error("Target is not binary after mapping. Please check the selected target/positive value in Step 1.")
            return

        stratify_opt = y if y.nunique() == 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_opt
        )

        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.feature_cols = X.columns.tolist()

        preprocessor = build_preprocessor(X_train)

        use_balanced = st.checkbox("Use class_weight='balanced' (helps imbalanced churn datasets)", value=True)

        lr = LogisticRegression(max_iter=2000, class_weight=("balanced" if use_balanced else None))
        dt = DecisionTreeClassifier(class_weight=("balanced" if use_balanced else None))
        gb = GradientBoostingClassifier()

        candidates = {
            "Logistic Regression": Pipeline([("preprocess", preprocessor), ("model", lr)]),
            "Decision Tree": Pipeline([("preprocess", preprocessor), ("model", dt)]),
            "Gradient Boosting": Pipeline([("preprocess", preprocessor), ("model", gb)]),
        }

        st.write("Training **Logistic Regression**, **Decision Tree**, and **Gradient Boosting**...")
        results = {}
        with st.spinner("Training..."):
            for name, pipe in candidates.items():
                pipe.fit(X_train, y_train)
                pred = pipe.predict(X_test)
                acc = accuracy_score(y_test, pred)
                results[name] = {"pipeline": pipe, "pred": pred, "acc": acc}

        st.session_state.models = results
        st.success("Training complete.")
        for name, d in results.items():
            st.metric(name, f"{d['acc']*100:.2f}%")

    # --- 4. Hyperparameter Tuning ---
    elif step == "4. Hyperparameter Tuning":
        st.header("4. Hyperparameter Tuning (GridSearchCV)")
        if st.session_state.X_train is None or st.session_state.y_train is None or st.session_state.target_col is None:
            st.warning("Train models first (Step 3).")
            return

        X_train = st.session_state.X_train
        y_train = st.session_state.y_train

        st.write("Example tuning for **Decision Tree** (works for any dataset).")
        preprocessor = build_preprocessor(X_train)
        base = Pipeline([("preprocess", preprocessor), ("model", DecisionTreeClassifier())])
        params = {
            "model__max_depth": [3, 5, 10, None],
            "model__min_samples_split": [2, 5, 10],
        }
        grid = GridSearchCV(base, param_grid=params, cv=5)
        with st.spinner("Running GridSearchCV (5-fold)..."):
            grid.fit(X_train, y_train)
        st.success(f"Best parameters: **{grid.best_params_}**")
        st.json(grid.best_params_)
        st.session_state.setdefault("best_tuned_model", None)
        st.session_state.best_tuned_model = grid.best_estimator_

    # --- 5. Model Comparison & Evaluation ---
    elif step == "5. Model Comparison & Evaluation":
        st.header("5. Model Comparison & Evaluation")
        if not st.session_state.models or st.session_state.y_test is None:
            st.warning("Train models first (Step 3).")
            return

        y_test = st.session_state.y_test
        models = st.session_state.models

        st.subheader("Model Comparison")
        comparison = pd.DataFrame(
            [{"Model": name, "Accuracy": d["acc"]} for name, d in models.items()]
        ).sort_values("Accuracy", ascending=False)
        comparison["Accuracy"] = (comparison["Accuracy"] * 100).map(lambda v: f"{v:.2f}%")
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        best_name = max(models, key=lambda k: models[k]["acc"])
        best_pipe = models[best_name]["pipeline"]
        pred_best = models[best_name]["pred"]

        cols = st.columns(2)
        with cols[0]:
            st.subheader("Confusion Matrix (Best Model)")
            fig, ax = plt.subplots(figsize=(5, 4))
            cm = confusion_matrix(y_test, pred_best)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                xticklabels=["Not Churn", "Churn"],
                yticklabels=["Not Churn", "Churn"],
            )
            ax.set_title(f"Confusion Matrix — {best_name}")
            st.pyplot(fig)
            plt.close()

        with cols[1]:
            st.subheader("Classification Report (Best Model)")
            st.text(classification_report(y_test, pred_best, target_names=["Not Churn", "Churn"]))

            try:
                proba = best_pipe.predict_proba(st.session_state.X_test)[:, 1]
                auc = roc_auc_score(y_test, proba)
                st.metric("ROC-AUC", f"{auc:.3f}")
            except Exception:
                st.caption("ROC-AUC not available for this model/dataset.")

    # --- 6. Predict Churn ---
    elif step == "6. Predict Churn":
        st.header("6. Predict Churn")
        if not st.session_state.models or st.session_state.df_ready is None:
            st.warning("Train models first (Step 3).")
            return

        models = st.session_state.models
        best_name = max(models, key=lambda k: models[k]["acc"])
        best_pipe = models[best_name]["pipeline"]
        feature_cols = st.session_state.feature_cols

        st.caption(f"Using best model: **{best_name}**")

        tab_a, tab_b = st.tabs(["Pick a row from dataset", "Upload single-row CSV"])

        with tab_a:
            df = st.session_state.df_ready
            target_col = st.session_state.target_col
            feat_df = df.drop(columns=[target_col], errors="ignore")
            idx = st.slider("Select row index", 0, max(len(feat_df) - 1, 0), 0)
            row = feat_df.iloc[[idx]]
            st.dataframe(row, use_container_width=True)
            X_row = ensure_training_schema(row, feature_cols)
            pred = int(best_pipe.predict(X_row)[0])
            proba_txt = ""
            try:
                p = float(best_pipe.predict_proba(X_row)[0, 1])
                proba_txt = f" | Churn probability: **{p:.2%}**"
            except Exception:
                pass
            st.success(f"Prediction: **{'Churn' if pred == 1 else 'Not Churn'}** ({pred}){proba_txt}")

        with tab_b:
            st.info("Upload a **single-row CSV** with feature columns. Extra columns are ignored; missing columns are allowed.")
            up = st.file_uploader("Upload single-row CSV", type=["csv"])
            if up:
                df_one = pd.read_csv(up)
                df_one = coerce_numeric_like(df_one)
                X_one = ensure_training_schema(df_one, feature_cols)
                st.dataframe(X_one, use_container_width=True)
                pred = int(best_pipe.predict(X_one)[0])
                proba_txt = ""
                try:
                    p = float(best_pipe.predict_proba(X_one)[0, 1])
                    proba_txt = f" | Churn probability: **{p:.2%}**"
                except Exception:
                    pass
                st.success(f"Prediction: **{'Churn' if pred == 1 else 'Not Churn'}** ({pred}){proba_txt}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Pipeline:** Upload → Configure target → Train → Tune → Evaluate → Predict")


if __name__ == "__main__":
    main()

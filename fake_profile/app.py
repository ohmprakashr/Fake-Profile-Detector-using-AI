import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

rf_model_path = os.path.join(BASE_DIR, "random_forest_model.pkl")
nb_model_path = os.path.join(BASE_DIR, "naive_bayes_model.pkl")
ensemble_model_path = os.path.join(BASE_DIR, "ensemble_model.pkl")

rf_model = joblib.load(rf_model_path)
nb_model = joblib.load(nb_model_path)
ensemble_model = joblib.load(ensemble_model_path)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import os

# -------------------- Page configuration --------------------
st.set_page_config(page_title="Fake Profile Detector", layout="wide")
st.title("🔍 AI‑Powered Fake Profile Detection")
st.markdown("""
This dashboard uses trained machine learning models to classify social media profiles as **Fake** or **Genuine**.
Select a model, then either enter a single profile manually or upload a batch CSV file.
""")

# -------------------- Load models (cached) --------------------
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "Random Forest": "random_forest_model.pkl",
        "Naive Bayes": "naive_bayes_model.pkl",
        "Ensemble (RF+NB)": "ensemble_model.pkl"
    }
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            st.error(f"❌ Model file `{path}` not found. Please run `rf.py` first to train and save the models.")
            st.stop()
    return models

models = load_models()

# -------------------- Sidebar: model selection --------------------
st.sidebar.header("⚙️ Settings")
model_name = st.sidebar.selectbox("Choose a classifier", list(models.keys()))
model = models[model_name]

st.sidebar.markdown("---")
st.sidebar.markdown("ℹ️ **Features used**")
st.sidebar.markdown("""
- `statuses_count`  
- `followers_count`  
- `friends_count`  
- `favourites_count`  
- `listed_count`  
- `sex_code` (0 = male, 1 = female)  
- `lang_code` (0–7, encoded from original `lang` column)
""")

# -------------------- Main tabs --------------------
tab1, tab2, tab3 = st.tabs(["📥 Single Prediction", "📂 Batch Prediction", "📊 Model Performance"])

# ==================== TAB 1: SINGLE PREDICTION ====================
with tab1:
    st.subheader("Enter profile details manually")

    col1, col2, col3 = st.columns(3)
    with col1:
        statuses = st.number_input("statuses_count", min_value=0, value=100)
        followers = st.number_input("followers_count", min_value=0, value=50)
        friends = st.number_input("friends_count", min_value=0, value=200)
    with col2:
        favourites = st.number_input("favourites_count", min_value=0, value=10)
        listed = st.number_input("listed_count", min_value=0, value=0)
    with col3:
        sex = st.selectbox("sex_code (0=male, 1=female)", [0, 1])
        lang = st.number_input("lang_code (0–7)", min_value=0, max_value=7, value=1, step=1)

    if st.button("🔮 Predict", key="predict_single"):
        # Build dataframe with the same column order as training
        input_df = pd.DataFrame([[statuses, followers, friends, favourites, listed, sex, lang]],
                                columns=["statuses_count", "followers_count", "friends_count",
                                         "favourites_count", "listed_count", "sex_code", "lang_code"])
        # Predict
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else [None, None]

        # Display result
        st.markdown("---")
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            if pred == 1:
                st.success("✅ **Genuine profile**")
            else:
                st.error("❌ **Fake profile**")
        with col_res2:
            if proba[0] is not None:
                st.metric("Confidence (Fake)", f"{proba[0]:.2%}")
                st.metric("Confidence (Genuine)", f"{proba[1]:.2%}")

        # Show feature importance if available (RF only)
        if model_name == "Random Forest" and hasattr(model.named_steps['rf'], 'feature_importances_'):
            importances = model.named_steps['rf'].feature_importances_
            feat_names = input_df.columns
            feat_imp = pd.DataFrame({"feature": feat_names, "importance": importances})
            fig = px.bar(feat_imp, x="importance", y="feature", orientation='h',
                         title="Feature Importance (Random Forest)")
            st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 2: BATCH PREDICTION ====================
with tab2:
    st.subheader("Upload a CSV file with multiple profiles")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        required_cols = ["statuses_count", "followers_count", "friends_count",
                         "favourites_count", "listed_count", "sex_code", "lang_code"]

        if all(col in df.columns for col in required_cols):
            # Make predictions
            X_batch = df[required_cols]
            preds = model.predict(X_batch)
            probs = model.predict_proba(X_batch) if hasattr(model, "predict_proba") else None

            df["prediction"] = preds
            df["profile_type"] = df["prediction"].map({0: "Fake", 1: "Genuine"})
            if probs is not None:
                df["prob_fake"] = probs[:, 0]
                df["prob_genuine"] = probs[:, 1]

            st.success(f"✅ Prediction completed for {len(df)} profiles.")

            # Show preview
            st.dataframe(df.head(10))

            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download predictions as CSV", data=csv,
                               file_name="predictions.csv", mime="text/csv")

            # Visualise distribution
            st.markdown("---")
            st.subheader("Prediction distribution")
            counts = df["profile_type"].value_counts().reset_index()
            counts.columns = ["profile_type", "count"]
            fig = px.bar(counts, x="profile_type", y="count", color="profile_type",
                         text="count", title="Fake vs Genuine")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        else:
            missing = set(required_cols) - set(df.columns)
            st.error(f"❌ Missing columns: {missing}")

# ==================== TAB 3: MODEL PERFORMANCE ====================
with tab3:
    st.subheader("Model Performance (on a test split of uploaded data)")

    if uploaded_file is not None and all(col in df.columns for col in required_cols):
        from sklearn.model_selection import train_test_split

        X = df[required_cols]
        y_true = df["prediction"]   # using predictions as ground truth for demo (replace with actual labels if available)
        st.caption("⚠️ **Note:** This evaluation uses the model's own predictions as ground truth – it shows perfect accuracy by design. In a real scenario, you would provide a separate test set with true labels.")

        # Split for illustration
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_true, test_size=0.2, random_state=42)
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None

        # Confusion matrix
        cm = confusion_matrix(y_te, y_pred)
        if cm.shape == (2, 2):
            x_labels = ["Fake", "Genuine"]
            y_labels = ["Fake", "Genuine"]
            # Annotated heatmap
            fig_cm = ff.create_annotated_heatmap(
                z=cm,
                x=x_labels,
                y=y_labels,
                colorscale='Blues',
                showscale=True
            )
            fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.warning("Confusion matrix is not 2x2 – only one class present in the test split?")

        # ROC curve
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_te, y_proba)
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                          name=f"AUC = {roc_auc:.2f}", line=dict(color="blue")))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                          name="Random guess", line=dict(color="red", dash="dash")))
            fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                                  yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc, use_container_width=True)
    else:
        st.info("ℹ️ Upload a batch file first to see performance metrics (using a random split of that data).")

# -------------------- Footer / Image analysis placeholder --------------------

st.markdown("---")

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Fake Profile Detector", layout="wide")
st.title("🔍 AI-Powered Fake Profile Detection")

st.markdown("""
This dashboard uses trained Machine Learning models to classify social media profiles as **Fake** or **Genuine**.
Choose a model and either test a single profile or upload a CSV file for batch prediction.
""")

# ---------------------------------------------------
# LOAD MODELS (ABSOLUTE PATH FIXED)
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    model_paths = {
        "Random Forest": os.path.join(BASE_DIR, "random_forest_model.pkl"),
        "Naive Bayes": os.path.join(BASE_DIR, "naive_bayes_model.pkl"),
        "Ensemble (RF+NB)": os.path.join(BASE_DIR, "ensemble_model.pkl")
    }

    models = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            st.error(f"❌ Model file not found: {path}")
            st.stop()

    return models

models = load_models()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.header("⚙️ Settings")
model_name = st.sidebar.selectbox("Choose a classifier", list(models.keys()))
model = models[model_name]

st.sidebar.markdown("---")
st.sidebar.markdown("### Features Used")
st.sidebar.markdown("""
- statuses_count  
- followers_count  
- friends_count  
- favourites_count  
- listed_count  
- sex_code (0=male, 1=female)  
- lang_code (0–7)
""")

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📥 Single Prediction", "📂 Batch Prediction", "📊 Model Performance"])

# ===================================================
# TAB 1 - SINGLE PREDICTION
# ===================================================
with tab1:
    st.subheader("Enter Profile Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        statuses = st.number_input("statuses_count", min_value=0, value=100)
        followers = st.number_input("followers_count", min_value=0, value=50)
        friends = st.number_input("friends_count", min_value=0, value=200)

    with col2:
        favourites = st.number_input("favourites_count", min_value=0, value=10)
        listed = st.number_input("listed_count", min_value=0, value=0)

    with col3:
        sex = st.selectbox("sex_code", [0, 1])
        lang = st.number_input("lang_code", min_value=0, max_value=7, value=1)

    if st.button("🔮 Predict"):

        input_df = pd.DataFrame([[statuses, followers, friends,
                                  favourites, listed, sex, lang]],
                                columns=["statuses_count", "followers_count",
                                         "friends_count", "favourites_count",
                                         "listed_count", "sex_code", "lang_code"])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None

        st.markdown("---")

        if prediction == 1:
            st.success("✅ Genuine Profile")
        else:
            st.error("❌ Fake Profile")

        if probability is not None:
            st.metric("Confidence (Fake)", f"{probability[0]:.2%}")
            st.metric("Confidence (Genuine)", f"{probability[1]:.2%}")

# ===================================================
# TAB 2 - BATCH PREDICTION
# ===================================================
with tab2:
    st.subheader("Upload CSV File")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    required_cols = ["statuses_count", "followers_count", "friends_count",
                     "favourites_count", "listed_count",
                     "sex_code", "lang_code"]

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if all(col in df.columns for col in required_cols):

            X_batch = df[required_cols]
            preds = model.predict(X_batch)
            probs = model.predict_proba(X_batch) if hasattr(model, "predict_proba") else None

            df["prediction"] = preds
            df["profile_type"] = df["prediction"].map({0: "Fake", 1: "Genuine"})

            if probs is not None:
                df["prob_fake"] = probs[:, 0]
                df["prob_genuine"] = probs[:, 1]

            st.success(f"✅ Predictions completed for {len(df)} profiles.")
            st.dataframe(df.head(10))

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions",
                               data=csv,
                               file_name="predictions.csv",
                               mime="text/csv")

            # Distribution Chart
            st.subheader("Prediction Distribution")
            counts = df["profile_type"].value_counts().reset_index()
            counts.columns = ["profile_type", "count"]

            fig = px.bar(counts,
                         x="profile_type",
                         y="count",
                         color="profile_type",
                         text="count")

            st.plotly_chart(fig, use_container_width=True)

        else:
            missing = set(required_cols) - set(df.columns)
            st.error(f"❌ Missing columns: {missing}")

# ===================================================
# TAB 3 - MODEL PERFORMANCE (SAFE VERSION)
# ===================================================
with tab3:
    st.subheader("Model Performance Visualization")

    st.info("⚠️ This demo evaluates predictions on a random split of uploaded data.")

    if uploaded_file is not None and all(col in df.columns for col in required_cols):

        X = df[required_cols]
        y = df["prediction"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Confusion Matrix (SAFE)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        if cm.shape == (2, 2):
            fig_cm = ff.create_annotated_heatmap(
                z=cm.tolist(),
                x=["Fake", "Genuine"],
                y=["Fake", "Genuine"],
                colorscale="Blues",
                showscale=True
            )

            fig_cm.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual"
            )

            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.warning("⚠️ Confusion matrix not available (only one class present).")

        # ROC Curve (SAFE)
        if y_proba is not None and len(set(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr,
                                         mode="lines",
                                         name=f"AUC = {roc_auc:.2f}"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                         mode="lines",
                                         name="Random Guess",
                                         line=dict(dash="dash")))

            fig_roc.update_layout(
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate"
            )

            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.warning("⚠️ ROC curve not available (only one class present).")

    else:
        st.info("Upload a batch CSV file to view performance charts.")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.caption("🚀 Developed using Machine Learning (Random Forest, Naive Bayes & Ensemble Model)")

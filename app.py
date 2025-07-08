# 💻 Imports
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ⚙️ Load model and thresholds
@st.cache_resource
def load_model():
    model = joblib.load("final_GBR_model.pkl")          # Preprocessing + Model
    thresholds = joblib.load("cost_thresholds.pkl")     # {"low": X, "high": Y}
    return model, thresholds

model, thresholds = load_model()

# 🎨 Page config
st.set_page_config(page_title="💊 Medical Cost Predictor", page_icon="💵", layout="centered")

# 🌈 Custom CSS
st.markdown("""
    <style>
        .main {
            background: linear-gradient(120deg, #f0f9ff 0%, #fef7ff 100%);
        }
        .stButton>button {
            background-color: #10b981;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.6rem 1rem;
        }
        .metric-cost {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .tier-card {
            padding: 1rem;
            border-radius: 0.8rem;
            font-size: 1.4rem;
            font-weight: bold;
            text-align: center;
        }
        .normal { background-color: #22c55e; color: white; }
        .medium { background-color: #facc15; color: black; }
        .premium { background-color: #ef4444; color: white; }
    </style>
""", unsafe_allow_html=True)

# 🧾 Sidebar Inputs
st.sidebar.header("📝 Patient Information")

age = st.sidebar.slider("🎂 Age", 0, 100, 30)
sex = st.sidebar.radio("⚧️ Sex", ["male", "female"], horizontal=True)
bmi = st.sidebar.slider("⚖️ BMI (Body Mass Index)", 10.0, 60.0, 25.0)
children = st.sidebar.slider("🧒 Children", 0, 5, 0)
smoker = st.sidebar.radio("🚬 Smoker", ["yes", "no"], horizontal=True)
region = st.sidebar.selectbox("🌍 Region", ["southeast", "southwest", "northwest", "northeast"])

# 📁 Upload CSV Input
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📄 Upload CSV to Predict in Bulk", type=["csv"])

# 🗂 Input conversion
input_df = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

# 🚀 Predict Button
if st.sidebar.button("📈 Predict Cost"):
    with st.spinner("Running prediction..."):
        predicted_cost = float(model.predict(input_df)[0])

    # 🏷️ Tiering logic
    if predicted_cost < thresholds["low"]:
        tier, css = "Normal", "normal"
    elif predicted_cost < thresholds["high"]:
        tier, css = "Medium", "medium"
    else:
        tier, css = "Premium", "premium"

    # 🎯 Show Prediction Results
    st.markdown("## 🎯 Prediction Result")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"<div class='metric-cost'>$ {predicted_cost:,.2f}</div>", unsafe_allow_html=True)
        st.caption("Estimated Annual Medical Expense")

    with col2:
        st.markdown(f"<div class='tier-card {css}'>{tier}</div>", unsafe_allow_html=True)
        st.caption("Cost Tier Based on Predicted Value")

    st.divider()

    with st.expander("🔍 Review Input Summary"):
        st.write(input_df)

    st.caption(f"📅 Prediction generated on {datetime.now().strftime('%d %b %Y • %I:%M %p')}")
    st.balloons()

elif uploaded_file is not None:
    df_bulk = pd.read_csv(uploaded_file)
    with st.spinner("🔄 Running bulk predictions..."):
        predictions = model.predict(df_bulk)
        df_bulk["Predicted Cost"] = predictions
        df_bulk["Tier"] = pd.cut(predictions,
                                  bins=[-np.inf, thresholds["low"], thresholds["high"], np.inf],
                                  labels=["Normal", "Medium", "Premium"])
        st.success("✅ Bulk Prediction Completed")
        st.dataframe(df_bulk)

        # Download CSV
        csv = df_bulk.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions CSV", csv, "predictions.csv", "text/csv")

else:
    st.info("💬 Please provide patient details or upload a CSV to begin.")

# 🔚 Footer
st.divider()
st.markdown("Made with ❤️ by Mrinmoy Nanda • [LinkedIn](https://www.linkedin.com/in/mrinmoynanda/) • [GitHub](https://github.com/MrinmoyNanda45)")

# 💊 Medical Cost Prediction App

A Streamlit-based web application that predicts annual medical insurance expenses based on patient demographics and health-related information using a Gradient Boosting Regressor model.

---

## 📈 Project Highlights

- 🔢 **ML Model**: Gradient Boosting Regressor (GBR)
- 🧹 **Pipeline**: Includes OneHotEncoder + StandardScaler
- 💻 **Frontend**: Built using Streamlit with custom UI
- 📂 **CSV Upload**: Predict in bulk using uploaded CSV
- 📊 **Tiering**: Categorizes expenses into Normal / Medium / Premium tiers

---

## 🚀 Live Demo

*Coming Soon – Deploy it using [Streamlit Cloud](https://mrinmoynanda-medical-expenses-prediction.streamlit.app/)*

---

## 🧠 Model Overview

| Feature         | Type       |
|----------------|------------|
| `age`          | Numerical  |
| `sex`          | Categorical (male/female) |
| `bmi`          | Numerical  |
| `children`     | Numerical  |
| `smoker`       | Categorical (yes/no) |
| `region`       | Categorical (4 regions) |

Trained with:
- 📊 MAE, RMSE, R² as performance metrics
- 🏆 Best model: **GradientBoostingRegressor**

---

## 💻 How to Run Locally

```bash
# Step 1: Clone the repo
git clone https://github.com/mrinmoynanda/Medical_Cost_Prediction.git
cd Medical_Cost_Prediction

# Step 2: Create virtual env (optional)
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the Streamlit app
streamlit run app.py

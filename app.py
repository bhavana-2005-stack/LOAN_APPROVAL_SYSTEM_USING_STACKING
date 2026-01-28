# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

# ----------------------------
# 1️⃣ Load dataset
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "loan_prediction.csv")

df = pd.read_csv(csv_path)

# Drop Loan_ID
df.drop('Loan_ID', axis=1, inplace=True)

# Fill missing values
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())

# Label encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 2️⃣ Train Base Models
# ----------------------------
lr = LogisticRegression()
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

base_models = [lr, dt, rf]

for model in base_models:
    model.fit(X_scaled, y)

# Generate meta features for stacking
train_meta_features = []
for model in base_models:
    oof_pred = cross_val_predict(model, X_scaled, y, cv=5, method="predict")
    train_meta_features.append(oof_pred)
X_meta_train = np.column_stack(train_meta_features)

# Meta model
meta_model = LogisticRegression()
meta_model.fit(X_meta_train, y)

# ----------------------------
# 3️⃣ Streamlit UI
# ----------------------------
st.set_page_config(page_title="Smart Loan Approval System", layout="wide")
st.title("🎯 Smart Loan Approval System – Stacking Model")
st.write("This system uses a **Stacking Ensemble ML model** to predict loan approval.")

# Sidebar Inputs
st.sidebar.header("Applicant Details")

# --- Numeric inputs ---
applicant_income = st.sidebar.number_input(
    "Applicant Income (₹)", min_value=0, step=1000, value=5000, format="%d"
)
coapplicant_income = st.sidebar.number_input(
    "Co-Applicant Income (₹)", min_value=0, step=1000, value=0, format="%d"
)

# --- Loan Amount proportional to income ---
max_loan = max(5000, (applicant_income + coapplicant_income) * 20)  # approx 20 months income
loan_amount = st.sidebar.number_input(
    "Loan Amount (₹)", min_value=1000, max_value=max_loan, step=500, value=applicant_income * 2
)

# --- Loan Term as selectbox with realistic options ---
loan_terms = [12, 24, 36, 48, 60, 84, 120, 180, 240, 300, 360]  # months
loan_term = st.sidebar.selectbox(
    "Loan Amount Term (months)",
    options=loan_terms,
    index=loan_terms.index(12)
)

# --- Categorical inputs ---
gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
married = st.sidebar.selectbox("Married", ("Yes", "No"))
dependents = st.sidebar.number_input("Dependents", min_value=0, max_value=5, value=0)
education = st.sidebar.selectbox("Education", ("Graduate", "Not Graduate"))
self_employed = st.sidebar.selectbox("Self-Employed", ("Yes", "No"))
credit_history = st.sidebar.radio("Credit History", ("Yes", "No"))
property_area = st.sidebar.selectbox("Property Area", ("Urban", "Semi-Urban", "Rural"))

# Encode inputs
gender_val = 1 if gender == "Male" else 0
married_val = 1 if married == "Yes" else 0
education_val = 1 if education == "Graduate" else 0
self_employed_val = 1 if self_employed == "Yes" else 0
credit_val = 1 if credit_history == "Yes" else 0
property_map = {"Urban": 2, "Semi-Urban": 1, "Rural": 0}
property_val = property_map[property_area]

# Create input dataframe
input_df = pd.DataFrame([[applicant_income, coapplicant_income, loan_amount, loan_term,
                          gender_val, married_val, dependents, education_val,
                          self_employed_val, credit_val, property_val]],
                        columns=X.columns)
input_scaled = scaler.transform(input_df)

# ----------------------------
# 4️⃣ Display stacking structure
# ----------------------------
st.subheader("📌 Model Architecture (Stacking)")
st.markdown("""
**Base Models Used:**  
- Logistic Regression  
- Decision Tree  
- Random Forest  

**Meta Model Used:**  
- Logistic Regression
""")

# ----------------------------
# 5️⃣ Prediction Button
# ----------------------------
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):
    # Base model predictions
    pred_lr = lr.predict(input_scaled)[0]
    pred_dt = dt.predict(input_scaled)[0]
    pred_rf = rf.predict(input_scaled)[0]
    pred_map = {1: "Approved", 0: "Rejected"}

    st.subheader("📊 Base Model Predictions")
    st.write(f"**Logistic Regression:** {pred_map[pred_lr]}")
    st.write(f"**Decision Tree:** {pred_map[pred_dt]}")
    st.write(f"**Random Forest:** {pred_map[pred_rf]}")

    # Stacking prediction
    meta_input = np.array([[pred_lr, pred_dt, pred_rf]])
    stacking_pred = meta_model.predict(meta_input)[0]
    stacking_prob = meta_model.predict_proba(meta_input)[0][stacking_pred] * 100

    st.subheader("🧠 Final Stacking Decision")
    if stacking_pred == 1:
        st.markdown(f"<h2 style='color:green;'>✅ Loan Approved</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='color:red;'>❌ Loan Rejected</h2>", unsafe_allow_html=True)
    
    st.write(f"📈 Confidence Score: {stacking_prob:.2f}%")

    # ----------------------------
    # 6️⃣ Business Explanation
    # ----------------------------
    st.subheader("💼 Business Explanation")
    decision_text = "likely to repay the loan" if stacking_pred == 1 else "unlikely to repay the loan"
    approval_text = "approval" if stacking_pred == 1 else "rejection"
    st.write(f"Based on income, credit history, and combined predictions from multiple models, "
             f"the applicant is **{decision_text}**. Therefore, the stacking model predicts **loan {approval_text}**.")
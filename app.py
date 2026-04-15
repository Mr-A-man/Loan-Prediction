import streamlit as st
import numpy as np
import joblib
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model.pkl")

# Load model bundle
bundle = joblib.load(model_path)
model = bundle['model']
imputer = bundle['imputer']
feature_cols = bundle['feature_cols']

st.title("Loan Prediction (Abinay Singh (CSDS-A))")

# Inputs
age = st.number_input("Age", 18, 100, value=35)
income = st.number_input("Annual Income (₹)", min_value=0, value=60000)
loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=20000)
credit_score = st.number_input("Credit Score", 300, 900, value=650)
employment_years = st.number_input("Employment Years", 0, 50, value=5)
education = st.selectbox("Education Level", ["High School", "Bachelors", "Masters", "PhD"])
housing = st.selectbox("Housing Status", ["Own", "Rent", "Mortgage"])

# Encode
edu_map = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
education_val = edu_map[education]

housing_own = 1 if housing == "Own" else 0
housing_rent = 1 if housing == "Rent" else 0
housing_mortgage = 1 if housing == "Mortgage" else 0

if st.button("Predict"):
    # Build input in correct feature order
    input_dict = {
        'Age': age,
        'Income': income,
        'Loan_Amount': loan_amount,
        'Credit_Score': credit_score,
        'Employment_Years': employment_years,
        'Education_Level': education_val,
        'Housing_Own': housing_own,
        'Housing_Rent': housing_rent,
        'Housing_Mortgage': housing_mortgage,
    }
    input_arr = np.array([[input_dict[col] for col in feature_cols]])
    input_imputed = imputer.transform(input_arr)

    prediction = model.predict(input_imputed)
    prob = model.predict_proba(input_imputed)[0][1]

    st.divider()
    if prediction[0] == 1:
        st.error(f"High Risk: {prob:.2%}")
    else:
        st.success(f"LikelyRisk : {prob:.2%}")
    
    st.caption(f"Model confidence in no-default: {1 - prob:.2%}")

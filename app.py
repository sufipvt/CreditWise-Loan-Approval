import streamlit as st
import pandas as pd
import pickle

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="CreditWise Loan Prediction",
    page_icon="🏦",
    layout="centered"
)

# ==================================================
# LOAD FILES
# ==================================================

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
ohe = pickle.load(open("ohe.pkl", "rb"))

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("🏦 CreditWise")

st.sidebar.write(
    "AI-Powered Loan Approval Prediction"
)

st.sidebar.info(
    "Built using Machine Learning + Streamlit"
)

# ==================================================
# TITLE
# ==================================================

st.title("🏦 CreditWise Loan Prediction")

st.markdown("""
Predict whether a loan application is likely
to be **approved or rejected** based on
financial and applicant information.
""")

# ==================================================
# PERSONAL DETAILS
# ==================================================

st.subheader("👤 Personal Details")

Age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

Marital_Status = st.selectbox(
    "Marital Status",
    ["Married", "Single"]
)

Dependents = st.selectbox(
    "Dependents",
    [0, 1, 2, 3]
)

Education_Level = st.selectbox(
    "Education Level",
    ["Graduate", "Not Graduate"]
)

# ==================================================
# EMPLOYMENT
# ==================================================

st.subheader("💼 Employment Information")

Employment_Status = st.selectbox(
    "Employment Status",
    ["Salaried", "Self-employed", "Unemployed"]
)

Employer_Category = st.selectbox(
    "Employer Category",
    ["Government", "MNC", "Private", "Unemployed"]
)

# ==================================================
# FINANCIAL INFORMATION
# ==================================================

st.subheader("💰 Financial Information")

Applicant_Income = st.number_input(
    "Applicant Income",
    min_value=0.0,
    value=10000.0
)

Coapplicant_Income = st.number_input(
    "Coapplicant Income",
    min_value=0.0,
    value=2000.0
)

Credit_Score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=650
)

Existing_Loans = st.number_input(
    "Existing Loans",
    min_value=0,
    value=1
)

Savings = st.number_input(
    "Savings",
    min_value=0.0,
    value=10000.0
)

Collateral_Value = st.number_input(
    "Collateral Value",
    min_value=0.0,
    value=25000.0
)

# ==================================================
# LOAN INFORMATION
# ==================================================

st.subheader("🏦 Loan Information")

Loan_Amount = st.number_input(
    "Loan Amount",
    min_value=0.0,
    value=15000.0
)

Loan_Term = st.selectbox(
    "Loan Term (Months)",
    [12, 24, 36, 48, 60, 72, 84]
)

Loan_Purpose = st.selectbox(
    "Loan Purpose",
    [
        "Car",
        "Business",
        "Education",
        "Home",
        "Personal"
    ]
)

Property_Area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

DTI_Ratio = st.slider(
    "DTI Ratio",
    0.0,
    1.0,
    0.30
)

# ==================================================
# PREDICT BUTTON
# ==================================================

if st.button("Predict Loan Approval"):

    # ---------------- Education Encoding ----------------

    education_encoded = (
        1 if Education_Level == "Graduate"
        else 0
    )

    # ---------------- Numerical Features ----------------

    raw_input = pd.DataFrame({
        "Applicant_Income": [Applicant_Income],
        "Coapplicant_Income": [Coapplicant_Income],
        "Age": [Age],
        "Dependents": [Dependents],
        "Credit_Score": [Credit_Score],
        "Existing_Loans": [Existing_Loans],
        "DTI_Ratio": [DTI_Ratio],
        "Savings": [Savings],
        "Collateral_Value": [Collateral_Value],
        "Loan_Amount": [Loan_Amount],
        "Loan_Term": [Loan_Term],
        "Education_Level": [education_encoded]
    })

    # ---------------- Categorical Features ----------------

    categorical_df = pd.DataFrame({
        "Employment_Status": [Employment_Status],
        "Marital_Status": [Marital_Status],
        "Employer_Category": [Employer_Category],
        "Property_Area": [Property_Area],
        "Gender": [Gender],
        "Loan_Purpose": [Loan_Purpose]
    })

    # ---------------- One Hot Encoding ----------------

    encoded = ohe.transform(categorical_df)

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out()
    )

    # ---------------- Merge Features ----------------

    final_input = pd.concat(
        [
            raw_input.reset_index(drop=True),
            encoded_df.reset_index(drop=True)
        ],
        axis=1
    )

    # ---------------- Scale Input ----------------

    scaled_input = scaler.transform(
        final_input
    )

    # ---------------- Prediction ----------------

    prediction = model.predict(
        scaled_input
    )[0]

    probability = model.predict_proba(
        scaled_input
    )[0]

    confidence = min(
        max(probability) * 100,
        99.5
    )

    st.subheader("📊 Prediction Result")

    # ==================================================
    # REJECTION REASONS
    # ==================================================

    reasons = []

    if Credit_Score < 600:
        reasons.append(
            "Low Credit Score"
        )

    if DTI_Ratio > 0.45:
        reasons.append(
            "High Debt-to-Income Ratio"
        )

    if Existing_Loans >= 3:
        reasons.append(
            "Too Many Existing Loans"
        )

    if Savings < Loan_Amount * 0.30:
        reasons.append(
            "Low Savings Compared to Loan Amount"
        )

    # ==================================================
    # RESULT DISPLAY
    # IMPORTANT FIX:
    # 0 = APPROVED
    # 1 = REJECTED
    # ==================================================

    if prediction == 0:

        st.success(
            f"✅ Loan Approved ({confidence:.2f}% confidence)"
        )

        st.success("""
Strong approval indicators detected:

✅ Good financial profile  
✅ Manageable debt ratio  
✅ Acceptable risk level
""")

    else:

        st.error(
            f"❌ Loan Rejected ({confidence:.2f}% confidence)"
        )

        if len(reasons) > 0:

            st.warning(
                "Key Risk Factors Detected"
            )

            for reason in reasons:
                st.write(f"• {reason}")

        else:

            st.info(
                "No major financial red flags detected."
            )

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "CreditWise | Built by Sufiyan Rizvi using Machine Learning & Streamlit"
)
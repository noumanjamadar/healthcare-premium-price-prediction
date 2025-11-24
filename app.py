import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# -------------------------------------------------
# SOFT GRADIENT + CLEAN GLASS UI (NO EXTRA BOXES)
# -------------------------------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }

        /* Soft Blue → Soft Teal Gradient */
        .stApp {
            background: linear-gradient(135deg, #3a7bd5 0%, #3a6073 100%);
            background-attachment: fixed;
        }

        /* Glass Card */
        .glass-card {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 16px;
            padding: 20px;
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.25);
        }

        /* Title */
        .title {
            font-size: 40px;
            font-weight: 700;
            text-align: center;
            color: white;
            margin-top: -10px;
            margin-bottom: 10px;
        }

        /* Section Headings */
        .sub-head {
            font-size: 22px;
            font-weight: 600;
            color: white;
            margin-bottom: 10px;
        }

        /* Predict Button */
        .stButton button {
            background: rgba(255,255,255,0.25);
            border-radius: 8px;
            padding: 12px 24px;
            font-size: 18px;
            color: white;
            border: 1px solid rgba(255,255,255,0.4);
            transition: all .25s ease;
        }

        .stButton button:hover {
            background: rgba(255,255,255,0.35);
            box-shadow: 0px 0px 10px rgba(255,255,255,0.4);
            transform: scale(1.03);
        }

        /* Prediction Box */
        .pred-box {
            background: rgba(255,255,255,0.20);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 26px;
            font-weight: 700;
            border: 1px solid rgba(255,255,255,0.35);
            color: #ffffff;
            margin-top: 15px;
        }
    </style>
""", unsafe_allow_html=True)



# -------------------------------------------------
# LOAD MODELS & SCALERS
# -------------------------------------------------
MODEL_OLD_PATH = "models/xgb_model_old_gr"
MODEL_YOUNG_PATH = "models/model_young_gr"
SCALER_OLD_PATH = "models/scaler_old_gr"
SCALER_YOUNG_PATH = "models/scaler_young_gr"


COLS_TO_SCALE = [
    "age", "number_of_dependants",
    "income_level", "income_lakhs",
    "insurance_plan", "genetical_risk"
]

MODEL_COLUMNS = [
    "age", "number_of_dependants",
    "income_lakhs", "insurance_plan",
    "genetical_risk", "normalized_risk_score",
    "gender_Male",
    "region_Northwest", "region_Southeast", "region_Southwest",
    "marital_status_Unmarried",
    "bmi_category_Obesity", "bmi_category_Overweight", "bmi_category_Underweight",
    "smoking_status_Occasional", "smoking_status_Regular",
    "employment_status_Salaried", "employment_status_Self-Employed"
]

RISK_SCORES = {
    "diabetes": 6, "high blood pressure": 6,
    "heart disease": 8, "thyroid": 5,
    "no disease": 0, "none": 0
}

MIN_SCORE, MAX_SCORE = 0, 14


def load_scaler(path):
    s = joblib.load(path)
    if isinstance(s, dict):
        return s.get("scaler", s.get("sc", s))
    return s


@st.cache_resource
def load_artifacts():
    return (
        joblib.load(MODEL_OLD_PATH),
        joblib.load(MODEL_YOUNG_PATH),
        load_scaler(SCALER_OLD_PATH),
        load_scaler(SCALER_YOUNG_PATH)
    )


model_old, model_young, scaler_old, scaler_young = load_artifacts()



# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def map_income_to_level(income):
    if income < 10: return 1
    if income < 25: return 2
    if income < 40: return 3
    return 4


def build_scaler_input(inputs):
    return pd.DataFrame([{c: inputs[c] for c in COLS_TO_SCALE}])


def build_model_input_from_scaled(scaled_vals, inp):
    X = pd.DataFrame(0, index=[0], columns=MODEL_COLUMNS)

    # Scaled numeric cols
    for col in ["age", "number_of_dependants", "income_lakhs", "insurance_plan", "genetical_risk"]:
        X.at[0, col] = scaled_vals[col]

    # Not scaled
    X.at[0, "normalized_risk_score"] = inp["normalized_risk_score"]

    # One-hot mapping
    mapping = {
        "gender": "gender",
        "region": "region",
        "marital": "marital_status",
        "bmi": "bmi_category",
        "smoking": "smoking_status",
        "employment": "employment_status",
    }

    for feat in MODEL_COLUMNS:
        if "_" in feat:
            left, right = feat.split("_", 1)
            key = mapping.get(left, left)
            val = str(inp.get(key, "")).lower()
            if val == right.lower():
                X.at[0, feat] = 1

    return X



# -------------------------------------------------
# UI LAYOUT
# -------------------------------------------------
st.markdown("<div class='title'>🏥 Healthcare Premium Price Predictor</div>", unsafe_allow_html=True)

left, right = st.columns([1.3, 1])


# ---------------- LEFT SIDE -------------------
with left:

    # Basic Info
    st.markdown("<div class='glass-card'><div class='sub-head'>📘 Basic Information</div></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 1, 120, 30)
    income_lakhs = c2.number_input("Annual Income (Lakhs)", 0.0, 200.0, 5.0)

    number_of_dependants = st.selectbox("Dependants", [0,1,2,3,4,5])

    # Lifestyle Info
    st.markdown("<div class='glass-card'><div class='sub-head'>💙 Lifestyle Information</div></div>", unsafe_allow_html=True)

    l1, l2, l3 = st.columns(3)
    gender = l1.selectbox("Gender", ["Male", "Female"])
    region = l2.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])
    marital_status = l3.selectbox("Marital Status", ["Married", "Unmarried"])

    bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obesity"])
    smoking_status = st.selectbox("Smoking Status", ["No Smoking", "Regular", "Occasional"])
    employment_status = st.selectbox("Employment Status", ["Salaried","Self-Employed","Freelancer"])



# ---------------- RIGHT SIDE -------------------
with right:

    # Medical History
    st.markdown("<div class='glass-card'><div class='sub-head'>🧬 Medical History</div></div>", unsafe_allow_html=True)

    diseases = st.multiselect(
        "Select Diseases (max 2)",
        ["No Disease","Diabetes","High blood pressure","Heart disease","Thyroid"],
        max_selections=2
    )

    insurance_plan = st.selectbox("Insurance Plan", ["Bronze","Silver","Gold"])

    # Prepare values
    insurance_code = {"Bronze":1, "Silver":2, "Gold":3}[insurance_plan]
    genetical_risk = 0
    total_disease_risk = sum(RISK_SCORES.get(d.lower(), 0) for d in diseases)
    normalized_risk_score = (total_disease_risk - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)
    income_level = map_income_to_level(income_lakhs)

    inputs = {
        "age": age,
        "number_of_dependants": number_of_dependants,
        "insurance_plan": insurance_code,
        "income_lakhs": income_lakhs,
        "income_level": income_level,
        "genetical_risk": genetical_risk,
        "normalized_risk_score": normalized_risk_score,
        "gender": gender,
        "region": region,
        "marital_status": marital_status,
        "bmi_category": bmi_category,
        "smoking_status": smoking_status,
        "employment_status": employment_status
    }


    # ------------ PREDICTION BUTTON + LOGIC -------------
    if st.button("Predict Premium"):

        try:
            # Build scaler input
            df_scale = build_scaler_input(inputs)

            # Select scaler + model based on age
            if age <= 25:
                scaler = scaler_young
                model = model_young
            else:
                scaler = scaler_old
                model = model_old

            # Scale
            scaled_arr = scaler.transform(df_scale)
            scaled_dict = dict(zip(COLS_TO_SCALE, scaled_arr.flatten()))

            # Build final model input
            X_final = build_model_input_from_scaled(scaled_dict, inputs)
            X_final = X_final.reindex(columns=MODEL_COLUMNS, fill_value=0)

            # Predict
            pred = model.predict(X_final)[0]

            st.markdown(f"<div class='pred-box'>💰 Estimated Premium: ₹ {pred:,.2f}</div>",
                        unsafe_allow_html=True)

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)


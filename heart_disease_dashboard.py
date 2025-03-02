import streamlit as st
import numpy as np
import joblib

# 🎯 Load Model & Scaler
MODEL_PATH = "heart_disease_model_balanced.pkl"
st.sidebar.title("🔍 Model Information")

try:
    model_data = joblib.load(MODEL_PATH)
    model, scaler, feature_names = model_data  # ✅ Unpack model components
    st.sidebar.success("✅ Model Loaded Successfully!")
    st.sidebar.write(f"📌 **Model Expects Features:** `{feature_names}`")
except Exception as e:
    st.sidebar.error(f"❌ **Error loading model:** {e}")
    st.stop()

# ---- 🎯 Title & Description ----
st.title("💖 AI-Powered Heart Disease Risk Prediction")
st.markdown("This tool predicts the likelihood of heart disease based on medical factors.")

# ---- 📊 User Input Section ----
st.subheader("🩺 Enter Patient Information")

col1, col2 = st.columns(2)  # Organizing inputs into two columns

with col1:
    age = st.number_input("📅 Age", min_value=18, max_value=120, value=45)
    sex = st.radio("🚻 Sex", ["Male", "Female"])
    cp = st.radio("💔 Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
    trestbps = st.number_input("🩸 Resting Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("🩸 Serum Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
    fbs = st.radio("🍬 Fasting Blood Sugar > 120 mg/dL", ["Yes", "No"])
    
with col2:
    restecg = st.radio("📊 Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.number_input("💓 Maximum Heart Rate Achieved", min_value=50, max_value=200, value=150)
    exang = st.radio("🏃 Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("📉 ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0)
    slope = st.radio("📈 Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.slider("🩺 Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, value=1)
    thal = st.radio("🧬 Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

# ✅ Convert categorical inputs to numerical format
sex = 1 if sex == "Male" else 0
cp = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal": 3, "Asymptomatic": 4}[cp]
fbs = 1 if fbs == "Yes" else 0
restecg = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}[restecg]
exang = 1 if exang == "Yes" else 0
slope = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}[slope]
thal = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}[thal]

# ✅ Prepare input data (ensure feature order matches training data)
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

# ---- 🔮 Prediction Section ----
if st.button("🩺 Predict Heart Disease Risk"):
    st.subheader("🔄 Processing Prediction...")
    try:
        # ✅ Scale input data
        input_scaled = scaler.transform(input_data)
        
        # ✅ Make prediction
        probabilities = model.predict_proba(input_scaled)[0]
        risk_percentage = round(probabilities[1] * 100, 2)

        # ✅ Display prediction result
        st.subheader("🎯 Prediction Result")
        if risk_percentage >= 35:
            st.error(f"🚨 **High Risk of Heart Disease: {risk_percentage}%**")
            st.warning("⚠️ Consider consulting a doctor immediately.")
        elif 20 <= risk_percentage < 35:
            st.warning(f"⚠️ **Moderate Risk of Heart Disease: {risk_percentage}%**")
            st.info("🩺 Lifestyle improvements and medical check-ups are recommended.")
        else:
            st.success(f"✅ **Low Risk of Heart Disease: {risk_percentage}%**")
            st.info("💚 Keep maintaining a healthy lifestyle!")

    except Exception as e:
        st.error(f"❌ **Prediction Error:** {e}")







# import streamlit as st
# import numpy as np
# import joblib

# # Load trained model
# model = joblib.load(r'B:\DiabetesApp\model\diabetes_predict_model.pkl')
# best_thresh = 0.51  # Based on your F1 optimization

# st.title('🩺 Diabetes Prediction App')
# st.markdown('Enter the details below to check for diabetes.')

# # Input fields
# pregnancies = st.number_input("Pregnancies", 0, 20)
# glucose = st.number_input("Glucose", 0, 200)
# blood_pressure = st.number_input("Blood Pressure", 0, 150)
# skin_thickness = st.number_input("Skin Thickness", 0, 100)
# insulin = st.number_input("Insulin", 0, 900)
# bmi = st.number_input("BMI", 0.0, 70.0)
# dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
# age = st.number_input("Age", 1, 120)

# if st.button("Predict"):
#     # Calculate derived features
#     age_bmi = age * bmi
#     glucose_insulin_ratio = glucose / (insulin + 1)  # +1 to avoid division by zero

#     # Prepare input for model (10 features)
#     input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin,
#                             bmi, dpf, age, age_bmi, glucose_insulin_ratio]])

#     # Get prediction probability
#     prob = model.predict_proba(input_data)[0][1]

#     # Apply custom threshold
#     prediction = int(prob > best_thresh)

#     st.write(f"🧪 **Probability of being diabetic:** `{prob*100:.2f}%`")

#     if prediction == 1:
#         st.error("⚠️ **Likely to have Diabetes!**")
#     else:
#         st.success("✅ **Not likely to have Diabetes!**")

import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load(r'B:\DiabetesApp\model\diabetes_predict_model.pkl')
best_thresh = 0.51  # Based on your F1 optimization

st.title('🩺 Diabetes Prediction App')
st.markdown('Enter the details below to check for diabetes.')

# Input fields
pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0, 200)
blood_pressure = st.number_input("Blood Pressure", 0, 150)
skin_thickness = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

if st.button("Predict"):
    # Derived features
    age_bmi = age * bmi
    glucose_insulin_ratio = glucose / (insulin + 1)  # avoid div by zero

    # Prepare input
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin,
                            bmi, dpf, age, age_bmi, glucose_insulin_ratio]])

    # Predict
    prob = model.predict_proba(input_data)[0][1]
    prediction = int(prob > best_thresh)

    # Display probability
    st.write(f"🧪 **Probability of being diabetic:** `{prob * 100:.2f}%`")

    # Risk category
    if prob > 0.80:
        risk = "🔴 High Risk"
    elif prob > 0.5:
        risk = "🟠 Moderate Risk"
    else:
        risk = "🟢 Low Risk"
    
    st.markdown(f"### Risk Category: **{risk}**")

    # Prediction Result
    st.markdown("### 🧬 Prediction:")
    if prediction == 1:
        st.error("🟥 **Diabetic (Positive)** - Please consult a doctor.")
    else:
        st.success("🟩 **Not Diabetic (Negative)** - Looks safe.")

    # Input Summary
    st.markdown("### 📋 Your Input Summary")
    st.json({
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "Blood Pressure": blood_pressure,
        "Skin Thickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "Diabetes Pedigree Function": dpf,
        "Age": age,
        "Derived - Age x BMI": round(age_bmi, 2),
        "Derived - Glucose/Insulin Ratio": round(glucose_insulin_ratio, 3)
    })

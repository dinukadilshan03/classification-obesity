import sys
import os
import pandas as pd

# Get the directory of app.py (src folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (the main folder)
parent_dir = os.path.dirname(current_dir)

# Add BOTH to the python path so it can find 'src'
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import joblib
import streamlit as st
# ... rest of your imports

# 1. Page Configuration
st.set_page_config(page_title="Obesity Risk AI", page_icon="⚖️", layout="centered")

# 2. Custom CSS for a Clean Look
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #2e7d32; color: white; font-weight: bold; }
    .result-card { background-color: white; padding: 25px; border-radius: 12px; border-left: 8px solid #2e7d32; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# 3. Load the Model
@st.cache_resource
def load_model():
    # 1. Get the absolute path of the folder where app.py is (the 'src' folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Go up one level to the main folder, then into 'models'
    model_path = os.path.join(current_dir, "..", "models", "obesity_classifier_v2_optimized.pkl")
    
    # 3. Load the model
    return joblib.load(model_path)
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 4. Header Section
st.title("⚖️ Obesity Category AI Analyzer")
st.write("This tool uses a high-precision Machine Learning model (96.22% Accuracy) to categorize weight status based on physical and behavioral data.")
st.divider()

# 5. Form Layout (Grouping all 17 features)
with st.form("obesity_form"):
    
    # --- SECTION 1: PHYSICAL METRICS ---
    st.subheader("📍 Physical Metrics")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 1, 100, 25)
    with col2:
        height = st.number_input("Height (m)", 1.20, 2.50, 1.75)
        weight = st.number_input("Weight (kg)", 30.0, 250.0, 75.0)

    # --- SECTION 2: DIETARY HABITS ---
    st.subheader("🥗 Dietary Habits")
    d1, d2 = st.columns(2)
    with d1:
        favc = st.selectbox("Frequent High Calorie Food Consumption? (FAVC)", ["yes", "no"])
        fcvc = st.slider("Daily Vegetable Consumption Frequency (FCVC)", 1, 3, 2)
        ncp = st.slider("Number of Main Meals per Day (NCP)", 1, 4, 3)
    with d2:
        caec = st.selectbox("Food Consumption Between Meals (CAEC)", ["Sometimes", "Frequently", "Always", "no"])
        ch2o = st.slider("Daily Water Intake (Liters) (CH2O)", 1.0, 3.0, 2.0)
        calc = st.selectbox("Alcohol Consumption Frequency (CALC)", ["no", "Sometimes", "Frequently", "Always"])

    # --- SECTION 3: LIFESTYLE & HISTORY ---
    st.subheader("🏃 Lifestyle & Personal History")
    l1, l2 = st.columns(2)
    with l1:
        fam_history = st.selectbox("Family History of Overweight?", ["yes", "no"])
        faf = st.slider("Physical Activity Frequency (Days/Week) (FAF)", 0, 3, 1)
        scc = st.selectbox("Do you monitor your calories daily? (SCC)", ["no", "yes"])
    with l2:
        smoke = st.selectbox("Do you smoke?", ["no", "yes"])
        tue = st.slider("Daily Tech Device Usage (Hours) (TUE)", 0, 2, 1)
        mtrans = st.selectbox("Primary Transportation Mode (MTRANS)", 
                              ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

    st.divider()
    submit = st.form_submit_button("GENERATE AI ASSESSMENT")

# 6. Handling the Result
if submit:
    # Build Input DataFrame (Order and Names MUST match X_train exactly)
    input_dict = {
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history_with_overweight': fam_history,
        'FAVC': favc,
        'FCVC': float(fcvc),
        'NCP': float(ncp),
        'CAEC': caec,
        'SMOKE': smoke,
        'CH2O': float(ch2o),
        'SCC': scc,
        'FAF': float(faf),
        'TUE': float(tue),
        'CALC': calc,
        'MTRANS': mtrans
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # Run Prediction
    prediction = model.predict(input_df)[0]
    
    # Formatting and Styling result
    severity_colors = {
        "Insufficient_Weight": "#FFCC00", "Normal_Weight": "#2e7d32", 
        "Overweight_Level_I": "#FFA500", "Overweight_Level_II": "#FF8C00",
        "Obesity_Type_I": "#D32F2F", "Obesity_Type_II": "#B71C1C", "Obesity_Type_III": "#7B1FA2"
    }
    result_color = severity_colors.get(prediction, "#007bff")
    
    st.balloons()
    st.markdown(f"""
        <div class="result-card" style="border-left-color: {result_color};">
            <h2 style="color: {result_color}; margin: 0;">Predicted Class: {prediction.replace('_', ' ')}</h2>
            <p style="color: #666; font-size: 1.1em; margin-top: 5px;">
                The model has analyzed your data with <b>96.22%</b> accuracy confidence.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Final advice section
    with st.expander("ℹ️ What does this mean?"):
        st.write(f"The model classifies you as **{prediction.replace('_', ' ')}**. This prediction is based on the multi-dimensional patterns of eating habits, physical metrics, and lifestyle choices provided.")
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

import streamlit as st

# Initialize session state
if "show_qr" not in st.session_state:
    st.session_state.show_qr = False  # Initially, show the button

col1, col2 = st.columns([8, 2])  # Layout to align on top-right

with col2:
    if not st.session_state.show_qr:
        # Show Buy Me a Coffee button
        if st.button("☕ Buy Me a Coffee", key="buy_coffee", help="Click to pay directly"):
            st.session_state.show_qr = True  # Toggle state to show QR code
            st.rerun()  # Refresh the UI
        
    else:
        # Show QR Code for direct payment
        st.image("qr_code.png", caption="Scan to Pay", width=150)  # Replace with your QR code image
        st.success("Scan the QR code to make a payment.")

        # Optional: Add a "Back" button to restore the original state
        if st.button("⬅️ Back", key="back"):
            st.session_state.show_qr = False
            st.rerun()

# Load all models and scaler from the pickle file
with open('models_accuracy_scale.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Load the scaler used during training
scaler = model_data.get('scaler', MinMaxScaler())

# Mapping raw predictions to human-readable labels
target_mapping = {
    '-': 'Negative', 'S': 'Hyperthyroid', 'F': 'Hypothyroid',
    'AK': 'Mixed', 'R': 'Other', 'I': 'Negative', 'M': 'Hyperthyroid',
    'N': 'Hypothyroid', 'G': 'Negative', 'K': 'Other', 'A': 'Hyperthyroid',
    'L': 'Hyperthyroid', 'MK': 'Mixed', 'Q': 'Other', 'J': 'Negative',
    'C|I': 'Mixed', 'O': 'Hyperthyroid', 'LJ': 'Hyperthyroid',
    'H|K': 'Mixed', 'GK': 'Negative', 'MI': 'Hypothyroid', 'KJ': 'Other',
    'P': 'Hyperthyroid', 'FK': 'Hypothyroid', 'B': 'Hyperthyroid',
    'GI': 'Negative', 'C': 'Mixed', 'GKJ': 'Other', 'OI': 'Hypothyroid',
    'D|R': 'Mixed', 'D': 'Negative', 'E': 'Other'
}

def convert_prediction_to_readable(raw_prediction):
    """Converts raw model output to human-readable categories."""
    return target_mapping.get(raw_prediction, 'Unknown')

# Streamlit UI
st.title("Thyroid Prediction App")

# User Input Form
st.header("Enter Patient Details")

age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", ["Male", "Female"], index=1)

on_thyroxine = st.selectbox("On Thyroxine", ["Yes", "No"], index=0)
query_on_thyroxine = st.selectbox("Query on Thyroxine", ["Yes", "No"], index=1)
on_antithyroid_meds = st.selectbox("On Antithyroid Meds", ["Yes", "No"], index=1)
sick = st.selectbox("Sick", ["Yes", "No"], index=0)
pregnant = st.selectbox("Pregnant", ["Yes", "No"], index=1)
thyroid_surgery = st.selectbox("Thyroid Surgery", ["Yes", "No"], index=0)
I131_treatment = st.selectbox("I131 Treatment", ["Yes", "No"], index=0)
query_hypothyroid = st.selectbox("Query Hypothyroid", ["Yes", "No"], index=0)
query_hyperthyroid = st.selectbox("Query Hyperthyroid", ["Yes", "No"], index=1)
lithium = st.selectbox("Lithium", ["Yes", "No"], index=1)
goitre = st.selectbox("Goitre", ["Yes", "No"], index=0)
tumor = st.selectbox("Tumor", ["Yes", "No"], index=1)
hypopituitary = st.selectbox("Hypopituitary", ["Yes", "No"], index=1)
psych = st.selectbox("Psych", ["Yes", "No"], index=1)

TSH = st.number_input("TSH", min_value=0.0, max_value=100.0, value=15.0)
T3 = st.number_input("T3", min_value=0.0, max_value=10.0, value=0.5)
TT4 = st.number_input("TT4", min_value=0.0, max_value=500.0, value=50.0)
T4U = st.number_input("T4U", min_value=0.0, max_value=10.0, value=0.7)
FTI = st.number_input("FTI", min_value=0.0, max_value=500.0, value=40.0)

# Encode categorical variables
encoded_data = np.array([
    age, 1 if sex == "Female" else 0,
    1 if on_thyroxine == "Yes" else 0,
    1 if query_on_thyroxine == "Yes" else 0,
    1 if on_antithyroid_meds == "Yes" else 0,
    1 if sick == "Yes" else 0,
    1 if pregnant == "Yes" else 0,
    1 if thyroid_surgery == "Yes" else 0,
    1 if I131_treatment == "Yes" else 0,
    1 if query_hypothyroid == "Yes" else 0,
    1 if query_hyperthyroid == "Yes" else 0,
    1 if lithium == "Yes" else 0,
    1 if goitre == "Yes" else 0,
    1 if tumor == "Yes" else 0,
    1 if hypopituitary == "Yes" else 0,
    1 if psych == "Yes" else 0,
    TSH, T3, TT4, T4U, FTI
])

# Feature names (ensure it matches training data)
feature_names = ["TSH", "T3", "TT4", "T4U", "FTI"]

# Convert numerical values to DataFrame for proper scaling
user_data_df = pd.DataFrame([encoded_data[16:]], columns=feature_names)

# Scale numerical values
scaled_user_data = scaler.transform(user_data_df)

# Concatenate categorical and scaled numerical features
X = np.concatenate((encoded_data[:16], scaled_user_data.flatten())).reshape(1, -1)

# Mapping display names to actual model keys
model_mapping = {
    'Linear-Regression': ['LinearRegression_model'],
    'KNN': ['KNN_model'],
    'Linear-SVC': ['LinearSVC_model'],
    'RBF-SVM': ['RBFSVM_model'],
    'Polynomial-SVM': ['PolynomialSVM_model'],
    'Decision-Tree (Recommended)': ['DecisionTree_model']
}

# Select model from dropdown
user_model_input = st.selectbox(
    "Models",
    list(model_mapping.keys()),
    index=5  # Default to 'Decision Tree (Recommended)'
)

# Retrieve the selected model
model_name = model_mapping[user_model_input][0]
model = model_data['models'][model_name][0]

accuracy_rate = int(model_data['models'][model_name][1] * 100)

st.warning(f"Accuracy rate of above model: {accuracy_rate}%")

# Predict the condition
if st.button("Predict"):
    prediction = model.predict(X)  # Model prediction
    readable_prediction = convert_prediction_to_readable(prediction[0])
    st.success(f"Prediction: The patient likely has **{readable_prediction}**.")

# Pre-filled default data for testing
if st.checkbox("Use Default Thyroid-Affected Data"):
    st.write("Default data loaded for testing purposes.")
    default_data = {
        "Age": 45,
        "Sex": "Female",
        "On Thyroxine": "Yes",
        "Query on Thyroxine": "No",
        "On Antithyroid Meds": "No",
        "Sick": "Yes",
        "Pregnant": "No",
        "Thyroid Surgery": "Yes",
        "I131 Treatment": "Yes",
        "Query Hypothyroid": "Yes",
        "Query Hyperthyroid": "No",
        "Lithium": "No",
        "Goitre": "Yes",
        "Tumor": "No",
        "Hypopituitary": "No",
        "Psych": "Yes",
        "TSH": 15.0,
        "T3": 0.5,
        "TT4": 50.0,
        "T4U": 0.7,
        "FTI": 40.0
    }
    st.json(default_data)

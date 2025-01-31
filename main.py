import streamlit as st
import numpy as np
import pandas as pd
import pickle
# from sklearn.preprocessing import MinMaxScaler

# Load models, scaler, and label encoder
with open('models_accuracy_scale.pkl', 'rb') as f:
    model_data = pickle.load(f)

scaler = model_data.get('scaler', None)
label_encoder = model_data.get('label_encoder', None)  # Extract LabelEncoder

# Mapping from character labels to user-readable labels
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

def convert_prediction_to_readable(encoded_prediction):
    """Converts numeric model output to character and then to a human-readable label."""
    if label_encoder:
        try:
            # Convert numeric to character label
            char_label = label_encoder.inverse_transform([encoded_prediction])[0]
            # Convert character label to user-readable label
            readable_label = target_mapping.get(char_label, "Unknown")
            return char_label, readable_label
        except Exception as e:
            st.error(f"Error decoding prediction: {e}")
            return "Error", "Unknown"
    else:
        st.error("LabelEncoder not found in the model file.")
        return "Error", "Unknown"

# Streamlit UI

# Initialize session state
if "show_qr" not in st.session_state:
    st.session_state.show_qr = False  # Initially, show the button

col1, col2 = st.columns([8, 2])  # Layout to align on top-right

with col2:
    if not st.session_state.show_qr:
        # Show Buy Me a Coffee button
        if st.button("Donate!", key="buy_coffee", help="Click to pay directly"):
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

st.markdown("<div style='text-align: left;'><a href='https://github.com/intrepid-ani/thyroid_detection_model' target='_blank'><img src='https://img.shields.io/badge/GitHub-Repo-blue?logo=github'></a></div>", unsafe_allow_html=True)


st.title("Thyroid Prediction App")
st.header("Enter Patient Details")

# User Input Form
age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", ["Male", "Female"], index=1)

on_thyroxine = st.selectbox("On Thyroxine", ["Yes", "No"])
query_on_thyroxine = st.selectbox("Query on Thyroxine", ["Yes", "No"])
on_antithyroid_meds = st.selectbox("On Antithyroid Meds", ["Yes", "No"])
sick = st.selectbox("Sick", ["Yes", "No"])
pregnant = st.selectbox("Pregnant", ["Yes", "No"])
thyroid_surgery = st.selectbox("Thyroid Surgery", ["Yes", "No"])
I131_treatment = st.selectbox("I131 Treatment", ["Yes", "No"])
query_hypothyroid = st.selectbox("Query Hypothyroid", ["Yes", "No"])
query_hyperthyroid = st.selectbox("Query Hyperthyroid", ["Yes", "No"])
lithium = st.selectbox("Lithium", ["Yes", "No"])
goitre = st.selectbox("Goitre", ["Yes", "No"])
tumor = st.selectbox("Tumor", ["Yes", "No"])
hypopituitary = st.selectbox("Hypopituitary", ["Yes", "No"])
psych = st.selectbox("Psych", ["Yes", "No"])

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

# Scale numerical values
feature_names = ["TSH", "T3", "TT4", "T4U", "FTI"]
user_data_df = pd.DataFrame([encoded_data[16:]], columns=feature_names)
scaled_user_data = scaler.transform(user_data_df)

# Combine categorical and numerical features
X = np.concatenate((encoded_data[:16], scaled_user_data.flatten())).reshape(1, -1)

# Model selection
model_mapping = {
    'Linear-Regression': 'LinearRegression_model',
    'KNN': 'KNN_model',
    'Linear-SVC': 'LinearSVC_model',
    'RBF-SVM': 'RBFSVM_model',
    'Polynomial-SVM': 'PolynomialSVM_model',
    'Decision-Tree (Recommended)': 'DecisionTree_model'
}

user_model_input = st.selectbox("Models", list(model_mapping.keys()), index=5)
model_name = model_mapping[user_model_input]
model = model_data['models'][model_name][0]
accuracy_rate = int(model_data['models'][model_name][1] * 100)

st.warning(f"Accuracy rate of above model: {accuracy_rate}%")

# Predict and display result
if st.button("Predict"):
    prediction = model.predict(X)  # Get numeric prediction
    raw_prediction = prediction[0]

    # Convert to character and user-readable label
    char_label, readable_prediction = convert_prediction_to_readable(raw_prediction)

    # Show results
    st.write(f"Raw Model Output: {raw_prediction} → Character: **{char_label}**")
    st.success(f"Final Prediction: **{readable_prediction}**")

import streamlit as st
import numpy as np
import pickle

# Load all models from the pickle file
with open('thyroid_dection_models.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Mapping for display names and actual model keys
model_mapping = {
    'Linear-Regression': 'LinearRegression_model',
    'KNN': 'KNN_model',
    'Linear-SVC': 'LinearSVC_model',
    'RBF-SVM': 'RBFSVM_model',
    'Polynomial-SVM': 'PolynomialSVM_model',
    'Decision-Tree (Recommended)': 'DecisionTree_model'
}

# Streamlit app
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

# Prepare input for the model
user_data = np.array([[age, 1 if sex == "Female" else 0,
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
                       TSH, T3, TT4, T4U, FTI]])


# Select model from dropdown
select_model_display = st.selectbox(
    "Models",
    list(model_mapping.keys()),
    index=5  # Default to 'DecisionTree_model (Recommended)'
)

# Map display name to actual model key
select_model = model_mapping[select_model_display]

# Retrieve the selected model
model = model_data[select_model]


# Retrieve the selected model
model = model_data[select_model]

# Use the selected model for prediction
st.write(f"Selected model: {select_model}")

# Predict the condition
if st.button("Predict"):
    prediction = model.predict(user_data)  #model['DecisionTree_model'].predict(X) 
    # Map prediction to categories
    if prediction == 0:
        st.success("The patient is likely **Negative** (No Thyroid Problem).")
    elif prediction == 1:
        st.warning("The patient likely has **Hyperthyroid**.")
    elif prediction == 2:
        st.warning("The patient likely has **Hypothyroid**.")
    elif prediction == 3:
        st.error("The patient likely has **Other Thyroid Issues**.")
    else:
        st.error("Unable to classify the condition. Please recheck inputs.")

# Pre-filled default data for testing
if st.checkbox("Use Default Thyroid-Affected Data"):
    st.write("Default data loaded for testing purposes.")
    st.write({
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
    })
import pickle
import numpy as np

with open('models_accuracy_scale.pkl', 'rb') as f:
    model_data = pickle.load(f)

# numerical_features = ['TSH', 'T3', 'TT4', 'T4U', 'FTI']

scale = model_data['scaler']
data = {
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


result = scale.transform(np.array([data['TSH'], data["T3"], data["TT4"], data["T4U"], data["FTI"]]).reshape(1, -1))

model = model_data['models']['PolynomialSVM_model'][0]

# result = model.predict(np.array(data.values()))

print(data.values())
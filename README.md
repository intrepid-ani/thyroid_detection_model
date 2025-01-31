# Thyroid Detection App

## Introduction
The **Thyroid Detection App** is a web-based application developed using **Streamlit** that helps in predicting thyroid conditions based on user input. The application leverages machine learning models trained on thyroid-related medical data to provide an accurate diagnosis.

This project was created as a **training project** to gain hands-on experience in machine learning, data preprocessing, and web app development.

## Features
- User-friendly interface built with Streamlit
- Accepts user input related to thyroid-related symptoms and test results
- Supports multiple machine learning models for prediction
- Displays the predicted condition along with confidence levels
- Integrated with a donation system using a QR code
- Hosted online for public access

## Project Motivation
This project was developed to gain practical experience in:
- Working with medical datasets
- Implementing machine learning algorithms for classification
- Developing an interactive web application
- Deploying a machine learning model for public use

## How It Works
1. Users enter relevant details, including medical history and thyroid test results.
2. Data is preprocessed and normalized using a **scaler**.
3. The selected machine learning model predicts the thyroid condition.
4. The output is mapped to a human-readable format (e.g., "Hypothyroid," "Hyperthyroid," or "Negative").
5. Users receive the final prediction along with model accuracy details.

## Available Machine Learning Models
The app allows users to choose from multiple machine learning models, including:
- Linear Regression
- K-Nearest Neighbors (KNN)
- Linear Support Vector Classifier (SVC)
- Radial Basis Function (RBF) SVM
- Polynomial SVM
- Decision Tree (Recommended)

## Technologies Used
- **Python** (Core language)
- **Streamlit** (Web framework)
- **Scikit-learn** (Machine learning)
- **Pandas & NumPy** (Data processing)
- **Pickle** (Model persistence)

## Installation & Setup
If you wish to run the project locally, follow these steps:

### Prerequisites
Ensure you have Python installed on your system. Then, install the required dependencies:
```sh
pip install -r requirements.txt
```

### Running the App
```sh
streamlit run main.py
```

## Live Demo
The application is available online at:
[Thyroid Detection App](https://thyroiddetedtion.streamlit.app/)

## Repository
Find the project source code on GitHub:
[GitHub Repository](https://github.com/intrepid-ani/thyroid_detection_model)

## Future Improvements
- Enhance model accuracy with more advanced machine learning techniques
- Improve UI/UX with additional visualizations
- Implement database storage for user data tracking
- Expand functionality to cover more thyroid-related conditions

## License
This project is released under the MIT License. You are free to modify and distribute it.

## Acknowledgments
Special thanks to **Scikit-learn** and **Streamlit** for providing open-source tools that made this project possible.

---
For any questions or contributions, feel free to reach out on **GitHub**!

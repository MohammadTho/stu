import streamlit as st
import pandas as pd
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Streamlit app UI for file upload
st.title("Student Employability Prediction")
st.write("Please upload your Excel file (.xlsx) containing the data.")

# File uploader widget for the dataset
uploaded_file = st.file_uploader("Choose an XLSX file", type=["xlsx"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Load the Excel file into a pandas DataFrame
    df = pd.read_excel(uploaded_file)

    # Show the first few rows of the DataFrame
    st.write("Here is a preview of your data:")
    st.dataframe(df)

    # Display basic info about the dataset
    st.write("Basic Information about the dataset:")
    st.write(df.info())

    # Phase 1 - DATA PROCESSING: Clean and Normalize the dataset
    data_cleaned = df.drop(columns=["Name of Student"])  # Drop the "Name of Student" column

    # Normalize numeric columns (Min-Max Scaling)
    scaler = MinMaxScaler()
    numeric_columns = [
        "GENERAL APPEARANCE", "MANNER OF SPEAKING", "PHYSICAL CONDITION",
        "MENTAL ALERTNESS", "SELF-CONFIDENCE", "ABILITY TO PRESENT IDEAS",
        "COMMUNICATION SKILLS", "Student Performance Rating"
    ]
    data_cleaned[numeric_columns] = scaler.fit_transform(data_cleaned[numeric_columns])

    # Encode the "CLASS" column
    le = LabelEncoder()
    data_cleaned["CLASS"] = le.fit_transform(data_cleaned["CLASS"])

    st.write("Cleaned dataset preview:")
    st.dataframe(data_cleaned.head())

    # Phase 2 - Feature Engineering: Check Feature Correlation
    correlation_matrix = data_cleaned.corr()

    st.write("Feature Correlation Heatmap:")
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    st.pyplot()

    # Aggregated Features
    data_cleaned['COMMUNICATION_SKILLS_TOTAL'] = (
        data_cleaned['COMMUNICATION SKILLS'] + data_cleaned['ABILITY TO PRESENT IDEAS']
    )
    st.write("Aggregated Feature - COMMUNICATION_SKILLS_TOTAL:")
    st.dataframe(data_cleaned[['COMMUNICATION SKILLS', 'ABILITY TO PRESENT IDEAS', 'COMMUNICATION_SKILLS_TOTAL']].head())

    # Phase 3 - Model Building

    # Split the data into features (X) and target (y)
    X = data_cleaned.drop(columns=['CLASS'])
    y = data_cleaned['CLASS']

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the RandomForestClassifier model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    rf_predictions = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)

    st.write(f"Random Forest Model Accuracy: {rf_accuracy:.4f}")
    st.write(f"Classification Report: \n{classification_report(y_test, rf_predictions)}")

    # Save the trained model, scaler, and label encoder
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')

    # Phase 4 - Model Prediction: Take input from the user for prediction
    st.write("### Make Predictions")
    communication_skills = st.slider("Communication Skills Total", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    confidence_alertness = st.slider("Confidence and Alertness", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    # Prediction button
    if st.button('Predict'):
        # Create a DataFrame from the user input
        user_data = pd.DataFrame([[communication_skills, confidence_alertness]], columns=['COMMUNICATION_SKILLS_TOTAL', 'CONFIDENCE_ALERTNESS'])

        # Scale the data (ensure you use the same scaler as when you trained the model)
        user_data_scaled = scaler.transform(user_data)

        # Load the trained model and make prediction
        rf_model = joblib.load('random_forest_model.pkl')
        prediction = rf_model.predict(user_data_scaled)

        # Decode the prediction back to original label if necessary
        prediction_class = le.inverse_transform(prediction)

        # Display the result
        st.write(f"Predicted Class: {prediction_class[0]}")

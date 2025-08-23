import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Load Data and Model
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("income_data.csv")
    return df

@st.cache_resource
def load_model():
    return joblib.load("logistic_income_model.pkl")

df = load_data()
model = load_model()

# ----------------------------
# App Title & Description
# ----------------------------
st.title("Income Level Prediction App")
st.write("This interactive app predicts whether a person earns **>50K or <=50K** per year based on demographic and work-related attributes.")

# Sidebar Navigation
page = st.sidebar.selectbox("Choose a page", ["Home", "EDA", "Prediction"])

# ----------------------------
# HOME PAGE
# ----------------------------
if page == "Home":
    st.subheader("Dataset Overview")
    st.dataframe(df.head(10))
    st.write("### Dataset Summary")
    st.write(df.describe(include="all"))

# ----------------------------
# EDA PAGE
# ----------------------------
elif page == "EDA":
    st.subheader("Exploratory Data Analysis")

    # Target Variable Distribution
    st.write("### Income Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="income", data=df, ax=ax)
    st.pyplot(fig)

    # Age Distribution
    st.write("### Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["age"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    # Correlation Heatmap (numeric features)
    st.write("### Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ----------------------------
# PREDICTION PAGE
# ----------------------------
elif page == "Prediction":
    st.subheader("Make a Prediction")

    # Example form (adjust feature names to match your model training)
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    education_num = st.number_input("Education Number", min_value=1, max_value=16, value=10)
    hours_per_week = st.number_input("Hours per week", min_value=1, max_value=100, value=40)

    workclass = st.selectbox("Workclass", df["workclass"].dropna().unique())
    occupation = st.selectbox("Occupation", df["occupation"].dropna().unique())
    marital_status = st.selectbox("Marital Status", df["marital_status"].dropna().unique())
    sex = st.selectbox("Sex", df["sex"].dropna().unique())
    native_country = st.selectbox("Native Country", df["native_country"].dropna().unique())

    # Prediction Button
    if st.button("Predict Income Level"):
        # Convert categorical to same encoding used in model training
        input_data = pd.DataFrame({
            "age": [age],
            "education_num": [education_num],
            "hours_per_week": [hours_per_week],
            "workclass": [workclass],
            "occupation": [occupation],
            "marital_status": [marital_status],
            "sex": [sex],
            "native_country": [native_country]
        })

        # Ensure preprocessing matches your training pipeline
        prediction = model.predict(input_data)
        prediction_label = prediction[0]

        st.success(f"Predicted Income Level: **{prediction_label}**")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("**Developed by Onyia Vivian Amara | YOUTHRIVE Capstone Project**")


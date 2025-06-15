import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Load original dataset for EDA
df = pd.read_csv('menu.csv')

# Set up Streamlit page
st.set_page_config(page_title="McDonald's Nutrition Project", layout="wide")

# 1. Introduction
st.title("🍟 McDonald's Nutrition Analysis & Calorie Predictor")
st.markdown("""
Welcome to the interactive Data Science project using McDonald's Nutrition dataset.  
This app allows you to:
- Explore the dataset visually
- Predict calories of food items based on nutritional values
""")

# 2. EDA with checkboxes
st.header("🔍 Exploratory Data Analysis (EDA)")

if st.checkbox("Show Dataset"):
    st.dataframe(df)

if st.checkbox("Summary Statistics"):
    st.write(df.describe())

if st.checkbox("Distribution of Calories"):
    fig, ax = plt.subplots()
    sns.histplot(df['Calories'], kde=True, ax=ax)
    st.pyplot(fig)

if st.checkbox("Correlation Heatmap"):
    try:
        numeric_df = df.select_dtypes(include='number')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    except:
        st.warning("Unable to generate correlation matrix.")

if st.checkbox("Calories vs Protein Scatter Plot"):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Protein', y='Calories', ax=ax)
    st.pyplot(fig)

# 3. Prediction section
st.header("🤖 Calorie Predictor")

# Inputs
col1, col2 = st.columns(2)
with col1:
    total_fat = st.number_input("Total Fat (g)", 0.0)
    sat_fat = st.number_input("Saturated Fat (g)", 0.0)
    trans_fat = st.number_input("Trans Fat (g)", 0.0)
    cholesterol = st.number_input("Cholesterol (mg)", 0.0)
with col2:
    sodium = st.number_input("Sodium (mg)", 0.0)
    carbs = st.number_input("Carbohydrates (g)", 0.0)
    sugars = st.number_input("Sugars (g)", 0.0)
    protein = st.number_input("Protein (g)", 0.0)

input_df = pd.DataFrame([[
    total_fat, sat_fat, trans_fat, cholesterol,
    sodium, carbs, sugars, protein
]], columns=[
    'Total Fat', 'Saturated Fat', 'Trans Fat', 'Cholesterol',
    'Sodium', 'Carbohydrates', 'Sugars', 'Protein'
])

if st.button("Predict Calories"):
    try:
        scaled = scaler.transform(input_df)
        prediction = model.predict(scaled)[0]
        st.success(f"🔥 Estimated Calories: **{prediction:.2f} kcal**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# 4. Conclusion
st.header("📌 Conclusion")
st.markdown("""
This project demonstrates a full **data science pipeline** using McDonald's nutritional dataset:
- We explored and visualized the data
- Built a machine learning model to predict **calories**
- Integrated everything into this **Streamlit web app**

🎯 **Key takeaway**: Nutritional features like fat, protein, and sugars strongly influence calorie content.

Thank you for exploring! 👋
""")

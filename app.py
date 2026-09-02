import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(page_title="Iris Flower Classification App", page_icon="🌸")

# Title and Description
st.title("🌸 Iris Flower Classification App")
st.write("This application predicts the species of an Iris flower based on its sepal and petal measurements using a trained Machine Learning model.")

# Sidebar for user inputs
st.sidebar.header("Input Features")

def get_user_input():
    sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
    petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.35, 0.1)
    petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)
    
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    return pd.DataFrame([data])

# Get input from user
input_df = get_user_input()

# Display user input values
st.subheader("Selected Input Features:")
st.write(input_df)

# Robust model loader
@st.cache_resource
def load_model():
    if os.path.exists("iris_model.joblib"):
        try:
            return joblib.load("iris_model.joblib")
        except Exception:
            pass
    # Fallback: train on the fly if file is absent or version mismatch
    iris = load_iris()
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(iris.data, iris.target)
    return clf

model = load_model()

# Prediction button
if st.button("Predict Species"):
    species_names = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
    
    # Make prediction
    prediction_index = model.predict(input_df)[0]
    prediction_probabilities = model.predict_proba(input_df)[0]
    
    # Display Result
    st.subheader("Prediction:")
    st.success(f"Predicted Species: **{species_names[prediction_index]}**")
    
    # Display Probabilities
    st.subheader("Prediction Probabilities:")
    prob_df = pd.DataFrame({
        'Species': species_names,
        'Probability (%)': [round(p * 100, 2) for p in prediction_probabilities]
    })
    st.write(prob_df)
    
    # Bar chart
    chart_data = prob_df.set_index('Species')
    st.bar_chart(chart_data)

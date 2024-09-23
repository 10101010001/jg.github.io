import streamlit as st
import joblib

# Load the model
model = joblib.load('model.pkl')

st.title("Multivariate Linear Regression Model")

st.write("Enter values to make a prediction:")

# Input fields for the two variables
input1 = st.number_input("Input 1", value=1)
input2 = st.number_input("Input 2", value=1)

# When the user clicks the "Predict" button
if st.button("Predict"):
    prediction = model.predict([[input1, input2]])
    st.write(f"Prediction: {prediction[0]}")
import streamlit as st
import numpy as np
import joblib

model = joblib.load("iris_model.pkl")

class_names = ["Setosa", "Versicolor", "Virginica"]

st.title("Iris Flower Prediction")
st.write("Enter the flower measurements below.")

sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

if st.button("Predict Flower"):
    measurements = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = model.predict(measurements)[0]
    st.success("Predicted Iris Species: " + class_names[prediction])
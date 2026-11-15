import streamlit as st
import zipfile
import os
import joblib

# Extract model from ZIP if needed
if not os.path.exists("bean_model.pkl"):
    with zipfile.ZipFile("bean_model.zip", "r") as zip_ref:
        zip_ref.extractall(".")

# Load model
model = joblib.load("bean_model.pkl")

# Load encoder
label_encoder = joblib.load("label_encoder.pkl")

st.title("Dry Bean Classification App")

area = st.number_input("Area")
perimeter = st.number_input("Perimeter")
major_axis = st.number_input("Major Axis Length")
minor_axis = st.number_input("Minor Axis Length")

if st.button("Predict"):
    st.success("Prediction button clicked successfully!")

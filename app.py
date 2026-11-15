import streamlit as st

st.title("Dry Bean Classification App")

area = st.number_input("Area")
perimeter = st.number_input("Perimeter")
major_axis = st.number_input("Major Axis Length")
minor_axis = st.number_input("Minor Axis Length")

if st.button("Predict"):
	st.success("Model prediction will appear here")

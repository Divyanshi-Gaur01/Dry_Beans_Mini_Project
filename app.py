import os
import zipfile

import joblib
import numpy as np
import streamlit as st


# ----------------------------------------------------------
# PAGE SETTINGS
# ----------------------------------------------------------

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="centered"
)


# ----------------------------------------------------------
# EXTRACT MODEL FROM ZIP
# ----------------------------------------------------------

if not os.path.exists("bean_model.pkl"):
    if os.path.exists("bean_model.zip"):
        with zipfile.ZipFile("bean_model.zip", "r") as zip_ref:
            zip_ref.extractall(".")
    else:
        st.error("bean_model.zip was not found.")
        st.stop()


# ----------------------------------------------------------
# LOAD MODEL, SCALER, AND LABEL ENCODER
# ----------------------------------------------------------

try:
    model = joblib.load("bean_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

except FileNotFoundError as error:
    st.error(f"Required file is missing: {error}")
    st.stop()

except Exception as error:
    st.error(f"Unable to load model files: {error}")
    st.stop()


# ----------------------------------------------------------
# APP TITLE
# ----------------------------------------------------------

st.title("🌱 Dry Bean Classification App")

st.write(
    "Enter all 16 physical measurements of a dry bean, "
    "then click **Predict Bean Type**."
)


# ----------------------------------------------------------
# INPUT FEATURES
# The default values are typical dataset values.
# ----------------------------------------------------------

area = st.number_input(
    "Area",
    min_value=0.0,
    value=44652.0,
    format="%.4f"
)

perimeter = st.number_input(
    "Perimeter",
    min_value=0.0,
    value=794.941,
    format="%.6f"
)

major_axis_length = st.number_input(
    "Major Axis Length",
    min_value=0.0,
    value=296.8833669,
    format="%.7f"
)

minor_axis_length = st.number_input(
    "Minor Axis Length",
    min_value=0.0,
    value=192.4317333,
    format="%.7f"
)

aspect_ration = st.number_input(
    "Aspect Ratio",
    min_value=0.0,
    value=1.551123666,
    format="%.9f"
)

eccentricity = st.number_input(
    "Eccentricity",
    min_value=0.0,
    value=0.764440806,
    format="%.9f"
)

convex_area = st.number_input(
    "Convex Area",
    min_value=0.0,
    value=45178.0,
    format="%.4f"
)

equiv_diameter = st.number_input(
    "Equivalent Diameter",
    min_value=0.0,
    value=238.4380258,
    format="%.7f"
)

extent = st.number_input(
    "Extent",
    min_value=0.0,
    value=0.759858948,
    format="%.9f"
)

solidity = st.number_input(
    "Solidity",
    min_value=0.0,
    value=0.988282998,
    format="%.9f"
)

roundness = st.number_input(
    "Roundness",
    min_value=0.0,
    value=0.883157289,
    format="%.9f"
)

compactness = st.number_input(
    "Compactness",
    min_value=0.0,
    value=0.801276684,
    format="%.9f"
)

shape_factor_1 = st.number_input(
    "Shape Factor 1",
    min_value=0.0,
    value=0.006645174,
    format="%.9f"
)

shape_factor_2 = st.number_input(
    "Shape Factor 2",
    min_value=0.0,
    value=0.001693531,
    format="%.9f"
)

shape_factor_3 = st.number_input(
    "Shape Factor 3",
    min_value=0.0,
    value=0.642044324,
    format="%.9f"
)

shape_factor_4 = st.number_input(
    "Shape Factor 4",
    min_value=0.0,
    value=0.996385938,
    format="%.9f"
)


# ----------------------------------------------------------
# PREDICTION
# Keep the feature order exactly the same as training.
# ----------------------------------------------------------

if st.button("Predict Bean Type", type="primary"):

    input_data = np.array([[
        area,
        perimeter,
        major_axis_length,
        minor_axis_length,
        aspect_ration,
        eccentricity,
        convex_area,
        equiv_diameter,
        extent,
        solidity,
        roundness,
        compactness,
        shape_factor_1,
        shape_factor_2,
        shape_factor_3,
        shape_factor_4
    ]])

    try:
        # Apply the same scaling used during model training
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction_encoded = model.predict(input_data_scaled)

        # Convert encoded result back to bean class name
        prediction_class = label_encoder.inverse_transform(
            prediction_encoded.astype(int)
        )

        st.success(
            f"Predicted Bean Type: {prediction_class[0]}"
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

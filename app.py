import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('model')

st.title("HDB Resale Price Prediction")

towns = ['Bedok', 'Punggol', 'Tampines']
flat_types = ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM']
storey_ranges = ['01 TO 03', '04 TO 06', '07 TO 09']

town_selected = st.selectbox("Select Town", towns)
flat_type_selected = st.selectbox("Select Flat Type", flat_types)
storey_range_selected = st.selectbox("Select Storey Range", storey_ranges)
floor_area_selected = st.slider("Select Floor Area (sqm)",  max_value=200, value=70)

if st.button("Predict HDB price"):
    input_data = {
        'town': town_selected,
        'flat_type': flat_type_selected,
        'storey_range': storey_range_selected,
        'floor_area': floor_area_selected
    }

    df_input = pd.DataFrame({
        'town': [town_selected],
        'flat_type': [flat_type_selected],
        'storey_range': [storey_range_selected],
        'floor_area': [floor_area_selected]
    })

    df_input = pd.get_dummies(df_input, columns = ['town', 'flat_type', 'storey_range'])

    df_input = df_input.reindex(columns = model.feature_names_in_,
                                fill_value = 0)

    y_unseen_pred = model.predict(df_input)[0]
    st.success(f"Predicted Resale Price: ${y_unseen_pred}:,.2f")

st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("https://upload.wikimedia.org/wikipedia/commons/b/b6/Image_created_with_a_mobile_phone.png")
        background-size: cover
    }}
    <style>
    """,
    unsafe_allow_html = True
)
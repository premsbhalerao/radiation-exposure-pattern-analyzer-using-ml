import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Radiation Analyzer", layout="centered")
st.title("☢ Radiation Exposure Pattern Analyzer")
st.write("Upload your radiation data CSV to check for safety risks.")

# Load trained model
model = joblib.load("model/rf_model.pkl")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df.head())

    # Encode 'Shift' column if present
    if 'Shift' in df.columns:
        le = LabelEncoder()
        df['Shift'] = le.fit_transform(df['Shift'])

    # Plot exposure over time
    if 'Date' in df.columns and 'Exposure_mSv' in df.columns:
        fig = px.line(df, x='Date', y='Exposure_mSv', title='Radiation Exposure Over Time')
        st.plotly_chart(fig)

    # Make predictions if relevant columns are present
    if all(col in df.columns for col in ['Exposure_mSv', 'Hours_Worked', 'Shift']):
        features = df[['Exposure_mSv', 'Hours_Worked', 'Shift']]
        predictions = model.predict(features)
        df['Risk'] = predictions
        st.subheader("Prediction Results")
        st.dataframe(df)
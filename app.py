import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline (scaler + model) and the encoders
pipeline = joblib.load("best_model_pipeline.pkl")
encoders = joblib.load("encoders.pkl")

# --- Streamlit App UI ---
st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼", layout="wide")

st.title("💼 Employee Salary Prediction")
st.markdown("This app predicts whether an employee's annual income is **more than $50K** or **less than/equal to $50K**.")
st.markdown("---")

# Create columns for input
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Demographics")
    age = st.slider("Age", 17, 75, 35)
    gender = st.selectbox("Gender", encoders['gender'].classes_)
    race = st.selectbox("Race", encoders['race'].classes_)
    native_country = st.selectbox("Native Country", encoders['native-country'].classes_)

with col2:
    st.header("Employment")
    workclass = st.selectbox("Work Class", encoders['workclass'].classes_)
    occupation = st.selectbox("Occupation", encoders['occupation'].classes_)
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)

with col3:
    st.header("Personal Life")
    educational_num = st.slider("Education (Numeric)", 5, 16, 10)
    marital_status = st.selectbox("Marital Status", encoders['marital-status'].classes_)
    relationship = st.selectbox("Relationship", encoders['relationship'].classes_)

# Create a button to trigger prediction
if st.button("Predict Salary Class", use_container_width=True):
    # Prepare the input data for the model
    # Note: We create a dataframe with all the columns the model was trained on.
    # We will use default values (mode) for features not included in the UI.
    input_data = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': 189778,  # Using a common value (mean/median)
        'educational-num': educational_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': 0, # Common value
        'capital-loss': 0, # Common value
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }
    input_df = pd.DataFrame([input_data])

    # Encode the categorical inputs using the loaded encoders
    for col in ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']:
        input_df[col] = encoders[col].transform(input_df[col])

    # Make prediction using the pipeline (it handles scaling and predicting)
    prediction_encoded = pipeline.predict(input_df)[0]
    prediction_proba = pipeline.predict_proba(input_df)[0]

    # Decode the prediction back to the original label ('<=50K' or '>50K')
    prediction_label = encoders['income'].inverse_transform([prediction_encoded])[0]
    probability_of_prediction = prediction_proba[prediction_encoded]

    # Display the result
    st.markdown("---")
    st.subheader("Prediction Result")
    if prediction_label == '>50K':
        st.success(f"**Predicted Income:** {prediction_label}")
    else:
        st.info(f"**Predicted Income:** {prediction_label}")

    st.write(f"**Confidence:** {probability_of_prediction:.2%}")


# @title 10. Running the Streamlit App
# This command will start the Streamlit web server.
# A URL will be printed below. You need to open this URL in your browser to see the app.


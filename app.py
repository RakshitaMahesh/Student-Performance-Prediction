
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Predictor")
st.write(
    "Predict a student's average exam score using demographic "
    "and educational background information."
)

@st.cache_resource
def train_model():
    df = pd.read_csv("StudentsPerformance.csv")

    df["average_score"] = df[
        ["math score", "reading score", "writing score"]
    ].mean(axis=1)

    X = df.drop(
        columns=[
            "math score", "reading score",
            "writing score", "average_score"
        ]
    )
    y = df["average_score"]

    categorical = X.select_dtypes(
        include=["object"]
    ).columns.tolist()
    numerical = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical),
        ("num", "passthrough", numerical)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100, random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    return model, categorical, numerical, mae

try:
    model, categorical, numerical, mae = train_model()
except Exception as error:
    st.error(
        "Could not load the dataset. Make sure "
        "StudentsPerformance.csv is in the same folder as app.py."
    )
    st.stop()

st.subheader("Enter Student Details")

with st.form("student_form"):
    gender = st.selectbox(
        "Gender", ["female", "male"]
    )
    race = st.selectbox(
        "Race/Ethnicity",
        ["group A", "group B", "group C", "group D", "group E"]
    )
    education = st.selectbox(
        "Parental Level of Education",
        [
            "some high school", "high school",
            "some college", "associate's degree",
            "bachelor's degree", "master's degree"
        ]
    )
    lunch = st.selectbox(
        "Lunch", ["standard", "free/reduced"]
    )
    preparation = st.selectbox(
        "Test Preparation Course", ["none", "completed"]
    )

    submitted = st.form_submit_button("Predict Average Score")

if submitted:
    student = pd.DataFrame([{
        "gender": gender,
        "race/ethnicity": race,
        "parental level of education": education,
        "lunch": lunch,
        "test preparation course": preparation
    }])

    prediction = model.predict(student)[0]

    st.subheader("Prediction Result")
    st.metric("Predicted Average Score", f"{prediction:.1f} / 100")
    st.caption(
        "This is an estimate from the trained model, "
        "not a guaranteed exam result."
    )

st.divider()
st.subheader("Model Evaluation")
st.write(f"Mean Absolute Error (MAE): {mae:.2f} points")
st.caption(
    "Evaluation uses a held-out test set. Predictions may be inaccurate."
)

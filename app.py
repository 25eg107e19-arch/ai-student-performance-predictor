
import streamlit as st
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="AI Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 AI Student Performance Predictor")
st.write(
    "Predict an estimated exam score using study habits "
    "and previous academic performance."
)

st.info(
    "Educational demonstration using synthetic data. "
    "Predictions are not real-world academic forecasts."
)


@st.cache_resource
def train_model():
    rng = np.random.default_rng(42)
    n = 1500

    study = rng.uniform(0, 12, n)
    attendance = rng.uniform(40, 100, n)
    previous = rng.uniform(20, 100, n)
    sleep = rng.uniform(3, 12, n)

    noise = rng.normal(0, 7, n)

    scores = (
        15
        + 3.5 * study
        + 0.20 * attendance
        + 0.35 * previous
        + 1.2 * sleep
        + noise
    )

    scores = np.clip(scores, 0, 100)

    X = pd.DataFrame({
        "Study Hours": study,
        "Attendance": attendance,
        "Previous Marks": previous,
        "Sleep Hours": sleep
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X, scores, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, mae, r2


model, mae, r2 = train_model()

st.subheader("Enter Student Details")

study = st.slider(
    "Daily study hours", 0.0, 12.0, 4.0, 0.5
)

attendance = st.slider(
    "Attendance (%)", 40, 100, 80
)

previous = st.slider(
    "Previous exam marks (%)", 20, 100, 65
)

sleep = st.slider(
    "Sleep hours per night", 3.0, 12.0, 8.0, 0.5
)

if st.button("Predict Exam Score", type="primary"):

    student = pd.DataFrame([{
        "Study Hours": study,
        "Attendance": attendance,
        "Previous Marks": previous,
        "Sleep Hours": sleep
    }])

    score = float(model.predict(student)[0])
    score = float(np.clip(score, 0, 100))

    st.subheader("Prediction Result")
    st.metric("Estimated score", f"{score:.1f}/100")
    st.progress(int(round(score)))

    st.caption(
        "This estimate comes from a model trained on "
        "synthetic data. It is not a reliable forecast."
    )

    st.subheader("Personalized Study Suggestions")

    suggestions = []

    if study < 3:
        suggestions.append(
            "Build a consistent daily study routine."
        )

    if attendance < 75:
        suggestions.append(
            "Try to improve class attendance."
        )

    if previous < 50:
        suggestions.append(
            "Revise foundational topics and practise questions."
        )

    if sleep < 7:
        suggestions.append(
            "Maintain a regular sleep schedule with adequate rest."
        )

    if not suggestions:
        suggestions.append(
            "Keep practising and review your progress regularly."
        )

    for item in suggestions:
        st.write("- " + item)


with st.expander("Model Evaluation and Feature Importance"):

    col1, col2 = st.columns(2)

    col1.metric("Test MAE", f"{mae:.2f}")
    col2.metric("Test R²", f"{r2:.3f}")

    importance = pd.Series(
        model.feature_importances_,
        index=[
            "Study Hours",
            "Attendance",
            "Previous Marks",
            "Sleep Hours"
        ]
    ).sort_values(ascending=False)

    st.write("Feature importance")
    st.bar_chart(importance)

    st.caption(
        "Evaluation metrics describe performance on synthetic "
        "test data and do not establish real-world accuracy."
    )

st.divider()

st.caption(
    "Built with Python, Pandas, NumPy, Scikit-learn and Streamlit."
)


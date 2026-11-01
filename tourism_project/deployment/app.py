"""
Streamlit app - Wellness Tourism Package purchase predictor ("Visit with Us").

Loads the model that the GitHub Actions pipeline committed to this folder, collects the customer
details in a form, stores them in a one-row DataFrame and shows the predicted purchase probability.
Deployed on Streamlit Community Cloud (main file: tourism_project/deployment/app.py).
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "best_tourism_model_v1.joblib"

st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧳", layout="wide")


@st.cache_resource
def load_model_bundle():
    """Model bundle written by train.py: pipeline + decision threshold + metadata."""
    return joblib.load(MODEL_PATH)


st.title("🧳 Wellness Tourism Package - Purchase Predictor")
st.write(
    "**Visit with Us** uses this tool to estimate, *before contacting a customer*, how likely they are "
    "to buy the new Wellness Tourism Package. Fill in the customer details and press **Predict**."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model file `{MODEL_PATH.name}` not found. Run the GitHub Actions pipeline - it trains the model "
        "and commits it to `tourism_project/deployment/`."
    )
    st.stop()

bundle = load_model_bundle()
model = bundle["pipeline"]
threshold = bundle["threshold"]
feature_names = bundle["feature_names"]

with st.sidebar:
    st.header("About the model")
    st.write("Random Forest, tuned with 5-fold cross-validated grid search (tracked in MLflow).")
    test_metrics = bundle["metrics"]["test"]
    st.metric("Test accuracy", f"{test_metrics['accuracy']:.1%}")
    st.metric("Test recall (buyers found)", f"{test_metrics['recall']:.1%}")
    st.metric("Test precision", f"{test_metrics['precision']:.1%}")
    st.caption(f"Customers with a purchase probability >= {threshold:.0%} are flagged as likely buyers.")

# ----------------------------------------------------------------------------------- input form
with st.form("customer_form"):
    st.subheader("Customer details")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=80, value=36)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital status", ["Single", "Married", "Divorced", "Unmarried"])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Monthly income (gross)", min_value=1000, max_value=200000,
                                         value=22000, step=500)
    with c2:
        city_tier = st.selectbox("City tier (1 = most developed)", [1, 2, 3])
        passport = st.selectbox("Holds a valid passport?", ["Yes", "No"])
        own_car = st.selectbox("Owns a car?", ["Yes", "No"])
        number_of_trips = st.number_input("Average trips per year", min_value=1, max_value=25, value=3)
        persons_visiting = st.number_input("People travelling with the customer", min_value=1, max_value=5, value=3)
        children_visiting = st.number_input("Children (under 5) travelling", min_value=0, max_value=5, value=1)
    with c3:
        type_of_contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
        product_pitched = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        preferred_star = st.selectbox("Preferred property star rating", [3, 4, 5])
        duration_of_pitch = st.number_input("Duration of pitch (minutes)", min_value=1, max_value=130, value=14)
        number_of_followups = st.number_input("Number of follow-ups", min_value=0, max_value=10, value=4)
        pitch_satisfaction = st.slider("Pitch satisfaction score", min_value=1, max_value=5, value=3)

    submitted = st.form_submit_button("Predict")

# ----------------------------------------------------------------------------------- prediction
if submitted:
    # collect the inputs into a dataframe (same columns / names as the training data)
    input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": persons_visiting,
        "NumberOfFollowups": number_of_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": pitch_satisfaction,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
    }])[feature_names]

    st.subheader("Input data")
    st.dataframe(input_df)

    probability = float(model.predict_proba(input_df)[0, 1])
    st.subheader("Prediction")
    st.metric("Probability of buying the Wellness Tourism Package", f"{probability:.1%}")
    st.progress(min(max(probability, 0.0), 1.0))
    if probability >= threshold:
        st.success("✅ Likely buyer - worth prioritising for the Wellness Tourism Package campaign.")
    else:
        st.warning("⚠️ Unlikely buyer - a lower-priority lead for this campaign.")

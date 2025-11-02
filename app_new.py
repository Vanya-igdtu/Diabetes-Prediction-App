import streamlit as st
import pandas as pd
import joblib
from transformers import pipeline

# ------------------ LOAD MODEL & ENCODERS ------------------
model = joblib.load("models/diabetes_model.pkl")
le_gender = joblib.load("models/le_gender.pkl")
le_smoking = joblib.load("models/le_smoking.pkl")

# ------------------ LOAD CHATBOT ------------------
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

def get_bot_response(user_input):
    response = chatbot(user_input, max_length=100, pad_token_id=50256)
    full_text = response[0]['generated_text']
    # Remove the user input from the start, keep only new reply
    return full_text[len(user_input):].strip()

# ------------------ SIDEBAR NAVIGATION ------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", [
    "Home", 
    "Diabetes Check", 
    "Your Profile", 
    "Doctor Appointment", 
    "About Us", 
    "Diabetes Knowledge", 
    "AI Chatbot"
])

# ------------------ HOME PAGE ------------------
if page == "Home":
    st.markdown("""
    <style>
        .main {background-color: #f5f5f5;}
        h1, h2, h3 {color: #2c3e50;}
        .stButton>button {background-color: #2c3e50; color: white;}
    </style>
    """, unsafe_allow_html=True)

    st.title("🏠 Welcome to Your Diabetes Health Dashboard")

    last_checkup = {
        "Date": "2025-09-15",
        "Diabetes Risk": "Moderate",
        "Probability": "0.62",
        "HbA1c Level": "6.1",
        "Glucose": "145",
        "BMI": "28.5"
    }
    st.markdown("### 🩺 Last Checkup Summary")
    st.table(pd.DataFrame([last_checkup]))

    st.markdown("---")
    st.markdown("### 📰 Health News Highlights")
    news_items = [
        {
            "title": "India launches largest-ever health outreach for women & children",
            "url": "https://health.economictimes.indiatimes.com/news/policy/pm-launches-largest-ever-health-scheme-for-women-and-children/123966618"
        },
        {
            "title": "Digital health revolution bridges rural–urban divide",
            "url": "https://health.economictimes.indiatimes.com/news/industry/transforming-rural-healthcare-in-india-a-digital-and-ai-driven-revolution/123898005"
        },
        {
            "title": "H3N2 virus outbreak hits Delhi-NCR, hospitals report surge",
            "url": "https://www.msn.com/en-in/health/health-news/delhi-ncr-health-news-live-updates-h3n2-virus-hits-india-doctors-alert-hospitals-report-massive-surge-in-cases/ar-AA1MMtt9"
        }
    ]
    for item in news_items:
        st.markdown(f"- [{item['title']}]({item['url']})")

    st.markdown("---")
    st.markdown("📌 Use the sidebar to explore predictions, appointments, and more.")

# ------------------ DIABETES PREDICTION ------------------
elif page == "Diabetes Check":
    st.title("🩺 Diabetes Prediction")

    st.markdown("### 👤 Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", le_gender.classes_, key="gender_check")
        age = st.number_input("Age", min_value=0, max_value=120, value=30, key="age_check")
    with col2:
        smoking_history = st.selectbox("Smoking History", le_smoking.classes_, key="smoking_check")
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, key="bmi_check")

    st.markdown("### 🧬 Medical History")
    col3, col4 = st.columns(2)
    with col3:
        hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1], key="hypertension_check")
        heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1], key="heart_check")
    with col4:
        hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5, key="hba1c_check")
        glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100, key="glucose_check")

    if st.button("Predict", key="predict_button"):
        gender_encoded = le_gender.transform([gender])[0]
        smoking_encoded = le_smoking.transform([smoking_history])[0]

        input_data = pd.DataFrame([{
            "gender": gender_encoded,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "smoking_history": smoking_encoded,
            "bmi": bmi,
            "HbA1c_level": hba1c,
            "blood_glucose_level": glucose
        }])

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.markdown("### 🧾 Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ The person is likely to have diabetes (Probability: {prob:.2f})")
        else:
            st.success(f"✅ The person is unlikely to have diabetes (Probability: {prob:.2f})")

# ------------------ PROFILE ------------------
elif page == "Your Profile":
    st.title("👤 Your Profile")
    name = st.text_input("Full Name", key="name_profile")
    age_profile = st.number_input("Age", min_value=0, max_value=120, key="age_profile")
    email = st.text_input("Email", key="email_profile")
    phone = st.text_input("Phone Number", key="phone_profile")
    st.write("📝 You can use this section to store your basic info for future appointments or reports.")

# ------------------ DOCTOR APPOINTMENT ------------------
elif page == "Doctor Appointment":
    st.title("🗓️ Book a Doctor Appointment")
    doc_type = st.selectbox("Specialist", ["Diabetologist", "General Physician", "Endocrinologist"], key="doc_type")
    preferred_date = st.date_input("Preferred Date", key="date_appointment")
    preferred_time = st.time_input("Preferred Time", key="time_appointment")
    symptoms = st.text_area("Describe your symptoms", key="symptoms_appointment")

    if st.button("Submit Appointment Request", key="submit_appointment"):
        st.success("✅ Appointment request submitted! You will be contacted soon.")

# ------------------ ABOUT US ------------------
elif page == "About Us":
    st.title("ℹ️ About Us")
    st.markdown("""
    Welcome to the Diabetes Prediction App!  
    Our mission is to empower individuals with early detection tools and reliable health insights.  
    This app was built using machine learning to help users assess their risk of diabetes based on clinical inputs.
    """)

# ------------------ DIABETES KNOWLEDGE ------------------
elif page == "Diabetes Knowledge":
    st.title("📚 Learn About Diabetes")
    st.markdown("""
    ### What is Diabetes?
    Diabetes is a chronic condition that affects how your body turns food into energy.  
    There are three main types:
    - **Type 1 Diabetes**: Autoimmune condition where the body attacks insulin-producing cells.
    - **Type 2 Diabetes**: Body becomes resistant to insulin or doesn’t produce enough.
    - **Gestational Diabetes**: Occurs during pregnancy and usually resolves after birth.

    ### Common Symptoms
    - Frequent urination  
    - Increased thirst  
    - Fatigue  
    - Blurred vision  
    - Slow healing wounds

    ### Prevention Tips
    - Maintain a healthy weight  
    - Exercise regularly  
    - Eat a balanced diet  
    - Avoid smoking and excessive alcohol  
    - Monitor blood sugar levels
    """)

# ------------------ AI CHATBOT ------------------
elif page == "AI Chatbot":
    st.title("💬 Ask Our AI Assistant")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("You:", key="chat_input")

    if user_input:
        response = get_bot_response(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

    for speaker, message in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {message}")

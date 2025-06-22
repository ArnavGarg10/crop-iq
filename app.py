import streamlit as st
import time
import requests
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tempfile
import os
from ultralytics import YOLO
from PIL import Image
import os
import gdown

@st.cache_data(show_spinner=False)
def download_video_from_drive(file_id: str, dest_path: str):
    if not os.path.exists(dest_path):
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, dest_path, quiet=False, fuzzy=True)
    return dest_path


hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)


st.set_page_config(
    page_title="CropIQ",
    page_icon="🌿",
    layout="wide", # Use "wide" layout for more space
    initial_sidebar_state="expanded" # Keep sidebar expanded by default
)

class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

model = load_model('new_model.h5', compile=False)

yolo = YOLO('best.pt')


def predict_image(img_path, model, class_names, target_size=(256, 256)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    preds = model.predict(img_array)
    pred_index = np.argmax(preds, axis=1)[0]
    pred_class = class_names[pred_index]
    confidence = preds[0][pred_index]

    status = "Healthy" if pred_class.lower().endswith("healthy") else "Diseased"

    return pred_class, confidence, status

def yolo_predict(image):
    class_names = {0: 'Flowering', 1: 'Germination', 2: 'Harvesting', 3: 'Vegetative'}
    result = yolo(image)[0]  # [0] to get the first (and only) result

    # Access boxes
    boxes = result.boxes

    # Extract details
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    coords = boxes.xyxy.cpu().numpy()

    class_names = {0: 'Flowering', 1: 'Germination', 2: 'Harvesting', 3: 'Vegetative'}

    # Example: get the first detection's class
    return class_names[class_ids[0]]


def rerun():
    st.rerun()

# --- Weather codes map ---
weather_code_map = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
    77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
    82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
}

# --- Weather forecast function ---
def get_weather_forecast():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 18.5944,
        "longitude": -72.3074,
        "daily": [
            "weathercode", "temperature_2m_min", "temperature_2m_max",
            "windspeed_10m_max", "precipitation_sum"
        ],
        "timezone": "auto"
    }

    res = requests.get(url, params=params)
    if res.status_code != 200:
        return None, "Failed to get weather data."
    
    data = res.json()
    forecast_list = []
    storm_warning = False

    for i, date in enumerate(data["daily"]["time"]):
        code = data["daily"]["weathercode"][i]
        description = weather_code_map.get(code, "Unknown")
        temp_min = data["daily"]["temperature_2m_min"][i]
        temp_max = data["daily"]["temperature_2m_max"][i]
        wind = data["daily"]["windspeed_10m_max"][i]
        rain = data["daily"]["precipitation_sum"][i]

        forecast_list.append({
            "date": date,
            "description": description,
            "temp_min": temp_min,
            "temp_max": temp_max,
            "wind": wind,
            "rain": rain,
            "storm": wind > 50 or rain > 20 or code in [82, 85, 95, 96, 99]
        })

        if forecast_list[-1]["storm"]:
            storm_warning = True

    return forecast_list, storm_warning

# --- GDACS feed function ---
def get_gdacs_alerts():
    url = "https://www.gdacs.org/xml/rss.xml"
    res = requests.get(url)
    if res.status_code != 200:
        return None, "Failed to get GDACS data."

    root = ET.fromstring(res.content)
    alerts = []

    for item in root.findall("./channel/item"):
        title = item.find("title").text or ""
        description = item.find("description").text or ""
        link = item.find("link").text or ""

        if "haiti" in title.lower() or "haiti" in description.lower():
            alerts.append({
                "title": title,
                "description": description,
                "link": link
            })

    return alerts, None

USER_CREDENTIALS = {
    "jaden_lakay": {"password": "farming", "role": "admin"},
    "jake54": {"password": "password", "role": "user"},
}

def user_videos_tab():
    st.header("Downloadable Videos")
    plant_care_id = "1UsXqwI9wzIxSFQ7RZIZpo9jh5Ssf_V2L"
    plant_care = download_video_from_drive(plant_care_id, "videos/plant_care.mp4")
    harvesting_id = "1z9bRAD9iROQmAN5mrll3nejuSPyhh0MM"  # replace with your ID
    harvesting = download_video_from_drive(harvesting_id, "videos/harvesting.mp4")
    pest_id = "1tHVdPmQeKR1XIyZypUjUnNLF1c_iDJa3"  # replace with your ID
    pest = download_video_from_drive(pest_id, "videos/pest.mp4")


    videos = {
        "Intro to Plant Care": plant_care,
        "Harvesting Tips": harvesting,
        "Pest Identification": pest
    }

    # Initialize watched videos list for user if not exists
    if "watched_videos" not in st.session_state:
        st.session_state["watched_videos"] = set()

    for title, path in videos.items():
        st.subheader(title)
        try:
            with open(path, "rb") as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)
                st.download_button(
                    label=f"Download '{title}'",
                    data=video_bytes,
                    file_name=path.split("/")[-1],
                    mime="video/mp4"
                )
                # Button to mark as watched
                if st.button(f"Mark '{title}' as watched", key=f"watch_{title}"):
                    st.session_state["watched_videos"].add(title)
                    st.success(f"Marked '{title}' as watched")
        except FileNotFoundError:
            st.error(f"Video file not found: {path}")
        st.markdown("---")


import streamlit as st

def login():
   st.markdown("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")


   col1, col2 = st.columns(2)


   with col1:
       st.write("")
       st.write("")
       st.write("")
       st.write("")
       st.write("")
       st.write("")
       st.write("")
       st.write("")
       st.write("")

       st.title("Welcome to CropIQ")
       st.markdown(
       """
       <div style="margin-right: 100px;">
       </div>
       """, unsafe_allow_html=True)


       username = st.text_input("Username")
       password = st.text_input("Password", type="password")
       if st.button("Login"):
           user = USER_CREDENTIALS.get(username)
           if user and password == user["password"]:
               # Initialize session_state values
               st.session_state["logged_in"] = True
               st.session_state["username"] = username
               st.session_state["role"] = user["role"]
               st.rerun()
           else:
               st.error("Invalid username/password")


   with col2:
       st.write("")
       st.write("")
       st.write("")
       st.write("")


       st.image("https://images.unsplash.com/photo-1501004318641-b39e6451bec6?auto=format&fit=crop&w=800&q=80", caption="Welcome to CropIQ 🌿", use_container_width=True)


def logout():
    for key in ["logged_in", "username", "role"]:
        if key in st.session_state:
            del st.session_state[key]
    rerun()

def admin_jobs_tab():
    st.header("Manage Job Postings (Admin)")
    st.write("""
        If a user applies for the job, you will receive an application under the applications tab 
        with their email and the amount of agricultural videos they watched to give you an idea 
        of their experience (and to avoid spam). If you would like to hire them, email them for 
        further stages such as an interview or to give them additional details.
    """)

    if "jobs" not in st.session_state:
        st.session_state["jobs"] = []

    new_job_title = st.text_input("New Job Title")
    new_job_description = st.text_area("Job Description")

    if st.button("Add Job"):
        if new_job_title and new_job_description:
            st.session_state["jobs"].append({
                "title": new_job_title,
                "description": new_job_description
            })
            st.success("Job added!")
        else:
            st.error("Please fill out both the job title and description.")

def user_jobs_tab():
    st.header("Available Jobs")
    st.write("You may apply for a job with your email and upload your resume.")

    jobs = st.session_state.get("jobs", [])

    # Email input (unique per user)
    email_key = f"email_{st.session_state['username']}"
    email = st.text_input("Enter your email", key=email_key)

    # Resume upload input (unique per user)
    resume_key = f"resume_{st.session_state['username']}"
    resume_file = st.file_uploader(
        "Upload your resume (PDF, DOC, DOCX)", 
        type=['pdf', 'doc', 'docx'], 
        key=resume_key
    )

    if not jobs:
        st.info("No jobs currently available.")
        return

    for i, job in enumerate(jobs):
        with st.expander(f"{job['title']}"):
            st.write(f"**Description:** {job['description']}")
            if st.button(f"Apply to '{job['title']}'", key=f"apply_{i}"):
                if not email:
                    st.error("Please enter your email before applying.")
                    continue
                if resume_file is None:
                    st.error("Please upload your resume before applying.")
                    continue

                if "applications" not in st.session_state:
                    st.session_state["applications"] = []

                watched_count = len(st.session_state.get("watched_videos", []))

                # Read the resume file bytes
                resume_bytes = resume_file.getvalue()

                # Store the application
                st.session_state["applications"].append({
                    "job": job['title'],
                    "applicant": st.session_state["username"],
                    "email": email,
                    "videos_watched": watched_count,
                    "resume_name": resume_file.name,
                    "resume_bytes": resume_bytes,
                    "resume_type": resume_file.type
                })
                st.success(f"Applied to '{job['title']}' with email {email}.")

def admin_applications_tab():
    import base64
    st.header("Job Applications (Admin)")
    applications = st.session_state.get("applications", [])
    
    if applications:
        for app in applications:
            st.write(f"Applicant: {app['applicant']} ({app['email']}) applied to {app['job']}")
            st.write(f"Videos watched: {app['videos_watched']}")
            
            # Provide a download link for the resume
            resume_name = app.get("resume_name", "resume")
            resume_bytes = app.get("resume_bytes")
            resume_type = app.get("resume_type", "application/octet-stream")
            
            if resume_bytes:
                b64 = base64.b64encode(resume_bytes).decode()
                href = f'<a href="data:{resume_type};base64,{b64}" download="{resume_name}">Download Resume</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.write("No resume uploaded.")
            
            st.markdown("---")
    else:
        st.write("No applications yet.")

def admin_settings_tab():
    st.header("Weather")
    st.subheader("🌤 7-Day Weather Forecast for Haiti")
    forecast, warning = get_weather_forecast()
    if forecast is None:
        st.error("Could not load weather forecast.")
    else:
            for day in forecast:
                st.markdown(f"**{day['date']}** – {day['description']}")
                st.write(f"🌡 Temp: {day['temp_min']}°C to {day['temp_max']}°C")
                st.write(f"🌬 Wind: {day['wind']} km/h, ☔ Rain: {day['rain']} mm")
                if day["storm"]:
                    st.warning("⚠️ Storm-like conditions!")
                st.markdown("---")
            if warning:
                st.error("⚠️ ALERT: Storm conditions expected in the week!")
            else:
                st.success("✅ No storm warnings.")

    st.subheader("🌍 GDACS Disaster Alerts")
    gdacs_alerts, error = get_gdacs_alerts()
    if gdacs_alerts:
            for alert in gdacs_alerts:
                st.error(f"🚨 {alert['title']}")
                st.write(alert["description"][:200] + "...")
                st.markdown(f"[Read more]({alert['link']})")
    elif error:
            st.error(error)
    else:
            st.success("✅ No current disaster alerts for Haiti.")

def admin_dashboard_tab():
    st.header("📷 Plant Health Analyzer")
    st.markdown("Upload a leaf image to analyze its health using AI-powered diagnostics.")

    col1, col2 = st.columns([1, 2])  # Left: Upload | Right: Output

    with col1:
        uploaded_file = st.file_uploader(
            "Upload an image of a plant leaf",
            type=["jpg", "jpeg", "png"]
        )

        run_analysis = False
        img_path = None

        if uploaded_file:
            # Save uploaded image temporarily
            temp_dir = tempfile.mkdtemp()
            img_path = os.path.join(temp_dir, uploaded_file.name)
            with open(img_path, "wb") as f:
                f.write(uploaded_file.read())

            run_analysis = st.button("🔍 Run Analysis")

            if run_analysis:
                with st.spinner("Analyzing..."):
                    predicted_class, confidence, health_status = predict_image(img_path, model, class_names)

                st.success("✅ Analysis Complete!")
                st.markdown(f"""
                **🧬 Prediction:** `{predicted_class}`  
                **🌿 Health Status:** `{health_status}`  
                **📊 Confidence:** `{confidence * 100:.2f}%`
                """)

    with col2:
        
        if uploaded_file:
                st.image(img_path, caption="Uploaded Image")

        else:
            st.info("Please upload a plant leaf image to get started.")

def admin_harvest_tab():
    st.header("🌾 Harvest Checker")

    uploaded_file = st.file_uploader("Upload an image of a plant", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Save uploaded image temporarily
        temp_dir = tempfile.mkdtemp()
        img_path = os.path.join(temp_dir, uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.read())

        st.image(img_path, caption="Uploaded Image", width = 100, use_container_width=False)

        if st.button("🔍 Harvest Check"):
            with st.spinner("Analyzing..."):
                stage = yolo_predict(img_path)

            st.success("✅ Analysis Complete!")
            st.subheader(f"""
            Predicted Stage: {stage}
            """)

def main():
    if "logged_in" not in st.session_state:
        login()
    else:
        role = st.session_state.get("role", None)

        if role == "admin":
            st.markdown(
            """
            <style>
            /* Make the tab labels bigger and with more padding */
            div[role="tablist"] > div[role="tab"] > button {
                font-size: 20px !important;
                padding: 12px 30px !important;
                font-weight: 700 !important;
            }
            </style>
            """, 
            unsafe_allow_html=True
            )
            st.title("CropIQ Portal")
            tabs = st.tabs(["Plant Health Analyzer", "Harvest Checker", "Jobs", "Applications", "Weather", "Settings"])
            with tabs[0]:
                admin_dashboard_tab()
            with tabs[1]:
                admin_harvest_tab()
            with tabs[2]:
                admin_jobs_tab()
            with tabs[3]:
                admin_applications_tab()
            with tabs[4]:
                admin_settings_tab()
            with tabs[5]:
                st.write(f"Logged in as: {st.session_state['username']}")
                if st.button("Logout"):
                    logout()
                role = st.session_state.get("role", None)
        else:
            tabs = st.tabs(["Videos", "Jobs", "Settings"])
            with tabs[0]:
                user_videos_tab()
            with tabs[1]:
                user_jobs_tab()
            with tabs[2]:
                st.write(f"Logged in as: {st.session_state['username']}")
                if st.button("Logout"):
                    logout()
                role = st.session_state.get("role", None)

if __name__ == "__main__":
    main()

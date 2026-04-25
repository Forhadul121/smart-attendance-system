import streamlit as st
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
import pytz

# --- 1. Page Configuration & Custom CSS ---
st.set_page_config(page_title="Smart Attendance", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #4F46E5; color: white; border: none; }
    .stButton>button:hover { background-color: #4338CA; border: none; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Firebase Connection ---
if not firebase_admin._apps:
    firebase_secrets = dict(st.secrets["firebase"])
    cred = credentials.Certificate(firebase_secrets)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# --- 3. Directory Setup & AI Model Loading ---
DATA_DIR = "student_faces"
os.makedirs(DATA_DIR, exist_ok=True)
BD_TIMEZONE = pytz.timezone('Asia/Dhaka') # Bangladesh Timezone

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# --- 4. Sidebar Navigation ---
st.sidebar.title("🛡️ Smart Access")
choice = st.sidebar.radio("Navigation", ["🏠 Home", "📸 Take Attendance", "👤 Register Student", "📊 Admin Dashboard"])

# --- 5. Home Page ---
if choice == "🏠 Home":
    st.title("📲 Smart Attendance System")
    
    # Date and Time Display
    now_bd = datetime.now(BD_TIMEZONE)
    st.markdown(f"**🇧🇩 Current Time:** `{now_bd.strftime('%I:%M %p')} | {now_bd.strftime('%d %b, %Y')}`")
    
    st.markdown("---")
    st.subheader("How to Use This System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("### 👤 Step 1: Registration\n"
                "**How to enroll a new student:**\n"
                "1. Go to **'Register Student'** from the left menu.\n"
                "2. Enter the student's **Full Name** and **ID**.\n"
                "3. Look directly at the webcam and capture a clear photo.\n"
                "4. Click **'Confirm Registration'** to save the profile.")
        
    with col2:
        st.success("### 📸 Step 2: Take Attendance\n"
                   "**How to log daily attendance:**\n"
                   "1. Click on **'Take Attendance'** from the menu.\n"
                   "2. Look straight into the camera when it opens.\n"
                   "3. Wait for the **green success message** confirming your identity.\n"
                   "4. Your attendance is automatically saved to the cloud database.")
    
    st.markdown("---")
    st.warning("📊 **For Admins:** Navigate to the **'Admin Dashboard'** to view today's complete attendance logs and export them as a CSV file.")

# --- 6. Student Registration ---
elif choice == "👤 Register Student":
    st.header("Student Enrollment")
    
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name")
    with col2:
        id_num = st.text_input("Student ID (Numbers only)")
        
    img_file = st.camera_input("Capture Enrollment Photo")
    
    if img_file and name and id_num.isdigit():
        if st.button("Confirm Registration"):
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    cv2.imwrite(f"{DATA_DIR}/{id_num}.{name}.jpg", face_roi)
                st.success(f"Profile created successfully for {name}!")
            else:
                st.error("Face not detected. Please make sure your face is clearly visible.")

# --- 7. Take Attendance ---
elif choice == "📸 Take Attendance":
    st.header("Attendance Scanner")
    
    faces, ids, name_map = [], [], {}
    for file in os.listdir(DATA_DIR):
        if file.endswith(".jpg"):
            parts = file.split(".")
            sid, sname = parts[0], parts[1]
            img = cv2.imread(f"{DATA_DIR}/{file}", cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            ids.append(int(sid))
            name_map[int(sid)] = sname
    
    if not faces:
        st.warning("No students registered yet! Please register a student first.")
    else:
        recognizer.train(faces, np.array(ids))
        img_file = st.camera_input("Scanning...")
        
        if img_file:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in found_faces:
                id_predicted, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                
                # Lower confidence means a better match in LBPH
                if confidence < 70:
                    student_name = name_map.get(id_predicted, "Unknown")
                    st.success(f"✅ Verified: {student_name}")
                    
                    # Fetching Bangladesh Time
                    now_bd = datetime.now(BD_TIMEZONE)
                    today_date = now_bd.strftime('%Y-%m-%d')
                    current_time = now_bd.strftime('%I:%M:%S %p') 
                    
                    # Save to Firebase
                    doc_id = f"{today_date}_{id_predicted}"
                    db.collection('attendance_logs').document(doc_id).set({
                        "ID": id_predicted,
                        "Name": student_name,
                        "Date": today_date,
                        "Time": current_time,
                        "Timestamp": now_bd 
                    })
                    st.toast(f"Attendance marked for {student_name}")
                else:
                    st.error("Face not recognized! Please try again or register.")

# --- 8. Admin Dashboard ---
elif choice == "📊 Admin Dashboard":
    st.header("Daily Attendance Records")
    
    now_bd = datetime.now(BD_TIMEZONE)
    today_date = now_bd.strftime('%Y-%m-%d')
    
    # Read from Firebase
    logs_ref = db.collection('attendance_logs').where("Date", "==", today_date).stream()
    records = [doc.to_dict() for doc in logs_ref]
    
    # Metric Cards
    col1, col2 = st.columns(2)
    col1.metric("Present Today", len(records))
    col2.metric("Date", now_bd.strftime('%d %b, %Y'))

    if records:
        df = pd.DataFrame(records)[["ID", "Name", "Time"]]
        st.dataframe(df, use_container_width=True)
        
        # Download Button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV Report", data=csv, file_name=f"attendance_{today_date}.csv", mime='text/csv')
    else:
        st.info("No attendance records found for today.")

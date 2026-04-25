import streamlit as st
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

# --- ১. পেজ কনফিগারেশন এবং স্টাইল ---
st.set_page_config(page_title="Universal Attendance", layout="wide")

# কাস্টম CSS স্টাইল
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #4F46E5; color: white; border: none; }
    .stButton>button:hover { background-color: #4338CA; border: none; }
    .css-1r6slb0 { border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; background: white; }
    </style>
    """, unsafe_allow_html=True)

# --- ২. ফায়ারবেস কানেকশন ---
if not firebase_admin._apps:
    firebase_secrets = dict(st.secrets["firebase"])
    cred = credentials.Certificate(firebase_secrets)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# --- ৩. ডিরেক্টরি এবং এআই মডেল লোড ---
DATA_DIR = "student_faces"
os.makedirs(DATA_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# --- ৪. সাইডবার ন্যাভিগেশন ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3589/3589030.png", width=100)
st.sidebar.title("Attendance System")
choice = st.sidebar.radio("Navigation", ["🏠 Home", "📸 Take Attendance", "👤 Register Student", "📊 Admin Dashboard"])

# --- ৫. হোম পেজ (নতুন সংযোজন) ---
if choice == "🏠 Home":
    st.title("📲 Smart Attendance System")
    st.info("Welcome! Select an option from the sidebar to begin.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### How it works?")
        st.write("1. **Register:** Add student face data.")
        st.write("2. **Scan:** Face ID marks attendance.")
        st.write("3. **Log:** Data is securely saved on Cloud.")
    with col2:
        # এখানে প্রজেক্টের একটি ডেমো ইমেজ বা আইকন দিতে পারেন
        st.image("https://cdn-icons-png.flaticon.com/512/2833/2833251.png", width=200)

# --- ৬. স্টুডেন্ট রেজিস্ট্রেশন ---
elif choice == "👤 Register Student":
    st.header("Student Enrollment")
    with st.container():
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
                    st.success(f"Profile created for {name}!")
                else:
                    st.error("Face not detected. Please look clearly at the camera.")

# --- ৭. হাজিরা গ্রহণ ---
elif choice == "📸 Take Attendance":
    st.header("Attendance Scanner")
    
    faces, ids, name_map = [], [], {}
    for file in os.listdir(DATA_DIR):
        if file.endswith(".jpg"):
            sid, sname = file.split(".")[0], file.split(".")[1]
            img = cv2.imread(f"{DATA_DIR}/{file}", cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            ids.append(int(sid))
            name_map[int(sid)] = sname
    
    if not faces:
        st.warning("No students registered yet!")
    else:
        recognizer.train(faces, np.array(ids))
        img_file = st.camera_input("Smile for the Camera!")
        
        if img_file:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in found_faces:
                id_predicted, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                
                if confidence < 70:
                    student_name = name_map.get(id_predicted, "Unknown")
                    st.toast(f"Welcome, {student_name}!")
                    st.success(f"✅ Verified: {student_name}")
                    
                    # Firebase Logging
                    today_date = datetime.now().strftime('%Y-%m-%d')
                    current_time = datetime.now().strftime("%H:%M:%S")
                    doc_id = f"{today_date}_{id_predicted}"
                    db.collection('attendance_logs').document(doc_id).set({
                        "ID": id_predicted, "Name": student_name, "Date": today_date, "Time": current_time
                    })
                else:
                    st.error("Face not recognized. Please register first.")

# --- ৮. অ্যাডমিন ড্যাশবোর্ড ---
elif choice == "📊 Admin Dashboard":
    st.header("Daily Attendance Insights")
    today_date = datetime.now().strftime('%Y-%m-%d')
    
    # Firebase থেকে ডেটা রিড
    logs_ref = db.collection('attendance_logs').where("Date", "==", today_date).stream()
    records = [doc.to_dict() for doc in logs_ref]
    
    # ম্যাট্রিক্স কার্ড (প্রফেশনাল লুক)
    m1, m2, m3 = st.columns(3)
    m1.metric("Attendance Today", len(records))
    m2.metric("Date", today_date)
    m3.metric("System Status", "Live")

    if records:
        df = pd.DataFrame(records)[["ID", "Name", "Time"]]
        
        tab1, tab2 = st.tabs(["📋 View Record", "📂 Export Data"])
        with tab1:
            st.dataframe(df, use_container_width=True)
        with tab2:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Attendance Report", data=csv, file_name=f"report_{today_date}.csv", mime='text/csv')
    else:
        st.info("No records found for today.")

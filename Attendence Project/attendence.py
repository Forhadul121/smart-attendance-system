import streamlit as st
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
import pytz  # বাংলাদেশি সময়ের জন্য এটি প্রয়োজন

# --- ১. পেজ কনফিগারেশন এবং স্টাইল ---
st.set_page_config(page_title="Universal Attendance", layout="wide")

# কাস্টম CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #4F46E5; color: white; border: none; }
    .stButton>button:hover { background-color: #4338CA; border: none; }
    </style>
    """, unsafe_allow_html=True)

# --- ২. ফায়ারবেস কানেকশন ---
if not firebase_admin._apps:
    firebase_secrets = dict(st.secrets["firebase"])
    cred = credentials.Certificate(firebase_secrets)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# --- ৩. সেটিংস এবং টাইমজোন ---
DATA_DIR = "student_faces"
os.makedirs(DATA_DIR, exist_ok=True)
BD_TIMEZONE = pytz.timezone('Asia/Dhaka') # বাংলাদেশ টাইমজোন

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# --- ৪. সাইডবার ---
st.sidebar.title("🛡️ Smart Access")
choice = st.sidebar.radio("Navigation", ["🏠 Home", "📸 Take Attendance", "👤 Register Student", "📊 Admin Dashboard"])

# --- ৫. হোম পেজ ---
if choice == "🏠 Home":
    st.title("📲 Smart Attendance System")
    now_bd = datetime.now(BD_TIMEZONE)
    st.write(f"### 🇧🇩 Current Time: {now_bd.strftime('%I:%M %p')} | {now_bd.strftime('%d %b, %Y')}")
    st.info("সঠিকভাবে হাজিরা দিতে বাম পাশের মেনু থেকে 'Take Attendance' সিলেক্ট করুন।")

# --- ৬. স্টুডেন্ট রেজিস্ট্রেশন ---
elif choice == "👤 Register Student":
    st.header("Student Enrollment")
    name = st.text_input("Full Name")
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
                st.error("Face not detected. Please try again.")

# --- ৭. হাজিরা গ্রহণ (এখানে টাইমজোন আপডেট করা হয়েছে) ---
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
        st.warning("No students registered yet!")
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
                
                if confidence < 70:
                    student_name = name_map.get(id_predicted, "Unknown")
                    st.success(f"✅ Verified: {student_name}")
                    
                    # বাংলাদেশি সময় নেওয়া
                    now_bd = datetime.now(BD_TIMEZONE)
                    today_date = now_bd.strftime('%Y-%m-%d')
                    current_time = now_bd.strftime('%I:%M:%S %p') # BST Format
                    
                    # Firebase এ সেভ
                    doc_id = f"{today_date}_{id_predicted}"
                    db.collection('attendance_logs').document(doc_id).set({
                        "ID": id_predicted,
                        "Name": student_name,
                        "Date": today_date,
                        "Time": current_time,
                        "Timestamp": now_bd # কুয়েরি করার সুবিধার জন্য
                    })
                    st.toast(f"Attendance marked for {student_name}")
                else:
                    st.error("Face not recognized!")

# --- ৮. অ্যাডমিন ড্যাশবোর্ড ---
elif choice == "📊 Admin Dashboard":
    st.header("Daily Attendance Records")
    now_bd = datetime.now(BD_TIMEZONE)
    today_date = now_bd.strftime('%Y-%m-%d')
    
    logs_ref = db.collection('attendance_logs').where("Date", "==", today_date).stream()
    records = [doc.to_dict() for doc in logs_ref]
    
    col1, col2 = st.columns(2)
    col1.metric("Present Today", len(records))
    col2.metric("Date", now_bd.strftime('%d-%m-%Y'))

    if records:
        df = pd.DataFrame(records)[["ID", "Name", "Time"]]
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name=f"attendance_{today_date}.csv")
    else:
        st.info("হাজিরা রেকর্ড পাওয়া যায়নি।")

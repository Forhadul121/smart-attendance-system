import streamlit as st
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

# 1. Page Config Setup
st.set_page_config(page_title="Universal Attendance", layout="centered")

# 2. Connect to Firebase securely
if not firebase_admin._apps:
    firebase_secrets = dict(st.secrets["firebase"])
    cred = credentials.Certificate(firebase_secrets)
    firebase_admin.initialize_app(cred)

# Create the database connection variable
db = firestore.client()

# 3. Setup directories
DATA_DIR = "student_faces"
os.makedirs(DATA_DIR, exist_ok=True)

# 4. Load the Face Detector and Recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

st.title("📲 Smart Attendance")

# 5. Sidebar Menu
menu = ["Take Attendance", "Register Student", "Admin Dashboard"]
choice = st.sidebar.selectbox("Navigation", menu)

# --- MODE: Register Student ---
if choice == "Register Student":
    st.subheader("Student Enrollment")
    name = st.text_input("Enter Name")
    id_num = st.text_input("Enter ID (Numbers only)")
    
    img_file = st.camera_input("Capture Enrollment Photo")
    
    if img_file and name and id_num.isdigit():
        if st.button("Save to Database"):
            # Process image
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect face and save crop
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    # Save as ID.Name.jpg locally for training
                    cv2.imwrite(f"{DATA_DIR}/{id_num}.{name}.jpg", face_roi)
                st.success(f"Profile created for {name}!")
            else:
                st.error("No face detected! Please try again.")

# --- MODE: Take Attendance ---
elif choice == "Take Attendance":
    st.subheader("Attendance Scanner")
    
    # Train the recognizer on saved images
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
        
        # Capture Live Image
        img_file = st.camera_input("Scan for Attendance")
        if img_file:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            found_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in found_faces:
                id_predicted, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                
                if confidence < 70:
                    student_name = name_map.get(id_predicted, "Unknown")
                    st.success(f"✅ Present: {student_name} (Match: {round(100 - confidence)}%)")
                    
                    # LOG TO FIREBASE FIRESTORE
                    today_date = datetime.now().strftime('%Y-%m-%d')
                    current_time = datetime.now().strftime("%H:%M:%S")
                    
                    # Create a unique document ID for this student today
                    doc_id = f"{today_date}_{id_predicted}"
                    
                    doc_ref = db.collection('attendance_logs').document(doc_id)
                    doc_ref.set({
                        "ID": id_predicted,
                        "Name": student_name,
                        "Date": today_date,
                        "Time": current_time
                    })
                else:
                    st.error("Unknown Face Detected.")

# --- MODE: Admin Dashboard ---
elif choice == "Admin Dashboard":
    st.subheader("Attendance Records (Firebase)")
    today_date = datetime.now().strftime('%Y-%m-%d')
    
    # READ FROM FIREBASE FIRESTORE
    logs_ref = db.collection('attendance_logs').where("Date", "==", today_date).stream()
    
    records = []
    for doc in logs_ref:
        records.append(doc.to_dict())
        
    if records:
        df = pd.DataFrame(records)
        # Reorder columns to look nice
        df = df[["ID", "Name", "Date", "Time"]]
        st.table(df)
        
        # Keep the download functionality
        st.download_button("Download CSV", df.to_csv(index=False), f"report_{today_date}.csv", "text/csv")
    else:
        st.info("No one has marked attendance today.")

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Connect to Firebase securely using Streamlit Secrets
if not firebase_admin._apps:
    # Convert Streamlit secrets to a dictionary that Firebase can read
    firebase_secrets = dict(st.secrets["firebase"])
    cred = credentials.Certificate(firebase_secrets)
    firebase_admin.initialize_app(cred)

# Create the database connection variable
db = firestore.client()
import streamlit as st  # type: ignore
import cv2  # type: ignore
import numpy as np  # type: ignore
import os
import pandas as pd  # type: ignore
from datetime import datetime

st.set_page_config(page_title="Python Attendance System")

# Setup directories
DATA_DIR = "student_faces"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Load the Face Detector (Built into OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Use LBPH Face Recognizer (Doesn't need dlib/CMake)
recognizer = cv2.face.LBPHFaceRecognizer_create()

st.set_page_config(page_title="Universal Attendance", layout="centered")
st.title("📲 Smart Attendance")

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
                    # Save as ID.Name.jpg
                    cv2.imwrite(f"{DATA_DIR}/{id_num}.{name}.jpg", face_roi)
                st.success(f"Profile created for {name}!")
            else:
                st.error("No face detected! Please try again.")

# --- MODE: Take Attendance ---
elif choice == "Take Attendance":
    st.subheader("Attendance Scanner")
    
    # 1. Train the recognizer on saved images
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
        
        # 2. Capture Live Image
        img_file = st.camera_input("Scan for Attendance")
        if img_file:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            found_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in found_faces:
                id_predicted, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                
                # Lower confidence score is better in LBPH
                if confidence < 70:
                    student_name = name_map[id_predicted]
                    st.success(f"✅ Present: {student_name} (Match: {round(100 - confidence)}%)")
                    
                    # Log to CSV
                    log_file = f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"
                    log_data = pd.DataFrame([{"ID": id_predicted, "Name": student_name, "Time": datetime.now().strftime("%H:%M:%S")}])
                    log_data.to_csv(log_file, mode='a', index=False, header=not os.path.exists(log_file))
                else:
                    st.error("Unknown Face Detected.")

# --- MODE: Admin Dashboard ---
elif choice == "Admin Dashboard":
    st.subheader("Attendance Records")
    log_file = f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"
    if os.path.exists(log_file):
        df = pd.read_csv(log_file).drop_duplicates(subset=['ID'], keep='first')
        st.table(df)
        st.download_button("Download CSV", df.to_csv(index=False), "report.csv", "text/csv")
    else:
        st.info("No one has marked attendance today.")

import cv2
import pandas as pd
import face_recognition
import streamlit as st
import numpy as np
import base64
from datetime import datetime
import os

# Load the Viola-Jones face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Load images and student names
images = [
    {"name": "Diab", "image_path": "images/Diab.jfif"},
    {"name": "Mostafa Hagag", "image_path": "images/Mostafa Hagag.jfif"},
    {"name": "Ahmed Sheaba", "image_path": "images/Ahmed Sheaba.jfif"},
    {"name": "Dr.Amany Sarhan", "image_path": "images/Dr.Amany.jpg"},
    {"name": "Basel Darwish", "image_path": "images/Basel.jpg"},
    {"name": "Hussein El-Sabagh", "image_path": "images/Hussien.jpg"},
]

attendance_file = "attendance_log.csv"

if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Name", "Time", "Present"]).to_csv(attendance_file, index=False)

known_faces = []
known_names = []

for student in images:
    image = face_recognition.load_image_file(student["image_path"])
    face_encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(face_encoding)
    known_names.append(student["name"])

def generate_excel_file():
    try:
        attendance_df = pd.read_csv(attendance_file)
    except pd.errors.EmptyDataError:
        attendance_df = pd.DataFrame(columns=["Name", "Time", "Present"])

    attendance_df.to_csv(attendance_file, index=False)

def get_recognized_names(result_image, temp_path):
    uploaded_image = face_recognition.load_image_file(temp_path)
    face_locations = face_recognition.face_locations(uploaded_image)
    face_encodings = face_recognition.face_encodings(uploaded_image, face_locations)

    recognized_names = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        if True in matches:
            name = known_names[matches.index(True)]

        recognized_names.append(name)

    return recognized_names

def log_attendance(recognized_names):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        attendance_df = pd.read_csv(attendance_file)
    except pd.errors.EmptyDataError:
        attendance_df = pd.DataFrame(columns=["Name", "Time", "Present"])

    for name in recognized_names:
        attendance_df = attendance_df.append({"Name": name, "Time": current_time, "Present": 1}, ignore_index=True)

    attendance_df.to_csv(attendance_file, index=False)

def recognize_faces_in_image(image_path):
    uploaded_image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(uploaded_image)
    face_encodings = face_recognition.face_encodings(uploaded_image, face_locations)

    result_image = cv2.imread(image_path)

    total_faces = 0
    correct_recognitions = 0

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        if True in matches:
            name = known_names[matches.index(True)]
            correct_recognitions += 1

        cv2.rectangle(result_image, (left, top), (right, bottom), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(result_image, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        total_faces += 1

    accuracy = correct_recognitions / total_faces if total_faces > 0 else 0
    formatted_accuracy = "{:.2%}".format(accuracy)
    st.write(f"Accuracy: {formatted_accuracy}")

    return result_image, accuracy

def main():
    st.title("Face Recognition Attendance System")

    menu = ["Home", "Upload", "Capture", "Download Attendance"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.write("Welcome to the Face Recognition Attendance System!")

    elif choice == "Upload":
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose a file")
        
        if uploaded_file is not None:
            temp_path = 'temp_image.jpg'
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            result_image, accuracy = recognize_faces_in_image(temp_path)
            recognized_names = get_recognized_names(result_image, temp_path)
            log_attendance(recognized_names)

            result_image_base64 = base64.b64encode(cv2.imencode('.jpg', result_image)[1]).decode()
            st.image(result_image, caption=f"Recognition Result - Accuracy: {accuracy}", use_column_width=True)
            st.write(f"Recognized Names: {', '.join(recognized_names)}")

    elif choice == "Capture":
        st.subheader("Capture Image")
        st.write("Click the button below to capture an image.")
        
        # Button to trigger image capture
        if st.button("Capture"):
            # Open a connection to the default camera (camera index 0)
            cap = cv2.VideoCapture(0)

            # Capture a single frame
            ret, frame = cap.read()

            # Release the camera
            cap.release()

            # Save the captured frame to a temporary location
            temp_path = 'temp_image.jpg'
            cv2.imwrite(temp_path, frame)

            # Call your existing face recognition function
            result_image, accuracy = recognize_faces_in_image(temp_path)

            # Get the recognized names
            recognized_names = get_recognized_names(result_image, temp_path)

            # Log the attendance
            log_attendance(recognized_names)

            # Convert result image to base64-encoded string
            result_image_base64 = base64.b64encode(cv2.imencode('.jpg', result_image)[1]).decode()

            # Display the result and accuracy
            st.image(result_image, caption=f"Recognition Result - Accuracy: {accuracy}", use_column_width=True)
            st.write(f"Recognized Names: {', '.join(recognized_names)}")

    elif choice == "Download Attendance":
        st.subheader("Download Attendance")
        if st.button("Download"):
            generate_excel_file()
            st.write("Attendance log downloaded successfully!")

if __name__ == '__main__':
    generate_excel_file()  # Ensure the Excel file is created before running the Streamlit app
    main()

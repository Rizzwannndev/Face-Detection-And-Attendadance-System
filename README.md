# Face Recognition Attendance System

## Overview
This project is a Face Recognition Attendance System that detects faces, recognizes them using K-Nearest Neighbors (KNN), and marks attendance in a CSV file. Additionally, it includes a web application built with Streamlit to display the attendance records.

## Features
- Face detection and recognition using OpenCV.
- Attendance marking in a CSV file.
- Audio confirmation of attendance.
- A Streamlit web app to display attendance records.

## Installation
### Prerequisites
Make sure you have Python installed along with the required dependencies.

### Install Dependencies
```sh
pip install opencv-python numpy scikit-learn pandas streamlit pyttsx3
```

## Project Structure
```
.
├── data/
│   ├── haarcascade_frontalface_default.xml
│   ├── names.pkl
│   ├── facesData.pkl
├── Attendance/
│   ├── Attendance_<date>.csv
├── image.png
├── main.py
├── marking_attendance.py
├── webapp.py
├── README.md
```

## How to Run
### Step 1: Collect Face Data
Run `main.py` to collect face data for a new user.
```sh
python main.py
```
It will prompt for the user’s name and capture 100 face images.

### Step 2: Start Face Recognition and Attendance Marking
Run `marking_attendance.py` to start recognizing faces and marking attendance.
```sh
python marking_attendance.py
```
Press 'w' to mark attendance and 'q' to quit.

### Step 3: View Attendance in Web App
Run `webapp.py` to launch the Streamlit web app.
```sh
streamlit run webapp.py
```

## Code Files
### `main.py`
This script captures and stores face data.

### `marking_attendance.py`
Recognizes faces and marks attendance in a CSV file.

### `webapp.py`
Displays the attendance records in a Streamlit web app and refreshes automatically every 3 seconds.

## Acknowledgments
- OpenCV for face detection
- Scikit-learn for KNN classification
- Streamlit for the web interface

## License
This project is open-source. Feel free to modify and use it for educational purposes.


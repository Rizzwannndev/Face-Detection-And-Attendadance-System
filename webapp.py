import streamlit as st
import pandas as pd
import time 
from datetime import datetime

t = time.time()
date = datetime.fromtimestamp(t).strftime("%d-%m-%Y")
timeStamp = datetime.fromtimestamp(t).strftime("%H-%M-%S")

dataFrame = pd.read_csv("Attendance/Attendance_" + date + ".csv")

st.title("📋 Attendance Monitoring")
st.dataframe(dataFrame.style.highlight_max(axis = 0))

time.sleep(3)
st.rerun()
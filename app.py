import streamlit as st
import pandas as pd
import time
from datetime import datetime

ts=time.time()
date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")
df = pd.read_csv("/home/tandinomu/Desktop/camera/face_recognition_project/Attendance/Attendance_" + date + ".csv")

st.dataframe(df.style.highlight_max(axis=0))
import streamlit as st
import pandas as pd
import os
from datetime import datetime

# Title for the Streamlit app
st.title("Face Recognition Attendance System")

# Get today's date for the attendance file
ts = datetime.now().strftime("%d-%m-%Y")
attendance_file = f"data/Attendance_{ts}.csv"

# Display the constructed file path for debugging
st.write("Looking for file:", attendance_file)

# Check if the attendance file exists and display it
if os.path.exists(attendance_file):
    df = pd.read_csv(attendance_file)
    if df.empty:
        st.write("The attendance file is empty.")
    else:
        st.dataframe(df)
else:
    st.write("No attendance recorded for today.")
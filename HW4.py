pip install seaborn
import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

st.markdown("# Car crashes Dataset Exploration")
col1, col2 = st.columns(2)

with col1:
    st.header("Introduction")
    st.write("This is a dataset having crash frequency and other fctors affecting the crash frequency for different states in USA.") 
col2.markdown("## Let's see our dataset")

dataset = col2.file_uploader("Upload a dataset")
# col2.success("Dataset uploaded successfully!")

# importing car crashes dataset from seaborn
df_crash = sns.load_dataset("car_crashes")
col2.dataframe(df_crash)

with col1:
    x_value = col1.selectbox("Select a variable for x-axis of the plot", df_crash.columns)
    y_value = col1.selectbox("Select a variable for y-axis of the plot (note: total is crash frequency)", df_crash.columns)

fig = sns.lmplot(data=df_crash,x= x_value, y= y_value,height=4)
st.pyplot(fig)


with st.expander("More info"):
    st.write("""Car crashes dataset reveals the number of crashes for different states and the factors that affect the crash frequency.""")











import streamlit as st
import seaborn as sns
import pandas as pd


@st.cache_data
def load_data():
    df = pd.read_csv("D:\Ph.D\Semester-III\CMSE-830\In-class Assignments\Cancer_data.csv")
    return df

df = load_data()

# Adding a title and description
st.title("""
# Wisconsin Breast Cancer Dataset
""")

x_value = st.selectbox("Select a variable for x axis of the plot", df.columns)
st.write('The current x-axis is', x_value)

y_value = st.selectbox("Select a variable for y axis of the plot", df.columns)
st.write('The current y-axis is', y_value)

# Creating a 3D scatter plot using Plotly
fig = sns.relplot(x=x_value, y=y_value, data = df)

# Showing the Plotly figure using Streamlit
st.pyplot(fig)

# updating text information about the Iris dataset
st.write("""
This is the relation plot for the selected variables.
""")

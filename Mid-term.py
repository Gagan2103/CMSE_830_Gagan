import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

st.sidebar.title("Navigation")
df_crash = sns.load_dataset("car_crashes")
option = st.sidebar.radio("Select an option", ["Data Exploration", "Data Visualization", "Data Analysis"])

#st.balloons()

if option == "Data Exploration":
    st.markdown("# Car Crashes Dataset Exploration")

    st.header("Introduction")
    st.write("This is a dataset having crash frequency and other factors affecting the crash frequency for different states in USA.") 

    # importing car crashes dataset from seaborn
    
    st.subheader("Do you want to see the dataset?")
    data_show = st.selectbox("", ["No", "Yes"])
    if data_show == "Yes":
        st.dataframe(df_crash)

    st.markdown("#### Dataset Description")
    st.write("There are 51 rows in the datset, one for each of the states in the USA, and 8 columns in the dataset.")
    selected_options = st.multiselect("Select column names to see their brief description", df_crash.columns)
    description = {"total": "Number of drivers that are involved in fatal crashes per billion vehicle miles traveled", 
               "speeding": "Percentage of drivers involved in fatal crashes and were involved in speeding", 
               "alcohol": "Percentage of drivers involved in fatal crashes and were impaired due to alcohol use while driving", 
               "not_distracted": "Percentage of drivers involved in fatal crashes and were not involved in distracted driving. This could be the texting, calling, etc.", 
               "no_previous": "Percentage of drivers involved in fatal crashes and had not involved in any previous crash/accident", 
               "ins_premium": "Car insurance premiums", 
               "ins_losses": "Losses incurred by insurance companies for crashes per insured drivers.", 
               "abbrev": "Abbreviation for states in the USA"}

    for option in selected_options:
        st.write(f"**Description of {option}:** {description[option]}")






if option == "Data Visualization":
    st.markdown("# Car crashes Dataset Visualization")
    # Visualization
    col1, col2 = st.columns(2)


    x_value = col1.selectbox("Select a variable for x-axis of the plot", df_crash.columns)
    y_value = col2.selectbox("Select a variable for y-axis of the plot", df_crash.columns)

    fig = sns.lmplot(data=df_crash,x= x_value, y= y_value,height=4)
    st.pyplot(fig)



    with st.expander("More info"):
        st.write("""Car crashes dataset reveals the number of crashes for different states and the factors that affect the crash frequency. 
                It also give relation between the increase in cost incurred to insurance companies and the total number of fatalities.""")


if option == "Data Analysis":
    st.markdown("# Car crashes Dataset Analysis")








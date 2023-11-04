import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
import numpy as np
from scipy.stats import zscore
import statsmodels.api as sm
import hiplot as hip

st.sidebar.title("Navigation")
df_crash = sns.load_dataset("car_crashes")

state_names_dict = {'AL': 'Alabama','AK': 'Alaska','AZ': 'Arizona','AR': 'Arkansas','CA': 'California', 'DC':'District of Columbia',
    'CO': 'Colorado','CT': 'Connecticut','DE': 'Delaware','FL': 'Florida','GA': 'Georgia','HI': 'Hawaii',
    'ID': 'Idaho','IL': 'Illinois','IN': 'Indiana','IA': 'Iowa','KS': 'Kansas','KY': 'Kentucky','LA': 'Louisiana',
    'ME': 'Maine','MD': 'Maryland','MA': 'Massachusetts','MI': 'Michigan','MN': 'Minnesota','MS': 'Mississippi',
    'MO': 'Missouri','MT': 'Montana','NE': 'Nebraska','NV': 'Nevada','NH': 'New Hampshire','NJ': 'New Jersey',
    'NM': 'New Mexico','NY': 'New York','NC': 'North Carolina','ND': 'North Dakota','OH': 'Ohio','OK': 'Oklahoma','OR': 'Oregon',
    'PA': 'Pennsylvania','RI': 'Rhode Island','SC': 'South Carolina','SD': 'South Dakota','TN': 'Tennessee','TX': 'Texas',
    'UT': 'Utah','VT': 'Vermont','VA': 'Virginia','WA': 'Washington','WV': 'West Virginia','WI': 'Wisconsin','WY': 'Wyoming'
}


df_crash['State_Name'] = df_crash['abbrev'].map(state_names_dict)
df_crash = df_crash.drop("abbrev", axis = 1)
df_columns = df_crash.columns


option = st.sidebar.radio("Select an option", ["Data Exploration", "Data Visualization", "Data Analysis"])


if option == "Data Exploration":
    st.markdown("# Car Crashes Dataset Exploration")

    st.header("Introduction")
    st.write(
        '''This is a dataset having crash frequency and other factors affecting the crash frequency for different states in USA.
        \n The goal of this project is to have a comparison analysis for factors affecting the number of fatal crashes across different
        states in the USA. It is of interest to road agencies to know the cause of fatal crashes because fatal crashes are those 
        crashes that lead to the death of one or more road users (drivers, pedestrians, bicyclists, etc.). It is also of interest 
        to road users, as it can help develop guidelines for road users to follow to have a safe and efficient transportation 
        system across the USA. 
        \nThis idea is important because millions of people die every year in road crashes around the world 
        and thousands of people in the USA.'''
        ) 

    # importing car crashes dataset from seaborn
    
    data_check = st.checkbox("**Show the dataset!**")
    if data_check:
        st.write(df_crash)

    st.markdown("#### Dataset Description")
    st.write(
        '''This dataset deals with the fatal crashes per billion vehicle miles traveled (VMT), their cause, 
        and the cost to the insurance companies. 
        \n This dataset have data for the total number of crashes per billion VMT, 
        different causes of these crashes, such as over-speeding, involving impaired (alcohol) driving, and driving with /without distraction. 
        This dataset also has data for driver percentage involved in fatal crashes for the first time in their lifetime,
        as well as the resulting insurance premiums as a result of crashes. Finally, dataset also have information about the losses 
        incurred by insurance companies as a result of crashes per insured driver. There is no issue of missingness in the dataset.
        \nThere are 51 rows in the datset, one for each of the states in the USA, and 8 columns in the dataset.'''
        )
    
    selected_options = st.multiselect("**Let's see the description of different columns present in the dataset. Select column names to see their brief description**", df_columns)
    description = {"total": "Number of drivers that are involved in fatal crashes per billion vehicle miles traveled", 
               "speeding": "Percentage of drivers involved in fatal crashes and were involved in speeding", 
               "alcohol": "Percentage of drivers involved in fatal crashes and were impaired due to alcohol use while driving", 
               "not_distracted": "Percentage of drivers involved in fatal crashes and were not involved in distracted driving. This could be the texting, calling, etc.", 
               "no_previous": "Percentage of drivers involved in fatal crashes and had not involved in any previous crash/accident", 
               "ins_premium": "Car insurance premiums", 
               "ins_losses": "Losses incurred by insurance companies for crashes per insured drivers.", 
               "State_Name": "Name of the states in the USA"}

    for option in selected_options:
        st.write(f"**Description of {option}:** {description[option]}")
    st.markdown("#### Descriptive statistics")
    st.write('''Descriptive statistics of a variable include the count, mean, standard deviation, minimum, maximum and various quartile values.''')
    datades_show = st.selectbox("**Would you like to see the descriptive statistics of the dataset?**", ["No", "Yes"])
    if datades_show == "Yes":

        selected_var = st.multiselect("**Select column names to see their descriptive statistics.**", df_columns.drop("State_Name"))
        
        st.write(df_crash.describe()[selected_var])

    with st.expander("More info"):
        st.write("""Car crashes dataset reveals the number of crashes for different states and the factors that affect the crash frequency. 
                It also give relation between the increase in cost incurred to insurance companies and the total number of fatalities.""")



if option == "Data Visualization":
    st.markdown("# Car Crashes Dataset Visualization")
    st.write("This section of the app deals with the visualization of data. Let's see what data tells about it.")
    st.write(
        '''The section provides an interactive interface for people who are interested
        to know about the number of fatal crashes and the reason behind them.'''
        )
    

    # Visualization
    graph = st.selectbox("Select the type of visualization you are interested", ['None', 'HiPlot', 'Bar plot', 'Multiple bar plot', 'Regression plot', 'Interactive Scatter plot'])
    def save_hiplot_to_html(exp):
        output_file = "hiplot_plot_1.html"
        exp.to_html(output_file)
        return output_file
    
    if graph == "HiPlot":
        st.write('This plot allows user to select required columns and visualize them using HiPlot.')
        selected_columns = st.multiselect("Select columns to visualize", df_columns)
        selected_data = df_crash[selected_columns]
        if not selected_data.empty:
            experiment = hip.Experiment.from_dataframe(selected_data)
            hiplot_html_file = save_hiplot_to_html(experiment)
            st.components.v1.html(open(hiplot_html_file, 'r').read(), height=1500, scrolling=True)
        else:
            st.write("No data selected. Please choose at least one column to visualize.")

    elif graph == "Bar plot":
        st.write("Bar plot is used to see the trend of different variables across different states in the USA.")
        y_value = df_crash['State_Name']
        x_value = st.selectbox("Select a variable for x-axis of the plot", df_columns.drop("State_Name"))
        abbrev_order = df_crash.sort_values(x_value, ascending=False)["State_Name"].tolist()
        plt.figure(figsize=(6, 12))
        bar_fig = sns.barplot(x = x_value, y = y_value, data=df_crash, label = x_value, color="g", order=abbrev_order)
        st.pyplot(bar_fig.get_figure())
        with st.expander("Key points"):
            st.write('''The key takeaway here is the states with the highest and lowest values of the selected variable. 
                    **For instance**, in the case of total crashes, South Carolina ranks highest, while the District of Columbia ranks lowest. 
                    Taking speeding as another example, Hawaii records the highest number of speeding cases, while New Jersey has the fewest.''')

        
    elif graph == "Multiple bar plot":
        st.write("Multiple bar plot is used to see the trend of different causes of crashes out of total crashes values across different states in the USA.")
        y_value = df_crash['State_Name']
        x_value = st.selectbox("Select a variable to overlap the barplot over total number of crashes.", df_columns.drop(["State_Name", 'total', "ins_losses", "ins_premium"]))
        abbrev_order = df_crash.sort_values('total', ascending=False)["State_Name"].tolist()
        plt.figure(figsize=(6, 12))
        sns.set_color_codes("pastel")
        fig = sns.barplot(x='total', y=y_value, data=df_crash, color="lightblue", order=abbrev_order)
        sns.set_color_codes("muted")
        fig = sns.barplot(x=x_value, y=y_value, data=df_crash, color="blue", order=abbrev_order)
        # Create a legend
        pastel_patch = plt.Rectangle((0, 0), 1, 1, fc="lightblue", edgecolor='none')
        muted_patch = plt.Rectangle((0, 0), 1, 1, fc="blue", edgecolor='none')

        plt.legend([pastel_patch, muted_patch], ['Total Crashes', x_value], loc="upper right")
        plt.xlabel("Value")
        plt.ylabel("State")
        plt.title("Bar plot for total crashes with "+ x_value + " related crashes")
        st.pyplot(fig.get_figure())
        with st.expander("Key points"):
            st.write('''Here, we can observe the trend of the total number of crashes across different states in the USA. 
                     Additionally, we can explore the impact of various other factors on crashes across the nation. 
                     **For instance**, in the case of speeding-related crashes, we notice that the total number of crashes is highest in South Carolina, 
                     followed by North Dakota and West Virginia. Conversely, the highest percentage of drivers involved in overspeeding can be found in Hawaii, 
                     with South Carolina and Pennsylvania following closely.''')

    
    elif graph == "Regression plot":
        st.write("Regression plot is used to see the relation and trend of one variable with the other variable.")
        col1, col2 = st.columns(2)
        x_value = col1.selectbox("Select a variable for x-axis of the plot", df_columns)
        y_value = col2.selectbox("Select a variable for y-axis of the plot", df_columns)
        reg_plot = sns.regplot(data=df_crash,x = x_value, y = y_value, lowess=False)
        st.pyplot(reg_plot.get_figure())
        with st.expander("Key points"):
            st.write('''When we use plots to explore the relationship between selected variables of interest, 
                    we are essentially visualizing how these variables interact with each other. 
                    By doing so, we can gain insights into patterns, trends, and dependencies that might not be immediately apparent when examining the data in tabular form.
                    **For instance**, let's consider the relationship between speeding (or alcohol) and the total number of crashes. 
                    We create a plot that allows us to analyze how changes in one variable affect the other. In this specific case, the plot reveals that 
                    there is a positive correlation between the percentage of drivers who are speeding (or impaired with alcohol) and the total number of crashes. 
                    In other words, as the percentage of drivers speeding (or impaired with alcohol) increases, the total number of crashes also tends to increase.
                    This visualization not only demonstrates the relationship but also helps us draw conclusions about causality or correlations that might exist in the dataset.''')
    
            
    elif graph == "Interactive Scatter plot":
        st.write("Scatter plot is used to see the variation of one variable with the other variable.")
        col1, col2 = st.columns(2)
        x_value = col1.selectbox("Select a variable for x-axis of the plot", df_columns)
        y_value = col2.selectbox("Select a variable for y-axis of the plot", df_columns)

        alt_fig = alt.Chart(df_crash).mark_point().encode(
            x = alt.X(x_value, axis=alt.Axis(title=x_value, labelFontWeight='bold')), 
            y = alt.Y(y_value, axis=alt.Axis(title=y_value, labelFontWeight='bold')), 
            tooltip=[x_value, y_value]
        ).properties(width=500, height=400,).interactive()
        st.altair_chart(alt_fig)
        with st.expander("Key points"):
            st.write('''Scatter plots are useful for identifying outliers and clusters, providing insights into how variables are related. 
                    They are a valuable tool in data analysis for making data-driven decisions and understanding trends.
                    It can help to see how changes in one variable (e.g., the percentage of drivers speeding or impaired with alcohol) affect another variable 
                    (e.g., the total number of crashes) for different states.''')



if option == "Data Analysis":
    st.markdown("# Car Crashes Dataset Analysis")
    st.write("This section of the app deals with the analysis of data. Before proceeding with the modelling, first let's see the correlation between different variables.")
    st.subheader("Correlation Matrix Heatmap")
    st.write('''Before diving into the dataset, let's see if different variables in the data are correlated to each other or not.
             You can hover on each cell to know the correlation value. 
             \n The positive value of correlation shows that with the increase in one variable, 
             other variable also increases, and vice versa. Correlation value close to zero means that the two variables are not having any trend between them. ''')
    
    check = st.checkbox("**Show Correlation Matrix Heatmap**")
    if check:
        fig_heat = px.imshow(df_crash.corr(), color_continuous_scale='RdBu_r')

        fig_heat.update_layout(
            width=600, height=500,
        )
        st.plotly_chart(fig_heat)

    st.write('''As, the variables are highly correlated with each other, they will have effect on each other in a model. 
            So, we will use simple linear-regression model to avoid the interaction between the variables.''')
    st.subheader("Regression Model")
    st.write('''**Simple linear regression (SLR) model** is a statistical method used to analyze the relationship between an independent variable
            (also known as predictor or explanatory variable) and a dependent variable (also known as the response variable). 
            \n In a SLR model, the goal is to create a linear equation that can predict the dependent variable based on the value of the independent variable. 
            \n The general form of the simple linear regression model is:''')
    st.write(''' $$Y =  \Beta_0 + \Beta_1X_1 + \epsilon $$, ''')
    st.write('''where, 
             \n $$Y$$ is the dependent variable (total number of crashes in our model),''')
    st.write('''$$\Beta_0$$ is the intercept or constant term,''')
    st.write('''$$\Beta_1$$ is the coefficients of the independent variable $$(X_1)$$.''')
    st.write('''$$\epsilon$$ is the error term.''')
    model_var = st.selectbox("##### Select independent variables names to add to the MLR model.", ['speeding', 'alcohol', 'not_distracted', 'no_previous','ins_premium'])

    X = sm.add_constant(df_crash[model_var])

    # Fitting the linear regression model
    model = sm.OLS(df_crash['total'], X).fit()
    summary = model.summary()
    table_data = summary.tables[1]
    result_table = pd.DataFrame(table_data.data, columns=table_data.data[0]).iloc[1:]
    result_table.columns = ['Label', 'Coefficient', 'Std. Error', 't-stat', 'P-value', "[0.025", '0.975]']
    st.dataframe(result_table)
    with st.expander("Results Description"):
        st.write("**Constant term** or the intercept of the model is the value of the dependent variable ($$Y$$) when all the independent variables are set to zero.")
        st.write('''**Coefficient term** of an independent variable shows how much the dependent variable is expected to change when the corresponding independent variable 
                increases or decreases by one unit while holding all other variables constant. In this case of simple linear-regression model, we have only independent variable.''', )
        if model_var == "ins_premium":
            st.write("Therefore, in case of "+ model_var + ", the coefficient is ", result_table['Coefficient'][2], ", which means that the one-unit increase in percent of drivers "
                    + model_var + " results in ", result_table['Coefficient'][2], " percent decrease in the percentage ot total fatal crashes.")
        else:
            st.write("Therefore, in case of "+ model_var + ", the coefficient is ", result_table['Coefficient'][2], ", which means that the one-unit increase in percent of drivers "
                    + model_var + " results in ", result_table['Coefficient'][2], " percent increase in the percentage ot total fatal crashes.")

        st.write("**Standard Error** is a measure of the variability or uncertainty in the coefficient estimate. Smaller standard errors indicate more precise estimates")
        st.write('''**P-value** is used to test the null hypothesis that the coefficient is equal to zero. 
                A small p-value (typically less than 0.05) suggests that the variable is statistically significant in predicting the dependent variable.
                \n **[0.025 0.975]** values represent the 95% confidence interval for the coefficient.''')
        
# Webapp link: https://cmse830-gagan-finalproject.streamlit.app/
# github repo link: https://github.com/Gagan2103/CMSE_830_Gagan

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.metrics import mean_squared_error


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


df_crash['state_name'] = df_crash['abbrev'].map(state_names_dict)
df_crash = df_crash.drop("abbrev", axis = 1)
df_columns = df_crash.columns


option = st.sidebar.radio("Select an option", ["Background", "Data Exploration", "Data Visualization", "Data Analysis", "About me"])


if option == "Background":
    st.markdown("# Introduction")

    st.image("Car-Crash.jpg")
    
    st.write('''Death and injuries from road traffic crashes remain a serious problem worldwide. 
             According to the National Highway Traffic Safety Administration, more than **40,000 fatal crashes** occur in USA every year [1].
             \n * Fatal crashes are defined as the crashes that lead to the death of one or more road users (drivers, pedestrians, bicyclists, etc.).
             \n * There are many possible factors making a crash fatal, such as overspeeding, drink and driving, using cell phone while driving, not wearing seat belt, etc. 
             To understand the reason behind such a high number of fatal crashes, we need to dive into the crash dataset.''')
    
    st.write(
        '''**Objective:** The goal of this webapp is to have a comparison analysis for factors affecting the number of fatal crashes across different
        states in the USA. 
        \nThis analysis is of interest to road agencies and road users, as it can help develop guidelines for road users to follow to have a safe and efficient transportation 
        system across the USA. This idea is important because millions of people die every year in road crashes around the world and thousands of people in the USA.''')
    
    st.markdown("##### References")

    st.write("1. https://www.nhtsa.gov/press-releases/2023-Q2-traffic-fatality-estimates#:~:text=More%20miles%20driven%20combined%20with,the%20first%20half%20of%202023.")    



    ######################################################################################################################################################################################
    # importing car crashes dataset from seaborn
elif option == "Data Exploration":
    st.markdown("# Car Crashes Dataset Exploration")
    st.write('''Let us explore a dataset having crash frequency and other factors affecting the crash frequency for different states in USA.''') 
    data_check = st.checkbox("**Dataset**")
    if data_check:
        st.write(df_crash)



    st.markdown("#### Dataset Description")

    st.write(
        '''This dataset contains following information: 
        \n* fatal crashes per billion vehicle miles traveled (VMT), 
        \n* factors affecting the number of fatal crashes such as: speeding, impaired (alcohol) driving, and driving with or without distraction.
        \n* percentage of drivers involved in fatal crashes for the first time in their lifetime. 
        \n* resulting insurance premiums stemming from these incidents. 
        \n* losses incurred by insurance companies due to crashes per insured driver. 
        \n It's important to note that there are no missing values in the dataset.
        \nThe dataset have **51 rows**, representing each of the states in the USA, and **8 columns** for different variables associated.'''
        )
    
    selected_options = st.multiselect("**Let's see the description of different columns present in the dataset. Select column names to see their brief description**", df_columns)
    description = {"total": "Number of drivers that are involved in fatal crashes per billion vehicle miles traveled", 
               "speeding": "Percentage of drivers involved in fatal crashes and were involved in speeding", 
               "alcohol": "Percentage of drivers involved in fatal crashes and were impaired due to alcohol use while driving", 
               "not_distracted": "Percentage of drivers involved in fatal crashes and were not involved in distracted driving. This could be the texting, calling, etc.", 
               "no_previous": "Percentage of drivers involved in fatal crashes and had not involved in any previous crash/accident", 
               "ins_premium": "Car insurance premiums", 
               "ins_losses": "Losses incurred by insurance companies for crashes per insured drivers.", 
               "state_name": "Name of the states in the USA"}

    for option in selected_options:
        st.write(f"**Description of {option}:** {description[option]}")
    st.markdown("#### Descriptive statistics")
    st.write('''Descriptive statistics of a variable include the count, mean, standard deviation, minimum, maximum and various quartile values.''')
    datades_show = st.selectbox("**Would you like to see the descriptive statistics of the dataset?**", ["No", "Yes"])
    if datades_show == "Yes":

        selected_var = st.multiselect("**Select column names to see their descriptive statistics.**", df_columns.drop("state_name"))
        
        st.write(df_crash.describe()[selected_var])

    with st.expander("More info"):
        st.write("""Car crashes dataset reveals the number of crashes for different states and the factors that affect the crash frequency. 
                It also give relation between the increase in cost incurred to insurance companies and the total number of fatalities.""")


    ######################################################################################################################################################################################
if option == "Data Visualization":
    st.markdown("# Car Crashes Dataset Visualization")
    st.write("This section of the app deals with the visualization of data. Let's see what data tells about it.")
    st.write(
        '''The section provides an interactive interface for people who are interested
        to know about the number of fatal crashes and the reason behind them.'''
        )
    

    # Visualization
    graph = st.selectbox("**Select the type of visualization you are interested**", ['None', 'HiPlot', 'Bar plot', 'Multiple bar plot', 'Regression plot', 'Interactive Scatter plot'])
    def save_hiplot_to_html(exp):
        output_file = "hiplot_plot_1.html"
        exp.to_html(output_file)
        return output_file
    
    if graph == "HiPlot":
        st.write('This plot allows you to select required columns and visualize them using HiPlot.')
        selected_columns = st.multiselect("Select columns to visualize", df_columns)
        selected_data = df_crash[selected_columns]
        if not selected_data.empty:
            experiment = hip.Experiment.from_dataframe(selected_data)
            hiplot_html_file = save_hiplot_to_html(experiment)
            st.components.v1.html(open(hiplot_html_file, 'r').read(), height=1500, scrolling=True)
            with st.expander("Key points"):
                st.write('''HiPlot makes it easy to explore and understand complex datasets with many variables or dimensions. 
                         To select a portion of one of the vertical axes, click on any of the features on the vertical column and drag it with left click.
                         The dataframe is showing the results of data appearing on the plot, for example, if we select some portion of vertical axes, dataframe will show data corresponsing to that range only.''')
        else:
            st.write("No data selected. Please choose at least one column to visualize.")
        

    elif graph == "Bar plot":
        st.write("Bar plot is used here to see the trend of different variables across different states in the USA.")
        y_value = df_crash['state_name']
        x_value = st.selectbox("Select a variable for x-axis of the plot", df_columns.drop("state_name"))
        abbrev_order = df_crash.sort_values(x_value, ascending=False)["state_name"].tolist()
        plt.figure(figsize=(6, 12))
        bar_fig = sns.barplot(x = x_value, y = y_value, data=df_crash, label = x_value, color="g", order=abbrev_order)
        st.pyplot(bar_fig.get_figure())
        with st.expander("Key points"):
            st.write('''The key takeaway here is the states with the highest and lowest values of the selected variable. 
                    **For instance**, in the case of total crashes, South Carolina ranks highest, while the District of Columbia ranks lowest. 
                    Taking speeding as another example, Hawaii records the highest number of speeding cases, while New Jersey has the fewest.''')

        
    elif graph == "Multiple bar plot":
        st.write("Multiple bar plot is used to see the trend of different causes of crashes out of total crashes values across different states in the USA.")
        y_value = df_crash['state_name']
        x_value = st.selectbox("Select a variable to overlap the barplot over total number of crashes.", df_columns.drop(["state_name", 'total', "ins_losses", "ins_premium"]))
        abbrev_order = df_crash.sort_values('total', ascending=False)["state_name"].tolist()
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
        st.write("Regression plot is used to see the relation and trend between variables.")
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


    ######################################################################################################################################################################################
if option == "Data Analysis":
    st.markdown("# Car Crashes Dataset Analysis")
    st.write("This section of the app deals with the analysis of data.")
    st.subheader("Correlation Matrix Heatmap")
    st.write('''Before diving into the dataset, let's see if different variables in the data are correlated to each other or not.
             You can hover on each cell to know the correlation value. 
             \n* The positive value of correlation shows that with the increase in one variable, 
             other variable also increases, and vice versa. 
             \n* Correlation value close to zero means that the two variables are not having any trend between them. ''')
    
    check = st.checkbox("**Show Correlation Matrix Heatmap**")
    if check:
        fig_heat = px.imshow(df_crash.corr(), color_continuous_scale='RdBu_r')

        fig_heat.update_layout(
            width=600, height=500,
        )
        st.plotly_chart(fig_heat)

        st.write('''We can see from the heatmap that independent variables are highly correlated with each other, so they might have effect on each other in a model. 
            So, we will see the effect of individual independent variables (e.g. speeding, alcohol, etc.) on the dependent variable (i.e. total fatal crashes).  
            ''')
    
    #################################################################################################################################################################################
    #################################################################################################################################################################################
    # Machine learning models


    st.subheader("Modeling")
    st.write("Now, we will try to explore the relationship between total number of fatal crashes and other variables present in the dataset.")

    tab1, tab2, tab3 = st.tabs(["Machine Learning Models", "Linear Regression Model", "Conclusion"])

    
    X = df_crash[['speeding', 'alcohol', 'not_distracted', 'no_previous','ins_premium', 'ins_losses']]
    y = df_crash[["total"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns =['speeding', 'alcohol', 'not_distracted', 'no_previous','ins_premium', 'ins_losses'])
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns = ['speeding', 'alcohol', 'not_distracted', 'no_previous','ins_premium', 'ins_losses'])

    


    def correlation(df,col1,col2):
        rmse=np.sqrt(mean_squared_error(df[col1], df[col2]))
        r=stats.pearsonr(df[col1], df[col2])
        r_square= pow(r[0], 2)
        return rmse,r,r_square

    def use_ML(data_train, label_train, data_test, est):
        """"Run Machine Learning Model.""" 

        if est == 'DT':

            rng = np.random.RandomState(42)
            
            regr = DecisionTreeRegressor(min_samples_leaf=leaf, max_features=max_featD)
            # fit the data and label
            regr.fit(data_train, label_train)

            # Make prediction of the model
            y_pred = regr.predict(data_test)

            important_feature = pd.DataFrame({'Feature': features, 'Importance': regr.feature_importances_})
            st.write("**Features Importance:** a measure of the contribution of each feature (or variable) to the accuracy or performance of the model")
            important_feature
            st.write("**Here, we can conclude that the variables: `not_distacted` and `no_previous` are having the highest contribution in the prediction of number of fatal crashes.**")

        elif est == 'RF':

            regr = RandomForestRegressor(n_estimators=n_est, max_features=max_featR)

            # fit the data and label
            regr.fit(data_train, label_train)

            # Make prediction of the model
            y_pred = regr.predict(data_test)

            important_feature = pd.DataFrame({'Feature': features, 'Importance': regr.feature_importances_})
            st.write("**Features Importance: a measure of the contribution of each feature (or variable) to the accuracy or performance of the model**")
            important_feature
            st.write("**Here, we can conclude that the variables: `not_distacted` and `no_previous` are having the highest contribution in the prediction of number of fatal crashes.**")


        elif est == "BR":
            regr = BayesianRidge()

            # fit the data and label
            regr.fit(data_train, label_train)

            # Make prediction of the model
            y_pred = regr.predict(data_test)
            
        elif est == 'LR':
            regr = LinearRegression()

            # fit the data and label
            regr.fit(data_train, label_train)

            # Make prediction of the model
            y_pred = regr.predict(data_test)

            y_pred = np.ravel(y_pred)

            important_feature = pd.DataFrame({'Feature': features, 'Importance': regr.coef_[0]})
            st.write("**Features Importance: a measure of the contribution of each feature (or variable) to the accuracy or performance of the model**")
            important_feature
            st.write("**Here, we can conclude that the variables: `not_distacted` and `no_previous` are having the highest contribution in the prediction of number of fatal crashes.**")
            
        
        elif est == "SVC":
            rng = np.random.RandomState(42)
            regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
            regr.fit(data_train, label_train)

            # Make prediction of the model
            y_pred = regr.predict(data_test)
            
        

        elif est == "GR":
            rng = np.random.seed(42)
            regr = GradientBoostingRegressor()
            regr.fit(data_train, label_train)

             # Make prediction of the model
            y_pred = regr.predict(data_test)
            important_feature = pd.DataFrame({'Feature': features, 'Importance': regr.feature_importances_})
            st.write("**Features Importance: a measure of the contribution of each feature (or variable) to the accuracy or performance of the model**")
            important_feature
            st.write("**Here, we can conclude that the variables: `not_distacted` and `no_previous` are having the highest contribution in the prediction of number of fatal crashes.**")


        return y_pred




    with tab1:
        st.write("First, we will use machine learning models to get the variables having high contribution in prediction of fatal crashes across various states in the USA.")
        modeling = st.selectbox('**Select the machine learning model of your choice:**', ["Random Forest", "Decision Tree", "Bayesian Ridge", 
            "Linear Regression", "Support Vector Classification (SVC)", "Gradient Boosting"])
        features = st.multiselect('Select features (i.e. independent variables):', ['speeding', 'alcohol', 'not_distracted', 'no_previous','ins_premium', 'ins_losses'])
        max_ft = len(features)                          
        if max_ft == 0:
            st.write('''#### Ooops! Please select atleast one feature of interest...... ''')
        else:

            if modeling == 'Random Forest':
                st.write('''**Number of trees:** It is used to built the decision trees and finally combining them to make predictions. 
                         More number of trees are used to avoid overfitting and improve the performance of the model.
                         Let us change the number of trees using the slider and see if it is having any impact on the model performance.
                         \nIf you select more than one features for model, you can select how many features you want in the actual model by sliding the maximum feature slider, 
                         and see how it is impacting the model performance.''')
                ml_col1, ml_col2 = st.columns(2)
                with ml_col2:
                    # Making a default value for the slider after rendering
                    if max_ft == 1:
                        max_featR = max_ft  # This will be used as the hyperparameter for Max Features
                    else: 
                        val = max_ft-1
                        max_featR = st.slider('Max Feature:', 1, max_ft, val)

                with ml_col1:
                    n_est = st.slider('Number of Trees:', 10, 100, 50)

                

                # Running the Random Forest Regressor Estimator
                y_pred = use_ML(X_train_scaled[features], y_train, X_test_scaled[features], est='RF')
                
            
                

            elif modeling == 'Decision Tree':
                st.write('''Here, the number of leafs is used to built the leaf nodes in a decision tree. Each leaf node in a decision tree represents a final decision or prediction.
                         More number of leafs can lead to overfitting and may reduce the performance of the model.
                         Let us change the number of leafs using the slider and see if it is having any impact on the model performance.
                         \nIf you select more than one features for model, you can select how many features you want in the actual model by sliding the maximum feature slider,
                         and see how it is impacting the model performance.''')
                ml_col_DT1, ml_col_DT2= st.columns(2)
                with ml_col_DT2:
                    # Making a default value for the slider after rendering
                    if max_ft == 1:
                        max_featD = max_ft  # This will be used as the hyperparameter for Max Features
                    else: 
                        val = max_ft-1
                        max_featD = st.slider('Max Feature:', 1, max_ft, val)

                with ml_col_DT1:
                    leaf = st.slider('Number of Leafs:', 1, 10, 1)

                # Running the Decision Tree Estimator
                y_pred = use_ML(X_train_scaled[features], y_train, X_test_scaled[features], est='DT')

            elif modeling == "Bayesian Ridge":
                y_pred = use_ML(X_train_scaled[features], y_train, X_test_scaled[features], est = "BR")

            elif modeling == 'Linear Regression':
                # Running the Linear Regressor Estimator
                y_pred= use_ML(X_train_scaled[features], y_train, X_test_scaled[features], est='LR')

            elif modeling == "Support Vector Classification (SVC)":
                y_pred= use_ML(X_train_scaled[features], y_train, X_test_scaled[features], est='SVC')

            elif modeling == "Gradient Boosting":
                y_pred = use_ML(X_train_scaled[features], y_train, X_test_scaled[features], est='GR')


            df = pd.DataFrame()
            df = X_test[features]
            df['Crash Predicted'] = y_pred
            df['Crash Observed'] = y_test
            
            #MM.visulaization_error()
            rmse_cal,r_cal,r_square_cal=correlation(df,'Crash Observed','Crash Predicted')
            st.write("""##### Model Performance:""")
            st.write('''We can see the difference in model performace by the changes in mean square error (MSE), root mean square error (RMSE) or the coefficient of determination (R-squared). 
                        \nA better model is characterized by: 
                     \n * lower Mean Squared Error (MSE), 
                     \n * lower Root Mean Squared Error (RMSE) values, and
                     \n *  higher R-squared value.''')
            df_score = pd.DataFrame(np.array([rmse_cal**2, rmse_cal,r_square_cal]).reshape(1, -1), columns=('MSE', 'RMSE', "R-squared"))
            st.table(df_score)

            st.write("Let us see a scatter plot of our predicted fatal crashes and actual fatal crashes observed. We can see the changes in scatter plot due to changes in the model and selected parameters.")

            scatter = alt.Chart(df).properties(width=350).mark_circle(size=100).encode(x='Crash Observed', y='Crash Predicted').interactive()
            reg_line=alt.Chart(df).properties(width=350).mark_circle(size=100).encode(x='Crash Observed', 
                y='Crash Predicted').transform_regression('Crash Observed','Crash Predicted').mark_line()
            scatter+reg_line


    with tab2:
        st.write('''**Linear regression (LR) model** is a statistical method used to analyze the relationship between independent variables
                    (also known as predictor or explanatory variables) and a dependent variable (also known as the response variable). 
                    \n In a LR model, the goal is to create a linear equation that can predict the dependent variable based on the value of the independent variables. 
                    \n The general form of the linear regression model is:''')
        st.write(''' $$Y =  \Beta_0 + \Beta_iX_i + \epsilon $$, ''')
        st.write('''where, 
                \n $$Y$$ is the dependent variable (total number of crashes in our model),''')
        st.write('''$$\Beta_0$$ is the intercept or constant term,''')
        st.write('''$$\Beta_i$$ is the vector of coefficients of the independent variables $$(X_i)$$.''')
        st.write('''$$\epsilon$$ is the error term.''')
        model_var = st.multiselect("##### Select independent variables names to add to the MLR model.", ['speeding', 'alcohol', 'not_distracted', 'no_previous','ins_premium', 'ins_losses'])

        X = sm.add_constant(df_crash[model_var])

        # Fitting the linear regression model
        model = sm.OLS(df_crash['total'], X).fit()
        summary = model.summary()
        table_data = summary.tables[1]
        result_table = pd.DataFrame(table_data.data, columns=table_data.data[0]).iloc[1:]
        result_table.columns = ['Label', 'Coefficient', 'Std. Error', 't-stat', 'P-value', "[0.025", '0.975]']
        st.dataframe(result_table)

        # Make predictions
        y_pred = model.predict(X)

        # Calculate Mean Squared Error (MSE)
        mse = np.mean((df_crash['total'] - y_pred)**2)

        # Calculate Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)

        # Get the R-squared value
        r_squared = model.rsquared



        check = st.checkbox("**Results Description**")
        if check:
            st.write("**Constant term** or the intercept of the model is the value of the dependent variable ($$Y$$) when all the independent variables are set to zero.")
            st.write('''**Coefficient term** of an independent variable shows how much the dependent variable is expected to change when the corresponding independent variable 
                    increases or decreases by one unit while holding all other variables constant.''')
            st.write("**Standard Error** is a measure of the variability or uncertainty in the coefficient estimate. Smaller standard errors indicate more precise estimates")
            st.write('''**P-value** is used to test the null hypothesis that the coefficient is equal to zero. 
                        A small p-value (typically less than 0.05) suggests that the variable is statistically significant in predicting the dependent variable.
                        \n **[0.025 0.975]** values represent the 95% confidence interval for the coefficient.''')
        st.write("""##### Model Performance:""")
        
        
        df_score = pd.DataFrame(np.array([mse, rmse,r_squared]).reshape(1, -1), columns=('MSE', 'RMSE', "R-squared"))
        st.table(df_score)

        





    with tab3:        
        st.write("Conclusions:")
        st.write("* Variables: `not_distacted` and `no_previous` are having the highest contribution in the prediction of number of fatal crashes.")
        st.write("* Using just these two variables in the Linear Regression model, we are getting $R^2=0.9432$.")
        st.write("* It means that $94.32 $%  of the variance in the dependent variable is explained by the independent variables in the model.")
        st.write("* The coefficient of both the variables is positive, meaning  that they are positively related to the number of fatal crashes.")

    ######################################################################################################################################################################################
if option == "About me":
    st.markdown("# About me")
    st.write('''👋 Hi there! 
             \n👦 My name is Gagan Gupta.
             \n🎓 I am a second-year Ph.D. student at Michigan State University.
             \n🚗 I am working in the field of Transportation Engineering. My work mainly focuses on safety of road users.
             🚦 I have been working in road safety for the last three years. 
             \n🏋️‍♂️ Outside of work, you'll find me in the IM West Gym or playing pool. 
             \n🍕 Pizza lover
             \n🔗 Connect with me on Linkedin (https://www.linkedin.com/in/gagan-gupta-2103/).''')


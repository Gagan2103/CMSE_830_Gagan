#My link to the project is: 
#MY link to my GitHub repository is: 
import streamlit as st
import altair as alt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
st.set_option('deprecation.showPyplotGlobalUse', False)
#********************************************************************************************************
st.set_page_config(page_title="Kicker Impact Analysis", layout="wide")
fg_data = pd.read_excel('nfl_data.xlsx', sheet_name='Last_2_mins', engine='openpyxl')
fg_data = fg_data.dropna()
min_attempts = 50
kicker_counts = fg_data['kicker'].value_counts()
qualified_kickers = kicker_counts[kicker_counts >= min_attempts].index
#********************************************************************************************************
menu = st.sidebar.radio("Jump to", ["Background", "About the Data", "Analysis Methodology","Predictions"], key="tabs")
#********************************************************************************************************
if menu == "Background":
    st.markdown('# Assessing the Field Goal Performance of NFL Kickers in 2- Minute Drives')
    image_url = "https://frontofficesports.com/wp-content/uploads/2023/06/USATSI_20001296.jpg"
    st.image(image_url, caption="(Credit: Kirby Lee -USA TODAY Sports)", use_column_width=True)
    st.write("The exploration centers around analyzing the data related to field goal attempts during 2-minute drives in NFL games from 2009 to 2018, with a specific focus on comparing the performances of prominent kickers over that time period. By scrutinizing variables such as kick distance, timeouts remaining, and goal-to-go situations, the goal is to uncover patterns and trends that illuminate the effectiveness of kickers during these high-pressure scenarios.")
    show = st.selectbox("", ["For the non-NFL enthusiasts","What is a Field Goal?", "What is a 2-minute Drive?", "Opting for Field Goal as a Strategic Decision-Making","Kickers as Game Changers"])
    if show == "What is a Field Goal?":
        st.markdown("A field goal in is a scoring play that occurs when a team successfully kicks the ball through the opponent's goalposts during a designated play. It is typically attempted from varying distances on the field, depending on the team's position and the strategic decision made by the coaching staff. A successful field goal is worth three points and is often employed when a team is within kicking range but may face challenges scoring a touchdown. Executing a field goal requires precision from the kicker and coordination with the snapper and holder, making it a fundamental and strategic element of the game. [More about field goal](https://en.wikipedia.org/wiki/Field_goal)")
        image_url = "https://i.ytimg.com/vi/SARBnsXNBgs/maxresdefault.jpg"
        st.image(image_url, caption="Kansas City Chiefs Kicker Harrison Butker attempts a 27-yard field goal to give his team a lead in SUperBowl LVII, which eventually turned out to be the penultimate play of the game and won them the Super Bowl (Credit: NFL)", use_column_width=False)
    if show == "What is a 2-minute Drive?":
        st.markdown("A 2-minute drive refers to a critical and time-sensitive period typically occurring at the end of each half of a game. During these intense moments, teams attempt to advance the ball down the field swiftly, aiming to score crucial points before the clock runs out. Known for heightened pressure and strategic decision-making, 2-minute drives often involve rapid plays, effective clock management, and can [significantly influence the outcome of a game](https://www.youtube.com/watch?v=JfVGoNf6gtI). These scenarios showcase the strategic prowess of teams as they navigate the limited time available to make impactful plays and secure a competitive advantage.")
        image_url = "https://media.cnn.com/api/v1/images/stellar/prod/231005220619-tom-brady-jersey-record-auction-spt.jpg?q=w_1110,c_fill/f_webp"
        st.image(image_url, caption="Thanks to his efficient 2-minute drills, Tom Brady has won a record 7 Super Bowls (Credit: Mike Ehrmann/Getty Images)", use_column_width=False)
    if show == "Opting for Field Goal as a Strategic Decision-Making":
        st.markdown("In the high-pressure context of a two-minute drill, opting for a field goal represents a strategic decision-making process for coaches. Assessing the remaining time, distance to the end zone, and the opposing defense's effectiveness, coaches may [strategically choose a field goal](https://www.youtube.com/watch?v=sgRZpFnrMdE) for swift points when a touchdown seems challenging within the limited time frame. Beyond scoring, the decision serves a tactical purpose by allowing teams to burn the opposing team's timeouts, disrupting their defensive strategies. This calculated approach illustrates the nuanced balance between time management, strategic decision-making, and the pursuit of points during a critical two-minute drill scenario.")
        image_url = "https://encrypted-tbn3.gstatic.com/licensed-image?q=tbn:ANd9GcSnujcUVVibMJ4XK4lNsqP6VVDyu2bcu6ulwu1gf17ArUTxYALlO3FpqByl0giJr0ycp8qYN_if_wLnfWU"
        st.image(image_url, caption="Kansas City Chiefs Head Coach Andy Reid's decision to take the QB kneel twice at the 2-yard line to opt for the field goal within final 2-minutes of the game left only 8 seconds for Philadelphia Eagles Offense, eventually helping Chiefs to win Super Bowl LVII (Credit: Don Juan Moore/Getty Images)", use_column_width=True)
    if show == "Kickers as Game Changers":
        st.markdown("In these critical moments, kickers emerge as potential game-changers. A successful field goal attempt by the kicker can turn the tide in favor of a team, [securing crucial points that may determine the game's outcome](https://www.youtube.com/watch?v=JJNLVT2luxg). Analyzing kickers' performances during 2-minute drives unveils valuable insights into their ability to thrive under pressure.")
        image_url = "https://cdn.vox-cdn.com/thumbor/RKWM__k62xOrxCBPJe8Te2HSm2A=/1400x1050/filters:format(jpeg)/cdn.vox-cdn.com/uploads/chorus_asset/file/22241086/usa_today_15444360.jpg"
        st.image(image_url, caption="Justin Tucker (Baltimore Ravens), often regarded as the greatest NFL kicker of all time, currently holds the NFL record for longest field goal at 66 yards.(Credit: Mark Konezny-USA TODAY Sports)", use_column_width=True)    
#********************************************************************************************************
if menu == "About the Data":
    st.title('Data Exploration')
    st.markdown('Data were obtained from an [open source](https://www.kaggle.com/datasets/maxhorowitz/nflplaybyplay2009to2016) that contained play-by-play data for NFL Regular Season games from the 2009 to 2018 seasons. To focus solely on field goal data, only the plays involving field goals were selected. Additionally, to analyze the performance of kickers in  "[clutch](https://en.wikipedia.org/wiki/Clutch_(sports))" moments, only the field goal plays that occurred within the final two minutes of either of the halves were taken. Ultimately, the dataset consisted of the following variables, which are explained below:')
    variable_info = {
    "year": "The year of the NFL game",
    "playID": "Unique identifier for each play",
    "posteam": "The team of the kicker",
    "Home_game": "Indicator if the game is a home game for the kicker's team",
    "defteam": "The opponent team",
    "yardline_100": "The yard line on the field, normalized to a 100-yard scale",
    "half_seconds_remaining": "Seconds remaining in the current half",
    "game_seconds_remaining": "Seconds remaining in the entire game",
    "scored": "Binary variable indicating if the field goal was scored (1) or not (0)",
    "down": "The down on which the field goal attempt was made",
    "goal_to_go": "Indicator if the kicker is in a goal-to-go situation",
    "ydstogo": "Yards to go for a first down",
    "kick_distance": "The distance of the field goal attempt in yards",
    "timeouts_remaining": "Number of timeouts remaining for the kicker's team",
    "score_differential": "Point difference between the kicker's team and the opponent",
    "kicker": "Name of the kicker",
    "kicker_player_id": "Unique identifier for the kicker player",
    }

    for variable, description in variable_info.items():
        st.write(f"**{variable}:** {description}")
    st.markdown("### Do you want to see the raw dataset?")
    show = st.selectbox("", ["No", "Yes"])
    if show == "Yes":
        st.write(fg_data)
    st.markdown('The interactive chart below aids in understanding the success rate for each kicker based on the minimum number of field goal attempts. Visualizing this chart also assists us in identifying kickers with a low number of attempts, potentially treating them as outliers during the statistical analysis.')
    min_attempts = st.slider("Select Minimum Attempts", min_value=1, max_value=100, value=50)
    st.write(f"Minimum Attempts: {min_attempts}")
    def calculate_success_rate(df):
        success_rate = df.groupby(['kicker', 'posteam'])['scored'].mean().reset_index()
        success_rate.columns = ['kicker', 'posteam', 'success_rate']
        return success_rate
    filtered_df = fg_data[fg_data.groupby('kicker')['kicker'].transform('count') >= min_attempts]
    success_rate_df = calculate_success_rate(filtered_df)
    chart = alt.Chart(success_rate_df).mark_bar().encode(
        x='kicker:N',
        y=alt.Y('success_rate:Q', axis=alt.Axis(title='Success Rate')),
        color=alt.Color('posteam:N', scale=alt.Scale(scheme='category20')),
        tooltip=['kicker:N', 'posteam:N', 'success_rate:Q']
    ).properties(
        title=f'Success Rate by Kicker and Team (Minimum Attempts: {min_attempts})',
        width=600,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)
#********************************************************************************************************
if menu == "Analysis Methodology":
    st.title('Logistic Regression')
    st.markdown('A logistic regression analysis was conducted, where a predictive model was developed to assess the success rate of field goal attempts by NFL kickers. The model utilized key features such as goal-to-go situations, kick distance, timeouts remaining, home game indicator, time remaining in the half, down, score differential, and yards to go for a first down. The logistic regression, trained on a subset of the dataset, aimed to predict whether a kicker would successfully score a field goal or not. The model coefficients were examined to understand the impact of each feature on the likelihood of a successful field goal attempt for the selected kicker.')
    st.markdown('For the purpose of this app, a minimum threshold of 50 field goal attempts was established to comprehensively assess the influence of in-game variables on the likelihood of a kicker successfully scoring a field goal. Each kicker underwent logistic regression modeling, wherein the resulting coefficients elucidate the impact of various game-related factors on the probability of the kicker making a successful field goal. The analysis aimed to unveil the nuanced relationships between key game dynamics and kicker performance, providing valuable insights into the factors shaping the outcomes of field goal attempts in critical two-minute scenarios.')
    st.subheader("Logistic Regression Model for Field Goal Success Prediction")
    selected_kicker = st.radio('Select a Kicker', qualified_kickers)
    df_kicker = fg_data[fg_data['kicker'] == selected_kicker]
    selected_columns = ['goal_to_go', 'kick_distance', 'timeouts_remaining', 'Home_game', 'half_seconds_remaining', 'down', 'score_differential', 'ydstogo', 'scored']
    df_selected = df_kicker[selected_columns].dropna()
    X = df_selected[['goal_to_go', 'kick_distance', 'timeouts_remaining', 'Home_game', 'half_seconds_remaining', 'down', 'score_differential', 'ydstogo']]
    y = df_selected['scored']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
    st.write(f"### Model Coefficients for {selected_kicker}:")
    st.table(coefficients)
    st.write(f"**Accuracy for {selected_kicker}:** {accuracy:.2f}")
    st.write(f"**Precision for {selected_kicker}:** {precision:.2f}")
    st.write(f"**Recall for {selected_kicker}:** {recall:.2f}")
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    st.write(f"### Confusion Matrix for {selected_kicker}:")
    st.table(cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f'Confusion Matrix for {selected_kicker}')
    st.pyplot(fig)
#********************************************************************************************************
if menu == "Predictions":
    st.markdown("Ultimately, the logistic regression model, trained on the historical dataset specific to each kicker, enables the prediction of a field goal attempt's success. This prediction capability extends to any of the 11 kickers in the dataset.")
    min_attempts = 50
    kicker_counts = fg_data['kicker'].value_counts()
    qualified_kickers = kicker_counts[kicker_counts >= min_attempts].index
    selected_kicker = st.radio('Select the Kicker:', qualified_kickers)    
    goal_to_go = st.slider('Goal to Go:', 0, 1, 1)
    kick_distance = st.slider('Kick Distance:', min_value=0, max_value=60, value=30)
    timeouts_remaining = st.slider('Timeouts Remaining:', min_value=0, max_value=3, value=2)
    home_game = st.slider('Home Game:', 0, 1, 1)
    half_seconds_remaining = st.slider('Half Seconds Remaining:', min_value=0, max_value=120, value=60)
    down = st.slider('Down:', min_value=1, max_value=4, value=1)
    score_differential = st.slider('Score Differential:', min_value=-50, max_value=50, value=0)
    ydstogo = st.slider('Yards to Go:', min_value=0, max_value=20, value=10)
    df_kicker = fg_data[fg_data['kicker'] == selected_kicker]
    selected_columns = ['goal_to_go', 'kick_distance', 'timeouts_remaining', 'Home_game', 'half_seconds_remaining', 'down', 'score_differential', 'ydstogo', 'scored']
    df_selected = df_kicker[selected_columns].dropna()
    X = df_selected[['goal_to_go', 'kick_distance', 'timeouts_remaining', 'Home_game', 'half_seconds_remaining', 'down', 'score_differential', 'ydstogo']]
    y = df_selected['scored']
    model = LogisticRegression()
    model.fit(X, y)
    input_data = [[goal_to_go, kick_distance, timeouts_remaining, home_game, half_seconds_remaining, down, score_differential, ydstogo]]
    probability = model.predict_proba(input_data)[0, 1]
    st.write(f"Probability of {selected_kicker} making the field goal: {probability:.2f}")


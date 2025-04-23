import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
import time

@st.cache_data(show_spinner="Fetching and preprocessing data...")
def load_and_preprocess_data(year, week):
    url = "https://api.collegefootballdata.com/drives"
    api_key = "FMvqMasSE6XNq09+QcGdshB+U2Ri9qQHZEV4JWd0p7Yr00nX5ZVilgNpA+JkBhwq"
    headers = {"Authorization": f"Bearer {api_key}"}

    all_drives = []
    for wk in range(1, week + 1):
        params = {"year": year, "classification": "fbs", "week": wk}
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200 and response.json():
            for row in response.json():
                row['week'] = wk
                all_drives.append(row)

    df = pd.DataFrame(all_drives)

    if not df.empty:
        df = df[df['start_period'] < 5]
        df = df[~df['drive_result'].isin(['Uncategorized', 'END OF HALF', 'END OF 4TH QUARTER'])]
        df['drive_result'] = df['drive_result'].apply(lambda x: 'Not Defended' if x in ['TD', 'FG'] else 'Defended')
        df['score_diff'] = (df['end_offense_score'] - df['start_offense_score']) - (df['end_defense_score'] - df['start_defense_score'])
        df['score_diff'] = df['score_diff'].shift(1)
        df['score_diff'] = df['score_diff'].fillna(0)
        df = df[~((df['start_period'] == 4) & (df['score_diff'] > 24))]

    return df

def stoprate(df, team_names):
    stop_rate_data = {'defense': [], 'Total Drives': [], 'Stops': [], 'Stop Rate%': []}
    for team in team_names:
        defense_drives = df[df['defense'] == team]
        total_drives = len(defense_drives)
        stops = len(defense_drives[defense_drives['drive_result'] == 'Defended'])    
        stop_rate = stops / total_drives if total_drives > 0 else 0
        stop_rate_data['defense'].append(team)
        stop_rate_data['Total Drives'].append(total_drives)
        stop_rate_data['Stops'].append(stops)
        stop_rate_data['Stop Rate%'].append(round(stop_rate * 100, 2))
    stop_rate_df = pd.DataFrame(stop_rate_data)
    stop_rate_df.sort_values(by='Stop Rate%', ascending=False, inplace=True)
    return stop_rate_df

def compute_weekly_stop_rate(df, teams, selected_week):
    weekly_data = {'defense': [], 'week': [], 'Stop Rate%': []}
    for team in teams:
        for wk in range(1, selected_week + 1):
            filtered = df[(df['defense'] == team) & (df['week'] == wk)]
            total = len(filtered)
            stops = len(filtered[filtered['drive_result'] == 'Defended'])
            rate = (stops / total) * 100 if total > 0 else 0

            weekly_data['defense'].append(team)
            weekly_data['week'].append(wk)
            weekly_data['Stop Rate%'].append(rate)
    return pd.DataFrame(weekly_data)


# Function to Preprocess Data for ASR Model
# Input-Data Frame
# Output- Preprocessed DataFrame, StandardScaler object

def model_preprocess(df):
    model_df = df[['offense','defense','start_yards_to_goal','drive_result']].copy()
    LE1=LabelEncoder()
    model_df['drive_result']=LE1.fit_transform(model_df['drive_result'])
    model_df['offense'] = model_df['offense'].astype(str)
    model_df['defense'] = model_df['defense'].astype(str)
    encoded_offense = pd.get_dummies(model_df['offense'], prefix='offense', dtype=int)
    encoded_defense = pd.get_dummies(model_df['defense'], prefix='defense', dtype=int)
    model_df = pd.concat([model_df, encoded_offense, encoded_defense], axis=1)
    model_df = model_df.drop(columns=['offense', 'defense'])
    SC=StandardScaler()
    model_df[['start_yards_to_goal']] = SC.fit_transform(model_df[['start_yards_to_goal']])

    return model_df,SC
    

# Function to Train the ASR Model
## Input- Prprocessed DataFrame from model_preprocess function, Team Names, StandardScaler object
def asr_model(model_df, team_names, SC, df):
    param_grid = {'C': [1/0.01, 1/0.1]}  

    lr_model = LogisticRegression(penalty='l2', solver='newton-cg')
    grid_search = GridSearchCV(lr_model, param_grid, cv=2, scoring='accuracy', n_jobs=1)
    grid_search.fit(model_df.drop('drive_result', axis=1), model_df['drive_result'])

    best_model = grid_search.best_estimator_
    #best_model=lr_model
    # === Generate Synthetic Test Set ===
    test_data = [
        {"offense": i, "defense": j, "start_yards_to_goal": k}
        for i in team_names for j in team_names for k in range(5, 100, 5)
    ]
    test_df = pd.DataFrame(test_data)

    # Encode
    encoded_offense = pd.get_dummies(test_df['offense'], prefix='offense', dtype=int)
    encoded_defense = pd.get_dummies(test_df['defense'], prefix='defense', dtype=int)
    test_df = pd.concat([test_df, encoded_offense, encoded_defense], axis=1)

    raw_yards = test_df['start_yards_to_goal'].copy()

    # Scale for prediction
    test_df[['start_yards_to_goal']] = SC.transform(test_df[['start_yards_to_goal']])

    # Predict
    X_test = test_df.drop(columns=['offense', 'defense'], errors='ignore')
    y_pred = best_model.predict_proba(X_test)[:, 1]
    test_df['predicted_prob'] = 1 - y_pred

    test_df['Yards to Goal'] = raw_yards

    # === Compute Bin Distribution from Original Data ===
    bins = np.arange(0, 110, 10)
    labels = [f"{i}-{i+10}" for i in bins[:-1]]
    counts = pd.cut(df['start_yards_to_goal'], bins=bins, labels=labels, right=False).value_counts().sort_index()

    distribution_df = pd.DataFrame(counts.reset_index())
    distribution_df.columns = ['start_yards_to_goal', 'count']
    distribution_df['start'] = distribution_df['start_yards_to_goal'].str.split('-').str[0].astype(int)
    distribution_df['end'] = distribution_df['start_yards_to_goal'].str.split('-').str[1].astype(int)

    final_bins = [0] + distribution_df['end'].tolist()
    final_labels = distribution_df['count'].tolist()

    test_df['count'] = pd.cut(
        test_df['Yards to Goal'],
        bins=final_bins,
        labels=final_labels,
        right=False,
        include_lowest=True
    )
    test_df['count'] = pd.to_numeric(test_df['count'], errors='coerce')

    # === Compute ASR & Weighted ASR ===
    asr_dict = {}
    weighted_asr_dict = {}

    for defense_team in team_names:
        team_data = test_df[test_df['defense'] == defense_team]
        probs = team_data['predicted_prob'].values
        counts = team_data['count'].values

        asr_value = probs.mean() * 100
        weighted_asr = (probs * counts).sum() / counts.sum() * 100 if counts.sum() > 0 else 0

        asr_dict[defense_team] = round(asr_value, 2)
        weighted_asr_dict[defense_team] = round(weighted_asr, 2)

    asr_df = pd.DataFrame({
        'defense': list(asr_dict.keys()),
        'ASR%': list(asr_dict.values()),
        'Weighted ASR%': list(weighted_asr_dict.values())
    })

    stop_data = stoprate(df, team_names)[['defense', 'Stops']]
    asr_df = asr_df.merge(stop_data, on='defense', how='left')
    asr_df.sort_values(by='ASR%', ascending=False, inplace=True)

    return asr_df




def defense_analysis():
    st.markdown("### 🏈 NCAA Defensive Analysis")
    st.markdown("""
     Welcome to the Defensive Analysis Page!
This section lets you dive into how teams perform on the defensive side of the game across different seasons.

You’ll find key metrics like:

**Stop Rate:** – This shows how effective a team’s defense is by measuring the percentage of drives they were able to stop from ending in a score. It's a simple yet powerful way to gauge overall defensive efficiency.

**Adjusted Stop Rate (ASR):** – A more refined metric that considers the strength of the opposing offense and the field position (start yards to goal). We use a logistic ridge regression model trained on drive-level data to estimate the likelihood of a stop across a range of realistic scenarios. Synthetic data is generated by simulating different combinations of offense, defense, and field position.

**Weighted Adjusted Stop Rate (WASR):** – This goes a step further by weighting those ASR predictions based on how often each field position actually occurred in the real data. It gives a more grounded view of a team’s defensive effectiveness considering real-world game flow.
    """)
    defense_bg = """
    <style>
    body {
        background-color: #f4f4f4;
    }

    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }

    .block-container {
        padding-top: 2rem;
    }

    h1, h2, h3 {
        color: #1f77b4;
    }
    </style>
"""
    st.markdown(defense_bg, unsafe_allow_html=True)


    st.sidebar.header("Select Parameters")
    year = st.sidebar.selectbox("Select Year", options=[2020, 2021, 2022, 2023, 2024, 2025])
    week = st.sidebar.selectbox("Select Week", options=list(range(1, 16)))

    if st.sidebar.button("Fetch Data"):
        df = load_and_preprocess_data(year, week)
        st.session_state['drive_data'] = df
        st.session_state['data_fetched'] = True
        st.session_state.pop('asr_df', None)
        st.success(f"Fetched {len(df)} drive records from Week 1 to Week {week} of {year}.")


    if not st.session_state.get('data_fetched', False):
        st.warning("Please fetch data from the sidebar to begin analysis.")
        return

    # TABS
    tab1, tab2,tab3,tab4= st.tabs(["**Stop Rate**", "**Weighted Adjusted Stop Rate(WASR)**",'**ASR VS WASR VS SR Comparison**','**Test VS Train Plots**'])

    # === TAB 1: STOP RATE ANALYSIS ===
    with tab1:
        st.markdown(f"### Top 10 Teams by Stop Rate ({year} Season)")
        #st.markdown("The Stop Rate is calculated as the percentage of drives that resulted in **no score** for the offense.")
        df = st.session_state['drive_data']
        team_names = df['defense'].unique()
        stop_rate_df = stoprate(df, team_names)
        stop_rate_df = stop_rate_df[stop_rate_df['Stops'] > 30]
        st.session_state['stop_rate_df'] = stop_rate_df
        st.table(stop_rate_df.head(10).style.format({"Stop Rate%": "{:.2f}"}))
        csv = stop_rate_df.to_csv(index=False).encode('utf-8')
        st.download_button(
        label="💾 Download Full Stop Rate Data as CSV",
        data=csv,
        file_name=f'stop_rate_data{year}.csv',
        mime='text/csv',
        key="download_sr"
        )
        st.markdown("""
        <style>
        div[data-testid="stDownloadButton"] > button {
            background-color: white !important;
            color: black !important;
            border: 2px solid #ff4d4d;
            border-radius: 6px;
            font-weight: 600;
            box-shadow: 0 0 5px rgba(255, 77, 77, 0.5);
            transition: box-shadow 0.3s ease-in-out, transform 0.2s ease-in-out;
        }

        div[data-testid="stDownloadButton"] > button:hover {
            box-shadow: 0 0 12px rgba(255, 77, 77, 0.8);
            transform: scale(1.03);
        }
        </style>
        """, unsafe_allow_html=True)


                
        st.markdown("### Select Comparison Type: ")
        option_button=st.radio("Please Select:",('Overall Stop Rate','Weekly Stop Rate'),index=0,horizontal=True)

        if option_button=='Weekly Stop Rate':
                
            st.markdown("### Weekly Stop Rate Trend")
            selected_teams = st.multiselect(
                "Select Defense Teams to Compare",
                options=sorted(stop_rate_df['defense'].unique()),
                key='team_select_sr'
            )

            available_weeks = sorted(df['week'].unique())
            selected_week = st.selectbox("Select Maximum Week", options=available_weeks, key="week_select_sr")

            if selected_teams and selected_week:
                weekly_df = compute_weekly_stop_rate(df, selected_teams, selected_week)

                fig, ax = plt.subplots(figsize=(10, 6))
                for team in selected_teams:
                    team_data = weekly_df[weekly_df['defense'] == team]
                    ax.plot(team_data['week'], team_data['Stop Rate%'], marker='o', label=team)

                ax.set_xlabel("Week")
                ax.set_ylabel("Stop Rate (%)")
                ax.set_title("Stop Rate Trend by Week")
                ax.set_xticks(sorted(weekly_df['week'].unique()))
                ax.legend()
                st.pyplot(fig)
            else:
                st.info("Please select at least one team and week.")
        else:
            st.markdown("### Overall Stop Rate Trend")
            selected_teams = st.multiselect(
                "Select Defense Teams to Compare",
                options=sorted(stop_rate_df['defense'].unique()),
                key='team_select_sr'
            )

            if selected_teams:
                fig, ax = plt.subplots(figsize=(10, 6))
                for team in selected_teams:
                    team_data = stop_rate_df[stop_rate_df['defense'] == team]
                    ax.bar(team_data['defense'], team_data['Stop Rate%'], label=team)

                ax.set_xlabel("Defense Teams")
                ax.set_ylabel("Stop Rate (%)")
                ax.set_title("Overall Stop Rate Comparison")
                ax.legend()
                st.pyplot(fig)
            else:
                st.info("Please select at least one team.")

    # === TAB 2: Adjusted Stop Rate===
    with tab2:
        st.markdown(f"### Weighted Adjusted Stop Rate (WASR) Analysis ({year} Season)")
        #st.markdown("The Adjusted Stop Rate (ASR) is a more advanced metric that considers the context of each drive.")
        #st.markdown("It is calculated using a logistic regression model that incorporates various factors, including score differential and drive start location.")

        # Train button
        if st.button("Train Logistic Ridge Regression Model", key="train_button"):
            progress = st.progress(0, text="Starting ASR Model Training...")

            df = st.session_state['drive_data']
            team_names = df['defense'].unique()
            
            # Step 1: Preprocess
            progress.progress(10, text="Preprocessing data...")
            model_df, SC = model_preprocess(df)
            time.sleep(0.5)
            
            # Step 2: Train model
            progress.progress(50, text="Training model...")
            asr_df = asr_model(model_df, team_names, SC, df)
            time.sleep(0.5)
            
            # Step 3: Filter & Save
            progress.progress(80, text="Postprocessing results...")
            asr_df = asr_df[asr_df['Stops'] > 30]
            st.session_state['asr_df'] = asr_df
            
            progress.progress(100, text="✅ Done! ASR Model Trained Successfully")

            info_placeholder.empty()

        # Display ASR table only if model is trained
        if 'asr_df' in st.session_state:
            asr_df = st.session_state['asr_df']
            st.markdown("### Top 10 Teams by Adjusted Stop Rate (ASR)")
            st.table(asr_df.head(10).reset_index(drop=True).style.format({
                "ASR%": "{:.2f}",
                "Weighted ASR%": "{:.2f}"
            }))

            # Download button
            asr_csv = asr_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Full Adjusted Stop Rate Data as CSV",
                data=asr_csv,
                file_name=f'adjusted_stop_rate_data_{year}.csv',
                mime='text/csv'
            )
        else:
            st.info("Click the button above to train the ASR model.")

    with tab3:
        st.markdown(f"### Stop Rate vs Adjusted Stop Rate vs Weighted ASR ({year} Season)")
        st.markdown("This section compares the Stop Rate, Adjusted Stop Rate (ASR), and Weighted ASR for each team.")

        if 'asr_df' not in st.session_state:
            st.warning("Please train the ASR model in the previous tab before viewing this comparison.")
        else:
            stop_rate_df = st.session_state['stop_rate_df']
            asr_df = st.session_state['asr_df']
             # Filter out teams with less than 30 stops
            asr_df = asr_df[asr_df['Stops'] >= 30]

            # Merge Stop Rate and ASR results
            merged_df = stop_rate_df.merge(asr_df, on='defense', how='inner')

            # Sort by Stop Rate descending for consistent ordering
            merged_df.sort_values(by='Weighted ASR%', ascending=False, inplace=True)

            fig, ax = plt.subplots(figsize=(12, 6))

            # Stop Rate - use blue with 'o' marker
            ax.plot(merged_df['defense'], merged_df['Stop Rate%'], label='Stop Rate',
                    marker='o', color='blue', linestyle='--', linewidth=2)

            # ASR - use orange with 's' square marker
            ax.plot(merged_df['defense'], merged_df['ASR%'], label='Adjusted Stop Rate',
                    marker='s', color='orange', linestyle='-', linewidth=2)

            # Weighted ASR - use green with 'p' pentagon marker
            ax.plot(merged_df['defense'], merged_df['Weighted ASR%'], label='Weighted Adjusted Stop Rate',
                    marker='p', color='green', linestyle='-', linewidth=2)

            ax.set_xticks([])  # Hide x-axis ticks
            ax.set_xlabel("Defense Teams")
            ax.set_ylabel("Rate (%)")
            ax.set_title("Stop Rate vs Adjusted Stop Rate Across Teams")
            ax.legend()
            ax.grid(True)

            st.pyplot(fig)
    with tab4:
        choice=st.radio("Please Select:",('Train (VS) Test Stop Rate','Train ASR (VS) Test SR'),index=0,horizontal=True)
        if choice=='Train (VS) Test Stop Rate':
            df=load_and_preprocess_data(year, 15)
            train_df=df[df['week']<9]
            test_df=df[df['week']>=9]
            st.markdown(f"### Train Stop Rate vs Test Stop Rate Comparison ({year} Season)")
            team_names=list(df['defense'].unique())
            train_stop_rate_df=stoprate(train_df, team_names)
            test_stop_rate_df=stoprate(test_df, team_names)
            train_stop_rate_df=train_stop_rate_df[train_stop_rate_df['Stops'] > 30]
            test_stop_rate_df=test_stop_rate_df[test_stop_rate_df['Stops'] > 30]
            # Renaming Columns According to Week
            train_stop_rate_df = train_stop_rate_df[['defense', 'Stop Rate%']].rename(columns={'Stop Rate%': 'Stop Rate Week 8'})
            test_stop_rate_df = test_stop_rate_df[['defense', 'Stop Rate%']].rename(columns={'Stop Rate%': 'Stop Rate Week 15'})
            # Combining Both DataFrames to only plot team that are there in both data frames
            stoprate_combined = pd.merge(train_stop_rate_df, test_stop_rate_df, on='defense', how='outer')
            fig=plt.figure(figsize=(12, 6))
            plt.scatter(stoprate_combined['Stop Rate Week 8'], stoprate_combined['Stop Rate Week 15'], color='blue', label='Stop Rate')
            plt.xlabel("Stop Rate Week 8")
            plt.ylabel("Stop Rate Week 15")
            plt.title("Train vs Test Stop Rate")
            st.pyplot(fig)     

            st.markdown("### Numerical Correlation")
            st.table(stoprate_combined[['Stop Rate Week 8','Stop Rate Week 15']].corr().style.background_gradient(cmap='grey'))  
        elif choice=='Train ASR (VS) Test SR':
            df = load_and_preprocess_data(year, 15)
            train_df = df[df['week'] < 9]
            test_df = df[df['week'] >= 9]
            st.markdown(f"### Train ASR vs Test Stop Rate Comparison ({year} Season)")

            team_names = list(train_df['defense'].unique())

            # Train: Compute ASR
            train_model_df, SC = model_preprocess(train_df)
            asr_df = asr_model(train_model_df, team_names, SC, train_df)
            asr_df = asr_df.rename(columns={'Weighted ASR%': 'Weighted ASR Week 8'})

            # Test: Compute Stop Rate
            stop_rate_df = stoprate(test_df, team_names)
            stop_rate_df = stop_rate_df[stop_rate_df['Stops'] > 30]
            stop_rate_df = stop_rate_df[['defense', 'Stop Rate%']].rename(columns={'Stop Rate%': 'Stop Rate Week 15'})

            # Filter ASR to only include teams in test stop rate
            asr_df = asr_df[asr_df['defense'].isin(stop_rate_df['defense'])]

            # Merge for plotting
            combined_df = pd.merge(asr_df, stop_rate_df, on='defense', how='inner')

            fig = plt.figure(figsize=(12, 6))
            plt.scatter(combined_df['Weighted ASR Week 8'], combined_df['Stop Rate Week 15'], color='purple', label='ASR vs Stop Rate')
            plt.xlabel("Train WASR")
            plt.ylabel("Test SR")
            plt.title("Train WASR vs Test Stop Rate Comparison")
            st.pyplot(fig)
            st.markdown("### Numerical Correlation")
            st.table(combined_df[['Weighted ASR Week 8','Stop Rate Week 15']].corr().style.background_gradient(cmap='grey'))










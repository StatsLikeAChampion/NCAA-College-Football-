# NCAA Defensive Analysis Streamlit App (No Session State)

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
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
        df['score_diff'] = df['score_diff'].shift(1).fillna(0)
        df = df[~((df['start_period'] == 4) & (df['score_diff'] > 24))]
    return df

def stoprate(df, team_names):
    data = []
    for team in team_names:
        drives = df[df['defense'] == team]
        total = len(drives)
        stops = len(drives[drives['drive_result'] == 'Defended'])
        stop_rate = (stops / total) * 100 if total > 0 else 0
        data.append((team, total, stops, round(stop_rate, 2)))
    stop_rate_df = pd.DataFrame(data, columns=['defense', 'Total Drives', 'Stops', 'Stop Rate%'])
    return stop_rate_df.sort_values(by='Stop Rate%', ascending=False)

def compute_weekly_stop_rate(df, teams, selected_week):
    records = []
    for team in teams:
        for wk in range(1, selected_week + 1):
            drives = df[(df['defense'] == team) & (df['week'] == wk)]
            total = len(drives)
            stops = len(drives[drives['drive_result'] == 'Defended'])
            stop_rate = (stops / total) * 100 if total > 0 else 0
            records.append((team, wk, stop_rate))
    return pd.DataFrame(records, columns=['defense', 'week', 'Stop Rate%'])

def model_preprocess(df):
    df_model = df[['offense', 'defense', 'start_yards_to_goal', 'drive_result']].copy()
    df_model['drive_result'] = LabelEncoder().fit_transform(df_model['drive_result'])
    df_model = pd.get_dummies(df_model, columns=['offense', 'defense'], dtype=int)
    scaler = StandardScaler()
    df_model[['start_yards_to_goal']] = scaler.fit_transform(df_model[['start_yards_to_goal']])
    return df_model, scaler

def train_asr_model(df_model, teams, scaler):
    X = df_model.drop('drive_result', axis=1)
    y = df_model['drive_result']
    best_score = -1
    best_model = None
    for C in [1, 0.1, 0.01]:
        model = LogisticRegression(C=C, solver='newton-cg', max_iter=500)
        score = cross_val_score(model, X, y, cv=5).mean()
        if score > best_score:
            best_model = model
            best_score = score
    best_model.fit(X, y)
    test_data = [{'offense': o, 'defense': d, 'start_yards_to_goal': y} for o in teams for d in teams for y in range(10, 100, 10)]
    test_df = pd.DataFrame(test_data)
    test_df = pd.get_dummies(test_df, columns=['offense', 'defense'], dtype=int)
    test_df = test_df.reindex(columns=X.columns, fill_value=0)
    test_df['Yards to Goal'] = range(10, 100, 10) * len(teams)**2
    test_df['predicted_prob'] = 1 - best_model.predict_proba(test_df)[:, 1]
    return test_df

def compute_asr(test_df, raw_df, teams):
    bins = np.arange(0, 110, 10)
    counts = pd.cut(raw_df['start_yards_to_goal'], bins=bins).value_counts().sort_index()
    counts.index = [f"{i.left}-{i.right}" for i in counts.index]
    bin_counts = dict(zip(range(10, 100, 10), counts))
    result = []
    for team in teams:
        team_data = test_df[test_df.columns[test_df.columns.str.startswith('defense_' + team)]]
        if team_data.empty:
            continue
        prob = test_df[test_df[f'defense_{team}'] == 1]['predicted_prob']
        wasr = np.average(prob, weights=test_df['Yards to Goal'].map(lambda y: bin_counts.get(y, 1)))
        result.append((team, round(prob.mean() * 100, 2), round(wasr * 100, 2)))
    return pd.DataFrame(result, columns=['defense', 'ASR%', 'Weighted ASR%'])

def defense_analysis():
    st.title("🏈 NCAA Defensive Analytics (No Session State)")
    year = st.sidebar.selectbox("Select Year", list(range(2020, 2026)))
    week = st.sidebar.selectbox("Select Week", list(range(1, 16)))
    if st.sidebar.button("Fetch Data"):
        df = load_and_preprocess_data(year, week)
        if df.empty:
            st.warning("No data available.")
            return
        team_names = df['defense'].unique()
        tab = st.radio("Choose Analysis", ["Stop Rate", "Weekly Stop Rate", "ASR / WASR"])

        if tab == "Stop Rate":
            stop_df = stoprate(df, team_names)
            st.write(stop_df.head(10))

        elif tab == "Weekly Stop Rate":
            selected_teams = st.multiselect("Select Teams", options=team_names)
            if selected_teams:
                weekly_df = compute_weekly_stop_rate(df, selected_teams, week)
                for team in selected_teams:
                    team_data = weekly_df[weekly_df['defense'] == team]
                    plt.plot(team_data['week'], team_data['Stop Rate%'], marker='o', label=team)
                plt.xlabel("Week")
                plt.ylabel("Stop Rate %")
                plt.title("Weekly Stop Rate Trend")
                plt.legend()
                st.pyplot(plt)

        elif tab == "ASR / WASR":
            st.markdown("Training model...")
            model_df, scaler = model_preprocess(df)
            test_df = train_asr_model(model_df, team_names, scaler)
            asr_df = compute_asr(test_df, df, team_names)
            st.dataframe(asr_df)



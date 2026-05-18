import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="PaySafe Dashboard", layout="wide")

st.title("💳 PaySafe Fraud Intelligence Dashboard")

API_URL = "http://127.0.0.1:8000"

# ---------------- LOGIN ---------------- #

username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):
    response = requests.post(
        f"{API_URL}/login",
        params={"username": username, "password": password}
    )

    data = response.json()

    if response.status_code == 200 and "access_token" in data:
        st.session_state.token = data["access_token"]
        st.success("Logged in!")
    else:
        st.error("Invalid credentials")

# ---------------- DASHBOARD ---------------- #

if "token" in st.session_state:

    headers = {"token": st.session_state.token}
    response = requests.get(f"{API_URL}/transactions", headers=headers)

    data = response.json()

    if len(data) == 0:
        st.warning("No transactions found. Add some using /predict API.")
        st.stop()

    df = pd.DataFrame(data)

    # Fix timestamp
    df["timestamp"] = pd.to_datetime(df["id"], errors="coerce")

    # ---------------- OVERVIEW ---------------- #

    st.subheader("📊 Overview")

    frauds = df[df["fraud"] == 1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(df))
    col2.metric("Frauds Detected", len(frauds))
    col3.metric("Avg Risk Score", round(df["risk_score"].mean(), 3))

    # ---------------- FILTERS ---------------- #

    st.sidebar.header("🔍 Filters")

    selected_city = st.sidebar.selectbox(
        "Select City",
        ["All"] + list(df["city"].dropna().unique())
    )

    show_fraud_only = st.sidebar.checkbox("Show Fraud Only")

    filtered_df = df.copy()

    if selected_city != "All":
        filtered_df = filtered_df[filtered_df["city"] == selected_city]

    if show_fraud_only:
        filtered_df = filtered_df[filtered_df["fraud"] == 1]

    # ---------------- TABLE ---------------- #

    st.subheader("📋 Transactions Table")
    st.dataframe(filtered_df, use_container_width=True)

    # ---------------- CHARTS ---------------- #

    st.subheader("📈 Risk Score Distribution")

    fig1 = px.histogram(df, x="risk_score", nbins=20,title="Risk Score Distribution",color_discrete_sequence=["#636EFA"])
    st.plotly_chart(fig1, use_container_width=True)

    # ---------------- FRAUD BAR ---------------- #

    st.subheader("📊 Fraud vs Normal")

    fraud_counts = df["fraud"].value_counts().reset_index()
    fraud_counts.columns = ["fraud", "count"]
    
    fig2 = px.bar(df["fraud"].value_counts().reset_index(),title="Fraud vs Normal",color="fraud",
                  x="fraud", y="count")
    st.plotly_chart(fig2, use_container_width=True)
      # ---------------- TOP RISKY CITIES ---------------- #

    st.subheader("🌆 Top Risky Cities")

    city_risk = df.groupby("city")["risk_score"].mean().reset_index()

    fig3 = px.bar(city_risk.sort_values("risk_score", ascending=False).head(5),x="city",y="risk_score",title="Top 5 Risky Cities",color="risk_score")

    st.plotly_chart(fig3, use_container_width=True)

    # ---------------- TREND ---------------- #

    st.subheader("📅 Fraud Trend Over Time")

    trend = df.groupby(df["timestamp"].dt.date)["fraud"].sum()
    fig_trend = px.line(trend)

    import plotly.express as px
    fig_trend = px.line(trend, title="Fraud Transactions Over Time")

    st.plotly_chart(fig_trend, use_container_width=True)

    # ---------------- GAUGE ---------------- #

    st.subheader("🎯 Risk Gauge")

    avg_risk = df["risk_score"].mean()

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_risk,
        title={'text': "System Risk Level"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "red"},
            'steps': [
                 {'range': [0, 0.3], 'color': "green"},
                 {'range': [0.3, 0.7], 'color': "yellow"},
                 {'range': [0.7, 1], 'color': "red"}
             ],
        }
))

    st.plotly_chart(fig_gauge, use_container_width=True)
    # ---------------- HEATMAP ---------------- #

    st.subheader("🌍 Risk Heatmap (City Level)")

    city_risk = df.groupby("city")["risk_score"].mean().reset_index()

    fig_map = px.density_heatmap(city_risk,x="city",y="risk_score",title="City Risk Heatmap")

    st.plotly_chart(fig_map, use_container_width=True)

 # ---------------- Metrices ---------------- #
    import json

    with open("model_metrics.json") as f:
         metrics = json.load(f)

    st.subheader("🤖 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", round(metrics["accuracy"], 3))
    col2.metric("Precision", round(metrics["precision"], 3))
    col3.metric("Recall", round(metrics["recall"], 3))
    col4.metric("F1 Score", round(metrics["f1_score"], 3))

    # ---------------- DOWNLOAD ---------------- #

    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False),
        file_name="report.csv",
        mime="text/csv"
    )

    # ---------------- ALERTS ---------------- #

    st.subheader("🚨 Alerts")

    for _, row in df.iterrows():
        if row["fraud"] == 1 or row.get("anomaly", 0) == -1:
            st.error(
                f"Fraud Alert! ₹{row['amount']} in {row['city']}"
                f"(Risk: {round(row['risk_score'], 2)})"
            )
            
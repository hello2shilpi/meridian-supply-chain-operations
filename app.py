import streamlit as st
import pandas as pd
import plotly.express as px
import ollama

st.title("AI-Powered Demand Planning")

st.header("Sales Data")

df = pd.read_csv("data/sales_data.csv")

st.dataframe(df)

fig = px.line(
    df,
    x="Month",
    y="UnitsSold",
    title="Historical Demand Trend"
)

st.plotly_chart(fig)

avg_sales = df["UnitsSold"].mean()
forecast = round(avg_sales * 1.1)

st.header("Forecast")

st.success(
    f"Forecasted Demand Next Month: {forecast} units"
)

st.write(
    "Recommendation: Review inventory levels and replenish stock accordingly."
)

# --- LLM-generated insight ---
st.header("AI Insight")

with st.spinner("Analyzing demand pattern..."):
    prompt = f"""You are a supply chain demand planning assistant.
Historical monthly sales data:
{df.to_string(index=False)}

Calculated forecast for next month: {forecast} units.

In 3-4 sentences, summarize the demand trend, flag any risk
(stockout or excess inventory), and give one concrete recommendation."""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    insight = response["message"]["content"]

st.write(insight)
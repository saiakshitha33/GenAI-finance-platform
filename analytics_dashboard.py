import streamlit as st
import pandas as pd
import hashlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Set page config
st.set_page_config(page_title="📊 RAG Analytics Dashboard", layout="wide")

# Title and subtitle
st.title("📊 RAG-Powered Financial QA Analytics Dashboard")
st.markdown("Get insights into usage patterns, performance, and model behavior.")

# Load query logs
df = pd.read_csv("query_logs.csv", names=["timestamp", "question", "answer"])
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Top Questions
st.subheader("🔝 Top 10 Most Asked Questions")
top_qs = df["question"].value_counts().head(10)
st.bar_chart(top_qs)

# Volume Trends
st.subheader("📈 Daily Query Volume")
df["date"] = df["timestamp"].dt.date
query_volume = df.groupby("date").size()
st.line_chart(query_volume)

# Context Overlap
st.subheader("🔁 Duplicate Answer Patterns (Context Overlap)")
df["context"] = df["answer"]
df["context_hash"] = df["context"].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
context_overlap = df["context_hash"].value_counts().head(10)
st.bar_chart(context_overlap)

# Latency Analysis (Simulated)
st.subheader("⏱️ Inference Latency Simulation")
latency = pd.Series(np.random.normal(loc=1.1, scale=0.3, size=len(df)), name="Latency (seconds)")
st.line_chart(latency)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | RAG Q&A Analytics")

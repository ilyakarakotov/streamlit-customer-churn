import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

@st.cache
def load_data():
    return pd.read_csv("Telco-Customer-Churn.csv")

df = load_data()

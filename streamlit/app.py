import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

filename = "rf.pkl"
try:
    loaded_model = pickle.load(open(filename, "rb"))
except FileNotFoundError:
    st.error(f"Model file '{filename}' not found. Ensure it's in the same directory.")

csv_file = "clustered_customer_data.csv"
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    st.error(f"Dataset file '{csv_file}' not found. Ensure it's in the same directory.")

# custom CSS for light theme with light grey as dominant color
st.markdown("""
    <style>
        body {
            background-color: #F7F7F7;
            color: #000000;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #D1D3D4;
            color: black;
            font-size: 18px;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px;
        }
        .stTextInput>div>div>input {
            border: 2px solid #D1D3D4;
            background-color: #F7F7F7;
            border-radius: 5px;
            padding: 10px;
        }
        .stNumberInput>div>div>input {
            border: 2px solid #D1D3D4;
            background-color: #F7F7F7;
            border-radius: 5px;
            padding: 10px;
        }
        .stForm>div {
            margin-bottom: 15px;
        }
        .stTextArea>div>div>textarea {
            border: 2px solid #D1D3D4;
            background-color: #F7F7F7;
            border-radius: 5px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Customer Segmentation Prediction")
st.markdown(
    "<style>body{background-color: #f7f7f7;}</style>", unsafe_allow_html=True
)

input_method = st.radio("How would you like to input customer data?", 
                        ("Comma Separated List", "Individual Entries", "Sliders"))

if input_method == "Comma Separated List":
    st.subheader("Enter customer data as a comma-separated list")
    input_string = st.text_area(
        "Enter values for Balance, Balance Frequency, Purchases, Oneoff Purchases, Installments Purchases, Cash Advance, "
        "Purchases Frequency, Oneoff Purchases Frequency, Purchases Installments Frequency, Cash Advance Frequency, "
        "Cash Advance Transactions, Purchases Transactions, Credit Limit, Payments, Minimum Payments, Percentage Full Payment, Tenure"
        "(in this order, separated by commas):"
    )
    
    if input_string:
        try:
            input_data = [float(x.strip()) for x in input_string.split(",")]
        except ValueError:
            st.error("Please ensure all values are numeric and correctly separated by commas.")
    
elif input_method == "Individual Entries":
    st.subheader("Enter customer data manually")

    col1, col2, col3 = st.columns(3)

    with col1:
        balance = st.number_input("Balance", step=0.01, format="%.2f")
        purchases = st.number_input("Purchases", step=0.01, format="%.2f")
        cash_advance = st.number_input("Cash Advance", step=0.01, format="%.2f")
    
    with col2:
        balance_frequency = st.number_input("Balance Frequency", step=0.01, format="%.2f")
        oneoff_purchases = st.number_input("Oneoff Purchases", step=0.01, format="%.2f")
        purchases_frequency = st.number_input("Purchases Frequency", step=0.01, format="%.2f")
    
    with col3:
        installments_purchases = st.number_input("Installments Purchases", step=0.01, format="%.2f")
        cash_advance_frequency = st.number_input("Cash Advance Frequency", step=0.01, format="%.2f")
        oneoff_purchases_frequency = st.number_input("Oneoff Purchases Frequency", step=0.01, format="%.2f")

    col1, col2, col3 = st.columns(3)

    with col1:
        purchases_installments_frequency = st.number_input("Purchases Installments Frequency", step=0.01, format="%.2f")
        cash_advance_trx = st.number_input("Cash Advance Transactions", step=1)
        credit_limit = st.number_input("Credit Limit", step=0.01, format="%.2f")
    
    with col2:
        purchases_trx = st.number_input("Purchases Transactions", step=1)
        payments = st.number_input("Payments", step=0.01, format="%.2f")
        minimum_payments = st.number_input("Minimum Payments", step=0.01, format="%.2f")
    
    with col3:
        prc_full_payment = st.number_input("Percentage Full Payment", step=0.01, format="%.2f")
        tenure = st.number_input("Tenure", step=1)

    input_data = [
        balance, balance_frequency, purchases, oneoff_purchases, installments_purchases, cash_advance,
        purchases_frequency, oneoff_purchases_frequency, purchases_installments_frequency, cash_advance_frequency,
        cash_advance_trx, purchases_trx, credit_limit, payments, minimum_payments, prc_full_payment, tenure
    ]

elif input_method == "Sliders":
    st.subheader("Enter customer data using sliders")

    col1, col2, col3 = st.columns(3)

    with col1:
        balance = st.slider("Balance", min_value=0.0, max_value=10000.0, value=1000.0, step=10.0)
        purchases = st.slider("Purchases", min_value=0.0, max_value=1000.0, value=200.0, step=10.0)
        cash_advance = st.slider("Cash Advance", min_value=0.0, max_value=500.0, value=50.0, step=10.0)
    
    with col2:
        balance_frequency = st.slider("Balance Frequency", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        oneoff_purchases = st.slider("Oneoff Purchases", min_value=0.0, max_value=1000.0, value=150.0, step=10.0)
        purchases_frequency = st.slider("Purchases Frequency", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    
    with col3:
        installments_purchases = st.slider("Installments Purchases", min_value=0.0, max_value=1000.0, value=100.0, step=10.0)
        cash_advance_frequency = st.slider("Cash Advance Frequency", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        oneoff_purchases_frequency = st.slider("Oneoff Purchases Frequency", min_value=0.0, max_value=1.0, value=0.4, step=0.01)

    col1, col2, col3 = st.columns(3)

    with col1:
        purchases_installments_frequency = st.slider("Purchases Installments Frequency", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        cash_advance_trx = st.slider("Cash Advance Transactions", min_value=0, max_value=100, value=5, step=1)
        credit_limit = st.slider("Credit Limit", min_value=1000.0, max_value=50000.0, value=10000.0, step=100.0)
    
    with col2:
        purchases_trx = st.slider("Purchases Transactions", min_value=0, max_value=100, value=20, step=1)
        payments = st.slider("Payments", min_value=0.0, max_value=5000.0, value=1000.0, step=10.0)
        minimum_payments = st.slider("Minimum Payments", min_value=0.0, max_value=5000.0, value=500.0, step=10.0)
    
    with col3:
        prc_full_payment = st.slider("Percentage Full Payment", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
        tenure = st.slider("Tenure", min_value=1, max_value=12, value=6, step=1)

    input_data = [
        balance, balance_frequency, purchases, oneoff_purchases, installments_purchases, cash_advance,
        purchases_frequency, oneoff_purchases_frequency, purchases_installments_frequency, cash_advance_frequency,
        cash_advance_trx, purchases_trx, credit_limit, payments, minimum_payments, prc_full_payment, tenure
    ]

submitted = st.button("Predict")

if submitted and input_data is not None:
    cluster = loaded_model.predict([input_data])[0]
    st.markdown(f"""
    <div style="background-color:#E0E0E0; padding:10px; border-radius:5px; color:black; text-align:center;">
        <b>The customer belongs to Cluster {cluster}</b>
    </div>
""", unsafe_allow_html=True)
    cluster_df = df[df["cluster"] == cluster].copy()

    if 'cluster' not in cluster_df.columns:
        cluster_df['cluster'] = cluster

    fig, ax = plt.subplots(figsize=(8, 6))
    
    st.subheader("Pairplot of Key Features")
    important_columns = ['balance', 'purchases', 'oneoff_purchases', 'installments_purchases', 'cash_advance', 
                         'purchases_frequency', 'cash_advance_trx', 'purchases_trx', 'credit_limit']
    pairplot = sns.pairplot(cluster_df[important_columns], diag_kind="scatter", plot_kws={'color': "#C15146"})
    fig = pairplot.fig
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    corr_matrix = cluster_df[important_columns].corr()
    mask = np.triu(np.ones(corr_matrix.shape))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, mask=mask, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    st.subheader(f"Feature Distributions for Cluster {cluster}")
    for col in important_columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(cluster_df[col], kde=True, bins=20, ax=ax, color='#FFE0CA')
        st.pyplot(fig)

    st.subheader("Cluster Size Distribution")
    cluster_sizes = df['cluster'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values, hue=cluster_sizes.index, palette="coolwarm", ax=ax, legend=False)
    st.pyplot(fig)

    st.subheader("Balance Distribution by Cluster")
    fig, ax = plt.subplots()
    sns.boxplot(x="cluster", y="balance", data=df, hue="cluster", palette="coolwarm", ax=ax, legend=False)
    st.pyplot(fig)
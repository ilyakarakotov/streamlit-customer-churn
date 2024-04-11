import pickle as pickle

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.impute import SimpleImputer


def get_clean_data():
    data = pd.read_csv('data/Telco-Customer-Churn.csv')

    # replace empty values with NaN
    data['TotalCharges'] = data['TotalCharges'].replace(' ', pd.NA)

    # convert TotalCharges to all numeric values
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])

    # Impute missing values in TotalCharges with median
    imputer = SimpleImputer(strategy='median')
    data[['TotalCharges']] = imputer.fit_transform(data[['TotalCharges']])

    # drop customerID column
    data = data.drop(columns=['customerID'])

    # convert Churn column to binary so ML can understand it
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

    # save the data before one hot encoding for use in slider labels
    data_before_encoding = data.copy()

    # one hot encode the data
    data = pd.get_dummies(data)

    st.write(data)

    return data, data_before_encoding


def add_sidebar(data, data_before_encoding):
    st.sidebar.header("Customer Information")

    input_labels = [
        ('Gender', 'gender'),
        ('Senior Citizen', 'SeniorCitizen'),
        ('Partner', 'Partner'),
        ('Dependents', 'Dependents'),
        ('Tenure (Months)', 'tenure'),
        ('Phone Service', 'PhoneService'),
        ('Multiple Lines', 'MultipleLines'),
        ('Internet Service', 'InternetService'),
        ('Online Security', 'OnlineSecurity'),
        ('Online Backup', 'OnlineBackup'),
        ('Device Protection', 'DeviceProtection'),
        ('Tech Support', 'TechSupport'),
        ('Streaming TV', 'StreamingTV'),
        ('Streaming Movies', 'StreamingMovies'),
        ('Contract', 'Contract'),
        ('Paperless Billing', 'PaperlessBilling'),
        ('Payment Method', 'PaymentMethod'),
        ('Monthly Charges (USD)', 'MonthlyCharges'),
        ('Total Charges (USD)', 'TotalCharges')
    ]

    input_dict = {}

    for label, key in input_labels:
        if data_before_encoding[key].dtype == 'object':  # if the column is categorical
            unique_values = data_before_encoding[key].unique()
            selected_value = st.sidebar.selectbox(label, unique_values, key=key)
        elif key == 'SeniorCitizen':
            unique_values = ['No', 'Yes']
            selected_value = st.sidebar.selectbox(label, unique_values, key=key)
        else:  # if the column is numerical
            selected_value = st.sidebar.slider(
                label,
                min_value=float(0),
                max_value=float(data_before_encoding[key].max()),
                value=float(data_before_encoding[key].mean()), key=key)
        input_dict[key] = [selected_value]  # Wrap the selected value in a list

    # Create a DataFrame from the input_dict
    input_df = pd.DataFrame(input_dict)

    # One-hot encode the DataFrame
    encoded_input = pd.get_dummies(input_df)

    # Ensure that the encoded_input has the same columns as the training data
    missing_cols = set(data.columns) - set(encoded_input.columns)
    for c in missing_cols:
        encoded_input[c] = 0

    # Ensure the order of column in the test query is in the same order than in train set
    encoded_input = encoded_input[data.columns]

    encoded_input = encoded_input.drop(columns=['Churn'])

    return encoded_input


def get_scaled_values(data):
    categories = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # Find mean values for churned and non-churned customers
    mean_values_churned = data[data['Churn'] == 1][categories].mean()
    mean_values_not_churned = data[data['Churn'] == 0][categories].mean()

    # Normalize the data for each category
    for category in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        max_value = data[category].max()
        min_value = data[category].min()
        range_value = max_value - min_value

        mean_values_churned[category] = (mean_values_churned[category] - min_value) / range_value
        mean_values_not_churned[category] = (mean_values_not_churned[category] - min_value) / range_value

    return mean_values_churned, mean_values_not_churned


def get_radar_chart(input_data, data):
    # categories for the radar chart
    categories = ['tenure', 'MonthlyCharges', 'TotalCharges']

    mean_values_churned, mean_values_not_churned = get_scaled_values(data)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=mean_values_churned.tolist(),
        theta=categories,
        fill='toself',
        name='Churned'
    ))
    fig.add_trace(go.Scatterpolar(
        r=mean_values_not_churned.tolist(),
        theta=categories,
        fill='toself',
        name='Not Churned'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, .75]
            )),
        showlegend=True
    )

    return fig


def add_predictions(input_data):
    model = pickle.load(open('model/model.pkl', 'rb'))
    scaler = pickle.load(open('model/scaler.pkl', 'rb'))

    input_array = input_data.values.reshape(1, -1)

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.write(prediction)


def main():
    st.set_page_config(
        page_title="Churn Prediction App",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    data, data_before_encoding = get_clean_data()

    input_data = add_sidebar(data, data_before_encoding)
    st.write(input_data)

    with st.container():
        st.title("Churn Prediction App")
        st.write("This app uses machine learning to predict customer churn, so you can see if a customer is at high "
                 "risk of leaving your business.")

    col1, col2 = st.columns([4, 1])  # first column will be 4 times wider than the second column

    with col1:
        radar_chart = get_radar_chart(input_data, data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)


if __name__ == '__main__':
    main()

import pickle as pickle

import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt


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

    return data, data_before_encoding


def add_sidebar(data, data_before_encoding):
    st.sidebar.header("Customer Analysis")

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
            selected_value = 1 if selected_value == 'Yes' else 0
        else:  # if the column is numerical
            selected_value = st.sidebar.slider(
                label,
                min_value=int(0),
                max_value=int(data_before_encoding[key].max()),
                value=int(data_before_encoding[key].mean()),
                step=int(1),
                key=key)
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


def get_altair_chart(data, input_data):
    # Group data by 'tenure' and count the number of churned customers in each group
    churned_data = data[data['Churn'] == 1].groupby('tenure').size().reset_index(name='Churned')
    not_churned_data = data[data['Churn'] == 0].groupby('tenure').size().reset_index(name='Churned')

    # Create a selection that is active on the line chart
    brush = alt.selection_interval(encodings=["x"])

    # Line chart of churned customers over time
    churn_chart = alt.Chart(churned_data).mark_line(color='#ff4c4b').encode(
        x=alt.X('tenure', title='Tenure (Months)'),
        y='Churned'
    ).properties(
        width=550,
        height=300
    ).add_selection(
        brush
    )

    not_churn_chart = alt.Chart(not_churned_data).mark_line(color='#479ce2').encode(
        x='tenure',
        y='Churned'
    ).properties(
        width=550,
        height=300
    ).add_selection(
        brush
    )

    # Add line at the place where customer information is chosen
    customer_tenure_df = pd.DataFrame({'tenure': [input_data['tenure'].values[0]]})
    customer_tenure_line = alt.Chart(customer_tenure_df).mark_rule(color='#22bb45').encode(x='tenure')

    # Layer tenure line to be in the front
    chart = alt.layer(churn_chart, not_churn_chart, customer_tenure_line)

    return chart


def add_predictions(input_data):
    model = pickle.load(open('model/model.pkl', 'rb'))
    scaler = pickle.load(open('model/scaler.pkl', 'rb'))

    input_array = input_data.values.reshape(1, -1)

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("Churn Prediction")
    st.write("Risk of customer churning:")
    if prediction == 1:
        st.write("<span class='risk high'>High</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='risk low'>Low</span>", unsafe_allow_html=True)

    st.write("The probability of customer churning is: ", model.predict_proba(input_array_scaled)[0][1])
    st.write("The probability of customer not churning is: ", model.predict_proba(input_array_scaled)[0][0])


def get_correlation_matrix(data):
    # Select numerical columns
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # Compute correlation matrix
    corr_matrix = data[numerical_cols].corr()

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.1)
    plt.title('Correlation Matrix')

    return plt


def get_stacked_bar_charts(data):
    # Create columns optimized for bar charts
    data['Gender'] = data['gender_Female'].replace({True: 'Female', False: 'Male'})
    data['Partner'] = data['Partner_Yes'].replace({True: 'Yes', False: 'No'})
    data['Dependents'] = data['Dependents_Yes'].replace({True: 'Yes', False: 'No'})
    data['Senior Citizen'] = data['SeniorCitizen'].replace({1: 'Yes', 0: 'No'})

    columns_to_plot = ['Senior Citizen', 'Gender', 'Partner', 'Dependents']

    num_columns = 2
    num_rows = 2

    # create a figure
    fig = plt.figure(figsize=(12, 5 * num_rows))
    fig.suptitle('Churn Proportions Per Category', fontsize=22, y=.95)

    # loop to each column name to create a subplot
    for index, column in enumerate(columns_to_plot, 1):

        # create a subplot
        ax = fig.add_subplot(num_rows, num_columns, index)

        # converts columns to what percentage of the total each category represents, adding up to 100%
        prop_by_independent = pd.crosstab(data[column], data['Churn']).apply(lambda x: x / x.sum() * 100, axis=1)
        prop_by_independent.plot(kind='bar', ax=ax, stacked=True, rot=0, color=['#22bb45', '#ff4c4b'])

        # set the legend in the upper right corner
        ax.legend(loc="upper right", bbox_to_anchor=(0.62, 0.5, 0.5, 0.5), title='Churn', fancybox=True)

        # set title and labels
        ax.set_title('Proportion of observations by ' + column, fontsize=16, loc='left')
        ax.tick_params(rotation='auto')

        # remove frame from the plot
        spine_names = ('top', 'right', 'bottom', 'left')
        for spine_name in spine_names:
            ax.spines[spine_name].set_visible(False)

    return fig


def main():
    st.set_page_config(
        page_title=" Customer Churn Prediction App",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    data, data_before_encoding = get_clean_data()

    input_data = add_sidebar(data, data_before_encoding)

    with st.container():
        st.title("Churn Prediction App")
    col1, col2 = st.columns([4, 1])  # first column will be 4 times wider than the second column

    # initialization of graphs
    altair_chart = get_altair_chart(data, input_data)
    bar_charts = get_stacked_bar_charts(data)
    correlation_matrix = get_correlation_matrix(data)

    with col1:
        st.write(
            "This app uses <span title='Logistic regression is a supervised machine learning algorithm used for "
            "classification tasks where the goal is to predict the probability that an instance belongs to a given "
            "class or not.'>Logistic Regression</span> to predict customer churn, so you can see if a customer "
            "is at <span style='color:#ff4c4b;'>high risk</span> of leaving your business. This information grants "
            "you an opportunity to take action in <span style='color:#22bb45;'>lowering risk</span> that a customer "
            "leaves. Having customers for a longer time means that the customer lifetime value (CLV) may be increase, "
            "<span style='color:#6591e7;'>boosting business</span> revenue.", unsafe_allow_html=True)
        st.write("")
        st.write("")
        st.write("The <span style='color:#ff4c4b;'>red line</span> represents the number of churned customers over "
                 "time, while the <span style='color:#6591e7;'>blue line</span> represents the number of customers who "
                 "did not churn. To see where <span style='color:#22bb45;'>your customer</span> falls in the data, "
                 "adjust the tenure slider.", unsafe_allow_html=True)
        st.write("")
        st.write("")
        st.altair_chart(altair_chart, use_container_width=True)

    with col2:
        add_predictions(input_data)
        st.write("For more useful metrics, expand the graphs below 👇")

        st.pyplot(bar_charts)
        st.pyplot(correlation_matrix)


if __name__ == '__main__':
    main()

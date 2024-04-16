import pandas as pd
import pickle as pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def create_model(data):
    # split the data into X and y
    X = data.drop(['Churn'], axis=1)
    y = data['Churn']

    # scales data so all values will be on the same level
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split the data. 80% for training, 20% for testing. Random state is random number
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # test model
    y_pred = model.predict(X_test)
    print("Accuracy of our model: ",
          accuracy_score(y_test, y_pred))  # y_test are the actual values, y_pred are the predicted values
    print("Classification report: \n", classification_report(y_test, y_pred))  # more in-depth information

    return model, scaler, X_train, X_test, y_train, y_test


def get_clean_data():
    data = pd.read_csv('data/customer-churn.csv')

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

    # one hot encode the data
    data = pd.get_dummies(data)

    print(data.columns.tolist())

    return data


def main():
    data = get_clean_data()

    model, scaler, X_train, X_test, y_train, y_test = create_model(data)

    # wb = write binary
    with open('../model/model.pkl', 'wb') as file:
        pickle.dump(model, file)
    with open('../model/scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    # Save the train-test split data
    with open('../model/X_train.pkl', 'wb') as file:
        pickle.dump(X_train, file)
    with open('../model/X_test.pkl', 'wb') as file:
        pickle.dump(X_test, file)
    with open('../model/y_train.pkl', 'wb') as file:
        pickle.dump(y_train, file)
    with open('../model/y_test.pkl', 'wb') as file:
        pickle.dump(y_test, file)


if __name__ == "__main__":
    main()

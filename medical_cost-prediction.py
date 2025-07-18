import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import streamlit as st
import os

# Title for the Streamlit app
st.title('Medical Cost Prediction 💰')

# File upload widget
uploaded_file = st.file_uploader("Upload a CSV file", type='csv')
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(data.head())
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # Encode categorical variables
    data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['age', 'bmi', 'children']
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Features (X) and target variable (y)
    X = data.drop(columns=['charges'])
    y = data['charges']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and save the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # User Inputs for Prediction
    st.subheader('Enter Your Details for Prediction 📊')
    age = st.slider("Age", 18, 100, 30)
    bmi = st.slider("BMI", 10, 50, 25)
    children = st.slider("Children", 0, 10, 1)
    sex = st.selectbox("Sex", ["Male", "Female"])
    smoker = st.selectbox("Smoker", ["Yes", "No"])
    region = st.selectbox("Region", ["Northwest", "Northeast", "Southeast", "Southwest"])

    # Prepare input for model
    user_input = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'smoker_yes': [1 if smoker == "Yes" else 0],
        'sex_male': [1 if sex == "Male" else 0],
        'region_northwest': [1 if region == "Northwest" else 0],
        'region_southeast': [1 if region == "Southeast" else 0],
        'region_southwest': [1 if region == "Southwest" else 0]
    })

    # Ensure column order matches training data
    user_input = user_input.reindex(columns=X_train.columns, fill_value=0)

    # Scale input features
    user_input[numerical_cols] = scaler.transform(user_input[numerical_cols])

    # Prediction
    if st.button("Predict Cost 💵"):
        try:
            prediction = model.predict(user_input)
            st.success(f"Predicted Medical Cost: ${prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

    # Model Evaluation
    y_pred = model.predict(X_test)
    st.subheader('Model Performance 📈')
    st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):,.2f}")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):,.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

    # Visualizations
    st.subheader('Visualizations 📊')
    plot_option = st.selectbox('Choose a visualization:',
                               ['Distribution of Charges', 'Charges by Smoking Status', 'BMI vs Charges',
                                'Age Distribution', 'Residuals Distribution', 'Actual vs Predicted'])

    if plot_option == 'Distribution of Charges':
        fig, ax = plt.subplots()
        sns.histplot(data['charges'], kde=True, ax=ax)
        ax.set_title('Distribution of Charges')
        st.pyplot(fig)

    elif plot_option == 'BMI vs Charges':
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x='bmi', y='charges', ax=ax)
        ax.set_title('BMI vs Charges')
        st.pyplot(fig)

    elif plot_option == 'Charges by Smoking Status':
        fig, ax = plt.subplots()
        sns.histplot(data=data, x='charges', hue='smoker_yes', kde=True, multiple='stack', ax=ax)
        ax.set_title('Distribution of Charges by Smoking Status')
        st.pyplot(fig)

    elif plot_option == 'Age Distribution':
        fig, ax = plt.subplots()
        sns.histplot(data=data, x='age', kde=True, ax=ax)
        ax.set_title('Distribution of Age')
        st.pyplot(fig)

    elif plot_option == 'Residuals Distribution':
        residuals = y_test - y_pred
        fig, ax = plt.subplots()
        sns.histplot(residuals, kde=True, color='purple', ax=ax)
        ax.set_title('Distribution of Residuals')
        st.pyplot(fig)

    elif plot_option == 'Actual vs Predicted':
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, color='blue', alpha=0.6)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        ax.set_title('Actual vs Predicted Charges')
        st.pyplot(fig)

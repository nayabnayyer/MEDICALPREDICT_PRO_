import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import streamlit as st
import os

# Title for the Streamlit app
st.title('Medical Cost Prediction 💰 using Random Forest 🌳🔥')

# File upload widget
uploaded_file = st.file_uploader("Upload a CSV file", type='csv')
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(data.head())
        st.write("Dataset Statistics:")
        st.write(data.describe())
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # Detect Outliers using IQR 🔍
    Q1, Q3 = data['charges'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    data['charges'] = np.where(data['charges'] > upper_limit, upper_limit, data['charges'])

    # Encode categorical variables
    data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)

    # Feature Engineering
    data['bmi_age_interaction'] = data['bmi'] * data['age']
    data['bmi_squared'] = data['bmi'] ** 2

    # Define all numerical columns (including the new ones)
    numerical_cols = ['age', 'bmi', 'children', 'bmi_age_interaction', 'bmi_squared']

    # Apply StandardScaler on all numerical features
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Features and Target
    X = data.drop(columns=['charges'])
    y = np.log(data['charges'])  # Apply log transformation

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load or Train Model
    model_path = "random_forest_model.pkl"
    if os.path.exists(model_path):
        best_model = joblib.load(model_path)
        st.success("Pre-trained model loaded ✅")
    else:
        st.warning("Training a new model, this may take a while ⏳")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='r2')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        joblib.dump(best_model, model_path)
        st.success("Model trained and saved! 🎉")

    # Model Performance
    y_pred = best_model.predict(X_test)
    st.subheader('Model Performance 📈')
    st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):,.2f}")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):,.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

    # Prediction Inputs
    st.subheader('Enter Your Details for Prediction 📊')
    age = st.slider("Age", 18, 100, 30)
    bmi = st.slider("BMI", 10, 50, 25)
    children = st.slider("Children", 0, 10, 1)
    sex = st.selectbox("Sex", ["Male", "Female"])
    smoker = st.selectbox("Smoker", ["Yes", "No"])
    region = st.selectbox("Region", ["Northwest", "Northeast", "Southeast", "Southwest"])

    # Prepare input
    user_input = pd.DataFrame({
        'age': [age], 'bmi': [bmi], 'children': [children],
        'smoker_yes': [1 if smoker == "Yes" else 0], 'sex_male': [1 if sex == "Male" else 0],
        'region_northwest': [1 if region == "Northwest" else 0],
        'region_southeast': [1 if region == "Southeast" else 0],
        'region_southwest': [1 if region == "Southwest" else 0]
    })

    # Feature Engineering for user input
    user_input['bmi_age_interaction'] = user_input['bmi'] * user_input['age']
    user_input['bmi_squared'] = user_input['bmi'] ** 2

    # Apply the same scaling transformation
    user_input[numerical_cols] = scaler.transform(user_input[numerical_cols])

    # Ensure all columns match the trained model
    user_input = user_input.reindex(columns=X_train.columns, fill_value=0)

    # Predict
    if st.button("Predict Cost 💵"):
        with st.spinner("Predicting... ⏳"):
            prediction = np.exp(best_model.predict(user_input))  # Reverse log transformation
            st.success(f"Predicted Medical Cost: ${prediction[0]:,.2f}")

    # Feature Importance
    st.subheader("Feature Importance 🌟")
    importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.write(importance)

    # Visualization
    st.subheader('Visualizations 📊')
    plot_option = st.selectbox('Choose a visualization:',
                               ['Distribution of Charges', 'Charges by Smoking Status', 'BMI vs Charges',
                                'Age Distribution', 'Residuals Distribution', 'Actual vs Predicted'])
    fig, ax = plt.subplots()

    if plot_option == 'Distribution of Charges':
        sns.histplot(data['charges'], kde=True, ax=ax)
        ax.set_title('Distribution of Charges')

    elif plot_option == 'BMI vs Charges':
        sns.scatterplot(data=data, x='bmi', y='charges', hue='smoker_yes', ax=ax)
        ax.set_title('BMI vs Charges')

    elif plot_option == 'Charges by Smoking Status':
        sns.boxplot(x='smoker_yes', y='charges', data=data, ax=ax)
        ax.set_title('Charges by Smoking Status')

    elif plot_option == 'Age Distribution':
        sns.histplot(data['age'], kde=True, ax=ax)
        ax.set_title('Age Distribution')

    elif plot_option == 'Residuals Distribution':
        residuals = y_test - y_pred
        sns.histplot(residuals, kde=True, color='purple', ax=ax)
        ax.set_title('Residuals Distribution')

    elif plot_option == 'Actual vs Predicted':
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        ax.set_title('Actual vs Predicted Charges')

    plt.tight_layout()
    st.pyplot(fig)

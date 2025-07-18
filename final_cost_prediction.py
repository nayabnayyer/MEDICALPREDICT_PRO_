import pandas as pd
import seaborn as sns
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
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # Detect Outliers using IQR 🔍
    Q1 = data['charges'].quantile(0.25)
    Q3 = data['charges'].quantile(0.75)
    IQR = Q3 - Q1

    outliers = data[(data['charges'] < Q1 - 1.5 * IQR) | (data['charges'] > Q3 + 1.5 * IQR)]
    st.write(f"Number of Outliers Detected: {len(outliers)}")

    st.write("Before Capping Outliers:")
    st.write(outliers)

    if len(outliers) > 0:
        st.warning("Outliers detected 🚨❗ Capping the outliers to make the model more accurate 🔥")
        upper_limit = Q3 + 1.5 * IQR
        data['charges'] = data['charges'].apply(lambda x: upper_limit if x > upper_limit else x)
        st.success("Outliers have been capped ✂️📌")
    else:
        st.write("No Outliers Found ✅")

    st.write("After Capping Outliers:")
    st.write(data.head())

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

    # Hyperparameter Tuning using GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    st.subheader("🔥 Best Parameters Found Using GridSearchCV")
    st.write(grid_search.best_params_)

    # Cross Validation Scoring
    cross_val_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
    st.subheader("Cross-Validation R² Scores 📌")
    st.write(cross_val_scores)
    st.write(f"Average Cross-Validation R² Score: {cross_val_scores.mean():.2f}")

    # Save model
    joblib.dump(best_model, "random_forest_model.pkl")

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
            prediction = best_model.predict(user_input)
            st.success(f"Predicted Medical Cost: ${prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

    # Model Evaluation
    y_pred = best_model.predict(X_test)
    st.subheader('Model Performance 📈')
    st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):,.2f}")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):,.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

    # Feature Importance 🔥
    st.subheader("Feature Importance 🌟")
    importance = pd.Series(best_model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)
    st.write(importance)

    # Visualizations
    st.subheader('Visualizations 📊')
    plot_option = st.selectbox('Choose a visualization:',
                               ['Distribution of Charges', 'Charges by Smoking Status',
                                'BMI vs Charges', 'Age Distribution',
                                'Residuals Distribution', 'Actual vs Predicted'])

    if plot_option == 'Distribution of Charges':
        fig, ax = plt.subplots()
        sns.histplot(data['charges'], kde=True, ax=ax)
        ax.set_title('Distribution of Charges')
        st.pyplot(fig)

    elif plot_option == 'BMI vs Charges':
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x='bmi', y='charges', hue='smoker_yes', ax=ax)
        ax.set_title('BMI vs Charges')
        st.pyplot(fig)

    elif plot_option == 'Charges by Smoking Status':
        fig, ax = plt.subplots()
        sns.boxplot(x='smoker_yes', y='charges', data=data, ax=ax)
        ax.set_title('Charges by Smoking Status')
        st.pyplot(fig)

    elif plot_option == 'Age Distribution':
        fig, ax = plt.subplots()
        sns.histplot(data['age'], kde=True, ax=ax)
        ax.set_title('Age Distribution')
        st.pyplot(fig)

    elif plot_option == 'Residuals Distribution':
        residuals = y_test - y_pred
        fig, ax = plt.subplots()
        sns.histplot(residuals, kde=True, color='purple', ax=ax)
        ax.set_title('Residuals Distribution')
        st.pyplot(fig)

    elif plot_option == 'Actual vs Predicted':
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, color='blue', alpha=0.6)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        ax.set_title('Actual vs Predicted Charges')
        st.pyplot(fig)



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import streamlit as st
import os

st.title('Medical Cost Prediction 💰 using Random Forest 🌳🔥')

uploaded_file = st.file_uploader("Upload a CSV file", type='csv')
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(data.head())
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    if 'charges' not in data.columns:
        st.error("Dataset must contain a 'charges' column for prediction.")
        st.stop()

    # Save original for UI sliders/dropdowns
    original_data = data.copy()

    # Drop missing values from target
    data.dropna(subset=['charges'], inplace=True)

    # Outlier Detection & Capping
    Q1 = data['charges'].quantile(0.25)
    Q3 = data['charges'].quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    data['charges'] = data['charges'].apply(lambda x: upper_limit if x > upper_limit else x)

    # Feature separation
    target_col = 'charges'
    feature_cols = [col for col in data.columns if col != target_col]
    categorical_cols = data[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = data[feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Encoding & Scaling
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Split features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model tuning
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    rf_model = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=50, cv=5, scoring='r2', random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    st.subheader("🔥 Best Parameters Found Using RandomizedSearchCV")
    st.write(random_search.best_params_)

    cross_val_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
    st.subheader("Cross-Validation R² Scores 📌")
    st.write(cross_val_scores)
    st.write(f"Average Cross-Validation R² Score: {cross_val_scores.mean():.2f}")

    joblib.dump(best_model, "random_forest_model.pkl")

    st.subheader('Enter Your Details for Prediction 📊')

    user_inputs = {}
    for col in numerical_cols:
        if col.lower() == 'age':
            user_inputs[col] = st.slider("Age", min_value=17, max_value=90, value=30)
        elif col.lower() == 'bmi':
            user_inputs[col] = st.slider("BMI", min_value=14.0, max_value=40.0, value=25.0)
        elif col.lower() == 'children':
            user_inputs[col] = st.slider("Children", min_value=0, max_value=20, value=1)
        else:
            col_min = float(original_data[col].min())
            col_max = float(original_data[col].max())
            default_val = float(original_data[col].median())
            user_inputs[col] = st.slider(f"{col.capitalize()}", min_value=round(col_min, 2), max_value=round(col_max, 2), value=round(default_val, 2))

    for col in categorical_cols:
        options = original_data[col].dropna().unique().tolist()
        selected = st.selectbox(f"{col.capitalize()}", options)
        for opt in options:
            user_inputs[f"{col}_{opt}"] = 1 if selected == opt else 0

    user_input_df = pd.DataFrame([user_inputs])
    user_input_df = user_input_df.reindex(columns=X_train.columns, fill_value=0)
    user_input_df[numerical_cols] = scaler.transform(user_input_df[numerical_cols])

    if st.button("Predict Cost 💵"):
        try:
            prediction = best_model.predict(user_input_df)
            st.success(f"Predicted Medical Cost: Rs.{prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

    y_pred = best_model.predict(X_test)
    st.subheader('Model Performance 📈')
    st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):,.2f}")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):,.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

    st.subheader("Feature Importance 🌟")
    importance = pd.Series(best_model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)
    st.write(importance)

    st.subheader('Visualizations 📊')
    plot_option = st.selectbox('Choose a visualization:',
                               ['Distribution of Charges', 'Residuals Distribution', 'Actual vs Predicted'])

    if plot_option == 'Distribution of Charges':
        fig, ax = plt.subplots()
        sns.histplot(y, kde=True, ax=ax)
        ax.set_title('Distribution of Charges')
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

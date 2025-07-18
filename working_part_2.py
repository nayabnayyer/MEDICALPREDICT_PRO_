# existing imports remain...
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import streamlit as st
from sklearn.preprocessing import PolynomialFeatures

# Title for the Streamlit app
st.title('📊 Medical Cost Prediction using Enhanced Random Forest 🌳')

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV only)", type='csv')

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("Data loaded successfully!")
        st.write(data.head())
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # Define numerical columns explicitly
    numerical_cols = ['age', 'bmi', 'children']

    # Verify numerical columns exist in data
    numerical_cols = [col for col in numerical_cols if col in data.columns]
    if not numerical_cols:
        st.error("No numerical columns found for processing")
        st.stop()

    # Cap Outliers in Charges
    if 'charges' in data.columns:
        Q1, Q3 = data['charges'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        upper_limit = Q3 + 1.5 * IQR
        data['charges'] = data['charges'].apply(lambda x: min(x, upper_limit))
    else:
        st.error("'charges' column not found in dataset")
        st.stop()

    # Encode Categorical Features
    categorical_cols = []
    if 'sex' in data.columns:
        categorical_cols.append('sex')
    if 'smoker' in data.columns:
        categorical_cols.append('smoker')
    if 'region' in data.columns:
        categorical_cols.append('region')

    if categorical_cols:
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # BMI Category
    if 'bmi' in data.columns:
        data['bmi_category'] = pd.cut(data['bmi'],
                                    bins=[0, 18.5, 24.9, 29.9, 60],
                                    labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        data = pd.get_dummies(data, columns=['bmi_category'], drop_first=True)

    # Scale and Transform Features
    scaler = StandardScaler()
    pt = PowerTransformer()  # Box-Cox or Yeo-Johnson

    # Only scale columns that exist
    existing_num_cols = [col for col in numerical_cols if col in data.columns]
    if existing_num_cols:
        data[existing_num_cols] = scaler.fit_transform(data[existing_num_cols])

    if 'charges' in data.columns:
        data[['charges']] = pt.fit_transform(data[['charges']])

    # Feature Engineering - Polynomial Features
    if existing_num_cols:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(data[existing_num_cols])
        poly_df = pd.DataFrame(poly_features,
                             columns=poly.get_feature_names_out(existing_num_cols))
        data = pd.concat([data.reset_index(drop=True), poly_df], axis=1)
        data.drop(columns=existing_num_cols, inplace=True)

    if 'charges' not in data.columns:
        st.error("Target variable 'charges' not found after preprocessing")
        st.stop()

    X = data.drop(columns=['charges'])
    y = data['charges']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Tuning
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [None, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 0.8]
    }

    rf = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='r2')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    st.subheader("Best Hyperparameters")
    st.write(grid.best_params_)

    # Cross-validation score
    cv_score = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
    st.subheader("Cross-Validation R²")
    st.write(cv_score)
    st.write(f"Average R²: {cv_score.mean():.4f}")

    # Save the model
    joblib.dump(best_model, "rf_model_final.pkl")

    # Evaluate
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    st.subheader("📈 Model Evaluation")
    st.write(f"R² Score: {r2:.4f}")
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    st.write(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")

    # Visualizations
    st.subheader("Visual Insights")

    # Determine available features for visualization
    plot_options = ['Distribution of Charges', 'Actual vs Predicted', 'Residuals Distribution']

    if 'bmi' in data.columns or any('bmi' in col for col in data.columns):
        plot_options.append('BMI vs Charges')
    if 'smoker_yes' in data.columns:
        plot_options.append('Charges by Smoking Status')
    if 'age' in data.columns or any('age' in col for col in data.columns):
        plot_options.append('Age Distribution')

    plot_option = st.selectbox('Choose a visualization:', plot_options)

    if plot_option == 'Distribution of Charges':
        fig, ax = plt.subplots()
        sns.histplot(data['charges'], kde=True, ax=ax)
        ax.set_title('Distribution of Charges')
        st.pyplot(fig)

    elif plot_option == 'BMI vs Charges':
        fig, ax = plt.subplots()
        bmi_col = 'bmi' if 'bmi' in data.columns else [col for col in data.columns if 'bmi' in col][0]
        hue_col = 'smoker_yes' if 'smoker_yes' in data.columns else None
        sns.scatterplot(data=data, x=bmi_col, y='charges', hue=hue_col, ax=ax)
        ax.set_title('BMI vs Charges')
        st.pyplot(fig)

    elif plot_option == 'Charges by Smoking Status':
        fig, ax = plt.subplots()
        sns.boxplot(x='smoker_yes', y='charges', data=data, ax=ax)
        ax.set_title('Charges by Smoking Status')
        st.pyplot(fig)

    elif plot_option == 'Age Distribution':
        fig, ax = plt.subplots()
        age_col = 'age' if 'age' in data.columns else [col for col in data.columns if 'age' in col][0]
        sns.histplot(data[age_col], kde=True, ax=ax)
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
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
st.title('📊 Universal Medical Cost Prediction using Enhanced Random Forest 🌳')

# File upload
uploaded_file = st.file_uploader("Upload your medical cost dataset (CSV only)", type='csv')

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        
        st.success("Data loaded successfully!")

        # Show basic info about the dataset
        st.subheader("Dataset Overview")
        st.write(f"Number of records: {len(data)}")
        st.write("First 5 rows:")
        st.write(data.head())

        # Check for required columns
        required_cols = {'age', 'bmi', 'charges'}  # Minimum required
        optional_cols = {'sex', 'smoker', 'region', 'children'}  # Common additional

        missing_required = required_cols - set(data.columns)
        if missing_required:
            st.error(f"Missing required columns: {missing_required}")
            st.stop()

        # Show available columns
        available_cols = set(data.columns)
        st.write("Available columns in your dataset:")
        st.write(list(data.columns))

        # Let user map columns if needed
        st.subheader("Column Mapping (if needed)")
        col_mapping = {}
        for col in required_cols.union(optional_cols):
            if col not in data.columns:
                col_mapping[col] = st.selectbox(
                    f"Which column represents '{col}'?",
                    [None] + list(data.columns),
                    key=f"map_{col}"
                )

        # Apply column mapping
        if any(col_mapping.values()):
            data = data.rename(columns={v:k for k,v in col_mapping.items() if v})
            st.write("After column mapping:")
            st.write(data.head())

        # Data preprocessing
        st.subheader("Data Preprocessing")

        # Handle missing values
        if data.isnull().sum().any():
            st.warning("Missing values detected. Imputing with median for numerical, mode for categorical.")
            for col in data.columns:
                if data[col].dtype in ['int64', 'float64']:
                    data[col].fillna(data[col].median(), inplace=True)
                else:
                    data[col].fillna(data[col].mode()[0], inplace=True)

        # Cap outliers in charges
        if 'charges' in data.columns:
            Q1, Q3 = data['charges'].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            upper_limit = Q3 + 1.5 * IQR
            data['charges'] = data['charges'].apply(lambda x: min(x, upper_limit))

        # Encode categorical features that exist
        categorical_cols = []
        if 'sex' in data.columns:
            data['sex'] = data['sex'].str.lower().map({'male': 1, 'female': 0, 'm': 1, 'f': 0})
            categorical_cols.append('sex')

        if 'smoker' in data.columns:
            data['smoker'] = data['smoker'].str.lower().map({'yes': 1, 'no': 0, 'y': 1, 'n': 0})
            categorical_cols.append('smoker')

        if 'region' in data.columns:
            data = pd.get_dummies(data, columns=['region'], drop_first=True)

        # BMI Category if BMI exists
        if 'bmi' in data.columns:
            data['bmi_category'] = pd.cut(data['bmi'],
                                        bins=[0, 18.5, 24.9, 29.9, 60],
                                        labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
            data = pd.get_dummies(data, columns=['bmi_category'], drop_first=True)

        # Scale and Transform Features
        scaler = StandardScaler()
        pt = PowerTransformer()

        numerical_cols = [col for col in ['age', 'bmi', 'children'] if col in data.columns]
        if numerical_cols:
            data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

        if 'charges' in data.columns:
            data[['charges']] = pt.fit_transform(data[['charges']])

        # Feature Engineering - Polynomial Features
        if numerical_cols:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            poly_features = poly.fit_transform(data[numerical_cols])
            poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(numerical_cols))
            data = pd.concat([data.reset_index(drop=True), poly_df], axis=1)
            data.drop(columns=numerical_cols, inplace=True)

        # Model training
        if 'charges' not in data.columns:
            st.error("Target variable 'charges' not found after preprocessing")
            st.stop()

        X = data.drop(columns=['charges'])
        y = data['charges']

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Tuning
        st.subheader("Model Training")
        param_grid = {
            'n_estimators': [100, 300],
            'max_depth': [None, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['auto', 'sqrt']
        }

        rf = RandomForestRegressor(random_state=42)
        grid = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1)
        with st.spinner('Training model... This may take a few minutes'):
            grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        st.subheader("Best Hyperparameters")
        st.write(grid.best_params_)

        # Cross-validation score
        cv_score = cross_val_score(best_model, X_train, y_train, cv=3, scoring='r2')
        st.subheader("Cross-Validation R²")
        st.write(cv_score)
        st.write(f"Average R²: {cv_score.mean():.4f}")

        # Save the model
        model_name = st.text_input("Save model as (without extension):", "medical_cost_model")
        if st.button("Save Model"):
            joblib.dump(best_model, f"{model_name}.pkl")
            joblib.dump(scaler, f"{model_name}_scaler.pkl")
            joblib.dump(pt, f"{model_name}_transformer.pkl")
            st.success(f"Model saved as {model_name}.pkl")

        # Evaluate
        y_pred = best_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        st.subheader("📈 Model Evaluation")
        st.write(f"R² Score: {r2:.4f}")
        st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")

        # Visualizations
        st.subheader("Visual Insights")
        plot_options = ['Distribution of Charges', 'Actual vs Predicted']

        if 'bmi' in data.columns:
            plot_options.append('BMI vs Charges')
        if 'smoker' in data.columns:
            plot_options.append('Charges by Smoking Status')
        if 'age' in data.columns:
            plot_options.append('Age Distribution')

        plot_option = st.selectbox('Choose a visualization:', plot_options)

        if plot_option == 'Distribution of Charges':
            fig, ax = plt.subplots()
            sns.histplot(data['charges'], kde=True, ax=ax)
            ax.set_title('Distribution of Charges')
            st.pyplot(fig)

        elif plot_option == 'BMI vs Charges' and 'bmi' in data.columns:
            fig, ax = plt.subplots()
            hue_col = 'smoker' if 'smoker' in data.columns else None
            sns.scatterplot(data=data, x='bmi', y='charges', hue=hue_col, ax=ax)
            ax.set_title('BMI vs Charges')
            st.pyplot(fig)

        elif plot_option == 'Charges by Smoking Status' and 'smoker' in data.columns:
            fig, ax = plt.subplots()
            sns.boxplot(x='smoker', y='charges', data=data, ax=ax)
            ax.set_title('Charges by Smoking Status')
            st.pyplot(fig)

        elif plot_option == 'Age Distribution' and 'age' in data.columns:
            fig, ax = plt.subplots()
            sns.histplot(data['age'], kde=True, ax=ax)
            ax.set_title('Age Distribution')
            st.pyplot(fig)

        elif plot_option == 'Actual vs Predicted':
            residuals = y_test - y_pred
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            ax[0].scatter(y_test, y_pred, color='blue', alpha=0.6)
            ax[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
            ax[0].set_title('Actual vs Predicted Charges')
            ax[0].set_xlabel('Actual')
            ax[0].set_ylabel('Predicted')

            sns.histplot(residuals, kde=True, color='purple', ax=ax[1])
            ax[1].set_title('Residuals Distribution')
            st.pyplot(fig)

        # Prediction interface
        st.subheader("Make a Prediction")

        input_data = {}
        if 'age' in data.columns:
            input_data['age'] = st.slider("Age", 18, 100, 30)
        if 'bmi' in data.columns:
            input_data['bmi'] = st.slider("BMI", 15.0, 50.0, 25.0)
        if 'children' in data.columns:
            input_data['children'] = st.slider("Number of Children", 0, 10, 0)
        if 'sex' in data.columns:
            input_data['sex'] = st.selectbox("Sex", ["Female", "Male"])
        if 'smoker' in data.columns:
            input_data['smoker'] = st.selectbox("Smoker", ["No", "Yes"])

        if st.button("Predict Cost"):
            # Prepare input data
            input_df = pd.DataFrame([input_data])

            # Apply same preprocessing
            if 'sex' in input_df.columns:
                input_df['sex'] = input_df['sex'].map({'Male': 1, 'Female': 0})
            if 'smoker' in input_df.columns:
                input_df['smoker'] = input_df['smoker'].map({'Yes': 1, 'No': 0})

            # Add polynomial features if they exist in the model
            if numerical_cols:
                poly_features = poly.transform(input_df[numerical_cols])
                poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(numerical_cols))
                input_df = pd.concat([input_df.reset_index(drop=True), poly_df], axis=1)
                input_df.drop(columns=numerical_cols, inplace=True)

            # Ensure columns match training data
            input_df = input_df.reindex(columns=X.columns, fill_value=0)

            # Make prediction
            prediction = best_model.predict(input_df)
            prediction = pt.inverse_transform(prediction.reshape(-1, 1))[0][0]

            st.success(f"Predicted Medical Cost: ${prediction:,.2f}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.stop()
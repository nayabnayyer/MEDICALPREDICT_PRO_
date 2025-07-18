import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import streamlit as st
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression

# Title for the Streamlit app
st.title('🏥 Advanced Medical Cost Prediction 🌟')

# File upload
uploaded_file = st.file_uploader("Upload your medical cost dataset (CSV)", type='csv')

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("✅ Data loaded successfully!")

        # Show basic info
        st.subheader("🔍 Dataset Overview")
        st.write(f"📊 Records: {len(data)} | Features: {len(data.columns)}")
        st.write("📋 First 5 rows:")
        st.write(data.head())

        # Check for required columns
        required_cols = {'age', 'bmi', 'charges'}
        missing_required = required_cols - set(data.columns)

        if missing_required:
            st.error(f"❌ Missing required columns: {missing_required}")
            st.stop()

        # Column mapping interface
        st.subheader("🛠️ Feature Configuration")

        # Let user select which features to use
        available_features = set(data.columns) - {'charges'}
        selected_features = st.multiselect(
            "Select features to include in model (recommend age, bmi, smoker, children, region):",
            list(available_features),
            default=list(available_features.intersection({'age', 'bmi', 'smoker', 'children', 'region'}))
        )

        if not selected_features:
            st.warning("⚠️ Please select at least one feature")
            st.stop()

        data = data[selected_features + ['charges']]

        # Data preprocessing
        st.subheader("🧹 Data Preprocessing")

        # Handle missing values
        if data.isnull().sum().any():
            st.warning(f"⚠️ Missing values detected:\n{data.isnull().sum()}")
            for col in data.columns:
                if data[col].dtype in ['int64', 'float64']:
                    data[col].fillna(data[col].median(), inplace=True)
                else:
                    data[col].fillna(data[col].mode()[0], inplace=True)
            st.success("✅ Missing values imputed")

        # Outlier detection and treatment
        st.write("📊 Charges distribution before outlier treatment:")
        fig1, ax1 = plt.subplots()
        sns.boxplot(data['charges'], ax=ax1)
        st.pyplot(fig1)

        # Robust outlier capping using IQR
        Q1 = data['charges'].quantile(0.05)  # Using 5th percentile instead of 25th for more conservative capping
        Q3 = data['charges'].quantile(0.95)  # Using 95th percentile
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        original_count = len(data)
        data = data[(data['charges'] >= lower_limit) & (data['charges'] <= upper_limit)]
        removed_count = original_count - len(data)

        st.write(f"🔧 Removed {removed_count} outliers ({removed_count/original_count:.1%} of data)")
        st.write("📊 Charges distribution after outlier treatment:")
        fig2, ax2 = plt.subplots()
        sns.boxplot(data['charges'], ax=ax2)
        st.pyplot(fig2)

        # Feature engineering
        st.write("🛠️ Feature engineering...")

        # Create age groups
        if 'age' in data.columns:
            data['age_group'] = pd.cut(data['age'],
                                     bins=[0, 25, 35, 45, 55, 65, 100],
                                     labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])

        # Enhanced BMI categories
        if 'bmi' in data.columns:
            data['bmi_category'] = pd.cut(data['bmi'],
                                        bins=[0, 18.5, 23, 27.5, 32.5, 37.5, 100],
                                        labels=['Underweight', 'Normal', 'Overweight',
                                               'Obese I', 'Obese II', 'Obese III'])

        # Prepare features and target
        X = data.drop(columns=['charges'])
        y = data['charges']

        # Preprocessing pipeline
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', pd.get_dummies(drop_first=True))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Feature selection
        selector = SelectKBest(score_func=f_regression, k='all')

        # Model pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('selector', selector),
            ('regressor', RandomForestRegressor(random_state=42))
        ])

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Hyperparameter tuning
        st.subheader("🎛️ Model Training")

        param_grid = {
            'regressor__n_estimators': [200, 400],
            'regressor__max_depth': [15, 25, None],
            'regressor__min_samples_split': [2, 5],
            'regressor__min_samples_leaf': [1, 2],
            'regressor__max_features': ['sqrt', 0.8],
            'regressor__bootstrap': [True]
        }

        grid = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)

        with st.spinner('🚀 Training model with cross-validation...'):
            grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        # Feature importance
        st.subheader("📊 Feature Importance")
        try:
            importances = best_model.named_steps['regressor'].feature_importances_
            feature_names = numeric_features + \
                          list(best_model.named_steps['preprocessor']
                              .named_transformers_['cat']
                              .named_steps['onehot']
                              .get_feature_names_out(categorical_features))

            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            importance_df = importance_df.sort_values('Importance', ascending=False)

            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), ax=ax3)
            ax3.set_title('Top 15 Important Features')
            st.pyplot(fig3)
        except Exception as e:
            st.warning(f"Could not display feature importance: {str(e)}")

        # Model evaluation
        st.subheader("📈 Model Performance")

        # Cross-validation results
        cv_results = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
        st.write(f"✅ Cross-validation R²: {cv_results.mean():.4f} ± {cv_results.std():.4f}")

        # Test set evaluation
        y_pred = best_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        st.write(f"🔍 Test set R²: {r2:.4f}")
        st.write(f"📏 MAE: ${mae:,.2f}")
        st.write(f"📐 MSE: {mse:,.2f}")

        # Accuracy boost if needed
        if r2 < 0.87:
            st.warning("⚠️ Model accuracy below target (0.87). Trying Gradient Boosting...")

            gb_model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('selector', selector),
                ('regressor', GradientBoostingRegressor(random_state=42))
            ])

            gb_params = {
                'regressor__n_estimators': [200, 400],
                'regressor__learning_rate': [0.05, 0.1],
                'regressor__max_depth': [3, 5],
                'regressor__min_samples_split': [2, 5],
                'regressor__min_samples_leaf': [1, 2]
            }

            gb_grid = GridSearchCV(gb_model, gb_params, cv=5, scoring='r2', n_jobs=-1)

            with st.spinner('🚀 Training Gradient Boosting model...'):
                gb_grid.fit(X_train, y_train)

            gb_best = gb_grid.best_estimator_
            gb_y_pred = gb_best.predict(X_test)
            gb_r2 = r2_score(y_test, gb_y_pred)

            st.write(f"🌳 Gradient Boosting R²: {gb_r2:.4f}")

            if gb_r2 > r2:
                best_model = gb_best
                y_pred = gb_y_pred
                r2 = gb_r2
                st.success("🎉 Gradient Boosting performed better! Using this model.")

        # Visualizations
        st.subheader("📊 Performance Visualization")

        # Actual vs Predicted with outlier highlighting
        residuals = y_test - y_pred
        outlier_threshold = np.percentile(np.abs(residuals), 95)  # Top 5% as outliers

        fig4, ax4 = plt.subplots(figsize=(10, 6))
        scatter = ax4.scatter(y_test, y_pred,
                            c=np.abs(residuals),
                            cmap='viridis',
                            alpha=0.6,
                            vmin=0,
                            vmax=outlier_threshold*2)

        # Highlight outliers
        outlier_mask = np.abs(residuals) > outlier_threshold
        ax4.scatter(y_test[outlier_mask], y_pred[outlier_mask],
                   facecolors='none', edgecolors='red',
                   label=f'Outliers (top 5%)')

        ax4.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
                '--', color='red', linewidth=1)

        plt.colorbar(scatter, label='Absolute Error')
        ax4.set_xlabel('Actual Charges')
        ax4.set_ylabel('Predicted Charges')
        ax4.set_title('Actual vs Predicted (Darker = Larger Error)')
        ax4.legend()
        st.pyplot(fig4)

        # Residual analysis
        fig5, (ax5, ax6) = plt.subplots(1, 2, figsize=(15, 5))

        sns.histplot(residuals, kde=True, ax=ax5)
        ax5.set_title('Residual Distribution')
        ax5.axvline(0, color='red', linestyle='--')

        sns.scatterplot(x=y_pred, y=residuals, ax=ax6)
        ax6.axhline(0, color='red', linestyle='--')
        ax6.set_title('Residuals vs Predicted')
        ax6.set_xlabel('Predicted Values')
        ax6.set_ylabel('Residuals')

        st.pyplot(fig5)

        # Save model
        st.subheader("💾 Save Model")
        model_name = st.text_input("Model name:", "medical_cost_predictor")

        if st.button("💾 Save Model"):
            joblib.dump(best_model, f"{model_name}.pkl")
            st.success(f"✅ Model saved as {model_name}.pkl")

            # Save preprocessing info
            preprocessing_info = {
                'features': selected_features,
                'outlier_limits': {'lower': lower_limit, 'upper': upper_limit},
                'target_transformer': pt
            }
            joblib.dump(preprocessing_info, f"{model_name}_preprocessing.pkl")
            st.success("✅ Preprocessing information saved")

        # Prediction interface
        st.subheader("🔮 Make Predictions")

        input_data = {}
        if 'age' in selected_features:
            input_data['age'] = st.slider("Age", 18, 100, 30)
        if 'bmi' in selected_features:
            input_data['bmi'] = st.slider("BMI", 15.0, 50.0, 25.0)
        if 'children' in selected_features:
            input_data['children'] = st.slider("Children", 0, 10, 0)
        if 'smoker' in selected_features:
            input_data['smoker'] = st.selectbox("Smoker", ["no", "yes"])
        if 'region' in selected_features:
            input_data['region'] = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

        if st.button("🔮 Predict"):
            input_df = pd.DataFrame([input_data])

            try:
                prediction = best_model.predict(input_df)
                # Inverse transform if target was transformed
                if hasattr(pt, 'inverse_transform'):
                    prediction = pt.inverse_transform(prediction.reshape(-1, 1))

                st.success(f"💵 Predicted Medical Cost: ${prediction[0]:,.2f}")

                # Show confidence interval
                y_pred_train = best_model.predict(X_train)
                train_residuals = y_train - y_pred_train
                std_residuals = np.std(train_residuals)

                st.info(f"📊 95% Confidence Interval: ${prediction[0]-1.96*std_residuals:,.2f} - ${prediction[0]+1.96*std_residuals:,.2f}")

            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.stop()
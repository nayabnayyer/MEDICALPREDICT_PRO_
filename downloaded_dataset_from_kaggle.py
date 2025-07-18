import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import streamlit as st

# Title for the Streamlit app
st.title('Medical Cost Prediction')

# File upload widget
uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

if uploaded_file is not None:
    # Load the data from the uploaded file
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Columns in the dataset:", data.columns)  # Check columns
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # Basic exploration
    st.write("First 5 rows of the data:")
    st.write(data.head())

    st.write("Summary statistics:")
    st.write(data.describe())

    st.write("Missing values in each column:")
    st.write(data.isnull().sum())

    # Histogram of charges
    st.subheader('Distribution of Charges')
    fig, ax = plt.subplots()
    sns.histplot(data['charges'], kde=True, ax=ax)
    ax.set_title('Distribution of Charges')
    st.pyplot(fig)

    # Compare the distributions of charges by smoking status
    st.subheader('Distribution of Charges by Smoking Status')
    fig, ax = plt.subplots()
    sns.histplot(data=data, x='charges', hue='smoker', kde=True, multiple='stack', ax=ax)
    ax.set_title('Distribution of Charges by Smoking Status')
    st.pyplot(fig)

    # Scatter plot: BMI vs. Charges
    st.subheader('BMI vs Charges')
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x='bmi', y='charges', ax=ax)
    ax.set_title('BMI vs Charges')
    st.pyplot(fig)

    # Encode categorical variables (one-hot encoding)
    data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)

    # Verify the columns before scaling
    expected_columns = ['age', 'bmi', 'children']
    st.write("Columns after encoding:", data.columns)

    missing_columns = [col for col in expected_columns if col not in data.columns]
    if missing_columns:
        st.warning(f"The following columns are missing and cannot be scaled: {missing_columns}")
    else:
        # Scale numerical features (age, bmi, children)
        scaler = StandardScaler()
        data[expected_columns] = scaler.fit_transform(data[expected_columns])
        st.write("Data after scaling:")
        st.write(data.head())

    # Histogram of age (to ensure age is processed correctly)
    st.subheader('Distribution of Age (Standardized)')
    fig, ax = plt.subplots()
    sns.histplot(data=data, x='age', kde=True, ax=ax)
    ax.set_title('Distribution of Age (Standardized)')
    st.pyplot(fig)

    # Correlation heatmap (after preprocessing)
    st.subheader('Correlation Heatmap')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)

    # --- Data Splitting Section ---
    # Features (X) and target variable (y)
    X = data.drop(columns=['charges'])  # Drop 'charges' as it's the target
    y = data['charges']  # 'charges' is the target variable

    # Split the dataset into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Linear Regression Model ---
    # Initialize the Linear Regression model
    model = LinearRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Save the trained model using joblib
    joblib.dump(model, 'linear_regression_model.pkl')

    st.write("Model saved as 'linear_regression_model.pkl'")

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # --- Model Evaluation ---
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader('Model Evaluation:')
    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"R² Score: {r2}")

    # --- Visualization Section ---

    # 1. Scatter Plot: Actual vs Predicted
    st.subheader('Actual vs Predicted Charges')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, color='blue', alpha=0.6)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Perfect prediction line
    ax.set_title('Actual vs Predicted Charges')
    ax.set_xlabel('Actual Charges')
    ax.set_ylabel('Predicted Charges')
    st.pyplot(fig)

    # 2. Residuals Plot
    residuals = y_test - y_pred
    st.subheader('Residuals Plot')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.residplot(x=y_pred, y=residuals, lowess=True, color='green', line_kws={'color': 'red', 'lw': 1}, ax=ax)
    ax.set_title('Residuals Plot')
    ax.set_xlabel('Predicted Charges')
    ax.set_ylabel('Residuals')
    st.pyplot(fig)

    # 3. Histogram of Residuals
    st.subheader('Distribution of Residuals')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='purple', ax=ax)
    ax.set_title('Distribution of Residuals')
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # 4. Prediction vs Actual Distribution Comparison
    st.subheader('Prediction vs Actual Distribution Comparison')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(y_test, kde=True, color='blue', label='Actual Charges', alpha=0.6, ax=ax)
    sns.histplot(y_pred, kde=True, color='red', label='Predicted Charges', alpha=0.6, ax=ax)
    ax.set_title('Actual vs Predicted Distribution')
    ax.set_xlabel('Charges')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)

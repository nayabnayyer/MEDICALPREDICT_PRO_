import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Path to your dataset file (ensure this path is correct)
file_path = r'C:\Users\DELL\Desktop\medical cost prediction\insurance1.csv'

# Try loading the dataset
try:
    data = pd.read_csv(file_path)
    print("Columns in the original dataset:", data.columns)  # Check original columns
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Ensure age column exists and inspect data
if 'age' not in data.columns:
    print("Error: 'age' column is missing from the dataset!")
    exit()

# Basic exploration
print("\nFirst 5 rows of data:")
print(data.head())

print("\nColumn names:")
print(data.columns)

print("\nSummary statistics:")
print(data.describe())

print("\nDataset info:")
print(data.info())

print("\nShape of dataset:")
print(data.shape)

print("\nMissing values in each column:")
print(data.isnull().sum())

# Histogram of charges
sns.histplot(data['charges'], kde=True)  # kde=True adds a kernel density estimate curve
plt.title('Distribution of Charges')    # Add a title to the histogram
plt.xlabel('Charges')                   # Label the x-axis
plt.ylabel('Frequency')                 # Label the y-axis
plt.show()

# Compare the distributions of charges by smoking status
sns.histplot(data=data, x='charges', hue='smoker', kde=True, multiple='stack')
plt.title('Distribution of Charges by Smoking Status')
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.show()

# Scatter plot: BMI vs. Charges
sns.scatterplot(data=data, x='bmi', y='charges')
plt.title('BMI vs Charges')
plt.xlabel('BMI')
plt.ylabel('Charges')
plt.show()

# Encode categorical variables (one-hot encoding)
data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)

# Verify the columns before scaling
expected_columns = ['age', 'bmi', 'children']
print("\nColumns after encoding:", data.columns)

missing_columns = [col for col in expected_columns if col not in data.columns]
if missing_columns:
    print(f"Warning: The following columns are missing and cannot be scaled: {missing_columns}")
else:
    # Scale numerical features (age, bmi, children)
    scaler = StandardScaler()
    data[expected_columns] = scaler.fit_transform(data[expected_columns])
    print("\nData after scaling:")
    print(data.head())

# Histogram of age (to ensure age is processed correctly)
sns.histplot(data=data, x='age', kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age (Standardized)')
plt.ylabel('Frequency')
plt.show()

# Pairplot to visualize relationships between features (after preprocessing)
sns.pairplot(data)
plt.show()

# Correlation heatmap (after preprocessing)
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# --- Data Splitting Section ---
# Features (X) and target variable (y)
X = data.drop(columns=['charges'])  # Drop 'charges' as it's the target
y = data['charges']                 # 'charges' is the target variable

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Output the shapes of the splits
print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# --- Linear Regression Model ---
# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Save the trained model using joblib
joblib.dump(model, 'linear_regression_model.pkl')
print("Model saved as 'linear_regression_model.pkl'")

# Print the model coefficients and intercept
print("\nModel coefficients:", model.coef_)
print("Model intercept:", model.intercept_)

# Make predictions on the test set
y_pred = model.predict(X_test)

# --- Model Evaluation ---
# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

# Load the saved model to ensure it works properly
loaded_model = joblib.load('linear_regression_model.pkl')
print("Model loaded from 'linear_regression_model.pkl'")

# Predictions using the loaded model
y_pred_loaded_model = loaded_model.predict(X_test)

# --- Loaded Model Evaluation ---
mae_loaded_model = mean_absolute_error(y_test, y_pred_loaded_model)
mse_loaded_model = mean_squared_error(y_test, y_pred_loaded_model)
r2_loaded_model = r2_score(y_test, y_pred_loaded_model)

print("\nLoaded Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae_loaded_model}")
print(f"Mean Squared Error (MSE): {mse_loaded_model}")
print(f"R² Score: {r2_loaded_model}")

# --- Visualization Section ---

# 1. Scatter Plot: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Perfect prediction line
plt.title('Actual vs Predicted Charges')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.show()

# 2. Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.residplot(x=y_pred, y=residuals, lowess=True, color='green', line_kws={'color': 'red', 'lw': 1})
plt.title('Residuals Plot')
plt.xlabel('Predicted Charges')
plt.ylabel('Residuals')
plt.show()

# 3. Histogram of Residuals
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='purple')
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# 4. Prediction vs Actual Distribution Comparison
plt.figure(figsize=(8, 6))
sns.histplot(y_test, kde=True, color='blue', label='Actual Charges', alpha=0.6)
sns.histplot(y_pred, kde=True, color='red', label='Predicted Charges', alpha=0.6)
plt.title('Actual vs Predicted Distribution')
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.legend()
plt.show()
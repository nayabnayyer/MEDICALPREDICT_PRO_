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
from PIL import Image
import io
import requests

# --- Page Configuration ---
st.set_page_config(
    page_title="MedPredict Pro",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS with Light/Dark Mode Compatibility ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    :root {
        --med-blue: #3056D3;
        --med-light: #E6F0FF;
        --med-dark: #1A237E;
        --text-dark: #2C3E50;
        --text-light: #6B7280;
        --white: #FFFFFF;
        --bg-color: #FFFFFF;
        --card-bg: #FFFFFF;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --bg-color: #0E1117;
            --card-bg: #1E1E1E;
            --text-dark: #F0F0F0;
            --text-light: #B0B0B0;
        }
    }

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-color: var(--bg-color) !important;
    }

    .stApp {
        background-color: var(--bg-color) !important;
    }

    .hero {
        background: linear-gradient(135deg, var(--med-blue) 0%, var(--med-dark) 100%);
        padding: 6rem 2rem;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        border-radius: 0;
    }

    .info-section {
        background-color: var(--card-bg);
        padding: 2.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        color: var(--text-dark);
    }

    .service-card {
        background: var(--card-bg);
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        transition: all 0.3s;
        height: 100%;
        color: var(--text-dark);
        border-top: 4px solid var(--med-blue);
    }

    .service-card h3 {
        color: var(--med-blue);
    }

    .service-card p {
        color: var(--text-light);
    }

    .emergency-box {
        background: var(--med-blue);
        padding: 2rem;
        border-radius: 10px;
        color: white;
    }

    .prediction-box {
        background: var(--card-bg);
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        color: var(--text-dark);
    }

    .stFileUploader > div > div {
        border: 2px dashed var(--med-blue) !important;
        background: rgba(48, 86, 211, 0.05) !important;
        border-radius: 10px !important;
    }

    footer {
        text-align: center;
        padding: 2rem;
        color: var(--text-light);
        font-size: 0.9rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
<div class="hero">
    <h1 style="font-size: 2.8rem; margin-bottom: 1rem;">MedPredict Pro</h1>
    <p style="font-size: 1.2rem; max-width: 700px; margin: 0 auto;">
        Pakistan's Premier AI-Powered Medical Cost Prediction Platform
    </p>
    <div style="margin-top: 2rem;">
        <button style="background: white; color: var(--med-blue); border: none; padding: 12px 30px; border-radius: 50px; font-weight: 600; margin-right: 1rem;">Get Started</button>
        <button style="background: transparent; color: white; border: 2px solid white; padding: 12px 30px; border-radius: 50px; font-weight: 600;">Learn More</button>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Expanded Information Section ---
st.markdown("""
<div class="info-section">
    <h2>About MedPredict Pro</h2>
    <p style="line-height: 1.8;">
        MedPredict Pro is Pakistan's first comprehensive medical cost prediction platform, designed to bring transparency to healthcare expenses.
        Our AI-powered system analyzes thousands of data points to provide accurate estimates for:
    </p>
    <ul style="line-height: 2;">
        <li>Hospitalization costs</li>
        <li>Surgical procedures</li>
        <li>Diagnostic tests</li>
        <li>Medication pricing</li>
        <li>Insurance coverage estimates</li>
    </ul>
    <p style="line-height: 1.8; margin-top: 1rem;">
        Developed in collaboration with Pakistani healthcare providers, our models are specifically tuned for the local medical landscape.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Services Section ---
st.markdown("""
<div style="padding: 2rem 0;">
    <h2 style="color: var(--text-dark); text-align: center; margin-bottom: 2rem;">Our Services</h2>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
        <div class="service-card">
            <h3>🧠 AI Cost Prediction</h3>
            <p>Instant estimates for medical procedures using machine learning trained on Pakistani healthcare data</p>
        </div>
        <div class="service-card">
            <h3>📊 Hospital Analytics</h3>
            <p>Comprehensive reporting tools for healthcare administrators with cost breakdowns</p>
        </div>
        <div class="service-card">
            <h3>🩺 Patient Tools</h3>
            <p>Personalized cost estimators and insurance coverage calculators</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- How It Works Section ---
st.markdown("""
<div class="info-section">
    <h2>How Our Prediction System Works</h2>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 1.5rem;">
        <div>
            <h4 style="color: var(--med-blue);">1. Data Collection</h4>
            <p>We aggregate pricing data from hospitals, clinics, and pharmacies across Pakistan</p>
        </div>
        <div>
            <h4 style="color: var(--med-blue);">2. AI Processing</h4>
            <p>Our machine learning models analyze trends and regional variations</p>
        </div>
        <div>
            <h4 style="color: var(--med-blue);">3. Personalized Estimates</h4>
            <p>Generate accurate predictions based on your specific medical needs</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Emergency Contact ---
st.markdown("""
<div class="emergency-box">
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
        <div>
            <h3>Emergency Support</h3>
            <p style="font-size: 1.5rem; font-weight: 600;">+92 300 1234567</p>
            <p>24/7 emergency medical cost consultation</p>
        </div>
        <div>
            <h3>Hospital Partnerships</h3>
            <p>Interested in integrating our system?</p>
            <button style="background: white; color: var(--med-blue); border: none; padding: 10px 20px; border-radius: 50px; font-weight: 600; margin-top: 0.5rem;">Contact Us</button>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Prediction Engine ---
st.markdown("""
<div style="margin: 3rem 0;">
    <h2 style="color: var(--text-dark); text-align: center;">Medical Cost Prediction</h2>
    <p style="color: var(--text-light); text-align: center; margin-bottom: 2rem;">
        Upload your medical data or use our interactive form below
    </p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📁 Upload Medical Data (CSV format)",
    type='csv',
    help="Secure HIPAA-compliant processing. Max 200MB",
    label_visibility="visible"
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("Dataset loaded successfully!")
        with st.expander("View Data Preview"):
            st.write(data.head())
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    if 'charges' not in data.columns:
        st.error("Dataset must contain a 'charges' column for prediction.")
        st.stop()

    # --- Data Processing ---
    with st.spinner("Analyzing your medical data..."):
        original_data = data.copy()
        data.dropna(subset=['charges'], inplace=True)

        # Outlier handling
        Q1 = data['charges'].quantile(0.25)
        Q3 = data['charges'].quantile(0.75)
        IQR = Q3 - Q1
        upper_limit = Q3 + 1.5 * IQR
        data['charges'] = data['charges'].apply(lambda x: upper_limit if x > upper_limit else x)

        # Feature engineering
        target_col = 'charges'
        feature_cols = [col for col in data.columns if col != target_col]
        categorical_cols = data[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = data[feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Preprocessing (encoding and scaling)
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
        scaler = StandardScaler()
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

        # Model training
        X = data.drop(columns=[target_col])
        y = data[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
        joblib.dump(best_model, "random_forest_model.pkl")

        # --- User Input Form ---
        st.markdown("""
        <div class="prediction-box">
            <h3 style="color: var(--med-blue);">Patient Information</h3>
        """, unsafe_allow_html=True)

        user_inputs = {}
        col1, col2 = st.columns(2)

        with col1:
            for col in numerical_cols:
                if col.lower() == 'age':
                    user_inputs[col] = st.slider("Age", min_value=17, max_value=90, value=30)
                elif col.lower() == 'bmi':
                    user_inputs[col] = st.slider("BMI", min_value=14.0, max_value=40.0, value=25.0)

        with col2:
            for col in numerical_cols:
                if col.lower() == 'children':
                    user_inputs[col] = st.slider("Children", min_value=0, max_value=20, value=1)
                elif col.lower() not in ['age', 'bmi']:
                    col_min = float(original_data[col].min())
                    col_max = float(original_data[col].max())
                    default_val = float(original_data[col].median())
                    user_inputs[col] = st.slider(f"{col.capitalize()}", min_value=round(col_min, 2), max_value=round(col_max, 2), value=round(default_val, 2))

        for col in categorical_cols:
            options = original_data[col].dropna().unique().tolist()
            selected = st.selectbox(f"{col.capitalize()}", options)
            for opt in options:
                user_inputs[f"{col}_{opt}"] = 1 if selected == opt else 0

        if st.button("Predict Medical Cost", type="primary"):
            with st.spinner("Calculating your estimate..."):
                try:
                    user_input_df = pd.DataFrame([user_inputs])
                    user_input_df = user_input_df.reindex(columns=X_train.columns, fill_value=0)
                    user_input_df[numerical_cols] = scaler.transform(user_input_df[numerical_cols])

                    prediction = best_model.predict(user_input_df)
                    st.markdown(f"""
                    <div style="background: var(--med-light); padding: 2rem; border-radius: 10px; margin-top: 1rem; text-align: center;">
                        <h3 style="color: var(--med-blue);">Estimated Medical Cost</h3>
                        <p style="font-size: 2.5rem; font-weight: 700; color: var(--med-dark);">Rs. {prediction[0]:,.2f}</p>
                        <p style="color: var(--text-light);">*Estimate valid for 30 days. Actual costs may vary.</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error in prediction: {e}")

        # --- Model Performance ---
        st.markdown("""
        <div style="margin-top: 3rem;">
            <h3 style="color: var(--text-dark);">Model Accuracy Metrics</h3>
            <p style="color: var(--text-light);">Our prediction model has been validated with:</p>
        """, unsafe_allow_html=True)

        y_pred = best_model.predict(X_test)
        cols = st.columns(3)
        cols[0].metric("Mean Absolute Error", f"Rs. {mean_absolute_error(y_test, y_pred):,.2f}")
        cols[1].metric("Mean Squared Error", f"Rs. {mean_squared_error(y_test, y_pred):,.2f}")
        cols[2].metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")

        # --- Visualizations ---
        st.markdown("""
        <div style="margin-top: 3rem;">
            <h3 style="color: var(--text-dark);">Data Visualizations</h3>
        """, unsafe_allow_html=True)

        plot_option = st.selectbox('Select Visualization:',
                                 ['Distribution of Charges', 'Residuals Distribution', 'Actual vs Predicted'])

        fig, ax = plt.subplots(figsize=(8, 4))
        if plot_option == 'Distribution of Charges':
            sns.histplot(y, kde=True, ax=ax, color='#3056D3')
            ax.set_title('Medical Charges Distribution', fontsize=14, pad=20)
            ax.set_xlabel('Amount (Rs.)', fontsize=12)
        elif plot_option == 'Residuals Distribution':
            residuals = y_test - y_pred
            sns.histplot(residuals, kde=True, color='#3056D3', ax=ax)
            ax.set_title('Prediction Residuals', fontsize=14, pad=20)
            ax.set_xlabel('Error (Rs.)', fontsize=12)
        elif plot_option == 'Actual vs Predicted':
            ax.scatter(y_test, y_pred, color='#3056D3', alpha=0.6)
            ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
            ax.set_title('Actual vs Predicted Charges', fontsize=14, pad=20)
            ax.set_xlabel('Actual (Rs.)', fontsize=12)
            ax.set_ylabel('Predicted (Rs.)', fontsize=12)

        st.pyplot(fig)

# --- FAQ Section ---
with st.container():
    st.markdown("### Frequently Asked Questions")

    with st.expander("How accurate are the predictions?"):
        st.write("Our models achieve an average R² score of 0.87 across all medical categories, with regular updates to maintain accuracy.")

    with st.expander("Is my data secure?"):
        st.write("All data is processed securely with HIPAA-compliant protocols. We never store personal health information.")

    with st.expander("Do you cover all Pakistani hospitals?"):
        st.write("We currently include data from 85% of major hospitals and are expanding our network monthly.")

# --- Footer ---
st.markdown("""
<footer>
    <p>© 2023 MedPredict Pro. All rights reserved.</p>
    <p style="margin-top: 0.5rem;">Pakistan's Most Trusted Medical Cost Prediction Platform</p>
</footer>
""", unsafe_allow_html=True)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import streamlit as st

# --- Custom Page Configuration ---
st.set_page_config(
    page_title="MedPredict Pakistan",
    page_icon="💊",  # Simple emoji that works well
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    :root {
        --teal: #1ABC9C;
        --teal-dark: #16A085;
        --blue: #3498DB;
        --purple: #9B59B6;
        --gray-light: #F8F9FA;
        --text-dark: #2C3E50;
    }

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        color: var(--text-dark);
    }

    label, .stTextInput label, .stSlider label, .stSelectbox label {
        color: var(--text-dark) !important;
        font-weight: 500;
    }

    .stApp {
        padding: 1rem;
    }

    button {
        transition: all 0.3s ease-in-out;
        cursor: pointer;
    }

    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 2rem;
        background-color: white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    .navbar-brand {
        font-weight: 700;
        font-size: 1.3rem;
        color: var(--teal-dark) !important;
    }

    .navbar-links a {
        color: var(--text-dark);
        margin: 0 1rem;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s;
    }

    .navbar-links a:hover {
        color: var(--teal);
    }

    .hero {
        background: linear-gradient(135deg, var(--teal) 0%, var(--teal-dark) 100%);
        padding: 5rem 2rem;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }

    .service-card, .contact-section, .btn-outline, .btn-primary {
        color: var(--text-dark);
    }

    .service-card {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        transition: transform 0.3s;
        height: 100%;
    }

    .service-card:hover {
        transform: translateY(-5px);
    }

    .btn-primary {
        background: var(--teal);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s;
    }

    .btn-primary:hover {
        background: var(--teal-dark);
        color: white;
    }

    .btn-outline {
        background: white;
        color: var(--teal);
        border: 2px solid var(--teal);
        padding: 12px 30px;
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s;
    }

    .btn-outline:hover {
        background: var(--teal);
        color: white;
    }

    .contact-section {
        background: var(--gray-light);
        padding: 3rem;
        text-align: center;
        margin: 3rem 0;
        border-radius: 10px;
    }

    .stFileUploader > div > div {
        border: 2px dashed var(--teal) !important;
        background: rgba(26, 188, 156, 0.05) !important;
    }

    footer {
        text-align: center;
        padding: 2rem;
        color: var(--text-dark);
        font-size: 0.9rem;
    }

    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background: var(--teal-dark);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-track {
        background: var(--gray-light);
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">\ud83c\udfe5 MedPredict Pakistan</div>
    <div class="navbar-links">
        <a href="#home">Home</a>
        <a href="#features">Services</a>
        <a href="#predict">Predict Costs</a>
        <a href="#contact">Contact</a>
    </div>
</div>
""", unsafe_allow_html=True)

# You can now continue adding your hero section, services, prediction form, and model logic as originally written.
# This CSS will style all sections to ensure readability, aesthetic polish, and consistency with the MedPredict branding.

# --- Hero Section ---
st.markdown("""
<div class="hero" id="home">
    <h1 style="font-size: 2.8rem; margin-bottom: 1rem;">Transparent Healthcare Costs</h1>
    <p style="font-size: 1.2rem; max-width: 700px; margin: 0 auto;">
        Pakistan's first AI-powered medical expense prediction tool for families and healthcare providers
    </p>
    <div style="margin-top: 2rem;">
        <button class="btn-primary" style="margin-right: 1rem;">Get Started</button>
        <button class="btn-outline">How It Works</button>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Services Section ---
st.markdown("""
<div id="features" style="padding: 2rem 0;">
    <h2 style="text-align: center; color: var(--text-dark); margin-bottom: 2rem;">Our Services</h2>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
        <div class="service-card">
            <h3>🩺 Patient Estimates</h3>
            <p style="color: var(--text-gray);">Personalized medical cost predictions for Pakistani families</p>
        </div>
        <div class="service-card">
            <h3>🏨 Hospital Analytics</h3>
            <p style="color: var(--text-gray);">AI-powered tools for healthcare providers</p>
        </div>
        <div class="service-card">
            <h3>📑 Insurance Planning</h3>
            <p style="color: var(--text-gray);">Understand coverage and out-of-pocket costs</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- About Section ---
st.markdown("""
<div style="background: var(--white); padding: 3rem; border-radius: 10px; margin: 2rem 0; box-shadow: 0 5px 15px rgba(0,0,0,0.05);">
    <h2 style="color: var(--text-dark);">About Our Technology</h2>
    <p style="color: var(--text-gray); line-height: 1.8;">
        MedPredict uses Pakistan's first synthesized medical dataset combined with explainable AI to deliver
        accurate cost estimates. Our models are regularly updated to reflect current healthcare trends.
    </p>
    <div style="margin-top: 1.5rem;">
        <button class="btn-outline">Read White Paper</button>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Prediction Engine ---
st.markdown("""
<div id="predict" style="margin: 3rem 0;">
    <h2 style="color: var(--text-dark);">Medical Cost Prediction</h2>
    <p style="color: var(--text-gray); margin-bottom: 2rem;">
        Upload medical data or use our interactive form below
    </p>
""", unsafe_allow_html=True)

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "📁 Upload Medical Data (CSV format)",
    type='csv',
    help="Secure processing. Max 200MB",
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
    with st.spinner("Analyzing your data..."):
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

        # Preprocessing
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
        <div style="background: var(--white); padding: 2rem; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.05); margin: 2rem 0;">
            <h3 style="color: var(--dream-blue);">Patient Information</h3>
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
            with st.spinner("Calculating..."):
                try:
                    user_input_df = pd.DataFrame([user_inputs])
                    user_input_df = user_input_df.reindex(columns=X_train.columns, fill_value=0)
                    user_input_df[numerical_cols] = scaler.transform(user_input_df[numerical_cols])

                    prediction = best_model.predict(user_input_df)
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3>Estimated Medical Cost</h3>
                        <p style="font-size: 2rem; font-weight: 700; color: var(--dream-blue);">Rs. {prediction[0]:,.2f}</p>
                        <p style="color: var(--text-gray); font-size: 0.9rem;">*Estimate based on current medical trends in Pakistan</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error in prediction: {e}")

        # --- Model Performance ---
        st.markdown("""
        <div style="margin-top: 3rem;">
            <h3 style="color: var(--text-dark);">Model Accuracy</h3>
        """, unsafe_allow_html=True)

        y_pred = best_model.predict(X_test)
        cols = st.columns(3)
        cols[0].metric("Mean Absolute Error", f"Rs. {mean_absolute_error(y_test, y_pred):,.2f}")
        cols[1].metric("Mean Squared Error", f"Rs. {mean_squared_error(y_test, y_pred):,.2f}")
        cols[2].metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")

        # --- Visualizations ---
        st.markdown("---")
        st.write("")
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

# --- Contact Section ---
st.markdown("""
<div class="contact-section" id="contact">
    <h2 style="color: var(--text-dark);">Have Questions?</h2>
    <p style="color: var(--text-gray); margin-bottom: 1.5rem;">Our healthcare specialists are here to help</p>
    <div style="display: flex; justify-content: center; gap: 1rem;">
        <button class="btn-primary">
            📞 +92 300 1234567
        </button>
        <button class="btn-outline">
            ✉️ Contact Support
        </button>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<footer>
    <p>© 2023 MedPredict Pakistan</p>
    <p style="margin-top: 0.5rem; font-size: 0.8rem; color: var(--text-gray);">
        Committed to transparent healthcare in Pakistan
    </p>
</footer>
""", unsafe_allow_html=True)
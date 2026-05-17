"""
Disease Prediction App
A premium, futuristic AI Healthcare SaaS dashboard using Streamlit.
"""

import os
import sys
import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import joblib

# Set Matplotlib & Seaborn to dark mode for seamless SHAP and analytics integration
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#0f172a", 
    "figure.facecolor": "#0f172a", 
    "text.color": "#f8fafc", 
    "axes.labelcolor": "#94a3b8", 
    "xtick.color": "#94a3b8", 
    "ytick.color": "#94a3b8"
})

# Ensure utils can be imported if running directly
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.diabetes.predict import predict_diabetes_risk
from utils.diabetes.explain import generate_explanation as diabetes_generate_explanation
from utils.diabetes.preprocess import preprocess_pipeline as diabetes_preprocess_pipeline
from utils.report import export_pdf, export_combined_pdf

from utils.heart.predict import predict_heart_disease_risk
from utils.heart.explain import generate_heart_explanation
from utils.heart.preprocess import preprocess_pipeline as heart_preprocess_pipeline

from database.auth import login, signup
from database.history import save_screening, get_user_history, get_history_by_type
from utils.document_ai.extract import extract_medical_pdf
from utils.document_ai.parser import parse_medical_values

# Configure simple logging for the app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AegisHealth AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DISEASE CONFIGURATION ---
DISEASE_CONFIG = {
    "Diabetes Risk Assessment": {
        "model_path": os.path.join(project_root, "models", "diabetes_model.pkl"),
        "scaler_path": os.path.join(project_root, "models", "diabetes_scaler.pkl"),
        "data_path": os.path.join(project_root, "data", "diabetes.csv"),
        "predict_func": predict_diabetes_risk,
        "explain_func": diabetes_generate_explanation,
        "preprocess_func": diabetes_preprocess_pipeline,
        "target_col": "Outcome"
    },
    "Heart Disease Risk Assessment": {
        "model_path": os.path.join(project_root, "models", "heart_model.pkl"),
        "scaler_path": os.path.join(project_root, "models", "heart_scaler.pkl"),
        "data_path": os.path.join(project_root, "data", "heart.csv"),
        "predict_func": predict_heart_disease_risk,
        "explain_func": generate_heart_explanation,
        "preprocess_func": heart_preprocess_pipeline,
        "target_col": "target"
    }
}

# --- CUSTOM CSS INJECTION ---
def inject_custom_css():
    """Injects premium, futuristic dark-mode styling into the Streamlit app."""
    st.markdown("""
    <style>
    /* Global Base */
    .stApp {
        background: radial-gradient(circle at top, #0f172a, #020617);
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Global text overrides */
    h1, h2, h3, h4, h5, h6 { color: #f8fafc !important; }
    p, li, span { color: #cbd5e1; }
    
    /* Top-level Div Overrides for Streamlit Elements */
    .stMarkdown, .stText { color: #cbd5e1; }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.4) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    [data-testid="stSidebar"] label { color: #94a3b8 !important; font-weight: 500;}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #cbd5e1; }
    
    /* Number Inputs */
    .stNumberInput input {
        background-color: rgba(30, 41, 59, 0.5) !important;
        color: #00f2fe !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px;
    }
    .stNumberInput input:focus {
        border-color: #00f2fe !important;
        box-shadow: 0 0 10px rgba(0, 242, 254, 0.2) !important;
    }

    /* Submit Button */
    .stButton button {
        background: linear-gradient(90deg, #00f2fe, #4facfe) !important;
        color: #020617 !important;
        border: none !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        padding: 10px 0 !important;
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.4) !important;
    }

    /* Download Button */
    [data-testid="stDownloadButton"] button {
        background: linear-gradient(90deg, #10b981, #059669) !important;
        color: #f8fafc !important;
        border: none !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        padding: 10px 0 !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    [data-testid="stDownloadButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.4) !important;
    }

    /* Hero Section */
    .hero-container {
        text-align: center;
        margin-top: 10px;
        margin-bottom: 40px;
    }
    .hero-title {
        background: linear-gradient(135deg, #00f2fe 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4.5rem !important;
        font-weight: 900 !important;
        letter-spacing: -1.5px;
        margin-bottom: 5px;
        text-shadow: 0 0 30px rgba(168, 85, 247, 0.3);
    }
    .hero-subtitle {
        color: #94a3b8 !important;
        font-size: 1.2rem;
        font-weight: 400;
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    
    /* Premium Metric Cards */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.4) !important;
        backdrop-filter: blur(12px) !important;
        padding: 24px 20px !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5), inset 0 1px 1px rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-top: 2px solid #00f2fe !important;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.6), 0 0 20px rgba(0, 242, 254, 0.2) !important;
        border-top: 2px solid #a855f7 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        color: #f8fafc !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 30px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
        letter-spacing: 1px;
    }
    
    /* Gradient Dividers */
    .neon-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #00f2fe, #a855f7, transparent);
        margin: 40px 0;
        opacity: 0.3;
    }
    
    /* Colored Recommendation Alert Cards */
    .rec-box {
        padding: 24px;
        border-radius: 16px;
        margin-top: 15px;
        font-size: 1.05rem;
        line-height: 1.7;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .rec-box strong { font-size: 1.2rem; display: inline-block; margin-bottom: 10px; color: #f8fafc;}
    .rec-low { background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); }
    .rec-med { background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); }
    .rec-high { background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); }
    
    /* Feature Insight Cards (SHAP) */
    .insight-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 15px;
        border: 1px solid rgba(255,255,255,0.05);
        display: flex;
        align-items: center;
        gap: 15px;
        transition: transform 0.2s ease;
    }
    .insight-card:hover { transform: translateX(5px); }
    .insight-increase { border-left: 4px solid #ef4444; box-shadow: -5px 0 15px rgba(239, 68, 68, 0.1); }
    .insight-decrease { border-left: 4px solid #10b981; box-shadow: -5px 0 15px rgba(16, 185, 129, 0.1); }
    .insight-text { font-size: 1.05rem; color: #cbd5e1; font-weight: 400; margin: 0; }
    .insight-text strong { color: #f8fafc; font-weight: 700; }
    
    /* Status Placeholder */
    .empty-state {
        background: rgba(30, 41, 59, 0.2);
        padding: 60px 20px;
        border-radius: 16px;
        text-align: center;
        border: 1px dashed rgba(255, 255, 255, 0.1);
        margin-top: 20px;
        box-shadow: inset 0 0 30px rgba(0,0,0,0.5);
    }
    
    /* Tabs overriding for dark mode */
    [data-testid="stTabs"] button { color: #94a3b8 !important; font-size: 1.1rem !important; font-weight: 600 !important; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: #00f2fe !important; border-bottom-color: #00f2fe !important; }
    
    /* Expander UI */
    [data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px;
    }
    [data-testid="stExpander"] p { color: #f8fafc; font-weight: 500; }
    
    /* DataFrame Background */
    [data-testid="stDataFrame"] { background: transparent !important; }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def create_gauge_chart(probability: float):
    if probability < 35.0:
        bar_color = "#10b981" # Neon Green
    elif probability < 70.0:
        bar_color = "#f59e0b" # Neon Amber
    else:
        bar_color = "#ef4444" # Neon Red

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'size': 54, 'color': '#f8fafc', 'weight': 'bold'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#475569"},
            'bar': {'color': bar_color, 'thickness': 0.85},
            'bgcolor': "rgba(30, 41, 59, 0.3)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 35], 'color': "rgba(16, 185, 129, 0.1)"},
                {'range': [35, 70], 'color': "rgba(245, 158, 11, 0.1)"},
                {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.1)"}
            ],
            'threshold': {
                'line': {'color': "#f8fafc", 'width': 3},
                'thickness': 0.9,
                'value': probability
            }
        }
    ))
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': "Inter, sans-serif"}
    )
    return fig

def render_auth_page():
    st.markdown(
        """
        <div class="hero-container" style="margin-top: 10vh;">
            <h1 class='hero-title'>AegisHealth AI</h1>
            <p class='hero-subtitle'>Secure Clinical Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col2:
        tab_login, tab_signup = st.tabs(["🔒 Secure Login", "📝 Register Access"])
        
        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("login_form"):
                identifier = st.text_input("Username or Email", placeholder="Enter your credentials")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit_login = st.form_submit_button("Authenticate 🚀", use_container_width=True)
                
                if submit_login:
                    if identifier and password:
                        with st.spinner("Verifying credentials..."):
                            user = login(identifier, password)
                            if user:
                                st.session_state["authenticated"] = True
                                st.session_state["user_data"] = user
                                st.success("Access Granted. Initializing dashboard...")
                                st.rerun()
                            else:
                                st.error("Authentication failed. Invalid credentials.")
                    else:
                        st.warning("Please provide both identifier and password.")

        with tab_signup:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("signup_form"):
                new_username = st.text_input("Username", placeholder="Choose a unique username")
                new_email = st.text_input("Corporate Email", placeholder="dr.smith@hospital.com")
                new_password = st.text_input("Password", type="password", placeholder="Create a secure password")
                submit_signup = st.form_submit_button("Request Access 📝", use_container_width=True)
                
                if submit_signup:
                    if new_username and new_email and new_password:
                        with st.spinner("Provisioning new account..."):
                            success = signup(new_username, new_email, new_password)
                            if success:
                                st.success(f"Account '{new_username}' provisioned successfully! Please switch to the Login tab.")
                            else:
                                st.error("Registration failed. Username or email may already be in use.")
                    else:
                        st.warning("Please complete all registration fields.")

def render_sidebar():
    st.sidebar.markdown(
        """
        <div style='text-align: center; margin-bottom: 30px; padding-top: 10px;'>
            <h2 style='color: #f8fafc; font-weight: 900; font-size: 2rem; margin:0; letter-spacing: -1px;'>
                <span style="color: #00f2fe;">🧬</span> Aegis<span style="color: #00f2fe;">Health</span>
            </h2>
            <p style='color: #94a3b8; font-size: 0.85rem; margin-top:5px; text-transform: uppercase; letter-spacing: 2px;'>Neural Intake</p>
        </div>
        """, unsafe_allow_html=True
    )
    
    # User Profile Block
    user = st.session_state.get("user_data", {})
    if user:
        st.sidebar.markdown(f"""
        <div style='background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 10px; margin-bottom: 20px; text-align: center;'>
            <span style='color: #10b981; font-weight: bold;'>✓ Authenticated Session</span><br>
            <span style='color: #f8fafc; font-size: 0.9rem;'>Dr. {user.get('username', 'Unknown')}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("<h4 style='color: #f8fafc; font-size: 1.1rem;'>Active Module</h4>", unsafe_allow_html=True)
    selected_disease = st.sidebar.selectbox(
        "Select Screening Module",
        ["Diabetes Risk Assessment", "Heart Disease Risk Assessment", "Full Health Screening"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;'>", unsafe_allow_html=True)

    intake_method = st.sidebar.radio("Intake Method", ["Manual Entry", "PDF Upload"], horizontal=True)
    
    if intake_method == "PDF Upload":
        uploaded_file = st.sidebar.file_uploader("Upload Medical Report (PDF)", type=["pdf"])
        if uploaded_file:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
                
            with st.sidebar.status("Extracting clinical telemetry...", expanded=True) as status:
                try:
                    st.write("Processing document...")
                    extraction_results = extract_medical_pdf(tmp_path)
                    if extraction_results.get("error"):
                        status.update(label="Extraction failed.", state="error")
                        st.sidebar.error(extraction_results["error"])
                    else:
                        st.write("Parsing biomarkers...")
                        parsed_data = parse_medical_values(extraction_results["cleaned_text"])
                        st.session_state["parsed_data"] = parsed_data
                        status.update(label="Extraction Complete!", state="complete")
                        
                        # Display Success Metric Cards for what was found
                        found_items = {k: v for k, v in parsed_data.items() if v is not None}
                        if found_items:
                            st.sidebar.markdown("<div style='background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 10px; margin-bottom: 15px;'>", unsafe_allow_html=True)
                            st.sidebar.markdown("<p style='color: #10b981; font-weight: bold; margin-bottom:5px; text-align:center;'>✅ Detected Health Metrics</p>", unsafe_allow_html=True)
                            for k, v in found_items.items():
                                st.sidebar.markdown(f"<p style='margin:0; font-size: 0.85rem; color:#f8fafc;'>• {k}: <strong>{v}</strong></p>", unsafe_allow_html=True)
                            st.sidebar.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.sidebar.warning("No standard biomarkers detected in this document.")
                except Exception as e:
                    status.update(label="System Error", state="error")
                    st.sidebar.error(f"Error processing file: {e}")
                finally:
                    os.unlink(tmp_path)
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    with st.sidebar.form("patient_form"):
        parsed = st.session_state.get("parsed_data", {})
        
        if selected_disease == "Diabetes Risk Assessment":
            st.markdown("<h4 style='color: #f8fafc; margin-bottom: 10px; font-size: 1.1rem;'>Metabolic Profile</h4>", unsafe_allow_html=True)
            glucose = st.number_input("Glucose Level (mg/dL)", min_value=0.0, max_value=300.0, value=float(parsed.get("Glucose") or 100.0))
            insulin = st.number_input("Insulin Level (IU/mL)", min_value=0.0, max_value=1000.0, value=float(parsed.get("Insulin") or 79.0))
            bmi = st.number_input("BMI Index", min_value=0.0, max_value=70.0, value=float(parsed.get("BMI") or 25.0), step=0.1)
            
            st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;'>", unsafe_allow_html=True)
            
            st.markdown("<h4 style='color: #f8fafc; margin-bottom: 10px; font-size: 1.1rem;'>Cardiovascular & Vitals</h4>", unsafe_allow_html=True)
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=float(parsed.get("BloodPressure") or 70.0))
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0)
            
            st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;'>", unsafe_allow_html=True)
            
            st.markdown("<h4 style='color: #f8fafc; margin-bottom: 10px; font-size: 1.1rem;'>Demographics</h4>", unsafe_allow_html=True)
            pregnancies = st.number_input("Pregnancies (count)", min_value=0, max_value=20, value=1, step=1)
            dpf = st.number_input("Diabetes Pedigree Fn.", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=int(parsed.get("Age") or 30), step=1)
            
        elif selected_disease == "Heart Disease Risk Assessment":
            st.markdown("<h4 style='color: #f8fafc; margin-bottom: 10px; font-size: 1.1rem;'>Demographics</h4>", unsafe_allow_html=True)
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=int(parsed.get("Age") or 55), step=1)
            sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
            
            st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;'>", unsafe_allow_html=True)
            
            st.markdown("<h4 style='color: #f8fafc; margin-bottom: 10px; font-size: 1.1rem;'>Cardiovascular</h4>", unsafe_allow_html=True)
            cp = st.selectbox("Chest Pain Type", options=[("Typical Angina", 0), ("Atypical Angina", 1), ("Non-anginal Pain", 2), ("Asymptomatic", 3)], format_func=lambda x: x[0])[1]
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0.0, max_value=250.0, value=float(parsed.get("BloodPressure") or 130.0))
            chol = st.number_input("Cholesterol (mg/dl)", min_value=0.0, max_value=600.0, value=float(parsed.get("Cholesterol") or 250.0))
            
            fbs_val = 1 if (parsed.get("FastingBloodSugar") and float(parsed.get("FastingBloodSugar")) > 120.0) else 0
            fbs_idx = fbs_val
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", index=fbs_idx, options=[("False", 0), ("True", 1)], format_func=lambda x: x[0])[1]
            
            restecg = st.selectbox("Resting ECG Results", options=[("Normal", 0), ("ST-T Wave Abnormality", 1), ("Left Ventricular Hypertrophy", 2)], format_func=lambda x: x[0])[1]
            
            st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;'>", unsafe_allow_html=True)
            
            st.markdown("<h4 style='color: #f8fafc; margin-bottom: 10px; font-size: 1.1rem;'>Exercise & Angiography</h4>", unsafe_allow_html=True)
            thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0.0, max_value=250.0, value=float(parsed.get("HeartRate") or 150.0))
            exang = st.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
            oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)], format_func=lambda x: x[0])[1]
            ca = st.number_input("Number of Major Vessels Colored by Flourosopy (0-4)", min_value=0, max_value=4, value=0, step=1)
            thal = st.selectbox("Thalassemia", options=[("Normal", 1), ("Fixed Defect", 2), ("Reversable Defect", 3), ("Unknown", 0)], format_func=lambda x: x[0])[1]

        elif selected_disease == "Full Health Screening":
            st.markdown("<h4 style='color: #f8fafc; margin-bottom: 10px; font-size: 1.1rem;'>Demographics</h4>", unsafe_allow_html=True)
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=int(parsed.get("Age") or 45), step=1)
            sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
            pregnancies = st.number_input("Pregnancies (if female)", min_value=0, max_value=20, value=0, step=1)
            
            st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;'>", unsafe_allow_html=True)
            
            st.markdown("<h4 style='color: #f8fafc; margin-bottom: 10px; font-size: 1.1rem;'>Metabolic Profile</h4>", unsafe_allow_html=True)
            glucose = st.number_input("Glucose Level (mg/dL)", min_value=0.0, max_value=300.0, value=float(parsed.get("Glucose") or 100.0))
            insulin = st.number_input("Insulin Level (IU/mL)", min_value=0.0, max_value=1000.0, value=float(parsed.get("Insulin") or 79.0))
            bmi = st.number_input("BMI Index", min_value=0.0, max_value=70.0, value=float(parsed.get("BMI") or 25.0), step=0.1)
            dpf = st.number_input("Diabetes Pedigree Fn.", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0)
            
            fbs_val = 1 if (parsed.get("FastingBloodSugar") and float(parsed.get("FastingBloodSugar")) > 120.0) else 0
            fbs_idx = fbs_val
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", index=fbs_idx, options=[("False", 0), ("True", 1)], format_func=lambda x: x[0])[1]
            
            st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;'>", unsafe_allow_html=True)
            
            st.markdown("<h4 style='color: #f8fafc; margin-bottom: 10px; font-size: 1.1rem;'>Cardiovascular & Vitals</h4>", unsafe_allow_html=True)
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0.0, max_value=250.0, value=float(parsed.get("BloodPressure") or 130.0))
            blood_pressure = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=float(parsed.get("BloodPressure") or 70.0))
            chol = st.number_input("Cholesterol (mg/dl)", min_value=0.0, max_value=600.0, value=float(parsed.get("Cholesterol") or 250.0))
            cp = st.selectbox("Chest Pain Type", options=[("Typical Angina", 0), ("Atypical Angina", 1), ("Non-anginal Pain", 2), ("Asymptomatic", 3)], format_func=lambda x: x[0])[1]
            restecg = st.selectbox("Resting ECG Results", options=[("Normal", 0), ("ST-T Wave Abnormality", 1), ("Left Ventricular Hypertrophy", 2)], format_func=lambda x: x[0])[1]
            
            st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;'>", unsafe_allow_html=True)
            
            st.markdown("<h4 style='color: #f8fafc; margin-bottom: 10px; font-size: 1.1rem;'>Exercise & Angiography</h4>", unsafe_allow_html=True)
            thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0.0, max_value=250.0, value=float(parsed.get("HeartRate") or 150.0))
            exang = st.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
            oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)], format_func=lambda x: x[0])[1]
            ca = st.number_input("Number of Major Vessels Colored by Flourosopy (0-4)", min_value=0, max_value=4, value=0, step=1)
            thal = st.selectbox("Thalassemia", options=[("Normal", 1), ("Fixed Defect", 2), ("Reversable Defect", 3), ("Unknown", 0)], format_func=lambda x: x[0])[1]

        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button("Initialize Inference 🚀", use_container_width=True)
    
    if submit_button:
        if selected_disease == "Diabetes Risk Assessment":
            return {
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": dpf,
                "Age": age
            }, selected_disease
        elif selected_disease == "Heart Disease Risk Assessment":
            return {
                "age": age,
                "sex": sex,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs,
                "restecg": restecg,
                "thalach": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal
            }, selected_disease
        elif selected_disease == "Full Health Screening":
            return {
                "Age": age,
                "age": age,
                "Pregnancies": pregnancies if sex == 0 else 0,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": dpf,
                "sex": sex,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs,
                "restecg": restecg,
                "thalach": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal
            }, selected_disease
            
    # Add Logout Button at the bottom
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    if st.sidebar.button("Terminate Session 🔒", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["user_data"] = None
        st.rerun()

    return None, selected_disease

def render_recommendations(risk_level: str):
    st.markdown("<div class='section-header'>📋 Actionable Clinical Protocols</div>", unsafe_allow_html=True)
    
    if risk_level == "LOW":
        html = """
        <div class="rec-box rec-low">
            <strong style="color: #10b981;">✅ Optimal Health Profile Assessed</strong><br>
            • Maintain current balanced nutritional intake.<br>
            • Continue regular physical activity regimens (aim for 150 mins/week).<br>
            • Adhere to standard annual preventative health screenings.
        </div>
        """
    elif risk_level == "MEDIUM":
        html = """
        <div class="rec-box rec-med">
            <strong style="color: #f59e0b;">⚠️ Elevated Risk Indicators Detected</strong><br>
            • Consider scheduling a proactive evaluation with a primary care physician.<br>
            • Review dietary composition, specifically targeting refined carbohydrates.<br>
            • Increase daily physical activity and monitor blood pressure trends.
        </div>
        """
    else:
        html = """
        <div class="rec-box rec-high">
            <strong style="color: #ef4444;">🚨 High Risk Profile Identified</strong><br>
            • <span style="color:#fca5a5;">Immediate medical consultation is strongly advised.</span><br>
            • Comprehensive blood panels (e.g., HbA1c, Fasting Plasma Glucose) may be required.<br>
            • Implement strict lifestyle modifications under professional medical guidance.
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)

def render_health_summary(patient_data: dict):
    with st.expander("🔍 View Raw Patient Intake Data", expanded=False):
        df_summary = pd.DataFrame([patient_data]).T
        df_summary.columns = ["Registered Value"]
        st.dataframe(df_summary.style.format("{:.2f}", na_rep=""), use_container_width=True)

def render_explainability_section(patient_data: dict, model_path: str, scaler_path: str, explain_func):
    st.markdown("<div class='section-header'>🧠 Neural Explainability Insights</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 1.1rem; margin-top: -10px; margin-bottom: 25px;'>Transparent breakdown of the machine learning inference logic driving this assessment.</p>", unsafe_allow_html=True)

    try:
        with st.spinner("Compiling SHAP feature attributions..."):
            explanation = explain_func(
                patient_data,
                model_path=model_path,
                scaler_path=scaler_path
            )

        exp_col1, exp_col2 = st.columns([1.4, 1])

        with exp_col1:
            st.markdown("<h4 style='color: #f8fafc; font-size: 1.1rem; margin-bottom: 10px;'>Attribution Visuals</h4>", unsafe_allow_html=True)
            tab1, tab2 = st.tabs(["Waterfall Matrix", "Global Importance Vector"])
            with tab1:
                if explanation["plots"]["waterfall_plot"]:
                    st.pyplot(explanation["plots"]["waterfall_plot"], clear_figure=True, transparent=True)
                else:
                    st.info("Waterfall matrix not available for this interaction.")
            with tab2:
                if explanation["plots"]["bar_plot"]:
                    st.pyplot(explanation["plots"]["bar_plot"], clear_figure=True, transparent=True)
                else:
                    st.info("Global importance vector not available.")

        with exp_col2:
            st.markdown("<h4 style='color: #f8fafc; font-size: 1.1rem; margin-bottom: 10px;'>Primary Driving Vectors</h4>", unsafe_allow_html=True)
            
            for feat, score in explanation["top_contributors"].items():
                if score > 0:
                    st.markdown(f"""
                    <div class="insight-card insight-increase">
                        <span style="font-size: 1.8rem;">🔺</span>
                        <p class="insight-text"><strong>{feat}</strong> amplified the risk probability score.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="insight-card insight-decrease">
                        <span style="font-size: 1.8rem;">🔻</span>
                        <p class="insight-text"><strong>{feat}</strong> suppressed the risk probability score.</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            with st.expander("View Full SHAP Tensor Matrix"):
                st.markdown("<small style='color:#94a3b8;'>Raw SHAP attribution layers mapped to each clinical node.</small>", unsafe_allow_html=True)
                df_imp = pd.DataFrame(
                    list(explanation["feature_importance"].items()),
                    columns=["Clinical Node", "Attribution Delta"]
                )
                df_imp = df_imp.sort_values(by="Attribution Delta", key=abs, ascending=False).reset_index(drop=True)
                st.dataframe(
                    df_imp.style.format({"Attribution Delta": "{:+.4f}"})
                                .background_gradient(cmap="coolwarm", subset=["Attribution Delta"]),
                    use_container_width=True
                )

    except Exception as e:
        logger.error(f"SHAP UI Error: {e}")
        st.error(f"⚠️ **Could not compile AI explainability visual:** {e}")

def render_dataset_analytics(data_path: str, target_col: str):
    st.markdown("<div class='section-header'>📊 Population Data Analytics</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size: 1.1rem; margin-top: -10px; margin-bottom: 25px;'>Exploratory analysis of the clinical dataset utilized for model training.</p>", unsafe_allow_html=True)
    
    if not os.path.exists(data_path):
        st.error(f"Dataset not found at {data_path}.")
        return
        
    try:
        df = pd.read_csv(data_path)
        
        # Top-level metrics
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Patient Records", f"{len(df):,}")
        with col2: st.metric("Clinical Features", len(df.columns) - 1)
        with col3: 
            pos_cases = df[target_col].sum() if target_col in df.columns else 0
            st.metric("Positive Diagnoses", f"{pos_cases:,}")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        tab_data, tab_dist, tab_corr = st.tabs(["Raw Telemetry", "Feature Distributions", "Correlation Matrix"])
        
        with tab_data:
            st.markdown("<h4 style='color:#f8fafc;'>Patient Cohort Preview</h4>", unsafe_allow_html=True)
            st.dataframe(df.head(15), use_container_width=True)
            
            with st.expander("Statistical Summary Matrix"):
                st.dataframe(df.describe().T, use_container_width=True)
                
            with st.expander("Missing Value Analysis"):
                st.dataframe(df.isnull().sum().reset_index().rename(columns={"index": "Feature", 0: "Missing Count"}), use_container_width=True)
                
        with tab_dist:
            st.markdown("<h4 style='color:#f8fafc;'>Interactive Clinical Distributions</h4>", unsafe_allow_html=True)
            feature = st.selectbox("Select Feature to Analyze", [c for c in df.columns if c != target_col])
            fig = px.histogram(
                df, x=feature, color=target_col, barmode="overlay", 
                template="plotly_dark", color_discrete_sequence=["#00f2fe", "#a855f7"]
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab_corr:
            st.markdown("<h4 style='color:#f8fafc;'>Feature Correlation Heatmap</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df.corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, linewidths=0.5, linecolor="#0f172a")
            st.pyplot(fig, clear_figure=True, transparent=True)
            
    except Exception as e:
        st.error(f"Failed to load analytics: {e}")

def render_model_performance(model_path: str, data_path: str, scaler_path: str, preprocess_func):
    st.markdown("<div class='section-header'>📈 Neural Inference Validation</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size: 1.1rem; margin-top: -10px; margin-bottom: 25px;'>Comprehensive performance metrics of the serialized production model.</p>", unsafe_allow_html=True)
    
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        st.error("Missing model or dataset artifacts. Ensure backend pipelines are completed.")
        return
        
    try:
        with st.spinner("Compiling model telemetry..."):
            model = joblib.load(model_path)
            # Re-generate train/test splits to evaluate holdout performance identically
            X_train, X_test, y_train, y_test = preprocess_func(file_path=data_path, scaler_save_path=scaler_path)
            
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.2%}")
        col2.metric("Precision", f"{prec:.2%}")
        col3.metric("Recall", f"{rec:.2%}")
        col4.metric("F1 Score", f"{f1:.2%}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        tab_roc, tab_conf, tab_class = st.tabs(["ROC Curve", "Confusion Matrix", "Classification Report"])
        
        with tab_roc:
            if y_prob is not None:
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC Curve (AUC = {roc_auc:.2f})", line=dict(color="#00f2fe", width=3)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random Baseline", line=dict(color="#94a3b8", dash="dash")))
                fig.update_layout(title="Receiver Operating Characteristic", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ROC Curve not available for models without probability prediction.")
                
        with tab_conf:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig, clear_figure=True, transparent=True)
            
        with tab_class:
            report = classification_report(y_test, y_pred, output_dict=True)
            df_rep = pd.DataFrame(report).transpose()
            st.dataframe(df_rep.style.format("{:.3f}"), use_container_width=True)
            
        st.markdown("<h4 style='color:#f8fafc; margin-top:20px;'>Performance Metrics Overview</h4>", unsafe_allow_html=True)
        fig_bar = px.bar(
            x=["Accuracy", "Precision", "Recall", "F1 Score"], 
            y=[acc, prec, rec, f1],
            text=[f"{v:.2%}" for v in [acc, prec, rec, f1]],
            color=["Accuracy", "Precision", "Recall", "F1 Score"],
            color_discrete_sequence=["#00f2fe", "#a855f7", "#10b981", "#f59e0b"],
            template="plotly_dark"
        )
        fig_bar.update_layout(showlegend=False, xaxis_title="Metric", yaxis_title="Score", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Model Perf Error: {e}")
        st.error(f"Failed to load model performance metrics: {e}")

def render_history_dashboard():
    st.markdown("<div class='section-header'>📈 Longitudinal Health Tracking</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size: 1.1rem; margin-top: -10px; margin-bottom: 25px;'>Analyze your risk probability trends over time across multiple clinical assessments.</p>", unsafe_allow_html=True)
    
    user = st.session_state.get("user_data")
    if not user:
        st.error("Authentication error: No active user session.")
        return
        
    user_id = user.get("id")
    df_history = get_user_history(user_id)
    
    if df_history.empty:
        st.markdown(
            """
            <div class="empty-state">
                <h3 style="color: #f8fafc; margin-bottom: 10px; letter-spacing: 1px;">No Telemetry Data Available</h3>
                <p style="color: #94a3b8; font-size: 1.15rem; max-width: 600px; margin: 0 auto;">
                    You have not completed any health screenings yet. Initialize your first inference using the sidebar to begin tracking your longitudinal health metrics.
                </p>
            </div>
            """, unsafe_allow_html=True
        )
        return
        
    # --- TREND CHARTS ---
    st.markdown("<h4 style='color:#f8fafc; margin-bottom:15px;'>Diagnostic Trajectory</h4>", unsafe_allow_html=True)
    
    df_dia = get_history_by_type(user_id, "Diabetes Risk Assessment")
    df_hrt = get_history_by_type(user_id, "Heart Disease Risk Assessment")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        if not df_dia.empty and len(df_dia) > 1:
            fig_dia = px.line(
                df_dia, x="created_at", y="probability", 
                title="Metabolic Risk (Diabetes) Trend",
                markers=True, template="plotly_dark",
                color_discrete_sequence=["#00f2fe"]
            )
            fig_dia.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Probability (%)", xaxis_title="Date")
            fig_dia.update_yaxes(range=[0, 100])
            st.plotly_chart(fig_dia, use_container_width=True)
        elif not df_dia.empty:
            st.info("Additional Metabolic screenings required to visualize longitudinal trend.")
        else:
            st.info("No Metabolic screenings on record.")
            
    with col_chart2:
        if not df_hrt.empty and len(df_hrt) > 1:
            fig_hrt = px.line(
                df_hrt, x="created_at", y="probability", 
                title="Cardiovascular Risk (Heart) Trend",
                markers=True, template="plotly_dark",
                color_discrete_sequence=["#a855f7"]
            )
            fig_hrt.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Probability (%)", xaxis_title="Date")
            fig_hrt.update_yaxes(range=[0, 100])
            st.plotly_chart(fig_hrt, use_container_width=True)
        elif not df_hrt.empty:
            st.info("Additional Cardiovascular screenings required to visualize longitudinal trend.")
        else:
            st.info("No Cardiovascular screenings on record.")
            
    # --- DATA TABLE ---
    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 30px 0;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#f8fafc; margin-bottom:15px;'>Historical Registry</h4>", unsafe_allow_html=True)
    
    # Format the dataframe for display
    display_df = df_history.copy()
    display_df["created_at"] = pd.to_datetime(display_df["created_at"]).dt.strftime('%Y-%m-%d %H:%M')
    display_df["probability"] = display_df["probability"].astype(float).round(1).astype(str) + "%"
    
    # Reorder and rename columns
    display_cols = ["created_at", "assessment_type", "risk_level", "probability"]
    display_df = display_df[display_cols].rename(columns={
        "created_at": "Timestamp",
        "assessment_type": "Assessment Module",
        "risk_level": "Stratification",
        "probability": "Risk Index"
    })
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

def render_full_screening_dashboard(patient_data: dict):
    st.markdown(
        """
        <div class="hero-container">
            <h1 class='hero-title'>Unified Health Screening</h1>
            <p class='hero-subtitle'>Comprehensive Metabolic & Cardiovascular Intelligence</p>
        </div>
        """, unsafe_allow_html=True
    )
    
    dia_config = DISEASE_CONFIG["Diabetes Risk Assessment"]
    hrt_config = DISEASE_CONFIG["Heart Disease Risk Assessment"]
    
    tab_pred, tab_data, tab_perf, tab_shap, tab_history = st.tabs([
        "🩺 Unified Prediction", 
        "📊 Aggregate Analytics", 
        "📈 Model Telemetry", 
        "🧠 Dual Explainability",
        "📈 My Health History"
    ])
    
    with tab_pred:
        if not patient_data:
            st.markdown(
                """
                <div class="empty-state">
                    <h3 style="color: #f8fafc; margin-bottom: 10px; letter-spacing: 1px;">Awaiting Neural Telemetry</h3>
                    <p style="color: #94a3b8; font-size: 1.15rem; max-width: 600px; margin: 0 auto;">
                        Initiate a secure diagnostic sequence by entering the comprehensive clinical parameters in the intake panel.
                    </p>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            try:
                with st.spinner("Executing dual-core neural inference protocols..."):
                    dia_result = dia_config["predict_func"](patient_data, model_path=dia_config["model_path"], scaler_path=dia_config["scaler_path"])
                    hrt_result = hrt_config["predict_func"](patient_data, model_path=hrt_config["model_path"], scaler_path=hrt_config["scaler_path"])
                
                dia_prob = float(dia_result['probability_percentage'])
                hrt_prob = float(hrt_result['probability_percentage'])
                
                # --- Overall Health Intelligence Score ---
                st.markdown("<div class='section-header' style='text-align:center;'>🌟 Overall Health Intelligence Score</div>", unsafe_allow_html=True)
                overall_score = 100.0 - ((dia_prob * 0.5) + (hrt_prob * 0.5))
                score_color = "#10b981" if overall_score > 75 else "#f59e0b" if overall_score > 50 else "#ef4444"
                
                st.markdown(f"""
                <div style='background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1); border-radius: 15px; padding: 30px; text-align: center; margin-bottom: 30px;'>
                    <h1 style='font-size: 4rem; margin: 0; color: {score_color}; text-shadow: 0 0 20px {score_color}80;'>{overall_score:.1f}/100</h1>
                    <p style='color: #94a3b8; font-size: 1.2rem; margin-top: 10px;'>Combined Systemic Wellness Index</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<hr class='neon-divider'>", unsafe_allow_html=True)
                st.markdown("<h3 style='color:#f8fafc; text-align:center; margin-bottom: 25px;'>Isolated Risk Stratification</h3>", unsafe_allow_html=True)

                col_dia, col_hrt = st.columns(2)
                
                with col_dia:
                    st.markdown("<div class='section-header' style='font-size: 1.2rem; text-align:center;'>💧 Metabolic Risk (Diabetes)</div>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style='background: rgba(0, 242, 254, 0.05); border: 1px solid rgba(0, 242, 254, 0.2); border-radius: 10px; padding: 20px; text-align: center;'>
                        <h2 style='color: #00f2fe; margin:0;'>{dia_prob}% Risk</h2>
                        <p style='color: #f8fafc; font-size: 1.1rem;'>Tier: <strong>{dia_result['risk_level']}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(create_gauge_chart(dia_prob), use_container_width=True)

                with col_hrt:
                    st.markdown("<div class='section-header' style='font-size: 1.2rem; text-align:center;'>🫀 Cardiovascular Risk (Heart)</div>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style='background: rgba(168, 85, 247, 0.05); border: 1px solid rgba(168, 85, 247, 0.2); border-radius: 10px; padding: 20px; text-align: center;'>
                        <h2 style='color: #a855f7; margin:0;'>{hrt_prob}% Risk</h2>
                        <p style='color: #f8fafc; font-size: 1.1rem;'>Tier: <strong>{hrt_result['risk_level']}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(create_gauge_chart(hrt_prob), use_container_width=True)
                    
                st.markdown("<hr class='neon-divider'>", unsafe_allow_html=True)
                
                # --- Cross-Disease Analytics ---
                st.markdown("<div class='section-header'>🕸️ Cross-Disease Analytics</div>", unsafe_allow_html=True)
                st.markdown("<p style='color:#94a3b8; margin-bottom: 20px;'>Normalized comparison of key biomarkers against physiological baselines.</p>", unsafe_allow_html=True)
                
                categories = ['Glucose', 'Cholesterol', 'BMI', 'Blood Pressure', 'Age']
                
                # Extract for visualization
                p_gluc = float(patient_data.get('Glucose', 100))
                p_chol = float(patient_data.get('Cholesterol', 200))
                p_bmi = float(patient_data.get('BMI', 25))
                p_bp = float(patient_data.get('trestbps', patient_data.get('BloodPressure', 120)))
                p_age = float(patient_data.get('Age', 45))
                
                # Normalize values (0-100 scale roughly based on risk thresholds)
                norm_gluc = min(100, (p_gluc / 140) * 100)
                norm_chol = min(100, (p_chol / 240) * 100)
                norm_bmi = min(100, (p_bmi / 30) * 100)
                norm_bp = min(100, (p_bp / 140) * 100)
                norm_age = min(100, (p_age / 80) * 100)
                
                patient_vals = [norm_gluc, norm_chol, norm_bmi, norm_bp, norm_age]
                healthy_vals = [70, 70, 75, 85, 50] 
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=patient_vals, theta=categories, fill='toself', name='Patient Biomarkers', line_color="#00f2fe"
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=healthy_vals, theta=categories, fill='toself', name='Healthy Baseline', line_color="#10b981"
                ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=True, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=20, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # --- Recommendations ---
                st.markdown("<hr class='neon-divider'>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>📋 Unified Recommendations</div>", unsafe_allow_html=True)
                col_rec1, col_rec2 = st.columns(2)
                with col_rec1:
                    st.markdown("<h4 style='color:#00f2fe; margin-bottom:10px;'>Metabolic Guidance</h4>", unsafe_allow_html=True)
                    render_recommendations(dia_result['risk_level'])
                with col_rec2:
                    st.markdown("<h4 style='color:#a855f7; margin-bottom:10px;'>Cardiovascular Guidance</h4>", unsafe_allow_html=True)
                    render_recommendations(hrt_result['risk_level'])
                    
                st.markdown("<hr class='neon-divider'>", unsafe_allow_html=True)
                st.markdown("<div class='section-header' style='font-size: 1.4rem;'>📄 Comprehensive Health Report</div>", unsafe_allow_html=True)
                st.markdown("<p style='color:#94a3b8;'>Download a unified diagnostic summary PDF containing dual risk metrics.</p>", unsafe_allow_html=True)
                
                try:
                    with st.spinner("Synthesizing unified patient report..."):
                        report_path = export_combined_pdf(patient_data, dia_result, hrt_result)
                        with open(report_path, "rb") as pdf_file:
                            pdf_bytes = pdf_file.read()
                            
                        st.download_button(
                            label="Download Unified PDF Report 📥",
                            data=pdf_bytes,
                            file_name=os.path.basename(report_path),
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                        # --- HISTORY HOOK ---
                        user_id = st.session_state.get("user_data", {}).get("id")
                        if user_id:
                            save_screening(user_id, "Diabetes Risk Assessment", dia_result['risk_level'], dia_result['probability_percentage'], report_path)
                            save_screening(user_id, "Heart Disease Risk Assessment", hrt_result['risk_level'], hrt_result['probability_percentage'], report_path)
                            
                except Exception as e:
                    logger.error(f"Unified PDF Generation Error: {e}")
                    st.error(f"⚠️ Failed to generate Unified PDF report: {e}")
                    
            except Exception as e:
                logger.error(f"Prediction UI error: {e}")
                st.error(f"❌ **System Exception:** {e}")

    with tab_data:
        st.info("Displaying underlying datasets for both active ML models.")
        sub_dia, sub_hrt = st.tabs(["Metabolic Dataset (Diabetes)", "Cardiovascular Dataset (Heart)"])
        with sub_dia:
            render_dataset_analytics(dia_config["data_path"], dia_config["target_col"])
        with sub_hrt:
            render_dataset_analytics(hrt_config["data_path"], hrt_config["target_col"])

    with tab_perf:
        st.info("Displaying holdout performance metrics for both serialized production models.")
        sub_dia, sub_hrt = st.tabs(["Metabolic Model Accuracy", "Cardiovascular Model Accuracy"])
        with sub_dia:
            render_model_performance(dia_config["model_path"], dia_config["data_path"], dia_config["scaler_path"], dia_config["preprocess_func"])
        with sub_hrt:
            render_model_performance(hrt_config["model_path"], hrt_config["data_path"], hrt_config["scaler_path"], hrt_config["preprocess_func"])

    with tab_shap:
        if patient_data:
            sub_dia, sub_hrt = st.tabs(["Metabolic Explainability", "Cardiovascular Explainability"])
            with sub_dia:
                render_explainability_section(patient_data, dia_config["model_path"], dia_config["scaler_path"], dia_config["explain_func"])
            with sub_hrt:
                render_explainability_section(patient_data, hrt_config["model_path"], hrt_config["scaler_path"], hrt_config["explain_func"])
        else:
            st.info("👈 Please initialize an AI Inference in the sidebar to generate custom Explainability Insights for this patient.")

    with tab_history:
        render_history_dashboard()

def render_main_dashboard(patient_data: dict, selected_disease: str):
    if selected_disease == "Full Health Screening":
        render_full_screening_dashboard(patient_data)
        return

    # Hero Header
    st.markdown(
        """
        <div class="hero-container">
            <h1 class='hero-title'>AI Health Screening Platform</h1>
            <p class='hero-subtitle'>Unified Predictive Diagnostic Intelligence</p>
        </div>
        """, unsafe_allow_html=True
    )
    
    if selected_disease not in DISEASE_CONFIG:
        st.markdown(
            """
            <div class="empty-state">
                <h3 style="color: #f8fafc; margin-bottom: 10px; letter-spacing: 1px;">Module Under Development</h3>
                <p style="color: #94a3b8; font-size: 1.15rem; max-width: 600px; margin: 0 auto;">
                    This comprehensive multi-organ screening module is currently under active development. Please select an active module from the sidebar.
                </p>
            </div>
            """, unsafe_allow_html=True
        )
        return

    config = DISEASE_CONFIG[selected_disease]
    model_path = config["model_path"]
    scaler_path = config["scaler_path"]
    data_path = config["data_path"]
    predict_func = config["predict_func"]
    explain_func = config["explain_func"]
    preprocess_func = config["preprocess_func"]
    target_col = config["target_col"]
    
    # Primary Dashboard Navigation
    tab_pred, tab_data, tab_perf, tab_shap, tab_history = st.tabs([
        "🩺 AI Prediction", 
        "📊 Dataset Analytics", 
        "📈 Model Performance", 
        "🧠 Explainable AI",
        "📈 My Health History"
    ])
    
    with tab_pred:
        if not patient_data:
            st.markdown(
                """
                <div class="empty-state">
                    <h3 style="color: #f8fafc; margin-bottom: 10px; letter-spacing: 1px;">Awaiting Neural Telemetry</h3>
                    <p style="color: #94a3b8; font-size: 1.15rem; max-width: 600px; margin: 0 auto;">
                        Initiate a secure diagnostic sequence by entering the required clinical parameters in the intake panel.
                    </p>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            try:
                with st.spinner("Executing secure neural inference protocol..."):
                    result = predict_func(
                        patient_data, 
                        model_path=model_path, 
                        scaler_path=scaler_path
                    )
                    
                prob = result["probability_percentage"]
                risk = result["risk_level"]
                label = result["prediction_label"]
                
                st.markdown("<div class='section-header'>🔬 Inference Telemetry</div>", unsafe_allow_html=True)
                
                # Primary Metrics Layer
                m1, m2, m3 = st.columns(3)
                with m1: st.metric(label="Model Output", value="Positive Flag" if label == 1 else "Negative Flag")
                with m2: st.metric(label="Probability Index", value=f"{prob}%")
                with m3: st.metric(label="Risk Stratification", value=risk)
                    
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Diagnostics Layout (Gauge + Recommendations)
                col_gauge, col_rec = st.columns([1, 1.2])
                with col_gauge:
                    st.markdown("<h4 style='text-align: center; color: #f8fafc; font-size: 1.1rem; margin-bottom: -15px;'>Risk Gauge Matrix</h4>", unsafe_allow_html=True)
                    st.plotly_chart(create_gauge_chart(prob), use_container_width=True)
                    render_health_summary(patient_data)
                with col_rec:
                    render_recommendations(risk)
                    
                    # --- DOWNLOAD REPORT SECTION ---
                    st.markdown("<div class='section-header' style='margin-top: 35px; font-size: 1.4rem;'>📄 Clinical Health Report</div>", unsafe_allow_html=True)
                    st.markdown("<p style='color:#94a3b8; font-size: 1.0rem; margin-top: -10px; margin-bottom: 20px;'>Download a comprehensive, patient-friendly diagnostic summary PDF for your medical records.</p>", unsafe_allow_html=True)
                    
                    try:
                        with st.spinner("Synthesizing patient report..."):
                            report_path = export_pdf(patient_data, result)
                            with open(report_path, "rb") as pdf_file:
                                pdf_bytes = pdf_file.read()
                                
                            st.download_button(
                                label="Download PDF Report 📥",
                                data=pdf_bytes,
                                file_name=os.path.basename(report_path),
                                mime="application/pdf",
                                use_container_width=True
                            )
                            st.success("✅ Report generated securely.")
                            
                            # --- HISTORY HOOK ---
                            user_id = st.session_state.get("user_data", {}).get("id")
                            if user_id:
                                save_screening(user_id, selected_disease, risk, prob, report_path)
                                
                    except Exception as e:
                        logger.error(f"PDF Generation Error: {e}")
                        st.error(f"⚠️ Failed to generate PDF report: {e}")
                    
            except FileNotFoundError:
                st.error("❌ **Models Offline:** Ensure backend training pipeline has fully synthesized `_model.pkl` & `_scaler.pkl` for this module.")
            except Exception as e:
                logger.error(f"Prediction UI error: {e}")
                st.error(f"❌ **System Exception:** {e}")

    with tab_data:
        render_dataset_analytics(data_path, target_col)

    with tab_perf:
        render_model_performance(model_path, data_path, scaler_path, preprocess_func)

    with tab_shap:
        if patient_data:
            render_explainability_section(patient_data, model_path, scaler_path, explain_func)
        else:
            st.info("👈 Please initialize an AI Inference in the sidebar to generate custom Explainability Insights for this patient.")

    with tab_history:
        render_history_dashboard()

# --- APP EXECUTION ---
def main():
    inject_custom_css()
    
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        st.session_state["user_data"] = None
        
    if "parsed_data" not in st.session_state:
        st.session_state["parsed_data"] = {}
        
    if not st.session_state["authenticated"]:
        render_auth_page()
    else:
        patient_data, selected_disease = render_sidebar()
        render_main_dashboard(patient_data, selected_disease)

if __name__ == "__main__":
    main()

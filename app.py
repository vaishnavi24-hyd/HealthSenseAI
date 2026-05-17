"""
Disease Prediction App
A Streamlit application for patient disease prediction.
"""

import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Disease Prediction Dashboard",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR ---
def render_sidebar():
    """Renders the sidebar navigation and settings."""
    st.sidebar.title("⚕️ Navigation")
    
    # Navigation options
    app_mode = st.sidebar.radio(
        "Select Mode",
        ["Home", "Patient Input", "Batch Prediction", "Model Insights"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This application uses machine learning to assist in "
        "predicting disease likelihood based on patient clinical records."
    )
    
    return app_mode

# --- MAIN CONTENT ---
def render_home():
    """Renders the home page overview."""
    st.title("Disease Prediction Dashboard")
    st.markdown("### Welcome to the Clinical Decision Support System")
    st.write(
        "Please use the sidebar to navigate to the Patient Input form "
        "or view Batch Predictions and Model Insights."
    )

def render_patient_form():
    """Renders the patient input form and prediction section."""
    st.title("Patient Information Form")
    st.markdown("Enter the patient's clinical data below to get a prediction.")
    
    # Placeholder Patient Form
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input("Age", min_value=0, max_value=120, value=45, step=1)
            st.selectbox("Gender", ["Male", "Female", "Other"])
            st.number_input("Blood Pressure (Systolic)", min_value=50, max_value=250, value=120)
            
        with col2:
            st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=400, value=200)
            st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=75)
            st.selectbox("Smoking Status", ["Never Smoked", "Former Smoker", "Current Smoker"])
            
        st.text_area("Additional Medical History Notes")
        
        submit_button = st.form_submit_button("Generate Prediction")
        
    # Placeholder Prediction Section
    if submit_button:
        st.markdown("---")
        st.subheader("Prediction Results")
        
        # Simulating a prediction result
        st.warning("⚠️ **High Risk**")
        st.progress(0.75, text="Probability: 75%")
        
        st.info(
            "Note: This is a placeholder prediction. "
            "Please integrate the trained model in `utils/predict.py`."
        )

# --- APP EXECUTION ---
def main():
    """Main function to run the Streamlit app."""
    app_mode = render_sidebar()
    
    if app_mode == "Home":
        render_home()
    elif app_mode == "Patient Input":
        render_patient_form()
    else:
        st.title(app_mode)
        st.info(f"The {app_mode} page is currently under construction.")

if __name__ == "__main__":
    main()

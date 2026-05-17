# Disease Prediction App

## Project Overview
A production-quality Streamlit application for predicting disease outcomes based on patient data. This application utilizes machine learning models to provide predictions and leverages interpretability tools to explain the results.

## Tech Stack
- **Frontend**: Streamlit
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Model Explainability**: SHAP
- **Model Serialization**: Joblib
- **Reporting**: FPDF

## Setup Instructions

1. Clone the repository or navigate to the project directory:
   ```bash
   cd disease-prediction-app
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Folder Structure
- `assets/`: Static assets like images and styles.
- `data/`: Datasets used for training and prediction.
- `models/`: Serialized machine learning models.
- `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA) and experimentation.
- `reports/`: Generated PDF reports.
- `utils/`: Helper scripts and modules.
  - `preprocess.py`: Data preprocessing utilities.
  - `train.py`: Model training scripts.
  - `predict.py`: Inference and prediction logic.
  - `explain.py`: Model explainability (SHAP).
  - `report.py`: PDF report generation.
- `app.py`: Main Streamlit application entry point.
- `requirements.txt`: Project dependencies.

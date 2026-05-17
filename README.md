# 🩺 HealthSense AI

## AI-Powered Preventive Healthcare Intelligence Platform

HealthSense AI is a full-stack AI healthcare screening platform designed to help users assess chronic disease risk using machine learning, explainable AI, advanced analytics, and patient-friendly healthcare reporting.

The platform currently supports:

* Diabetes Risk Assessment
* Heart Disease Risk Assessment
* Full Health Screening
* Explainable AI Insights
* PDF Health Reports
* Patient Authentication & History Tracking

---

# 🚀 Features

## 🔍 AI Health Screening

* Diabetes risk prediction
* Heart disease risk prediction
* Unified health screening workflow
* Risk probability analysis
* Intelligent health recommendations

## 🧠 Explainable AI

* SHAP-based explainability
* Feature contribution analysis
* Interactive explainability charts
* Patient-friendly health factor insights

## 📊 Advanced Analytics Dashboard

* Interactive visualizations
* Risk trend analytics
* Correlation heatmaps
* Model performance analysis
* Longitudinal health tracking

## 📄 PDF Report Generation

* Patient-friendly downloadable reports
* Health summaries
* Risk interpretation
* Personalized recommendations
* Clinical-style formatting

## 🔐 Authentication System

* Secure signup/login
* Password hashing using bcrypt
* Session-based access control
* Protected dashboard access

## 🗂 Patient History System

* Historical screening records
* Health trend tracking
* Report archive management
* Longitudinal analytics

## 🎨 Premium Healthcare UI

* Futuristic dark SaaS design
* Responsive layouts
* Interactive dashboards
* Modern healthcare visualizations

---

# 🏗 System Architecture

```text
User Input
    ↓
Authentication Layer
    ↓
AI Screening Engine
    ↓
Disease-Specific ML Models
    ↓
Explainable AI (SHAP)
    ↓
Analytics Dashboard
    ↓
PDF Report Generation
    ↓
Patient History Database
```

---

# 🧠 Machine Learning Workflow

## 1. Data Preprocessing

* Data cleaning
* Feature scaling
* Train-test splitting
* Standardization using StandardScaler

## 2. Model Training

Models used:

* RandomForestClassifier
* LogisticRegression
* Support Vector Machine (SVM)

## 3. Model Evaluation

Metrics:

* Accuracy
* Precision
* Recall
* F1 Score

## 4. Explainable AI

* SHAP explainability
* Feature importance analysis
* Waterfall visualizations
* Risk contribution analysis

---

# 🧪 Supported Assessments

| Assessment               | Status |
| ------------------------ | ------ |
| Diabetes Risk Prediction | ✅      |
| Heart Disease Prediction | ✅      |
| Full Health Screening    | ✅      |
| Explainable AI Insights  | ✅      |
| PDF Health Reports       | ✅      |
| Patient History Tracking | ✅      |

---

# 🛠 Tech Stack

## Frontend

* Streamlit
* Plotly
* HTML/CSS

## Backend

* Python
* SQLite
* Pandas
* NumPy

## Machine Learning

* Scikit-learn
* SHAP
* Joblib

## Reporting

* ReportLab
* Matplotlib

## Authentication

* bcrypt
* SQLite

---

# 📁 Project Structure

```text
disease-prediction-app/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── diabetes.csv
│   └── heart.csv
│
├── models/
│   ├── diabetes_model.pkl
│   ├── diabetes_scaler.pkl
│   ├── heart_model.pkl
│   └── heart_scaler.pkl
│
├── reports/
│
├── database/
│   ├── db.py
│   ├── auth.py
│   ├── history.py
│   ├── models.py
│   └── healthcare.db
│
├── utils/
│   ├── diabetes/
│   └── heart/
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/healthsense-ai.git
cd healthsense-ai
```

## 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run Application

```bash
streamlit run app.py
```


# 👨‍💻 Developer

Developed as an end-to-end AI healthcare analytics and preventive screening platform integrating:

* Machine Learning
* Explainable AI
* Full-Stack Development
* Healthcare Analytics
* Data Visualization
* Authentication Systems
* PDF Reporting

---

# 📌 Key Highlights

✅ Multi-Disease AI Architecture
✅ Explainable AI using SHAP
✅ Advanced Analytics Dashboard
✅ Secure Authentication System
✅ Patient History Tracking
✅ PDF Health Reports
✅ Premium SaaS UI Design
✅ Modular Scalable Architecture

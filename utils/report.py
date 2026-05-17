"""
Patient-Friendly Health Report Generator
Generates clinical summary PDFs using ReportLab.
"""
import os
import logging
import tempfile
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable

# Configure logging
logger = logging.getLogger(__name__)

def generate_header(elements: list, styles: dict):
    """Generates the professional report header."""
    title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=15,
        alignment=1 # Center
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=30,
        alignment=1
    )
    
    elements.append(Paragraph("<b>AegisHealth Diagnostics</b>", title_style))
    date_str = datetime.now().strftime("%B %d, %Y - %H:%M")
    elements.append(Paragraph(f"Confidential Patient Health Assessment<br/>Generated: {date_str}", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceBefore=0, spaceAfter=20))

def generate_patient_summary(elements: list, styles: dict, patient_data: Dict[str, Any]):
    """Generates the patient clinical profile table."""
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#2563eb"), # Blue branding
        spaceBefore=10,
        spaceAfter=15
    )
    elements.append(Paragraph("<b>Patient Clinical Profile</b>", section_style))
    
    friendly_keys = {
        "Pregnancies": "Pregnancies (count)",
        "Glucose": "Glucose Level (mg/dL)",
        "BloodPressure": "Blood Pressure (mmHg)",
        "SkinThickness": "Skin Thickness (mm)",
        "Insulin": "Insulin Level (IU/mL)",
        "BMI": "Body Mass Index (BMI)",
        "DiabetesPedigreeFunction": "Family History Factor",
        "Age": "Age (years)"
    }
    
    table_data = [["Clinical Metric", "Recorded Value"]]
    for k, v in patient_data.items():
        name = friendly_keys.get(k, k)
        val = f"{v:.2f}" if isinstance(v, float) else str(v)
        table_data.append([name, val])
        
    t = Table(table_data, colWidths=[4 * inch, 2 * inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor("#334155")),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 25))

def generate_risk_assessment(elements: list, styles: dict, prediction_results: Dict[str, Any]) -> str:
    """Generates the primary risk output and dynamic risk gauge image."""
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#2563eb"),
        spaceBefore=10,
        spaceAfter=15
    )
    normal_style = ParagraphStyle(
        'NormalText',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor("#334155"),
        spaceAfter=15,
        leading=16
    )
    
    elements.append(Paragraph("<b>Diagnostic Risk Assessment</b>", section_style))
    
    prob = prediction_results.get("probability_percentage", 0.0)
    risk = prediction_results.get("risk_level", "UNKNOWN")
    
    if risk == "LOW":
        risk_color = "#10b981"
        explanation = "Your clinical metrics align with a <b>low risk</b> category for developing diabetes. Your parameters are generally within healthy biological ranges."
    elif risk == "MEDIUM":
        risk_color = "#f59e0b"
        explanation = "Your clinical metrics indicate an <b>elevated risk</b> for developing diabetes. Certain parameters fall outside of optimal biological ranges."
    else:
        risk_color = "#ef4444"
        explanation = "Your clinical metrics strongly indicate a <b>high risk</b> for developing diabetes. Several key parameters are significantly outside safe medical thresholds."
        
    result_text = f"<b>Assessed Risk Tier:</b> <font color='{risk_color}'>{risk}</font><br/><b>Estimated Probability Score:</b> {prob}%"
    
    elements.append(Paragraph(result_text, normal_style))
    elements.append(Paragraph(explanation, normal_style))
    
    # Generate Matplotlib Gauge Image
    chart_path = ""
    try:
        fig, ax = plt.subplots(figsize=(6, 1.0))
        
        # Background segments (Low, Medium, High)
        ax.barh(0, 35, color='#d1fae5', height=0.5) 
        ax.barh(0, 35, left=35, color='#fef3c7', height=0.5) 
        ax.barh(0, 30, left=70, color='#fee2e2', height=0.5) 
        
        # Actual score bar
        color = '#10b981' if risk == "LOW" else '#f59e0b' if risk == "MEDIUM" else '#ef4444'
        ax.barh(0, prob, color=color, height=0.5)
        
        # Formatting
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xticks([0, 35, 70, 100])
        ax.set_xticklabels(['0%', '35%', '70%', '100%'], color='#475569', fontweight='bold', fontsize=9)
        
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        ax.text(prob, 0.30, f"{prob}%", ha='center', va='bottom', fontweight='bold', color=color, fontsize=11)
        
        fd, chart_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', transparent=True)
        plt.close(fig)
        
        img = Image(chart_path, width=6*inch, height=1.0*inch)
        elements.append(img)
        elements.append(Spacer(1, 15))
        
    except Exception as e:
        logger.error(f"Failed to generate risk chart image: {e}")
        
    return chart_path

def generate_recommendations(elements: list, styles: dict, risk_level: str):
    """Generates patient-friendly clinical recommendations based on risk."""
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#2563eb"),
        spaceBefore=10,
        spaceAfter=15
    )
    bullet_style = ParagraphStyle(
        'BulletItem',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor("#334155"),
        spaceAfter=8,
        leftIndent=15,
        bulletIndent=5,
        leading=14
    )
    
    elements.append(Paragraph("<b>Health & Lifestyle Recommendations</b>", section_style))
    
    if risk_level == "LOW":
        recs = [
            "Maintain your current balanced nutritional diet.",
            "Continue engaging in regular physical activity (aim for 150 minutes per week).",
            "Keep up with your standard annual preventative health checkups."
        ]
    elif risk_level == "MEDIUM":
        recs = [
            "Schedule a proactive checkup with your primary care physician to discuss these results.",
            "Review your dietary habits, focusing on reducing added sugars and refined carbohydrates.",
            "Aim to incrementally increase your daily physical activity levels.",
            "Monitor your blood pressure and fasting glucose levels periodically."
        ]
    else:
        recs = [
            "<b>Please consult a healthcare professional immediately for a comprehensive medical evaluation.</b>",
            "Comprehensive blood panels (such as HbA1c or Fasting Plasma Glucose) are strongly recommended.",
            "Work closely with your doctor to establish a proactive health management plan.",
            "Implement critical lifestyle and dietary modifications under professional medical guidance."
        ]
        
    for r in recs:
        elements.append(Paragraph(f"• {r}", bullet_style))
        
    elements.append(Spacer(1, 20))

def generate_disclaimer(elements: list, styles: dict):
    """Generates the legal medical disclaimer."""
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Italic'],
        fontSize=9,
        textColor=colors.HexColor("#94a3b8"),
        spaceBefore=20,
        alignment=1,
        leading=12
    )
    
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceBefore=20, spaceAfter=15))
    
    disclaimer = (
        "<b>Medical Disclaimer:</b> This report was generated by an artificial intelligence diagnostic tool "
        "and is intended strictly for informational and educational purposes. It does <b>NOT</b> constitute a "
        "definitive medical diagnosis. Human health involves complex biological factors that algorithms cannot fully evaluate. "
        "Always seek the advice of your physician or other qualified healthcare provider regarding any medical condition."
    )
    elements.append(Paragraph(disclaimer, disclaimer_style))

def export_pdf(patient_data: Dict[str, Any], prediction_results: Dict[str, Any], output_dir: str = None) -> str:
    """
    Orchestrates the generation of the professional PDF report.
    
    Args:
        patient_data (Dict[str, Any]): Dictionary of patient clinical inputs.
        prediction_results (Dict[str, Any]): Dictionary of model prediction outputs.
        output_dir (str, optional): Destination directory for the PDF. If None, defaults to disease-prediction-app/reports/
        
    Returns:
        str: Absolute path to the generated PDF.
    """
    logger.info("Initializing patient-friendly PDF report generation.")
    
    if output_dir is None:
        # Resolve path robustly to project_root/reports
        project_root = Path(__file__).resolve().parent.parent
        out_path = project_root / "reports"
    else:
        out_path = Path(output_dir)
        
    out_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Patient_Health_Summary_{timestamp}.pdf"
    filepath = str(out_path / filename)
    
    doc = SimpleDocTemplate(
        filepath,
        pagesize=letter,
        rightMargin=1*inch, leftMargin=1*inch,
        topMargin=1*inch, bottomMargin=1*inch
    )
    
    styles = getSampleStyleSheet()
    elements = []
    
    generate_header(elements, styles)
    generate_patient_summary(elements, styles, patient_data)
    chart_path = generate_risk_assessment(elements, styles, prediction_results)
    generate_recommendations(elements, styles, prediction_results.get("risk_level", "UNKNOWN"))
    generate_disclaimer(elements, styles)
    
    try:
        doc.build(elements)
        logger.info(f"Successfully generated PDF report at {filepath}")
    except Exception as e:
        logger.error(f"Failed to build PDF document: {e}")
        raise e
    finally:
        # Cleanup temporary matplotlib chart
        if chart_path and os.path.exists(chart_path):
            os.remove(chart_path)
            
    return filepath

def generate_combined_risk_assessment(elements: list, styles: dict, dia_results: Dict[str, Any], hrt_results: Dict[str, Any]) -> List[str]:
    """Generates dual risk output and dynamic risk gauge images."""
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#2563eb"),
        spaceBefore=10,
        spaceAfter=15
    )
    normal_style = ParagraphStyle(
        'NormalText',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor("#334155"),
        spaceAfter=15,
        leading=16
    )
    
    chart_paths = []
    
    # --- Diabetes Assessment ---
    elements.append(Paragraph("<b>Metabolic Diagnostic Risk Assessment (Diabetes)</b>", section_style))
    dia_prob = dia_results.get("probability_percentage", 0.0)
    dia_risk = dia_results.get("risk_level", "UNKNOWN")
    dia_color = '#10b981' if dia_risk == "LOW" else '#f59e0b' if dia_risk == "MEDIUM" else '#ef4444'
    dia_text = f"<b>Assessed Risk Tier:</b> <font color='{dia_color}'>{dia_risk}</font><br/><b>Estimated Probability Score:</b> {dia_prob}%"
    elements.append(Paragraph(dia_text, normal_style))
    
    try:
        fig, ax = plt.subplots(figsize=(6, 1.0))
        ax.barh(0, 35, color='#d1fae5', height=0.5) 
        ax.barh(0, 35, left=35, color='#fef3c7', height=0.5) 
        ax.barh(0, 30, left=70, color='#fee2e2', height=0.5) 
        ax.barh(0, dia_prob, color=dia_color, height=0.5)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xticks([0, 35, 70, 100])
        ax.set_xticklabels(['0%', '35%', '70%', '100%'], color='#475569', fontweight='bold', fontsize=9)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.text(dia_prob, 0.30, f"{dia_prob}%", ha='center', va='bottom', fontweight='bold', color=dia_color, fontsize=11)
        fd, chart_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', transparent=True)
        plt.close(fig)
        
        img = Image(chart_path, width=6*inch, height=1.0*inch)
        elements.append(img)
        chart_paths.append(chart_path)
    except Exception as e:
        logger.error(f"Failed to generate diabetes risk chart image: {e}")
        
    elements.append(Spacer(1, 15))
    
    # --- Heart Assessment ---
    elements.append(Paragraph("<b>Cardiovascular Diagnostic Risk Assessment (Heart)</b>", section_style))
    hrt_prob = hrt_results.get("probability_percentage", 0.0)
    hrt_risk = hrt_results.get("risk_level", "UNKNOWN")
    hrt_color = '#10b981' if hrt_risk == "LOW" else '#f59e0b' if hrt_risk == "MEDIUM" else '#ef4444'
    hrt_text = f"<b>Assessed Risk Tier:</b> <font color='{hrt_color}'>{hrt_risk}</font><br/><b>Estimated Probability Score:</b> {hrt_prob}%"
    elements.append(Paragraph(hrt_text, normal_style))
    
    try:
        fig, ax = plt.subplots(figsize=(6, 1.0))
        ax.barh(0, 35, color='#d1fae5', height=0.5) 
        ax.barh(0, 35, left=35, color='#fef3c7', height=0.5) 
        ax.barh(0, 30, left=70, color='#fee2e2', height=0.5) 
        ax.barh(0, hrt_prob, color=hrt_color, height=0.5)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xticks([0, 35, 70, 100])
        ax.set_xticklabels(['0%', '35%', '70%', '100%'], color='#475569', fontweight='bold', fontsize=9)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.text(hrt_prob, 0.30, f"{hrt_prob}%", ha='center', va='bottom', fontweight='bold', color=hrt_color, fontsize=11)
        fd, chart_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', transparent=True)
        plt.close(fig)
        
        img = Image(chart_path, width=6*inch, height=1.0*inch)
        elements.append(img)
        chart_paths.append(chart_path)
    except Exception as e:
        logger.error(f"Failed to generate heart risk chart image: {e}")

    elements.append(Spacer(1, 15))
    return chart_paths

def export_combined_pdf(patient_data: Dict[str, Any], dia_results: Dict[str, Any], hrt_results: Dict[str, Any], output_dir: str = None) -> str:
    """
    Orchestrates the generation of the unified professional PDF report.
    """
    logger.info("Initializing unified patient-friendly PDF report generation.")
    
    if output_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        out_path = project_root / "reports"
    else:
        out_path = Path(output_dir)
        
    out_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Unified_Health_Summary_{timestamp}.pdf"
    filepath = str(out_path / filename)
    
    doc = SimpleDocTemplate(
        filepath,
        pagesize=letter,
        rightMargin=1*inch, leftMargin=1*inch,
        topMargin=1*inch, bottomMargin=1*inch
    )
    
    styles = getSampleStyleSheet()
    elements = []
    
    generate_header(elements, styles)
    generate_patient_summary(elements, styles, patient_data)
    chart_paths = generate_combined_risk_assessment(elements, styles, dia_results, hrt_results)
    
    elements.append(Paragraph("<b>Metabolic Recommendations</b>", ParagraphStyle('Sub', parent=styles['Heading3'], textColor=colors.HexColor("#2563eb"))))
    generate_recommendations(elements, styles, dia_results.get("risk_level", "UNKNOWN"))
    
    elements.append(Paragraph("<b>Cardiovascular Recommendations</b>", ParagraphStyle('Sub', parent=styles['Heading3'], textColor=colors.HexColor("#2563eb"))))
    generate_recommendations(elements, styles, hrt_results.get("risk_level", "UNKNOWN"))
    
    generate_disclaimer(elements, styles)
    
    try:
        doc.build(elements)
        logger.info(f"Successfully generated Unified PDF report at {filepath}")
    except Exception as e:
        logger.error(f"Failed to build Unified PDF document: {e}")
        raise e
    finally:
        for chart_path in chart_paths:
            if chart_path and os.path.exists(chart_path):
                os.remove(chart_path)
            
    return filepath

if __name__ == "__main__":
    # Test block
    logging.basicConfig(level=logging.INFO)
    
    sample_patient = {
        "Pregnancies": 2,
        "Glucose": 135.0,
        "BloodPressure": 75.0,
        "SkinThickness": 30.0,
        "Insulin": 120.0,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.45,
        "Age": 42
    }
    
    sample_prediction = {
        "prediction_label": 1,
        "probability_percentage": 68.5,
        "risk_level": "MEDIUM"
    }
    
    print("Generating sample patient report...")
    try:
        report_path = export_pdf(sample_patient, sample_prediction)
        print(f"Success! Report saved to: {report_path}")
    except Exception as ex:
        print(f"Error generating report: {ex}")

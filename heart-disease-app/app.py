"""
CardioCare AI — enhanced Streamlit application
Keeps the existing trained assets unchanged:
    RF_KNN_heart.pkl
    scaler.pkl
    columns.pkl

Run:
    streamlit run app.py

Optional PDF dependency:
    pip install reportlab
"""

import os
import io
import html
import re
import json
import sqlite3
import requests
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------- PAGE CONFIG -----------------------------
st.set_page_config(
    page_title="CardioCare AI | Heart Health Insights",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "RF_KNN_heart.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "columns.pkl")
HISTORY_PATH = os.path.join(BASE_DIR, "prediction_history.csv")
CLINICAL_DB_PATH = os.path.join(BASE_DIR, "cardiocare_clinical.db")

# ----------------------------- PREMIUM THEME ----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
    :root {
        --navy:#10243A; --navy-2:#183B55; --teal:#087F83; --aqua:#20B8AE;
        --mint:#E8F7F5; --ink:#183247; --muted:#52677A; --line:#DCE8EC;
    }
    html, body, [class*="css"], .stApp { font-family:'DM Sans',sans-serif; }
    .stApp { background:radial-gradient(circle at 88% 4%,#E2F5F2 0,transparent 25%),linear-gradient(180deg,#F7FAFC 0%,#EEF4F7 100%); color:var(--ink); }
    .block-container { max-width:1500px; padding-top:1.2rem; padding-bottom:3.2rem; }
    h1,h2,h3,h4 { font-family:'Manrope',sans-serif; color:var(--navy) !important; letter-spacing:-.035em; }
    h2 { font-size:1.85rem !important; margin-top:.4rem !important; }
    h3 { font-size:1.28rem !important; }
    p, label, [data-testid="stCaptionContainer"] { color:#43596B; }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#10243A 0%,#153C50 100%); border-right:1px solid rgba(255,255,255,.08); }
    [data-testid="stSidebar"] * { color:#F8FAFC !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { padding:.55rem .7rem; border-radius:10px; transition:background .18s ease; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:rgba(255,255,255,.10); }
    [data-testid="stSidebar"] [data-testid="stRadio"] input:checked + div { color:#8DE5DC !important; }
    [data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.15); }
    [data-testid="stSidebar"] [data-testid="stTextInput"] input { background:#F8FBFC !important; color:#183247 !important; border-radius:10px !important; }
    div.stButton > button, div.stDownloadButton > button, [data-testid="stFormSubmitButton"] button {
        border-radius:13px; border:0; font-weight:800; min-height:2.85rem;
        background:linear-gradient(120deg,#087F83,#13A89F); color:#fff !important;
        box-shadow:0 7px 20px rgba(8,127,131,.20); transition:all .2s ease;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
        transform:translateY(-2px); box-shadow:0 11px 25px rgba(8,127,131,.28); border:0;
    }
    /* Explicit metric colours prevent pale/white labels in dark-mode browsers. */
    div[data-testid="stMetric"] {
        background:linear-gradient(145deg,#FFFFFF 0%,#F5FBFB 100%) !important;
        border:1px solid #D9E9EC !important; border-radius:19px !important;
        padding:19px 20px !important; min-height:116px;
        box-shadow:0 8px 24px rgba(16,36,58,.07) !important;
        transition:transform .2s ease, box-shadow .2s ease;
    }
    div[data-testid="stMetric"]:hover { transform:translateY(-3px); box-shadow:0 13px 28px rgba(16,36,58,.11) !important; }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
    div[data-testid="stMetric"] label { color:#52677A !important; opacity:1 !important; font-weight:700 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] * { color:#10243A !important; opacity:1 !important; font-weight:800 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] { color:#087F83 !important; }
    .hero {
        position:relative; overflow:hidden; padding:2.35rem 2.5rem; border-radius:26px; color:white;
        background:linear-gradient(112deg,#10243A 0%,#155365 52%,#0B9692 100%);
        box-shadow:0 18px 42px rgba(16,36,58,.16); margin-bottom:1.55rem;
        border:1px solid rgba(255,255,255,.18);
    }
    .hero:after { content:''; position:absolute; width:260px; height:260px; right:5%; top:-150px; border:1px solid rgba(255,255,255,.16); border-radius:50%; box-shadow:0 0 0 28px rgba(255,255,255,.035),0 0 0 58px rgba(255,255,255,.025); }
    .hero h1 { color:white; font-size:2.55rem; margin:0 0 .45rem 0; }
    .hero p { color:#D8ECEE; font-size:1.02rem; margin:0; }
    .eyebrow { text-transform:uppercase; letter-spacing:.15em; font-size:.72rem; font-weight:800; color:#9BD8D3; }
    .section-card {
        background:linear-gradient(145deg,#FFFFFF 0%,#F8FCFC 100%); border:1px solid #DCE8EC; border-radius:20px;
        padding:1.35rem 1.5rem; box-shadow:0 8px 24px rgba(16,36,58,.055);
        margin-bottom:1rem; color:#183247;
    }
    .section-card b { color:#10243A; }
    .small-muted { color:#52677A; font-size:.9rem; }
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stDateInput"] input { border-radius:11px !important; border-color:#C9D9DF !important; }
    [data-testid="stTextInput"] input:focus, [data-testid="stNumberInput"] input:focus { border-color:#0F9995 !important; box-shadow:0 0 0 2px rgba(15,153,149,.12) !important; }
    [data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid #DCE8EC; border-radius:14px; overflow:hidden; }
    [data-testid="stAlert"] { border-radius:13px; }
    [data-testid="stExpander"] { background:#fff; border:1px solid #DCE8EC; border-radius:13px; }
    .stProgress > div > div > div > div { background:linear-gradient(90deg,#087F83,#20B8AE); }
    .pill { display:inline-block; background:#E8F5F3; color:#126B78; border-radius:99px;
            padding:.3rem .7rem; font-size:.75rem; font-weight:700; }
    .warning-note { background:#FFF7E8; border-left:4px solid #E4A23A; padding:.9rem 1rem;
                    border-radius:8px; color:#694B17; }
    .footer { color:#718096; text-align:center; padding:1.5rem 0 .5rem; font-size:.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- MODEL LOADING ----------------------------
@st.cache_resource
def load_model_assets():
    """Load the user's existing model/scaler/column files without modifying them."""
    missing = [p for p in (MODEL_PATH, SCALER_PATH, COLUMNS_PATH) if not os.path.exists(p)]
    if missing:
        return None, None, None, missing
    try:
        return (
            joblib.load(MODEL_PATH),
            joblib.load(SCALER_PATH),
            joblib.load(COLUMNS_PATH),
            [],
        )
    except Exception as exc:
        return None, None, None, [f"Could not load model assets: {exc}"]


model, scaler, expected_columns, model_errors = load_model_assets()

# ----------------------------- HELPERS ----------------------------------
def build_patient_frame(values):
    """
    Preserve the original app's one-hot feature naming and saved column order.
    Do not change this mapping unless the original training preprocessing changes.
    """
    raw = {
        "Age": values["Age"],
        "RestingBP": values["RestingBP"],
        "Cholesterol": values["Cholesterol"],
        "FastingBS": values["FastingBS"],
        "MaxHR": values["MaxHR"],
        "Oldpeak": values["Oldpeak"],
        "Sex_" + values["Sex"]: 1,
        "ChestPainType_" + values["ChestPainType"]: 1,
        "RestingECG_" + values["RestingECG"]: 1,
        "ExerciseAngina_" + values["ExerciseAngina"]: 1,
        "ST_Slope_" + values["ST_Slope"]: 1,
    }
    frame = pd.DataFrame([raw])
    ordered_columns = list(expected_columns)
    frame = frame.reindex(columns=ordered_columns, fill_value=0)
    frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0)
    return frame


def positive_class_index():
    """Find class 1 where possible; otherwise use the second class for binary models."""
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None, None
    classes = list(classes)
    for i, label in enumerate(classes):
        if label == 1 or str(label).strip() == "1":
            return i, label
    if len(classes) == 2:
        return 1, classes[1]
    return None, None


def run_prediction(values):
    if model is None or scaler is None or expected_columns is None:
        raise RuntimeError("Model, scaler, or columns file is missing. Check the three .pkl files.")
    patient_frame = build_patient_frame(values)
    scaled = scaler.transform(patient_frame)
    prediction = model.predict(scaled)[0]
    classes = list(getattr(model, "classes_", []))
    positive_idx, positive_label = positive_class_index()
    score = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled)[0]
        if positive_idx is not None and positive_idx < len(probabilities):
            score = float(probabilities[positive_idx])
    is_positive = (prediction == positive_label) if positive_label is not None else bool(prediction == 1)
    return {
        "prediction": prediction,
        "is_positive": bool(is_positive),
        "model_score": score,
        "class_labels": classes,
        "input_frame": patient_frame,
        "scaled_input": scaled,
    }


def load_history():
    if not os.path.exists(HISTORY_PATH):
        return pd.DataFrame()
    try:
        return pd.read_csv(HISTORY_PATH)
    except Exception:
        return pd.DataFrame()


def save_history(profile, values, result):
    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Patient": profile["name"],
        "Patient ID": profile["patient_id"],
        "Prediction": "Model class flagged" if result["is_positive"] else "Model class not flagged",
        "Predicted class": str(result["prediction"]),
        "Model score (not calibrated risk)": (
            round(result["model_score"] * 100, 2) if result["model_score"] is not None else ""
        ),
    }
    record.update(values)
    old = load_history()
    updated = pd.concat([old, pd.DataFrame([record])], ignore_index=True)
    updated.to_csv(HISTORY_PATH, index=False)


def feature_importance_frame():
    """Show native model-level importance only when the estimator exposes it."""
    if model is None:
        return None
    importance = getattr(model, "feature_importances_", None)
    if importance is None and hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            importance = getattr(step, "feature_importances_", None)
            if importance is not None:
                break
    if importance is None:
        coef = getattr(model, "coef_", None)
        if coef is None and hasattr(model, "named_steps"):
            for step in reversed(list(model.named_steps.values())):
                coef = getattr(step, "coef_", None)
                if coef is not None:
                    break
        if coef is not None:
            importance = np.mean(np.abs(np.asarray(coef)), axis=0)
    if importance is None:
        return None
    names = list(expected_columns) if expected_columns is not None else []
    vals = np.asarray(importance).ravel()
    if len(names) != len(vals):
        names = [f"Feature {i+1}" for i in range(len(vals))]
    return pd.DataFrame({"Feature": names, "Importance": vals}).sort_values(
        "Importance", ascending=False
    )


def make_pdf(profile, values, result):
    """Create a compact PDF report using ReportLab, if installed."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    except ImportError:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=16*mm, bottomMargin=16*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Brand", parent=styles["Title"], textColor=colors.HexColor("#10243A"),
                              fontSize=21, leading=26, spaceAfter=5))
    styles.add(ParagraphStyle(name="Subtle", parent=styles["Normal"], textColor=colors.HexColor("#64748B"),
                              fontSize=9, leading=13))
    story = [
        Paragraph("CardioCare AI", styles["Brand"]),
        Paragraph("Heart Health Screening Summary", styles["Heading2"]),
        Paragraph(f"Generated: {html.escape(datetime.now().strftime('%d %b %Y, %I:%M %p'))}", styles["Subtle"]),
        Spacer(1, 8*mm),
        Paragraph("Patient profile", styles["Heading2"]),
    ]
    profile_data = [
        ["Patient name", profile["name"] or "Not provided"],
        ["Patient ID", profile["patient_id"] or "Not provided"],
    ]
    story.append(Table(profile_data, colWidths=[45*mm, 115*mm], style=TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#E8F5F3")),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#10243A")),
        ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#DCE7EA")),
        ("PADDING", (0,0), (-1,-1), 7),
    ])))
    story += [Spacer(1, 7*mm), Paragraph("Model output", styles["Heading2"])]
    label = "Class flagged by model" if result["is_positive"] else "Class not flagged by model"
    score_text = (
        f"{result['model_score']*100:.1f}% (model probability; not clinically calibrated)"
        if result["model_score"] is not None else "Not available"
    )
    story.append(Table([["Classification", label], ["Model score", score_text]],
                       colWidths=[45*mm, 115*mm], style=TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#E8F5F3")),
        ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#DCE7EA")),
        ("PADDING", (0,0), (-1,-1), 7),
    ])))
    story += [Spacer(1, 7*mm), Paragraph("Entered measurements", styles["Heading2"])]
    rows = [["Parameter", "Value"]]
    rows.extend([[str(k), str(v)] for k, v in values.items()])
    story.append(Table(rows, colWidths=[75*mm, 85*mm], repeatRows=1, style=TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#10243A")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#DCE7EA")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F6FAFB")]),
        ("PADDING", (0,0), (-1,-1), 6),
    ])))
    story += [
        Spacer(1, 8*mm),
        Paragraph("Important medical notice", styles["Heading2"]),
        Paragraph(
            "This report is generated by an educational machine-learning screening project. "
            "It is not a diagnosis, does not establish the presence or absence of heart disease, "
            "and must not be used to make treatment decisions. Model probabilities may not be "
            "calibrated to real-world clinical risk. Discuss health concerns and test results with "
            "a qualified healthcare professional. If you may be having a heart attack, seek "
            "emergency medical care immediately.", styles["BodyText"]
        ),
    ]
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def lifestyle_tips(values):
    tips = [
        "Choose a varied eating pattern with vegetables, fruit, whole grains, beans, nuts and suitable protein sources.",
        "Build toward regular moderate physical activity if it is safe for you; ask a clinician before starting if you have symptoms or known disease.",
        "Avoid tobacco and seek support to stop smoking or vaping.",
        "Discuss blood pressure, cholesterol and blood sugar measurements with a healthcare professional.",
        "Prioritize consistent sleep and practical stress-management habits.",
    ]
    if values["RestingBP"] >= 130:
        tips.insert(0, "Your entered resting blood pressure is elevated relative to common adult reference targets. Recheck correctly and discuss repeated high readings with a clinician.")
    if values["Cholesterol"] >= 200:
        tips.insert(0, "Your entered total cholesterol is at or above 200 mg/dL. A clinician can interpret a complete lipid panel, including LDL and HDL.")
    if values["FastingBS"] == 1:
        tips.insert(0, "You selected fasting blood sugar above 120 mg/dL. Confirm the actual laboratory result and discuss glucose testing with a clinician.")
    return tips


def chatbot_answer(question, chat_history=None):
    import os
    import requests

    # Read API keys from Streamlit Secrets or environment variables
    try:
        openai_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        openai_key = ""

    if not openai_key:
        openai_key = os.getenv("OPENAI_API_KEY", "")

    try:
        tavily_key = st.secrets.get("TAVILY_API_KEY", "")
    except Exception:
        tavily_key = ""

    if not tavily_key:
        tavily_key = os.getenv("TAVILY_API_KEY", "")

    if not openai_key:
        return {
            "answer": (
                "OpenAI API key is missing. Please add OPENAI_API_KEY "
                "in your Streamlit Cloud app settings under Secrets."
            ),
            "images": [],
            "sources": []
        }

    images = []
    sources = []
    web_context = ""

    # Search the web for current heart-health information when available
    if tavily_key:
        try:
            search_response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": tavily_key,
                    "query": question + " heart health medical information",
                    "search_depth": "advanced",
                    "max_results": 5,
                    "include_images": True
                },
                timeout=30
            )

            if search_response.status_code == 200:
                search_data = search_response.json()

                for item in search_data.get("results", []):
                    title = item.get("title", "Medical reference")
                    url = item.get("url", "")
                    content = item.get("content", "")

                    if url:
                        sources.append({
                            "title": title,
                            "url": url
                        })

                    if content:
                        web_context += (
                            f"\nTitle: {title}\n"
                            f"Information: {content}\n"
                        )

                for item in search_data.get("images", []):
                    if isinstance(item, str):
                        images.append(item)
                    elif isinstance(item, dict) and item.get("url"):
                        images.append(item["url"])

        except Exception:
            pass

    # Instructions for the AI assistant
    system_prompt = """
You are CardioCare AI, an intelligent heart-health educational assistant.

Your responsibilities:
1. Understand natural language questions, even if grammar or spelling is incorrect.
2. Answer the exact question instead of giving generic responses.
3. Explain heart disease, symptoms, causes, prevention, tests,
   ECG, blood pressure, cholesterol, medicines and nutrition.
4. Explain difficult medical concepts in simple English.
5. Give detailed, structured answers using headings and bullet points
   when useful.
6. Use the supplied web research when relevant and do not invent sources.
7. Maintain conversational context and understand follow-up questions.
8. Explain general medicine information, but never prescribe medicines,
   invent dosages, or tell users to start, stop or change medication.
9. Never claim to diagnose a person or replace a doctor.
10. For possible heart attack symptoms, advise immediate emergency care.

If the user asks what happens when heart disease develops,
explain what happens to the heart and blood vessels, possible symptoms,
complications and why medical evaluation matters.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # Include recent conversation for follow-up understanding
    if chat_history:
        for item in chat_history[-10:]:
            role = item.get("role")
            content = item.get("content", "")

            if role in ["user", "assistant"] and isinstance(content, str):
                messages.append({
                    "role": role,
                    "content": content
                })

    # Add current question and available web research
    user_message = "User question: " + question

    if web_context:
        user_message += (
            "\n\nCurrent web research for reference:\n"
            + web_context[:12000]
        )

    messages.append({
        "role": "user",
        "content": user_message
    })

    # Ask the OpenAI AI model
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openai_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": messages,
                "temperature": 0.5,
                "max_tokens": 1500
            },
            timeout=60
        )

        if response.status_code != 200:
            try:
                error_message = response.json().get("error", {}).get(
                    "message", "Unknown API error"
                )
            except Exception:
                error_message = "Unable to read the API error."

            return {
                "answer": (
                    "The AI service returned an error: "
                    + str(error_message)
                ),
                "images": images,
                "sources": sources
            }

        data = response.json()
        answer = data["choices"][0]["message"]["content"]

        return {
            "answer": answer,
            "images": images[:4],
            "sources": sources[:5]
        }

    except requests.exceptions.Timeout:
        return {
            "answer": "The AI request timed out. Please try again.",
            "images": [],
            "sources": []
        }

    except Exception as exc:
        return {
            "answer": "Unable to contact the AI service: " + str(exc),
            "images": [],
            "sources": []
        }



# ----------------------------- SESSION STATE -----------------------------
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None
if "latest_values" not in st.session_state:
    st.session_state.latest_values = None
if "latest_profile" not in st.session_state:
    st.session_state.latest_profile = {"name": "", "patient_id": ""}


# ---------------------- PROFESSIONAL PLATFORM HELPERS ----------------------
def init_clinical_db():
    """Local prototype database. Use encrypted managed storage for real patient deployment."""
    with sqlite3.connect(CLINICAL_DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, patient_id TEXT,
            patient_name TEXT, filename TEXT, extracted_json TEXT, report_text TEXT)""")


def extract_report_text(uploaded_file):
    """Extract text from PDF or image. OCR requires Tesseract installed on the host."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    if name.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                return "\n".join((page.extract_text() or "") for page in pdf.pages)
        except ImportError:
            raise RuntimeError("PDF support is missing. Install pdfplumber: pip install pdfplumber")
    if name.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        try:
            from PIL import Image
            import pytesseract
            return pytesseract.image_to_string(Image.open(io.BytesIO(raw)))
        except ImportError:
            raise RuntimeError("Image OCR needs Pillow and pytesseract. Install them and install Tesseract OCR on your computer.")
    raise RuntimeError("Please upload a PDF, PNG, JPG or JPEG file.")


CLINICAL_PATTERNS = {
    "Total Cholesterol": [r"total\s+cholesterol", r"cholesterol\s*\(total\)"],
    "LDL Cholesterol": [r"\bldl\b"], "HDL Cholesterol": [r"\bhdl\b"],
    "Triglycerides": [r"triglycerides?"], "Resting Blood Pressure": [r"blood\s+pressure", r"\bbp\b"],
    "Fasting Blood Glucose": [r"fasting\s+(?:blood\s+)?(?:glucose|sugar)", r"\bfbs\b"],
    "HbA1c": [r"\bhba1c\b", r"glycated\s+haemoglobin", r"glycated\s+hemoglobin"],
    "Heart Rate": [r"heart\s+rate", r"pulse\s+rate"], "Troponin": [r"troponin(?:\s+i|\s+t)?"],
    "Haemoglobin": [r"haemoglobin", r"hemoglobin"],
}


def extract_clinical_values(report_text):
    """Conservative label/value matching; user must verify against the source report."""
    found = []
    lines = [re.sub(r"\s+", " ", x).strip() for x in report_text.splitlines() if x.strip()]
    for line in lines:
        for label, patterns in CLINICAL_PATTERNS.items():
            if any(re.search(p, line, re.I) for p in patterns):
                # Extract first numeric result after a recognized label, excluding ranges where possible.
                m = re.search(r"(?:[:=]|\s)\s*(-?\d+(?:\.\d+)?)\s*(mg/dL|mmol/L|mmHg|bpm|%|g/dL|ng/L|pg/mL)?", line, re.I)
                if m:
                    found.append({"Parameter": label, "Value": m.group(1), "Unit": m.group(2) or "Not detected", "Source line": line[:240]})
                break
    # Keep first occurrence per parameter to avoid duplicate rows from repeated report sections.
    unique = {}
    for item in found:
        unique.setdefault(item["Parameter"], item)
    return list(unique.values())


def save_clinical_report(profile, filename, extracted, report_text):
    init_clinical_db()
    with sqlite3.connect(CLINICAL_DB_PATH) as conn:
        conn.execute("INSERT INTO reports(created_at, patient_id, patient_name, filename, extracted_json, report_text) VALUES(?,?,?,?,?,?)",
                     (datetime.now().isoformat(timespec="seconds"), profile.get("patient_id", ""),
                      profile.get("name", ""), filename, json.dumps(extracted), report_text[:30000]))


def load_clinical_reports(patient_id=""):
    init_clinical_db()
    with sqlite3.connect(CLINICAL_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        if patient_id.strip():
            rows = conn.execute("SELECT id, created_at, patient_id, patient_name, filename, extracted_json FROM reports WHERE patient_id=? ORDER BY id DESC", (patient_id.strip(),)).fetchall()
        else:
            rows = conn.execute("SELECT id, created_at, patient_id, patient_name, filename, extracted_json FROM reports ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def report_explanation(values):
    if not values:
        return "No clearly labelled numeric results were detected. Please check the uploaded report manually."
    lines = []
    for item in values:
        lines.append(f"**{item['Parameter']}: {item['Value']} {item['Unit']}**  \nExtracted from: `{item['Source line']}`")
    return "\n\n".join(lines)



def find_nearby_healthcare(place, category="all", radius=10000):
    """Search OpenStreetMap using multiple Overpass servers."""
    headers = {
        "User-Agent": "CardioCareAI-StudentProject/1.0"
    }

    geo = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": place, "format": "json", "limit": 1},
        headers=headers,
        timeout=30
    )
    geo.raise_for_status()

    locations = geo.json()
    if not locations:
        return pd.DataFrame(), None

    lat = float(locations[0]["lat"])
    lon = float(locations[0]["lon"])

    filters = {
        "all": 'node["amenity"~"hospital|clinic|doctors"],way["amenity"~"hospital|clinic|doctors"];node["healthcare"~"hospital|clinic|doctor|laboratory"],way["healthcare"~"hospital|clinic|doctor|laboratory"]',
        "hospitals": 'node["amenity"="hospital"],way["amenity"="hospital"];node["healthcare"="hospital"],way["healthcare"="hospital"]',
        "labs": 'node["healthcare"="laboratory"],way["healthcare"="laboratory"];node["amenity"="clinic"],way["amenity"="clinic"]',
        "doctors": 'node["amenity"="doctors"],way["amenity"="doctors"];node["healthcare"="doctor"],way["healthcare"="doctor"]'
    }

    selectors = filters[category].split(";")
    scoped = [
        f"{s.strip()}(around:{radius},{lat},{lon})"
        for s in selectors if s.strip()
    ]

    query = "[out:json][timeout:45];(" + ";".join(scoped) + ");out center tags;"

    servers = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.private.coffee/api/interpreter"
    ]

    elements = []
    last_error = None

    for server in servers:
        try:
            response = requests.post(
                server,
                data={"data": query},
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            elements = response.json().get("elements", [])
            break
        except requests.RequestException as error:
            last_error = error
        except ValueError as error:
            last_error = error

    else:
        raise RuntimeError(
            f"Healthcare search is temporarily unavailable. "
            f"Please try again later. Details: {last_error}"
        )

    rows = []

    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name")

        if not name:
            continue

        elat = el.get("lat", el.get("center", {}).get("lat"))
        elon = el.get("lon", el.get("center", {}).get("lon"))

        if elat is None or elon is None:
            continue

        rows.append({
            "Name": name,
            "Type": tags.get("healthcare", tags.get("amenity", "healthcare facility")),
            "Address": ", ".join(
                x for x in [
                    tags.get("addr:street"),
                    tags.get("addr:city"),
                    tags.get("addr:postcode")
                ] if x
            ) or "Address not listed",
            "Phone": tags.get("phone", tags.get("contact:phone", "Not listed")),
            "Website": tags.get("website", tags.get("contact:website", "")),
            "Latitude": elat,
            "Longitude": elon,
            "Map": f"https://www.openstreetmap.org/?mlat={elat}&mlon={elon}#map=17/{elat}/{elon}"
        })

    df = (
        pd.DataFrame(rows).drop_duplicates(
            subset=["Name", "Latitude", "Longitude"]
        ) if rows else pd.DataFrame()
    )

    return df, (lat, lon)



TEST_GUIDE = [
    ("ECG", "Records the heart's electrical activity. A clinician interprets the tracing."),
    ("Lipid profile", "Measures total cholesterol, LDL, HDL and triglycerides; interpretation depends on overall clinical context."),
    ("Blood glucose / HbA1c", "Assesses blood glucose or longer-term glucose patterns."),
    ("Echocardiogram", "Ultrasound imaging used to assess heart structure, valves and pumping function."),
    ("Troponin", "A blood marker used in the evaluation of possible heart muscle injury, including urgent symptoms."),
    ("CBC", "Measures blood cells and may help assess conditions such as anaemia or infection."),
    ("Exercise stress test", "Assesses the heart during exertion when a clinician considers it appropriate."),
    ("Holter monitoring", "Records heart rhythm over an extended period to investigate intermittent rhythm symptoms."),
]

init_clinical_db()

# ----------------------------- SIDEBAR ----------------------------------
with st.sidebar:
    st.markdown("## 🫀 CardioCare AI")
    st.caption("Heart health insights")
    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Heart Prediction", "Detailed Report", "Explainable AI",
         "Patient History", "Lifestyle Guide", "AI Health Assistant", "ECG Visualizer", "Heart Education",
          "Medical Report Analyzer", "Find Healthcare Near Me", "Heart Test Guide", "Patient Clinical Records"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**Patient profile**")
    profile_name = st.text_input("Patient name", value=st.session_state.latest_profile["name"])
    profile_id = st.text_input("Patient ID / reference", value=st.session_state.latest_profile["patient_id"])
    st.session_state.latest_profile = {"name": profile_name, "patient_id": profile_id}
    st.caption("History is stored locally in prediction_history.csv on this machine.")

# ----------------------------- HEADER -----------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Intelligent screening • Clearer health insights</div>
      <h1>CardioCare AI</h1>
      <p>A modern heart-health learning and prediction dashboard, powered by your existing ML model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model_errors:
    st.error("Model assets could not be loaded. Keep all three original .pkl files in the same folder as app.py.")
    for err in model_errors:
        st.code(err)

# ----------------------------- INPUT FORM -------------------------------
def render_prediction_form():
    st.markdown("### Patient measurements")
    st.caption("Enter values from a reliable measurement or clinical report. Do not guess medical values.")
    with st.form("heart_prediction_form"):
        a, b, c = st.columns(3)
        with a:
            prefill = st.session_state.get("prediction_prefill", {})
            age = st.slider("Age (years)", 18, 100, int(prefill.get("Age", 48)))
            sex = st.selectbox("Sex", ["M", "F"], format_func=lambda x: "Male" if x == "M" else "Female")
            chest_pain = st.selectbox("Chest pain type", ["ATA", "NAP", "ASY", "TA"])
            resting_bp = st.number_input("Resting blood pressure (mmHg)", 80, 220, int(prefill.get("RestingBP", 130)))
        with b:
            cholesterol = st.number_input("Total cholesterol (mg/dL)", 100, 600, int(prefill.get("Cholesterol", 220)))
            fasting_bs = st.selectbox("Fasting blood sugar > 120 mg/dL", [0, 1], index=int(prefill.get("FastingBS", 0)), format_func=lambda x: "Yes" if x else "No")
            resting_ecg = st.selectbox("Resting ECG category", ["Normal", "ST", "LVH"])
            max_hr = st.slider("Maximum heart rate achieved (bpm)", 60, 220, int(prefill.get("MaxHR", 145)))
        with c:
            exercise_angina = st.selectbox("Exercise-induced angina", ["N", "Y"], format_func=lambda x: "No" if x == "N" else "Yes")
            st_slope = st.selectbox("ST slope", ["Up", "Flat", "Down"])
            oldpeak = st.number_input("Oldpeak / ST depression", min_value=0.0, max_value=6.0, value=float(prefill.get("Oldpeak", 1.2)), step=0.1)
        submitted = st.form_submit_button("Run heart-health screening", use_container_width=True)
    if submitted:
        values = {
            "Age": age, "Sex": sex, "ChestPainType": chest_pain,
            "RestingBP": resting_bp, "Cholesterol": cholesterol,
            "FastingBS": fasting_bs, "RestingECG": resting_ecg,
            "MaxHR": max_hr, "ExerciseAngina": exercise_angina,
            "ST_Slope": st_slope, "Oldpeak": oldpeak,
        }
        try:
            result = run_prediction(values)
            st.session_state.latest_values = values
            st.session_state.latest_result = result
            st.session_state.latest_profile = {"name": profile_name, "patient_id": profile_id}
            save_history(st.session_state.latest_profile, values, result)
            st.success("Prediction completed and saved to local history.")
        except Exception as exc:
            st.error("Prediction could not be completed. Verify that the model, scaler and columns files match.")
            st.exception(exc)


def show_latest_result():
    result = st.session_state.latest_result
    if result is None:
        st.info("No prediction yet. Open **Heart Prediction** and submit patient measurements.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Model classification", "Flagged" if result["is_positive"] else "Not flagged")
    c2.metric("Predicted class", str(result["prediction"]))
    if result["model_score"] is None:
        c3.metric("Model probability", "Unavailable")
    else:
        c3.metric("Model probability", f"{result['model_score']*100:.1f}%")
    if result["is_positive"]:
        st.warning("The model returned its flagged class for these inputs. This is a screening output, not a diagnosis.")
    else:
        st.success("The model did not return its flagged class for these inputs. This does not rule out heart disease.")
    st.caption("A model probability is not necessarily calibrated to real-world clinical risk.")

# ----------------------------- PAGES ------------------------------------
if page == "Dashboard":
    st.markdown("## Health dashboard")
    history = load_history()
    total = len(history)
    flagged = int((history["Prediction"] == "Model class flagged").sum()) if total and "Prediction" in history else 0
    latest_date = history.iloc[-1]["Timestamp"] if total and "Timestamp" in history else "—"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Saved screenings", total)
    m2.metric("Flagged classifications", flagged)
    m3.metric("Not flagged", total - flagged)
    m4.metric("Latest screening", str(latest_date))
    st.markdown("### Latest screening")
    show_latest_result()
    if total:
        st.markdown("### Screening activity")
        if "Timestamp" in history and "Prediction" in history:
            trend = history.copy()
            trend["Date"] = pd.to_datetime(trend["Timestamp"], errors="coerce").dt.date
            daily = trend.groupby(["Date", "Prediction"]).size().unstack(fill_value=0)
            st.line_chart(daily)
        st.dataframe(history.tail(8).iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.markdown('<div class="section-card"><b>Your dashboard is ready.</b><br><span class="small-muted">Run your first screening to see patient metrics and history here.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="warning-note">This dashboard summarizes entries saved on this device. It is not a clinical patient record system.</div>', unsafe_allow_html=True)

elif page == "Heart Prediction":
    st.markdown("## Heart disease prediction")
    st.markdown('<div class="section-card"><span class="pill">Existing ML model retained</span><p class="small-muted">The original feature encoding, saved column order and scaler transformation are preserved.</p></div>', unsafe_allow_html=True)
    render_prediction_form()
    show_latest_result()

elif page == "Detailed Report":
    st.markdown("## Detailed prediction report")
    show_latest_result()
    if st.session_state.latest_result is not None:
        values = st.session_state.latest_values
        result = st.session_state.latest_result
        st.markdown("### Entered patient measurements")
        st.dataframe(pd.DataFrame([values]), use_container_width=True, hide_index=True)
        pdf_bytes = make_pdf(st.session_state.latest_profile, values, result)
        if pdf_bytes:
            st.download_button("Download PDF report", data=pdf_bytes,
                               file_name="CardioCare_Heart_Health_Report.pdf",
                               mime="application/pdf", use_container_width=True)
        else:
            st.warning("PDF export needs ReportLab. Install it with: pip install reportlab")
    else:
        st.info("Complete a prediction first to generate a report.")

elif page == "Explainable AI":
    st.markdown("## Explainable AI · Feature importance")
    st.write("This page displays native feature-importance information exposed by the loaded estimator. It does not invent explanations when the model cannot provide them.")
    importance_df = feature_importance_frame()
    if importance_df is not None and not importance_df.empty:
        top = importance_df.head(15).sort_values("Importance")
        st.bar_chart(top.set_index("Feature"))
        st.dataframe(importance_df, use_container_width=True, hide_index=True)
        st.caption("Importance indicates model reliance under the estimator's own definition; it does not prove causation or explain an individual patient's outcome.")
    else:
        st.info("The loaded model does not expose feature_importances_ or coefficients. A reliable SHAP/permutation explanation would require compatible model internals and representative validation/training data, which are not included with the three model assets.")
    if st.session_state.latest_values is not None:
        st.markdown("### Current prediction inputs")
        st.dataframe(pd.DataFrame([st.session_state.latest_values]), use_container_width=True, hide_index=True)

elif page == "Patient History":
    st.markdown("## Patient profile & prediction history")
    st.caption("Records are appended to a local CSV file. This is suitable for a local demo, not a secure multi-user medical-record system.")
    history = load_history()
    if history.empty:
        st.info("No history saved yet. Run a prediction to create the first record.")
    else:
        if "Patient ID" in history.columns and profile_id.strip():
            view = history[history["Patient ID"].astype(str) == profile_id.strip()]
        else:
            view = history
        st.metric("Records in current view", len(view))
        st.dataframe(view.iloc[::-1], use_container_width=True, hide_index=True)
        if len(view) >= 2 and "Model score (not calibrated risk)" in view.columns:
            chart_data = view.copy()
            chart_data["Model score (not calibrated risk)"] = pd.to_numeric(
                chart_data["Model score (not calibrated risk)"], errors="coerce"
            )
            if chart_data["Model score (not calibrated risk)"].notna().any():
                st.markdown("### Model-score history (not clinical risk)")
                st.line_chart(chart_data.set_index("Timestamp")["Model score (not calibrated risk)"])
        csv_data = view.to_csv(index=False).encode("utf-8")
        st.download_button("Download visible history as CSV", csv_data, "cardiocare_history.csv", "text/csv")
        if st.button("Delete all local prediction history", type="secondary"):
            try:
                os.remove(HISTORY_PATH)
                st.success("Local history file deleted.")
                st.rerun()
            except FileNotFoundError:
                st.info("No local history file exists.")

elif page == "Lifestyle Guide":
    st.markdown("## Personalized lifestyle guide")
    st.caption("Suggestions are general education based on the values entered. They are not treatment instructions.")
    if st.session_state.latest_values is None:
        st.info("Run a prediction first to tailor the educational tips to the entered values.")
    else:
        for tip in lifestyle_tips(st.session_state.latest_values):
            st.markdown(f"- {tip}")
    st.markdown("### Everyday heart-health habits")
    st.markdown(
        "- Prefer vegetables, fruits, whole grains, pulses and minimally processed foods.\n"
        "- Be physically active at a level that is safe and sustainable for you.\n"
        "- Avoid tobacco; seek evidence-based help to quit.\n"
        "- Monitor blood pressure and discuss cholesterol and glucose results with a clinician.\n"
        "- Maintain regular sleep and follow prescribed treatment plans."
    )
    st.warning("Do not start, stop or change medicines based on this application.")


elif page == "AI Health Assistant":
    st.markdown("## 🫀 CardioCare AI Health Assistant")
    st.caption(
        "Ask questions naturally. Get AI-powered explanations, "
        "current web references and relevant visual resources."
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! Welcome to CardioCare AI. 🫀\n\n"
                    "I can help explain heart health, blood pressure, "
                    "cholesterol, ECG reports, nutrition, exercise and "
                    "medicines.\n\n"
                    "What would you like to understand today?"
                ),
                "images": [],
                "sources": [],
            }
        ]

    # Clear conversation.
    top_left, top_right = st.columns([4, 1])
    with top_right:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()

    # Display conversation.
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                image_urls = message.get("images", [])
                if image_urls:
                    st.markdown("#### Visual references")
                    image_columns = st.columns(min(2, len(image_urls)))

                    for index, image_url in enumerate(image_urls):
                        with image_columns[index % len(image_columns)]:
                            try:
                                st.image(
                                    image_url,
                                    use_container_width=True
                                )
                            except Exception:
                                st.caption("This image could not be displayed.")

                source_list = message.get("sources", [])
                if source_list:
                    with st.expander("📚 Sources and further reading"):
                        for source in source_list:
                            st.markdown(
                                f"- [{source['title']}]({source['url']})"
                            )

    # User input.
    prompt = st.chat_input(
        "Ask anything about heart health, medicines, ECG, diet..."
    )

    if prompt:
        previous_history = list(st.session_state.chat_messages)
        report_context = st.session_state.get("current_report_text", "")
        if report_context:
            previous_history.append({"role":"assistant", "content":"The user has an uploaded report available for discussion. Extracted source text (may contain OCR errors):\n" + report_context[:8000]})

        st.session_state.chat_messages.append({
            "role": "user",
            "content": prompt,
            "images": [],
            "sources": [],
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("CardioCare AI is researching and preparing your answer..."):
                result = chatbot_answer(
                    prompt,
                    previous_history
                )

            st.markdown(result["answer"])

            if result.get("images"):
                st.markdown("#### Visual references")
                image_columns = st.columns(
                    min(2, len(result["images"]))
                )

                for index, image_url in enumerate(result["images"]):
                    with image_columns[index % len(image_columns)]:
                        try:
                            st.image(
                                image_url,
                                use_container_width=True
                            )
                        except Exception:
                            st.caption("This image could not be displayed.")

            if result.get("sources"):
                with st.expander("📚 Sources and further reading"):
                    for source in result["sources"]:
                        st.markdown(
                            f"- [{source['title']}]({source['url']})"
                        )

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": result["answer"],
            "images": result.get("images", []),
            "sources": result.get("sources", []),
        })

elif page == "ECG Visualizer":
    st.markdown("## ECG waveform visualizer")
    st.caption("Illustrative waveform only — not recorded from a person and not suitable for diagnosis.")
    duration = st.slider("Display duration (seconds)", 4, 12, 6)
    sample_rate = 500
    t = np.linspace(0, duration, duration * sample_rate)
    signal = np.zeros_like(t)
    beat_interval = 0.85
    for beat in np.arange(0.5, duration, beat_interval):
        # Synthetic P-QRS-T-like shape for educational illustration.
        signal += 0.12 * np.exp(-((t - (beat - 0.18)) / 0.045) ** 2)
        signal += -0.16 * np.exp(-((t - (beat - 0.035)) / 0.012) ** 2)
        signal += 1.00 * np.exp(-((t - beat) / 0.014) ** 2)
        signal += -0.25 * np.exp(-((t - (beat + 0.035)) / 0.018) ** 2)
        signal += 0.30 * np.exp(-((t - (beat + 0.25)) / 0.075) ** 2)
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(t, signal, linewidth=1.5)
    ax.set_facecolor("#FBFEFE")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Illustrative amplitude")
    ax.set_title("Synthetic ECG-style waveform")
    ax.grid(True, alpha=.25)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.warning("A real ECG requires clinical equipment and interpretation by a qualified professional.")

elif page == "Heart Education":
    st.markdown("## Heart health education")
    with st.expander("Common warning symptoms", expanded=True):
        st.write("Chest pressure or discomfort, breathlessness, pain spreading to the arm/jaw/back, cold sweating, nausea or faintness can be warning signs. Symptoms vary.")
    with st.expander("What does this model use?"):
        st.write("Age, sex, chest-pain category, resting blood pressure, cholesterol, fasting blood sugar category, resting ECG category, maximum heart rate, exercise angina, ST slope and Oldpeak.")
    with st.expander("What this project cannot do"):
        st.write("It cannot diagnose heart disease, interpret an actual ECG, assess an emergency, prescribe treatment or guarantee that a person is healthy.")
    st.markdown('<div class="warning-note"><b>Emergency:</b> If you suspect a heart attack or have severe/new chest symptoms, call your local emergency service immediately. Do not wait for an app prediction.</div>', unsafe_allow_html=True)


elif page == "Medical Report Analyzer":
    st.markdown("## Medical Report Analyzer")
    st.write("Upload a laboratory report or clinical document. CardioCare extracts readable text and labelled values for your review.")
    st.info("Extraction is assistive, not a diagnosis. Always compare extracted values with the original report, units and laboratory reference ranges.")
    uploaded = st.file_uploader("Upload medical report", type=["pdf","png","jpg","jpeg"], key="clinical_report_upload")
    if uploaded is not None:
        if st.button("Analyze uploaded report", type="primary"):
            try:
                with st.spinner("Reading report..."):
                    report_text = extract_report_text(uploaded)
                    extracted = extract_clinical_values(report_text)
                st.session_state.current_report_text = report_text
                st.session_state.current_report_values = extracted
                st.session_state.current_report_filename = uploaded.name
                st.success("Report text extracted. Please verify every value against the original document.")
            except Exception as exc:
                st.error(str(exc))
    if st.session_state.get("current_report_text"):
        st.markdown("### Extracted clinical information")
        extracted = st.session_state.get("current_report_values", [])
        if extracted:
            st.dataframe(pd.DataFrame(extracted), use_container_width=True, hide_index=True)
        else:
            st.warning("No labelled numeric values were confidently detected. This can happen with handwritten, low-quality or differently formatted reports.")
        with st.expander("View extracted report text"):
            st.text(st.session_state.current_report_text[:20000])
        st.markdown("### Extracted report summary")
        st.markdown(report_explanation(extracted))
        st.caption("The analyzer does not decide whether a result is clinically normal unless the original report's reference range is reliably available.")
        consent_ai = st.checkbox("I consent to sending the extracted report text to my configured AI service for an educational explanation.", key="report_ai_consent")
        if st.button("Generate AI explanation", disabled=not consent_ai):
            try:
                prompt_for_ai = ("Explain this uploaded medical report in simple language for education. "
                                 "Use only the supplied values, do not diagnose, do not recommend medication, "
                                 "identify uncertainty and advise clinician review. Extracted values: "
                                 + json.dumps(extracted) + "\nReport text: " + st.session_state.current_report_text[:7000])
                with st.spinner("Preparing report explanation..."):
                    ai_result = chatbot_answer(prompt_for_ai)
                st.session_state.current_report_ai_explanation = ai_result.get("answer", "No explanation returned.")
            except Exception as exc:
                st.error(f"AI explanation unavailable: {exc}")
        if st.session_state.get("current_report_ai_explanation"):
            st.markdown("#### AI explanation (educational only)")
            st.markdown(st.session_state.current_report_ai_explanation)
        if st.button("Save report to patient clinical records"):
            try:
                save_clinical_report(st.session_state.latest_profile, st.session_state.get("current_report_filename","uploaded_report"), extracted, st.session_state.current_report_text)
                st.success("Report summary saved to the local clinical records database.")
            except Exception as exc:
                st.error(f"Could not save report: {exc}")
        # Only map fields with compatible units and meaning; never fill categorical/model features from guesswork.
        compatible = {}
        for item in extracted:
            try: number=float(item["Value"])
            except (ValueError, TypeError): continue
            if item["Parameter"] == "Total Cholesterol" and item["Unit"].lower() in ("mg/dl","not detected") and 100 <= number <= 600:
                compatible["Cholesterol"] = int(round(number))
            if item["Parameter"] == "Resting Blood Pressure" and "/" not in item["Source line"] and item["Unit"].lower() in ("mmhg","not detected") and 80 <= number <= 220:
                compatible["RestingBP"] = int(round(number))
            if item["Parameter"] == "Fasting Blood Glucose" and item["Unit"].lower() in ("mg/dl","not detected"):
                compatible["FastingBS"] = int(number > 120)
        if compatible:
            st.markdown("### Transfer compatible values to Heart Prediction")
            st.write("Only fields with a compatible measurement were detected. Remaining model inputs must be entered from verified clinical information.")
            st.json(compatible)
            if st.button("Use these values in Heart Prediction"):
                st.session_state.prediction_prefill = compatible
                st.success("Compatible values prepared. Open Heart Prediction and verify them before running the model.")
        pdf = make_pdf(st.session_state.latest_profile, {x["Parameter"]:x["Value"]+" "+x["Unit"] for x in extracted} or {"Report":"Text extracted; manual review required"},
                       {"is_positive":False,"prediction":"Not run","model_score":None})
        if pdf:
            st.download_button("Download extracted report summary PDF", pdf, "CardioCare_Extracted_Report_Summary.pdf", "application/pdf")

elif page == "Find Healthcare Near Me":
    st.markdown("## Find hospitals, laboratories and doctors")
    st.caption("Enter a city, locality or PIN code. This demo uses OpenStreetMap community listings; verify details directly before travelling or booking.")
    place = st.text_input("Your area / city / PIN code", placeholder="e.g. Mohali, Punjab")
    category_label = st.selectbox("Facility type", ["All healthcare facilities","Hospitals","Diagnostic labs / clinics","Doctors / clinics"])
    category = {"All healthcare facilities":"all","Hospitals":"hospitals","Diagnostic labs / clinics":"labs","Doctors / clinics":"doctors"}[category_label]
    radius_km = st.select_slider("Search radius", options=[2,5,10,15,25], value=10)
    if st.button("Search nearby healthcare", type="primary"):
        if not place.strip():
            st.warning("Enter your area or city to search.")
        else:
            try:
                with st.spinner("Searching mapped healthcare facilities..."):
                    facilities, center = find_nearby_healthcare(place.strip(), category, radius_km*1000)
                st.session_state.nearby_facilities = facilities
                st.session_state.nearby_center = center
                st.session_state.nearby_search_place = place.strip()
                if facilities.empty: st.warning("No mapped results were found. Try a nearby city or a larger radius.")
                else: st.success(f"Found {len(facilities)} mapped facilities.")
            except Exception as exc:
                st.error(f"Location search is temporarily unavailable: {exc}")
    facilities = st.session_state.get("nearby_facilities", pd.DataFrame())
    if isinstance(facilities, pd.DataFrame) and not facilities.empty:
        st.dataframe(facilities[["Name","Type","Address","Phone","Website"]], use_container_width=True, hide_index=True)
        st.markdown("### Map locations")
        st.map(facilities.rename(columns={"Latitude":"lat","Longitude":"lon"})[["lat","lon"]])
        for _, row in facilities.iterrows():
            website = f" · [Website]({row['Website']})" if row["Website"] else ""
            st.markdown(f"**{row['Name']}** · {row['Type']}  \n{row['Address']} · {row['Phone']} · [Open map]({row['Map']}){website}")
        st.caption("Listings, telephone numbers, service availability, opening hours and home collection are not independently verified by CardioCare AI.")

elif page == "Heart Test Guide":
    st.markdown("## Heart Health Test Guide")
    st.write("Learn what common investigations are used for. A clinician should decide which tests are appropriate for an individual.")
    for test_name, description in TEST_GUIDE:
        with st.expander(test_name):
            st.write(description)
            if test_name in ("ECG","Troponin"):
                st.warning("If you have new or severe chest pain, breathlessness, sweating, faintness or pain spreading to your arm/jaw/back, seek emergency care immediately. Do not use this app to arrange a routine test first.")
    st.markdown("### What should I ask a laboratory? ")
    st.markdown("- Is the requested test available at this branch?\n- Is fasting required, and for how long? Follow the clinician/lab instructions.\n- What is the total price and expected report time?\n- Is home sample collection available at my address?\n- How will the verified report be delivered?")

elif page == "Patient Clinical Records":
    st.markdown("## Patient Clinical Records")
    st.caption("Local prototype storage. Do not use this local database for real patient care or sensitive production records without encryption, authentication, access controls, backups and applicable legal/privacy compliance.")
    records = load_clinical_reports(profile_id)
    st.metric("Saved report records", len(records))
    if records:
        for record in records:
            with st.expander(f"{record['created_at']} · {record['filename']} · {record['patient_name'] or 'Unnamed patient'}"):
                try:
                    items=json.loads(record["extracted_json"])
                    if items: st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)
                    else: st.write("No structured values were extracted.")
                except Exception:
                    st.write("Saved report metadata could not be displayed.")
    else:
        st.info("No saved report summaries found for this patient ID. Upload and save a report in Medical Report Analyzer.")
    st.markdown("### Existing prediction history")
    history = load_history()
    if not history.empty:
        if profile_id.strip() and "Patient ID" in history.columns:
            history = history[history["Patient ID"].astype(str) == profile_id.strip()]
        st.dataframe(history.iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("No prediction history is available yet.")


# ----------------------------- FOOTER -----------------------------------
st.markdown(
    '<div class="footer">CardioCare AI · Educational screening project · Not a substitute for professional medical care</div>',
    unsafe_allow_html=True,
)

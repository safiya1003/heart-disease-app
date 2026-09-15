import streamlit as st
import pandas as pd
import joblib
import os
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Heart Disease Prediction | Better Insights • Healthier Tomorrows",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Design & Grid Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #fff5f8;
        color: #9C4561;
    }
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1240px !important;
    }

    [data-testid="stSidebarNav"], footer, header {
        display: none !important;
    }

    .top-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0 1.2rem 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    .brand-group {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-icon {
        background: linear-gradient(135deg, #f43f5e 0%, #e11d48 100%);
        color: white;
        width: 44px;
        height: 44px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        box-shadow: 0 4px 14px rgba(225, 29, 72, 0.28);
    }
    .brand-title {
        margin: 0;
        font-size: 1.25rem;
        font-weight: 800;
        color: #0f172a;
    }
    .brand-sub {
        margin: 0;
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 500;
    }

    .hero-badge {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.75rem;
    }
    .hero-title-dark {
        font-size: 3.4rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
        margin: 0;
    }
    .hero-title-red {
        font-size: 3.4rem;
        font-weight: 800;
        color: #e11d48;
        line-height: 1.1;
        margin: 0 0 1.2rem 0;
    }
    .hero-description {
        font-size: 1.05rem;
        color: #475569;
        line-height: 1.6;
        margin-bottom: 1.8rem;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #e11d48 0%, #be123c 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 26px !important;
        padding: 0.5rem 1.4rem !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        box-shadow: 0 4px 14px rgba(225, 29, 72, 0.25) !important;
        transition: all 0.25s ease-in-out !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 22px rgba(225, 29, 72, 0.35) !important;
    }

    .features-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.2rem;
        margin: 2.2rem 0;
    }
    .feature-item {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    .feature-circle {
        width: 46px;
        height: 46px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        flex-shrink: 0;
    }
    .feature-txt h5 {
        margin: 0;
        font-size: 0.92rem;
        font-weight: 700;
        color: #0f172a;
    }
    .feature-txt p {
        margin: 2px 0 0 0;
        font-size: 0.77rem;
        color: #64748b;
    }

    .section-title-wrap {
        margin: 2.5rem 0 1.2rem 0;
    }
    .section-title-wrap h3 {
        margin: 0;
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a;
    }
    .section-title-wrap p {
        margin: 0.3rem 0 0 0;
        font-size: 0.85rem;
        color: #64748b;
    }

    .topic-tile {
        border-radius: 16px;
        padding: 1.1rem 1rem;
        border: 1px solid rgba(0,0,0,0.03);
        margin-bottom: 0.5rem;
    }
    .topic-tile-icon {
        font-size: 1.35rem;
        margin-bottom: 0.4rem;
    }
    .topic-tile h4 {
        margin: 0 0 0.3rem 0;
        font-size: 0.92rem;
        font-weight: 700;
        color: #0f172a;
    }
    .topic-tile p {
        margin: 0;
        font-size: 0.76rem;
        color: #64748b;
        line-height: 1.4;
    }

    .content-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 3px 12px rgba(0,0,0,0.03);
    }

    .clinical-footer {
        background: #090e17;
        color: #94a3b8;
        padding: 1.6rem;
        text-align: center;
        font-size: 0.85rem;
        border-radius: 18px;
        margin-top: 3rem;
        letter-spacing: 0.02em;
    }
</style>
""", unsafe_allow_html=True)
# Safe Loader for Model Assets
@st.cache_resource
def get_model_assets():
    import os
    import joblib

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "RF_KNN_heart.pkl")
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    columns_path = os.path.join(base_dir, "columns.pkl")

    model = joblib.load(model_path) if os.path.exists(model_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    columns = joblib.load(columns_path) if os.path.exists(columns_path) else None
    return model, scaler, columns

model, scaler, expected_columns = get_model_assets()

# Modal Dialog Functions
@st.dialog("🩺 Cardiovascular Assessment Diagnostic")
def open_prediction_dialog():
    st.caption("Enter clinical markers below to evaluate cardiac health status:")
    d1, d2 = st.columns(2)
    with d1:
        age_in = st.slider("Age (Years)", 18, 100, 42)
        sex_in = st.selectbox("Sex", ["M", "F"], format_func=lambda x: "Male" if x == "M" else "Female")
        cp_in = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
        rbp_in = st.number_input("Resting BP (mm Hg)", 80, 220, 120)
        chol_in = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, 205)
    with d2:
        fbs_in = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        ecg_in = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        mhr_in = st.slider("Max Heart Rate (bpm)", 60, 220, 150)
        ang_in = st.selectbox("Exercise Angina", ["Y", "N"], format_func=lambda x: "Yes" if x == "Y" else "No")
        slope_in = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
        oldpeak_in = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
        if model is not None and scaler is not None and expected_columns is not None:
            raw_dict = {
                'Age': age_in, 'RestingBP': rbp_in, 'Cholesterol': chol_in,
                'FastingBS': fbs_in, 'MaxHR': mhr_in, 'Oldpeak': oldpeak_in,
                'Sex_' + sex_in: 1, 'ChestPainType_' + cp_in: 1,
                'RestingECG_' + ecg_in: 1, 'ExerciseAngina_' + ang_in: 1,
                'ST_Slope_' + slope_in: 1
            }
            df = pd.DataFrame([raw_dict])
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_columns]
        scaled_df = scaler.transform(df)
        pred = model.predict(scaled_df)[0]
        proba = model.predict_proba(scaled_df)[0][1] if hasattr(model, "predict_proba") else (1.0 if pred == 1 else 0.0)

        if pred == 1:
            st.error(f"⚠️ **Elevated Risk Detected** (Confidence: {proba*100:.1f}%)")
            st.warning("Clinical markers suggest high risk for cardiac anomalies. Please consult a healthcare professional.")
        else:
            st.success(f"✅ **Normal Range: Low Risk** (Confidence: {(1 - proba)*100:.1f}%)")
            st.info("Clinical markers indicate normal parameters.")
@st.dialog("❤️ Comprehensive Guide: Heart Disease", width="large")
def show_heart_disease_modal():
    c_img, c_txt = st.columns([1.1, 1.4], gap="medium")
    with c_img:
        st.image("https://images.unsplash.com/photo-1628348068343-c6a848d2b6dd?auto=format&fit=crop&w=700&q=80", caption="Arterial Structure & Anatomy", use_container_width=True)
        st.markdown("""
        | Condition | Primary Focus |
        | :--- | :--- |
        | **CAD** | Coronary Arteries |
        | **Arrhythmia** | Electrical Conduction |
        | **Cardiomyopathy** | Ventricular Muscle |
        """)
    with c_txt:
        st.markdown("#### Clinical Overview & Pathophysiology")
        st.markdown("""
        * **Atherosclerosis Initiation:** Starts with arterial lining damage, followed by LDL cholesterol calcification.
        * **Coronary Artery Narrowing:** Plaque buildup restricts oxygenated blood delivery to heart tissues.
        * **Ischemic Cascades:** Prolonged oxygen deprivation leads to cellular injury and potential myocardial infarction.
        * **Global Mortality:** Cardiovascular conditions remain the leading global cause of death according to WHO data.
        * **Genetic Risks:** Inherited lipoprotein levels and family history significantly elevate baseline risks.
        * **Hypertension Impact:** Chronic high blood pressure strains heart chambers, inducing left ventricular hypertrophy.
        * **Metabolic Syndrome:** Concomitant elevated glucose and high triglycerides accelerate arterial stiffness.
        """)

@st.dialog("⚠️ Clinical Symptoms & Early Warning Signs", width="large")
def show_symptoms_modal():
    c_img, c_txt = st.columns([1.1, 1.4], gap="medium")
    with c_img:
        st.image("https://images.unsplash.com/photo-1576091160550-2173dba999ef?auto=format&fit=crop&w=700&q=80", caption="Clinical Symptom Triage", use_container_width=True)
        st.markdown("""
        | Angina Class | Clinical Description |
        | :--- | :--- |
        | **Typical (TA)** | Pressure relieved with rest |
        | **Atypical (ATA)** | Dyspnea, fatigue, nausea |
        | **Silent (ASY)** | Painless ischemia in diabetics |
        """)
    with c_txt:
        st.markdown("#### Common Warning Signs & Variants")
        st.markdown("""
        * **Substernal Pressure:** Squeezing discomfort or heavy tightness centered in the chest.
        * **Referred Pain Pathways:** Discomfort spreading outward to the neck, jaw, shoulder, or left arm.
        * **Exertional Dyspnea:** Unexplained shortness of breath during light daily activity or rest.
        * **Cold Diaphoresis:** Clammy, sudden cold sweat occurring independently of room temperature.
        * **Female Presentation:** Women frequently exhibit atypical signs such as nausea, dizziness, or indigestion.
        * **Silent Ischemia:** Diabetic neuropathy can mask typical pain cues entirely.
        * **Syncope & Presyncope:** Unexplained lightheadedness indicating poor cardiac perfusion.
        """)

@st.dialog("📊 Algorithmic Risk Stratification", width="large")
def show_risk_prediction_modal():
    c_img, c_txt = st.columns([1.1, 1.4], gap="medium")
    with c_img:
        st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=700&q=80", caption="Risk Stratification Analytics", use_container_width=True)
        st.markdown("""
        | Clinical Parameter | Baseline Target | Elevated Risk |
        | :--- | :--- | :--- |
        | **Blood Pressure** | < 120/80 mmHg | ≥ 130/80 mmHg |
        | **Cholesterol** | < 200 mg/dL | ≥ 240 mg/dL |
        | **ST Depression** | 0.0 mm | > 1.5 mm |
        """)
    with c_txt:
        st.markdown("#### Machine Learning Analytics Architecture")
        st.markdown("""
        * **Multivariable Mapping:** Algorithms evaluate interconnected metrics across 11 key diagnostic markers.
        * **Standardized Scaling:** Z-score normalization balances disparate units (blood pressure vs. heart rate).
        * **Functional Capacity:** Maximum achieved heart rate provides insights into coronary reserve strength.
        * **ST Wave Dynamics:** ST slope (Up, Flat, Down) and depression detect reversible myocardial strain.
        * **Glycemic Risk:** Elevated fasting blood sugar (>120 mg/dL) shifts statistical weight upwards.
        * **Classification Sensitivity:** Supervised KNN models prioritize high sensitivity to reduce false negatives.
        * **Proactive Re-testing:** Periodic metric re-entry supports continuous preventive tracking.
        """)

@st.dialog("🩺 Diagnostic Pathways & Screenings", width="large")
def show_diagnosis_modal():
    c_img, c_txt = st.columns([1.1, 1.4], gap="medium")
    with c_img:
        st.image("https://images.unsplash.com/photo-1516549655169-df83a0774514?auto=format&fit=crop&w=700&q=80", caption="Cardiovascular Diagnostic Modalities", use_container_width=True)
        st.markdown("""
        | Modality | Diagnostic Utility |
        | :--- | :--- |
        | **12-Lead ECG** | Rhythm & ST abnormalities |
        | **Echocardiogram** | Ejection fraction, chamber size |
        | **Angiography** | Direct visual arterial stenosis |
        """)
    with c_txt:
        st.markdown("#### Primary Diagnostic Modalities")
        st.markdown("""
        * **12-Lead ECG:** Real-time electrical waveform tracing identifying ischemia and rhythm disorders.
        * **Cardiac Biomarkers:** Blood tests measuring Troponin I/T to detect acute heart muscle injury.
        * **Echocardiography:** Non-invasive ultrasound measuring chamber wall motion and ejection fraction.
        * **Stress Testing:** Monitored treadmill exercise evaluating cardiac responses to physical stress.
        * **Coronary CT (CCTA):** High-resolution scans quantifying calcium deposits in arterial walls.
        * **Coronary Angiography:** Gold-standard catheter procedure mapping exact blockage percentages.
        * **Holter Monitoring:** Ambulatory continuous recording capturing sporadic palpitations over 24–48 hours.
        """)

@st.dialog("🛡️ Prevention & Risk Mitigation", width="large")
def show_prevention_modal():
    c_img, c_txt = st.columns([1.1, 1.4], gap="medium")
    with c_img:
        st.image("https://images.unsplash.com/photo-1498837167922-ddd27525d352?auto=format&fit=crop&w=700&q=80", caption="Preventive Lifestyle Interventions", use_container_width=True)
        st.markdown("""
        | Factor | Recommendation |
        | :--- | :--- |
        | **Aerobic Activity** | 150 min/week |
        | **Sodium Intake** | < 2,000 mg/day |
        | **Sleep Duration** | 7–8 hours/night |
        """)
    with c_txt:
        st.markdown("#### Evidence-Based Interventions")
        st.markdown("""
        * **Mediterranean/DASH Diet:** Prioritize leafy greens, berries, legumes, olive oil, and omega-3s.
        * **Sodium Control:** Keeping sodium below 2,000 mg per day reduces vessel wall tension and blood pressure.
        * **Routine Exercise:** 30 minutes of daily moderate aerobic activity maintains vascular elasticity.
        * **Smoking Cessation:** Halves personal cardiovascular event risks within one year of quitting.
        * **Metabolic Tracking:** Maintaining healthy HbA1c and LDL cholesterol prevents plaque formation.
        * **Adequate Sleep:** 7 to 8 hours of restorative sleep regulates nocturnal blood pressure dips.
        * **Stress Management:** Controlled breathing and mindfulness exercises decrease chronic cortisol spikes.
        """)

@st.dialog("🌱 Heart Health & Everyday Habits", width="large")
def show_heart_health_modal():
    c_img, c_txt = st.columns([1.1, 1.4], gap="medium")
    with c_img:
        st.image("https://images.unsplash.com/photo-1506126613408-eca07ce68773?auto=format&fit=crop&w=700&q=80", caption="Cardiovascular Wellness & Habits", use_container_width=True)
        st.markdown("""
        | Health Target | Recommended Target |
        | :--- | :--- |
        | **Resting Heart Rate** | 60–80 bpm |
        | **Body Mass Index** | 18.5–24.9 kg/m² |
        | **Daily Water Intake** | 2.5–3.0 Liters |
        """)
    with c_txt:
        st.markdown("#### Practical Wellness Principles")
        st.markdown("""
        * **Resting Heart Rate:** Lower resting rates indicate efficient stroke volume and muscle conditioning.
        * **Soluble Fiber:** Daily intake of oats, beans, and seeds helps reduce intestinal cholesterol absorption.
        * **Omega-3 Fatty Acids:** Consuming fatty fish twice weekly supports healthy cell membrane fluidity.
        * **Resistance Training:** Performing moderate strength training twice weekly enhances insulin sensitivity.
        * **Consistent Hydration:** Drinking 2 to 3 liters of water daily helps maintain optimal blood viscosity.
        * **Limiting Alcohol:** Avoiding excessive alcohol consumption reduces arrhythmia and cardiomyopathy risks.
        * **Annual Checkups:** Consistent blood pressure monitoring and lipid profiling ensure early detection.
        """)

# Main Header & Navigation
st.markdown("""
<div class="top-navbar">
    <div class="brand-group">
        <div class="brand-icon">📈</div>
        <div class="brand-text">
            <h2 class="brand-title">Heart Disease Prediction</h2>
            <p class="brand-sub">Better insights • Healthier Tomorrows</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

active_tab = st.tabs([
    "🏠 Home", "♡ Heart Disease", "⚠️ Symptoms", "📊 Risk Prediction", "🛡️ Prevention", "ⓘ About"
])

# Tab 1: Home
with active_tab[0]:
    hero_col1, hero_col2 = st.columns([1.1, 1], gap="large")

    with hero_col1:
        st.markdown("""
        <div style="margin-top: 1.5rem;">
            <div class="hero-badge">AI-POWERED HEALTHCARE SOLUTION</div>
            <h1 class="hero-title-dark">Heart Disease</h1>
            <h1 class="hero-title-red">Prediction</h1>
            <p class="hero-description">
                Understand your heart. Predict the risk.<br>
                Take control of your health.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("♡ Start Prediction →", key="btn_hero_start"):
            open_prediction_dialog()

    with hero_col2:
        components.html(
        """
        <html>
        <head>
        <style>
        body {
            margin: 0;
            background: transparent;
            font-family: Arial, sans-serif;
        }

        .monitor {
            text-align: center;
            padding: 20px;
        }

        .heart {
            font-size: 85px;
            animation: heartbeat 1s infinite;
        }

        @keyframes heartbeat {
            0%, 100% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.15);
            }
        }

        .bpm {
            margin-top: 5px;
        }

        .number {
            font-size: 32px;
            font-weight: bold;
            color: #e11d48;
        }

        .label {
            font-size: 14px;
            color: #777;
        }

        .ecg {
            width: 100%;
            overflow: hidden;
            margin-top: 20px;
        }

        .track {
            display: flex;
            width: 200%;
            animation: move 3s linear infinite;
        }

        svg {
            width: 50%;
            height: 120px;
        }

        polyline {
            fill: none;
            stroke: #e11d48;
            stroke-width: 4;
        }

        @keyframes move {
            from {
                transform: translateX(0);
            }
            to {
                transform: translateX(-50%);
            }
        }

        .status {
            margin-top: 10px;
            font-size: 13px;
            color: #e11d48;
            font-weight: bold;
        }
        </style>
        </head>

        <body>

        <div class="monitor">

            <div class="heart">❤️</div>

            <div class="bpm">
                <div class="number" id="bpm">72</div>
                <div class="label">BPM</div>
            </div>

            <div class="ecg">
                <div class="track">

                    <svg viewBox="0 0 800 160">
                        <polyline points="0,80 100,80 130,80 150,40 170,120 190,80 250,80 280,80 300,30 320,130 340,80 420,80 450,80 470,40 490,120 510,80 600,80 630,80 650,35 670,125 690,80 800,80"/>
                    </svg>

                    <svg viewBox="0 0 800 160">
                        <polyline points="0,80 100,80 130,80 150,40 170,120 190,80 250,80 280,80 300,30 320,130 340,80 420,80 450,80 470,40 490,120 510,80 600,80 630,80 650,35 670,125 690,80 800,80"/>
                    </svg>

                </div>
            </div>

            <div class="status">● LIVE HEART MONITOR</div>

        </div>

        <script>
        let bpm = 72;
        let direction = 1;

        setInterval(function() {
            bpm += direction;

            if (bpm >= 75) {
                direction = -1;
            }

            if (bpm <= 70) {
                direction = 1;
            }

            document.getElementById("bpm").innerText = bpm;
        }, 700);
        </script>

        </body>
        </html>
        """,
        height=390,
        scrolling=False
    )
    # 4 Feature Badges
    st.markdown("""
    <div class="features-row">
        <div class="feature-item">
            <div class="feature-circle" style="background:#e0f2fe; color:#0284c7;">⚡</div>
            <div class="feature-txt">
                <h5>Machine Learning</h5>
                <p>Powered Predictions</p>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-circle" style="background:#dcfce7; color:#16a34a;">🛡️</div>
            <div class="feature-txt">
                <h5>Accurate Results</h5>
                <p>Based on Real Data</p>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-circle" style="background:#f3e8ff; color:#9333ea;">⏱️</div>
            <div class="feature-txt">
                <h5>Quick & Easy</h5>
                <p>Just a Few Steps</p>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-circle" style="background:#ffe4e6; color:#e11d48;">❤️</div>
            <div class="feature-txt">
            <h5>Better Decisions</h5>
                <p>For a Healthier Life</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-title-wrap">
        <h3>📈 Explore More About Heart Health</h3>
        <p>Learn about heart disease, its symptoms, risk factors and how you can prevent it.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.markdown("""
        <div class="topic-tile" style="background:#fff1f2; min-height:160px;">
            <div class="topic-tile-icon">❤️</div>
            <h4>Heart Disease</h4>
            <p>What is heart disease, its types and clinical risks.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Learn more →", key="btn_c1", use_container_width=True):
            show_heart_disease_modal()

    with c2:
        st.markdown("""
        <div class="topic-tile" style="background:#fefce8; min-height:160px;">
            <div class="topic-tile-icon">⚠️</div>
            <h4>Symptoms</h4>
            <p>Know the warning signs, angina and silent cues.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Learn more →", key="btn_c2", use_container_width=True):
            show_symptoms_modal()

    with c3:
        st.markdown("""
        <div class="topic-tile" style="background:#f0f9ff; min-height:160px;">
            <div class="topic-tile-icon">📊</div>
            <h4>Risk Prediction</h4>
            <p>How metrics calculate cardiovascular likelihood.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Learn more →", key="btn_c3", use_container_width=True):
            show_risk_prediction_modal()

    with c4:
        st.markdown("""
        <div class="topic-tile" style="background:#faf5ff; min-height:160px;">
            <div class="topic-tile-icon">🩺</div>
            <h4>Diagnosis</h4>
            <p>How conditions are identified and clinically screened.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Learn more →", key="btn_c4", use_container_width=True):
            show_diagnosis_modal()

    with c5:
        st.markdown("""
        <div class="topic-tile" style="background:#f0fdf4; min-height:160px;">
            <div class="topic-tile-icon">🛡️</div>
            <h4>Prevention</h4>
            <p>Guidelines and proactive steps for long-term health.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Learn more →", key="btn_c5", use_container_width=True):
            show_prevention_modal()

    with c6:
        st.markdown("""
        <div class="topic-tile" style="background:#f0fdfa; min-height:160px;">
            <div class="topic-tile-icon">🌱</div>
            <h4>Heart Health</h4>
            <p>Cardioprotective habits, diet and daily exercises.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Learn more →", key="btn_c6", use_container_width=True):
            show_heart_health_modal()

# Tab 2: Heart Disease
with active_tab[1]:
    st.markdown("""
    <div class="content-box">
        <h2 style="color:#0f172a; margin-top:0;">Understanding Cardiovascular Diseases (CVD)</h2>
        <p style="color:#475569; font-size:1.05rem;">
            Cardiovascular diseases represent a spectrum of structural and functional disorders affecting blood vessels and the heart muscle. Atherosclerosis—the arterial hardening caused by calcified lipid plaques—is the fundamental driver across most acute coronary events.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c_hd1, c_hd2, c_hd3 = st.columns(3)
    with c_hd1:
        st.markdown("""
        <div class="content-box" style="height:100%;">
            <h4>🫀 Coronary Artery Disease (CAD)</h4>
            <p style="font-size:0.88rem; color:#475569;">
                <b>Mechanism:</b> Gradual narrowing of the coronary arteries by cholesterol plaque diminishes myocardial perfusion.<br><br>
                <b>Clinical Implication:</b> Results in stable angina during exercise or unstable acute coronary syndromes if the fibrous cap ruptures.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c_hd2:
        st.markdown("""
        <div class="content-box" style="height:100%;">
            <h4>🧠 Cerebrovascular Disease</h4>
            <p style="font-size:0.88rem; color:#475569;">
                <b>Mechanism:</b> Occlusion of carotid or cerebral vessels by thrombi causes ischemic strokes, while vessel ruptures induce intracranial hemorrhage.<br><br>
                <b>Clinical Implication:</b> Rapid loss of neurological function requiring emergency thrombolytic intervention within a 4.5-hour therapeutic window.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c_hd3:
        st.markdown("""
        <div class="content-box" style="height:100%;">
            <h4>⚡ Heart Failure & Arrhythmias</h4>
            <p style="font-size:0.88rem; color:#475569;">
                <b>Mechanism:</b> Structural myocyte death reduces cardiac ejection fraction (systolic/diastolic heart failure), while conduction defects cause atrial fibrillation.<br><br>
                <b>Clinical Implication:</b> Elevates peripheral edema and substantially increases systemic thromboembolism risk.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Tab 3: Symptoms
with active_tab[2]:
    st.markdown("""
    <div class="content-box">
        <h2 style="color:#0f172a; margin-top:0;">Recognizing Warning Signs & Symptom Classifications</h2>
        <p style="color:#475569;">Timely triage and clinical identification of chest discomfort reduce myocardial necrosis and improve long-term functional recovery.</p>
    </div>
    """, unsafe_allow_html=True)

    sym_c1, sym_c2 = st.columns(2)
    with sym_c1:
        st.markdown("""
        <div class="content-box">
            <h4 style="color:#e11d48;">🚨 Common Physical Indicators</h4>
            <ul style="color:#334155; line-height:1.9; font-size:0.92rem;">
                <li><b>Central Retrosternal Pressure:</b> Crushing, heavy tightness behind the breastbone lasting longer than 5 minutes.</li>
                <li><b>Pain Radiation Pathways:</b> Discomfort traveling along sensory dermatomes to the left shoulder, inner arm, jaw, or scapula.</li>
                <li><b>Dyspnea:</b> Unexplained shortness of breath on mild exertion or when lying flat (orthopnea).</li>
                <li><b>Cold Diaphoresis:</b> Sudden cold sweats accompanied by ashen pallor without physical exertion.</li>
                <li><b>Atypical Female Signs:</b> Pronounced fatigue, epigastric heartburn sensations, sleep disturbance, and dizziness.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with sym_c2:
        st.markdown("""
        <div class="content-box">
            <h4 style="color:#0284c7;">🔍 Clinical Chest Pain Types</h4>
            <ul style="color:#334155; line-height:1.9; font-size:0.92rem;">
                <li><b>Typical Angina (TA):</b> Exertion-provoked retrosternal discomfort promptly relieved within minutes by rest or sublingual nitroglycerin.</li>
                <li><b>Atypical Angina (ATA):</b> Exertional discomfort exhibiting non-standard features like sharp burning, prominent dyspnea, or nausea.</li>
                <li><b>Non-Anginal Pain (NAP):</b> Aches stemming from musculoskeletal (costochondritis), gastrointestinal (GERD), or pulmonary causes.</li>
                <li><b>Asymptomatic (ASY):</b> Complete absence of chest discomfort; frequent in diabetic individuals with autonomic nerve degradation.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        # Tab 4: Risk Prediction
with active_tab[3]:

    st.markdown("""
    <div class="content-box">
        <h2 style="color:#0f172a; margin-top:0;">
            Algorithmic Risk Stratification & Model Metrics
        </h2>
        <p style="color:#475569;">
            Enter patient physiological indicators to calculate cardiovascular risk
            using the trained machine learning model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_t1, col_t2 = st.columns(2)

    # LEFT SIDE
    with col_t1:

        t_age = st.slider(
            "Patient Age (Years)",
            18, 100, 48,
            key="tab_age"
        )

        t_sex = st.selectbox(
            "Biological Sex",
            ["M", "F"],
            format_func=lambda x: "Male" if x == "M" else "Female",
            key="tab_sex"
        )

        t_cp = st.selectbox(
            "Chest Pain Type",
            ["ATA", "NAP", "ASY", "TA"],
            key="tab_cp"
        )

        t_rbp = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=80,
            max_value=220,
            value=130,
            key="tab_rbp"
        )

        t_chol = st.number_input(
            "Serum Cholesterol (mg/dL)",
            min_value=100,
            max_value=600,
            value=220,
            key="tab_chol"
        )

    # RIGHT SIDE
    with col_t2:

        t_fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            key="tab_fbs"
        )

        t_ecg = st.selectbox(
            "Resting ECG Result",
            ["Normal", "ST", "LVH"],
            key="tab_ecg"
        )

        t_mhr = st.slider(
            "Maximum Heart Rate Achieved (bpm)",
            60, 220, 145,
            key="tab_mhr"
        )

        t_ang = st.selectbox(
            "Exercise-Induced Angina",
            ["Y", "N"],
            format_func=lambda x: "Yes" if x == "Y" else "No",
            key="tab_ang"
        )

        t_slope = st.selectbox(
            "Peak Exercise ST Slope",
            ["Up", "Flat", "Down"],
            key="tab_slope"
        )

        t_oldpeak = st.slider(
            "ST Depression (Oldpeak)",
            0.0, 6.0, 1.2,
            key="tab_oldpeak"
        )

    # PREDICTION BUTTON
    if st.button(
        "❤️ Evaluate Cardiac Risk",
        use_container_width=True,
        key="tab_predict_btn"
    ):

        # Check model files
        if model is None:
            st.error("❌ Model file not found: RF_KNN_heart.pkl")
            st.stop()

        if scaler is None:
            st.error("❌ Scaler file not found: scaler.pkl")
            st.stop()

        if expected_columns is None:
            st.error("❌ Column file not found: columns.pkl")
            st.stop()

        try:

            # Create patient input
            tab_raw = {
                "Age": t_age,
                "RestingBP": t_rbp,
                "Cholesterol": t_chol,
                "FastingBS": t_fbs,
                "MaxHR": t_mhr,
                "Oldpeak": t_oldpeak,

                "Sex_" + t_sex: 1,
                "ChestPainType_" + t_cp: 1,
                "RestingECG_" + t_ecg: 1,
                "ExerciseAngina_" + t_ang: 1,
                "ST_Slope_" + t_slope: 1
            }

            # Convert to DataFrame
            tab_df = pd.DataFrame([tab_raw])

            # Make sure expected columns are a normal list
            expected_cols = list(expected_columns)

            # Add missing columns
            for col in expected_cols:
                if col not in tab_df.columns:
                    tab_df[col] = 0

            # Remove unexpected columns
            tab_df = tab_df.reindex(
                columns=expected_cols,
                fill_value=0
            )

            # Make sure values are numeric
            tab_df = tab_df.apply(
                pd.to_numeric,
                errors="coerce"
            ).fillna(0)

            # Scale input
            scaled_input = scaler.transform(tab_df)

            # Prediction
            tab_pred = model.predict(scaled_input)[0]

            # Probability
            if hasattr(model, "predict_proba"):

                prob_arr = model.predict_proba(scaled_input)[0]

                # Find probability safely
                if len(prob_arr) == 2:
                    risk_p = float(prob_arr[1]) * 100
                    safe_p = float(prob_arr[0]) * 100
                else:
                    risk_p = 100.0 if tab_pred == 1 else 0.0
                    safe_p = 100.0 - risk_p

            else:
                risk_p = 100.0 if tab_pred == 1 else 0.0
                safe_p = 100.0 - risk_p

            # Results
            st.markdown("### 📊 Prediction Result")

            res_c1, res_c2 = st.columns(2)

            with res_c1:
                st.metric(
                    "❤️ Risk Probability",
                    f"{risk_p:.1f}%"
                )

            with res_c2:
                st.metric(
                    "🛡️ Low-Risk Probability",
                    f"{safe_p:.1f}%"
                )

            # Final message
            if int(tab_pred) == 1:

                st.error(
                    "⚠️ Elevated Risk Detected"
                )

                st.warning(
                    "The machine-learning model classified the submitted "
                    "parameters as higher risk. This is a screening result "
                    "and is not a medical diagnosis."
                )

            else:

                st.success(
                    "✅ Lower Risk Predicted"
                )

                st.info(
                    "The machine-learning model classified the submitted "
                    "parameters as lower risk. This is a screening result "
                    "and does not guarantee absence of heart disease."
                )

        except Exception as e:

            st.error("❌ Prediction could not be completed.")

            st.write(
                "Please check that your model, scaler and columns files "
                "were created from the same dataset/features."
            )

            st.code(str(e))
# Tab 5: Prevention
with active_tab[4]:
    st.markdown("""
    <div class="content-box">
        <h2 style="color:#0f172a; margin-top:0;">Evidence-Based Prevention & Risk Mitigation</h2>
        <p style="color:#475569;">Up to 80% of premature cardiovascular disease can be averted by implementing cardioprotective behavioral interventions.</p>
    </div>
    """, unsafe_allow_html=True)

    prev_c1, prev_c2, prev_c3 = st.columns(3)
    with prev_c1:
        st.markdown("""
        <div class="content-box" style="height:100%;">
            <h4>🥗 Cardioprotective Nutrition</h4>
            <p style="font-size:0.88rem; color:#475569;">
                • Prioritize Mediterranean and DASH dietary patterns.<br>
                • Consume extra virgin olive oil, walnuts, and cold-water fatty fish (rich in EPA/DHA).<br>
                • Eliminate trans-fatty acids and ultra-processed snacks.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with prev_c2:
        st.markdown("""
        <div class="content-box" style="height:100%;">
            <h4>🏃 Exercise Prescription</h4>
            <p style="font-size:0.88rem; color:#475569;">
                • At least 150 minutes of moderate aerobic training weekly.<br>
                • Incorporate resistance training twice weekly to enhance muscle glucose uptake.<br>
                • Avoid prolonged sedentary intervals with hourly micro-walks.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with prev_c3:
        st.markdown("""
        <div class="content-box" style="height:100%;">
            <h4>🛡️ Vascular Risk Factor Control</h4>
            <p style="font-size:0.88rem; color:#475569;">
                • Total smoking cessation (halves risk within 12 months).<br>
                • Limit dietary sodium intake to under 2,000 mg/day.<br>
                • Maintain 7–8 hours of uninterrupted sleep to avoid nocturnal hypertension.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Tab 6: About
with active_tab[5]:
    st.markdown("""
    <div class="content-box">
        <h2 style="color:#0f172a; margin-top:0;">About CardioCare Intelligence</h2>
        <p style="color:#475569;">
            This web application is a clinical decision-support and educational screening system developed with Streamlit and Scikit-Learn. It leverages supervised machine learning models to assess non-linear cardiovascular risk parameters from standardized cardiac datasets.
        </p>
        <p style="font-size:0.85rem; color:#64748b; margin-top:1rem; border-top: 1px solid #e2e8f0; padding-top: 1rem;">
            ⚕️ <b>Clinical Disclaimer:</b> The predictions, probability scores, and indicators rendered by this platform are algorithmic approximations intended solely for screening and educational presentation. They do not replace formal clinical diagnosis or physician oversight.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="clinical-footer">
    A healthier heart leads to a brighter future ♡
</div>
""", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import joblib
import os

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Heart Disease Prediction | Better Insights • Healthier Tomorrows",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #fcfdfe;
    color: #1e293b;
}

.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1240px !important;
}

[data-testid="stSidebarNav"],
footer,
header {
    display: none !important;
}

/* =========================================================
   NAVBAR
   ========================================================= */

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

/* =========================================================
   HERO
   ========================================================= */

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

/* =========================================================
   BUTTON
   ========================================================= */

div.stButton > button {
    background: linear-gradient(
        135deg,
        #e11d48 0%,
        #be123c 100%
    ) !important;

    color: #ffffff !important;
    border: none !important;
    border-radius: 26px !important;
    padding: 0.5rem 1.4rem !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;

    box-shadow:
        0 4px 14px rgba(225, 29, 72, 0.25) !important;

    transition: all 0.25s ease-in-out !important;
}

div.stButton > button:hover {
    transform: translateY(-2px) !important;

    box-shadow:
        0 8px 22px rgba(225, 29, 72, 0.35) !important;
}

/* =========================================================
   HEART ANIMATION CONTAINER
   ========================================================= */

.heart-animation-container {
    width: 100%;
    height: 390px;
    position: relative;
    overflow: hidden;

    border-radius: 28px;

    background:
        radial-gradient(
            circle at 50% 45%,
            rgba(244, 63, 94, 0.12),
            transparent 38%
        ),
        linear-gradient(
            145deg,
            #fff1f2 0%,
            #ffffff 50%,
            #fef2f2 100%
        );

    border: 1px solid #fecdd3;

    box-shadow:
        0 18px 45px rgba(225, 29, 72, 0.13);

    display: flex;
    align-items: center;
    justify-content: center;
}

/* Background circles */

.pulse-circle {
    position: absolute;
    width: 210px;
    height: 210px;

    border-radius: 50%;

    border: 2px solid rgba(225, 29, 72, 0.13);

    animation: circlePulse 2s infinite ease-out;
}

.pulse-circle.two {
    animation-delay: 0.65s;
}

.pulse-circle.three {
    animation-delay: 1.3s;
}

@keyframes circlePulse {
    0% {
        transform: scale(0.65);
        opacity: 0.8;
    }

    100% {
        transform: scale(1.7);
        opacity: 0;
    }
}

/* Heart */

.animated-heart {
    position: absolute;
    top: 82px;

    font-size: 6.8rem;

    filter:
        drop-shadow(
            0 10px 22px rgba(225, 29, 72, 0.28)
        );

    animation: heartbeat 1.05s infinite;
    z-index: 5;
}

@keyframes heartbeat {
    0% {
        transform: scale(1);
    }

    15% {
        transform: scale(1.14);
    }

    30% {
        transform: scale(1);
    }

    45% {
        transform: scale(1.12);
    }

    65% {
        transform: scale(1);
    }

    100% {
        transform: scale(1);
    }
}

/* ECG */

.ecg-wrapper {
    position: absolute;

    left: 8%;
    right: 8%;
    bottom: 55px;

    height: 100px;

    overflow: hidden;

    border-radius: 14px;

    background: rgba(255,255,255,0.72);

    border:
        1px solid rgba(225,29,72,0.12);
}

.ecg-line {
    position: absolute;

    width: 200%;
    height: 100%;

    left: 0;
    top: 0;

    animation: ecgMove 2.4s linear infinite;
}

@keyframes ecgMove {
    from {
        transform: translateX(0);
    }

    to {
        transform: translateX(-50%);
    }
}

.ecg-svg {
    width: 100%;
    height: 100%;
}

.ecg-path {
    fill: none;

    stroke: #e11d48;

    stroke-width: 3;

    stroke-linecap: round;
    stroke-linejoin: round;

    filter:
        drop-shadow(
            0 0 5px rgba(225,29,72,0.45)
        );
}

/* Monitor label */

.monitor-label {
    position: absolute;

    bottom: 15px;

    left: 50%;

    transform: translateX(-50%);

    font-size: 0.72rem;

    font-weight: 800;

    letter-spacing: 0.12em;

    color: #be123c;

    text-transform: uppercase;

    background: rgba(255,255,255,0.88);

    padding: 6px 13px;

    border-radius: 20px;

    border: 1px solid #fecdd3;
}

/* BPM */

.bpm-display {
    position: absolute;

    right: 20px;
    top: 18px;

    background: white;

    border-radius: 12px;

    padding: 8px 13px;

    box-shadow:
        0 5px 18px rgba(0,0,0,0.07);

    text-align: center;

    z-index: 10;
}

.bpm-number {
    color: #e11d48;

    font-size: 1.1rem;

    font-weight: 800;

    margin: 0;
}

.bpm-text {
    color: #64748b;

    font-size: 0.65rem;

    margin: 0;

    font-weight: 600;
}

/* =========================================================
   FEATURE CARDS
   ========================================================= */

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

    box-shadow:
        0 2px 8px rgba(0,0,0,0.02);
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

/* =========================================================
   SECTION TITLES
   ========================================================= */

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

/* =========================================================
   TOPIC TILES
   ========================================================= */

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

/* =========================================================
   CONTENT BOX
   ========================================================= */

.content-box {
    background: #ffffff;

    border: 1px solid #e2e8f0;

    border-radius: 18px;

    padding: 2rem;

    margin-bottom: 1.5rem;

    box-shadow:
        0 3px 12px rgba(0,0,0,0.03);
}

/* =========================================================
   FOOTER
   ========================================================= */

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

/* =========================================================
   MOBILE RESPONSIVE
   ========================================================= */

@media (max-width: 900px) {

    .hero-title-dark,
    .hero-title-red {
        font-size: 2.5rem;
    }

    .features-row {
        grid-template-columns: repeat(2, 1fr);
    }

    .heart-animation-container {
        height: 320px;
    }
}

@media (max-width: 600px) {

    .features-row {
        grid-template-columns: 1fr;
    }

    .hero-title-dark,
    .hero-title-red {
        font-size: 2.2rem;
    }

    .heart-animation-container {
        height: 300px;
    }

    .animated-heart {
        font-size: 5.5rem;
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL LOADER
# =========================================================

@st.cache_resource
def get_model_assets():

    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(
        base_dir,
        "RF_KNN_heart.pkl"
    )

    scaler_path = os.path.join(
        base_dir,
        "scaler.pkl"
    )

    columns_path = os.path.join(
        base_dir,
        "columns.pkl"
    )

    model = (
        joblib.load(model_path)
        if os.path.exists(model_path)
        else None
    )

    scaler = (
        joblib.load(scaler_path)
        if os.path.exists(scaler_path)
        else None
    )

    columns = (
        joblib.load(columns_path)
        if os.path.exists(columns_path)
        else None
    )

    return model, scaler, columns


model, scaler, expected_columns = get_model_assets()


# =========================================================
# HELPER FUNCTION FOR PREDICTION
# =========================================================

def make_prediction(
    age,
    sex,
    chest_pain,
    resting_bp,
    cholesterol,
    fasting_bs,
    ecg,
    max_hr,
    exercise_angina,
    slope,
    oldpeak
):

    if (
        model is None
        or scaler is None
        or expected_columns is None
    ):
        return None, None

    raw_dict = {

        "Age": age,

        "RestingBP": resting_bp,

        "Cholesterol": cholesterol,

        "FastingBS": fasting_bs,

        "MaxHR": max_hr,

        "Oldpeak": oldpeak,

        "Sex_" + sex: 1,

        "ChestPainType_" + chest_pain: 1,

        "RestingECG_" + ecg: 1,

        "ExerciseAngina_" + exercise_angina: 1,

        "ST_Slope_" + slope: 1
    }

    df = pd.DataFrame([raw_dict])

    for col in expected_columns:

        if col not in df.columns:
            df[col] = 0

    df = df[expected_columns]

    scaled_df = scaler.transform(df)

    prediction = model.predict(scaled_df)[0]

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            scaled_df
        )[0]

        risk_probability = probabilities[1] * 100

        safe_probability = probabilities[0] * 100

    else:

        risk_probability = (
            100.0 if prediction == 1 else 0.0
        )

        safe_probability = 100.0 - risk_probability

    return (
        prediction,
        risk_probability,
        safe_probability
    )


# =========================================================
# PREDICTION DIALOG
# =========================================================

@st.dialog("🩺 Cardiovascular Assessment Diagnostic")
def open_prediction_dialog():

    st.caption(
        "Enter clinical markers below to evaluate cardiac health status:"
    )

    d1, d2 = st.columns(2)

    with d1:

        age_in = st.slider(
            "Age (Years)",
            18,
            100,
            42
        )

        sex_in = st.selectbox(
            "Sex",
            ["M", "F"],
            format_func=lambda x:
                "Male" if x == "M" else "Female"
        )

        cp_in = st.selectbox(
            "Chest Pain Type",
            ["ATA", "NAP", "ASY", "TA"]
        )

        rbp_in = st.number_input(
            "Resting BP (mm Hg)",
            80,
            220,
            120
        )

        chol_in = st.number_input(
            "Serum Cholesterol (mg/dL)",
            100,
            600,
            205
        )

    with d2:

        fbs_in = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )

        ecg_in = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"]
        )

        mhr_in = st.slider(
            "Max Heart Rate (bpm)",
            60,
            220,
            150
        )

        ang_in = st.selectbox(
            "Exercise Angina",
            ["Y", "N"],
            format_func=lambda x:
                "Yes" if x == "Y" else "No"
        )

        slope_in = st.selectbox(
            "ST Slope",
            ["Up", "Flat", "Down"]
        )

        oldpeak_in = st.slider(
            "Oldpeak (ST Depression)",
            0.0,
            6.0,
            1.0
        )

    st.divider()

    if st.button(
        "🔍 Analyze Heart Risk",
        use_container_width=True
    ):

        if (
            model is None
            or scaler is None
            or expected_columns is None
        ):

            st.warning(
                "Model files are missing. Please make sure "
                "RF_KNN_heart.pkl, scaler.pkl and columns.pkl "
                "are present in the project folder."
            )

            return

        result = make_prediction(
            age_in,
            sex_in,
            cp_in,
            rbp_in,
            chol_in,
            fbs_in,
            ecg_in,
            mhr_in,
            ang_in,
            slope_in,
            oldpeak_in
        )

        pred, risk, safe = result

        if pred == 1:

            st.error(
                f"⚠️ Elevated Risk Detected "
                f"(Model probability: {risk:.1f}%)"
            )

            st.warning(
                "These results are for educational screening only "
                "and should not replace evaluation by a healthcare professional."
            )

        else:

            st.success(
                f"✅ Lower Predicted Risk "
                f"(Model probability: {safe:.1f}% lower-risk class)"
            )

            st.info(
                "The submitted parameters fall into the model's lower-risk class. "
                "This does not guarantee absence of heart disease."
            )


# =========================================================
# HEART DISEASE MODAL
# =========================================================

@st.dialog(
    "❤️ Comprehensive Guide: Heart Disease",
    width="large"
)
def show_heart_disease_modal():

    c_img, c_txt = st.columns(
        [1.1, 1.4],
        gap="medium"
    )

    with c_img:

        st.image(
            "https://images.unsplash.com/photo-1628348068343-c6a848d2b6dd?auto=format&fit=crop&w=700&q=80",
            caption="Arterial Structure & Anatomy",
            use_container_width=True
        )

        st.markdown("""
| Condition | Primary Focus |
|---|---|
| **CAD** | Coronary Arteries |
| **Arrhythmia** | Electrical Conduction |
| **Cardiomyopathy** | Ventricular Muscle |
""")

    with c_txt:

        st.markdown(
            "#### Clinical Overview & Pathophysiology"
        )

        st.markdown("""
* **Atherosclerosis:** Plaque buildup can narrow arteries and reduce blood flow.
* **Coronary Artery Narrowing:** Reduced blood flow can limit oxygen delivery to heart muscle.
* **Ischemia:** Insufficient oxygen supply may cause chest discomfort and other symptoms.
* **Risk Factors:** Blood pressure, cholesterol, diabetes, smoking, age and family history can contribute to cardiovascular risk.
* **Hypertension:** Long-term high blood pressure can place additional strain on the heart.
* **Metabolic Health:** Blood glucose and lipid abnormalities can contribute to cardiovascular risk.
""")


# =========================================================
# SYMPTOMS MODAL
# =========================================================

@st.dialog(
    "⚠️ Clinical Symptoms & Early Warning Signs",
    width="large"
)
def show_symptoms_modal():

    c_img, c_txt = st.columns(
        [1.1, 1.4],
        gap="medium"
    )

    with c_img:

        st.image(
            "https://images.unsplash.com/photo-1576091160550-2173dba999ef?auto=format&fit=crop&w=700&q=80",
            caption="Clinical Symptom Awareness",
            use_container_width=True
        )

        st.markdown("""
| Angina Class | Clinical Description |
|---|---|
| **Typical (TA)** | Symptoms fitting common exertional angina patterns |
| **Atypical (ATA)** | Some, but not all, typical features |
| **Non-Anginal (NAP)** | Symptoms less consistent with angina |
| **Asymptomatic (ASY)** | No reported chest pain |
""")

    with c_txt:

        st.markdown(
            "#### Common Warning Signs & Variants"
        )

        st.markdown("""
* **Chest Pressure:** Pressure, squeezing or discomfort in the chest.
* **Radiating Discomfort:** Symptoms can sometimes extend toward the arm, shoulder, jaw or back.
* **Shortness of Breath:** Breathlessness can occur with or without chest discomfort.
* **Sweating:** Unexpected sweating can accompany cardiovascular symptoms.
* **Nausea or Dizziness:** Some people may experience these symptoms.
* **Important:** Symptoms can vary between people. Sudden or severe symptoms require urgent medical attention.
""")


# =========================================================
# RISK MODAL
# =========================================================

@st.dialog(
    "📊 Algorithmic Risk Stratification",
    width="large"
)
def show_risk_prediction_modal():

    c_img, c_txt = st.columns(
        [1.1, 1.4],
        gap="medium"
    )

    with c_img:

        st.image(
            "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=700&q=80",
            caption="Risk Stratification Analytics",
            use_container_width=True
        )

        st.markdown("""
| Clinical Parameter | Example |
|---|---|
| **Blood Pressure** | Measured in mmHg |
| **Cholesterol** | Measured in mg/dL |
| **Max Heart Rate** | Measured in bpm |
| **ST Depression** | Model input parameter |
""")

    with c_txt:

        st.markdown(
            "#### Machine Learning Analytics Architecture"
        )

        st.markdown("""
* **Multivariable Mapping:** The model evaluates several clinical parameters together.
* **Standardized Scaling:** Numerical inputs are transformed using the saved scaler.
* **Feature Encoding:** Categorical values such as chest-pain type and ECG result are encoded.
* **Classification:** The trained model predicts a class from the supplied parameters.
* **Probability:** When supported by the model, prediction probabilities are displayed.
* **Screening:** Results are intended for educational and project demonstration purposes.
""")


# =========================================================
# DIAGNOSIS MODAL
# =========================================================

@st.dialog(
    "🩺 Diagnostic Pathways & Screenings",
    width="large"
)
def show_diagnosis_modal():

    c_img, c_txt = st.columns(
        [1.1, 1.4],
        gap="medium"
    )

    with c_img:

        st.image(
            "https://images.unsplash.com/photo-1516549655169-df83a0774514?auto=format&fit=crop&w=700&q=80",
            caption="Cardiovascular Diagnostic Modalities",
            use_container_width=True
        )

        st.markdown("""
| Modality | Diagnostic Utility |
|---|---|
| **12-Lead ECG** | Rhythm & electrical abnormalities |
| **Echocardiogram** | Heart structure & function |
| **Angiography** | Visualization of coronary arteries |
""")

    with c_txt:

        st.markdown(
            "#### Primary Diagnostic Modalities"
        )

        st.markdown("""
* **12-Lead ECG:** Records the electrical activity of the heart.
* **Cardiac Biomarkers:** Blood tests such as troponin can help identify heart muscle injury.
* **Echocardiography:** Uses ultrasound to evaluate heart structure and function.
* **Stress Testing:** Evaluates cardiac response to physical or pharmacological stress.
* **Coronary CT:** Can provide detailed images of coronary anatomy.
* **Coronary Angiography:** Provides detailed visualization of coronary arteries.
* **Holter Monitoring:** Records heart rhythm over an extended period.
""")


# =========================================================
# PREVENTION MODAL
# =========================================================

@st.dialog(
    "🛡️ Prevention & Risk Mitigation",
    width="large"
)
def show_prevention_modal():

    c_img, c_txt = st.columns(
        [1.1, 1.4],
        gap="medium"
    )

    with c_img:

        st.image(
            "https://images.unsplash.com/photo-1498837167922-ddd27525d352?auto=format&fit=crop&w=700&q=80",
            caption="Preventive Lifestyle Interventions",
            use_container_width=True
        )

        st.markdown("""
| Factor | General Recommendation |
|---|---|
| **Aerobic Activity** | About 150 min/week |
| **Healthy Diet** | Vegetables, fruits & whole foods |
| **Sleep** | Regular adequate sleep |
""")

    with c_txt:

        st.markdown(
            "#### Evidence-Based Interventions"
        )

        st.markdown("""
* **Healthy Diet:** Emphasize vegetables, fruits, whole grains, legumes and healthy fats.
* **Physical Activity:** Regular moderate activity supports cardiovascular health.
* **Smoking:** Avoiding tobacco substantially reduces cardiovascular risk.
* **Blood Pressure:** Regular monitoring helps identify hypertension.
* **Cholesterol:** Appropriate lipid management can reduce cardiovascular risk.
* **Sleep:** Maintaining a consistent sleep schedule supports overall health.
* **Stress Management:** Relaxation techniques can support healthy lifestyle habits.
""")


# =========================================================
# HEART HEALTH MODAL
# =========================================================

@st.dialog(
    "🌱 Heart Health & Everyday Habits",
    width="large"
)
def show_heart_health_modal():

    c_img, c_txt = st.columns(
        [1.1, 1.4],
        gap="medium"
    )

    with c_img:

        st.image(
            "https://images.unsplash.com/photo-1506126613408-eca07ce68773?auto=format&fit=crop&w=700&q=80",
            caption="Cardiovascular Wellness & Habits",
            use_container_width=True
        )

        st.markdown("""
| Health Target | General Guidance |
|---|---|
| **Physical Activity** | Regular activity |
| **Healthy Diet** | Balanced nutrition |
| **Blood Pressure** | Regular monitoring |
""")

    with c_txt:

        st.markdown(
            "#### Practical Wellness Principles"
        )

        st.markdown("""
* **Regular Activity:** Supports cardiovascular fitness.
* **Fiber-Rich Foods:** Oats, beans, fruits and vegetables can support healthy cholesterol levels.
* **Healthy Fats:** Nuts, seeds and fish can be part of a balanced diet.
* **Strength Training:** Can complement aerobic activity.
* **Hydration:** Drink fluids according to your individual needs.
* **Avoid Tobacco:** Tobacco exposure increases cardiovascular risk.
* **Regular Checkups:** Blood pressure, glucose and lipid monitoring can help identify risk factors.
""")


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="top-navbar">

    <div class="brand-group">

        <div class="brand-icon">
            📈
        </div>

        <div class="brand-text">

            <h2 class="brand-title">
                Heart Disease Prediction
            </h2>

            <p class="brand-sub">
                Better insights • Healthier Tomorrows
            </p>

        </div>

    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# NAVIGATION TABS
# =========================================================

active_tab = st.tabs([
    "🏠 Home",
    "♡ Heart Disease",
    "⚠️ Symptoms",
    "📊 Risk Prediction",
    "🛡️ Prevention",
    "ⓘ About"
])


# =========================================================
# TAB 1 — HOME
# =========================================================

with active_tab[0]:

    hero_col1, hero_col2 = st.columns(
        [1.1, 1],
        gap="large"
    )

    # -----------------------------------------------------
    # LEFT SIDE
    # -----------------------------------------------------

    with hero_col1:

        st.markdown("""
<div style="margin-top: 1.5rem;">

    <div class="hero-badge">
        AI-POWERED HEALTHCARE SOLUTION
    </div>

    <h1 class="hero-title-dark">
        Heart Disease
    </h1>

    <h1 class="hero-title-red">
        Prediction
    </h1>

    <p class="hero-description">
        Understand your heart. Predict the risk.<br>
        Take control of your health.
    </p>

</div>
""", unsafe_allow_html=True)

        if st.button(
            "♡ Start Prediction →",
            key="btn_hero_start"
        ):
            open_prediction_dialog()


    # -----------------------------------------------------
    # RIGHT SIDE — ANIMATED HEART + ECG
    # -----------------------------------------------------

    with hero_col2:

        st.markdown("""
<div class="heart-animation-container">

    <div class="pulse-circle"></div>
    <div class="pulse-circle two"></div>
    <div class="pulse-circle three"></div>

    <div class="bpm-display">
        <p class="bpm-number">72 BPM</p>
        <p class="bpm-text">HEART RATE</p>
    </div>

    <div class="animated-heart">
        ❤️
    </div>

    <div class="ecg-wrapper">

        <div class="ecg-line">

            <svg
                class="ecg-svg"
                viewBox="0 0 1000 100"
                preserveAspectRatio="none"
            >

                <path
                    class="ecg-path"
                    d="
                    M0,55
                    L60,55
                    L75,55
                    L90,55
                    L105,55
                    L120,55
                    L135,55
                    L150,55
                    L160,55
                    L170,55
                    L180,55
                    L190,55
                    L200,55
                    L210,55
                    L220,55
                    L230,55
                    L240,55
                    L250,55
                    L260,55
                    L270,55
                    L280,55
                    L290,55
                    L300,55
                    L310,55
                    L320,55
                    L330,55
                    L340,55
                    L350,55
                    L360,55
                    L370,55
                    L380,55
                    L390,55
                    L400,55
                    L410,55
                    L420,55
                    L430,55
                    L440,55
                    L450,55
                    L460,55
                    L470,55
                    L480,55
                    L490,55
                    L500,55
                    L510,55
                    L520,55
                    L530,55
                    L540,55
                    L550,55
                    L560,55
                    L570,55
                    L580,55
                    L590,55
                    L600,55
                    L610,55
                    L620,55
                    L630,55
                    L640,55
                    L650,55
                    L660,55
                    L670,55
                    L680,55
                    L690,55
                    L700,55
                    L710,55
                    L720,55
                    L730,55
                    L740,55
                    L750,55
                    L760,55
                    L770,55
                    L780,55
                    L790,55
                    L800,55
                    L810,55
                    L820,55
                    L830,55
                    L840,55
                    L850,55
                    L860,55
                    L870,55
                    L880,55
                    L890,55
                    L900,55
                    L910,55
                    L920,55
                    L930,55
                    L940,55
                    L950,55
                    L960,55
                    L970,55
                    L980,55
                    L990,55
                    L1000,55

                    M0,55
                    L25,55
                    L35,55
                    L45,55
                    L55,55
                    L65,55
                    L75,55

                    L85,55
                    L95,55
                    L105,55

                    L115,55
                    L125,55

                    L135,55
                    L145,55

                    L155,55
                    L165,55

                    L175,55
                    L185,55

                    L195,55
                    L205,55

                    L215,55
                    L225,55

                    L235,55
                    L245,55

                    L255,55
                    L265,55

                    L275,55
                    L285,55

                    L295,55
                    L305,55

                    L315,55
                    L325,55

                    L335,55
                    L345,55

                    L355,55
                    L365,55

                    L375,55
                    L385,55

                    L395,55
                    L405,55

                    L415,55
                    L425,55

                    L435,55
                    L445,55

                    L455,55
                    L465,55

                    L475,55
                    L485,55

                    L495,55
                    L505,55

                    L515,55
                    L525,55

                    L535,55
                    L545,55

                    L555,55
                    L565,55

                    L575,55
                    L585,55

                    L595,55
                    L605,55

                    L615,55
                    L625,55

                    L635,55
                    L645,55

                    L655,55
                    L665,55

                    L675,55
                    L685,55

                    L695,55
                    L705,55

                    L715,55
                    L725,55

                    L735,55
                    L745,55

                    L755,55
                    L765,55

                    L775,55
                    L785,55

                    L795,55
                    L805,55

                    L815,55
                    L825,55

                    L835,55
                    L845,55

                    L855,55
                    L865,55

                    L875,55
                    L885,55

                    L895,55
                    L905,55

                    L915,55
                    L925,55

                    L935,55
                    L945,55

                    L955,55
                    L965,55

                    L975,55
                    L985,55

                    L995,55
                    L1000,55
                    "
                />

                <!-- Main ECG heartbeat -->
                <path
                    class="ecg-path"
                    d="
                    M0,55
                    L100,55
                    L130,55
                    L145,55
                    L160,55

                    L175,55
                    L185,55

                    L195,55
                    L205,55

                    L215,55

                    L225,55
                    L235,55

                    L245,55
                    L255,55

                    L265,55
                    L275,55

                    L285,55
                    L295,55

                    L305,55

                    L315,55
                    L325,55

                    L335,55

                    L345,55
                    L355,55

                    L365,55

                    L375,55
                    L385,55

                    L395,55

                    L405,55
                    L415,55

                    L425,55

                    L435,55
                    L445,55

                    L455,55

                    L465,55
                    L475,55

                    L485,55

                    L495,55
                    L505,55

                    L515,55

                    L525,55
                    L535,55

                    L545,55

                    L555,55
                    L565,55

                    L575,55

                    L585,55
                    L595,55

                    L605,55

                    L615,55
                    L625,55

                    L635,55

                    L645,55
                    L655,55

                    L665,55

                    L675,55
                    L685,55

                    L695,55

                    L705,55
                    L715,55

                    L725,55

                    L735,55
                    L745,55

                    L755,55

                    L765,55
                    L775,55

                    L785,55

                    L795,55
                    L805,55

                    L815,55

                    L825,55
                    L835,55

                    L845,55

                    L855,55
                    L865,55

                    L875,55

                    L885,55
                    L895,55

                    L905,55

                    L915,55
                    L925,55

                    L935,55

                    L945,55
                    L955,55

                    L965,55
                    L975,55
                    L985,55
                    L1000,55
                    "
                />

            </svg>

        </div>

    <div class="monitor-label">
        LIVE HEART MONITOR
    </div>

</div>
""", unsafe_allow_html=True)


    # =====================================================
    # FEATURE BADGES
    # =====================================================

    st.markdown("""
<div class="features-row">

    <div class="feature-item">

        <div
            class="feature-circle"
            style="background:#e0f2fe; color:#0284c7;"
        >
            ⚡
        </div>

        <div class="feature-txt">

            <h5>
                Machine Learning
            </h5>

            <p>
                Powered Predictions
            </p>

        </div>

    </div>


    <div class="feature-item">

        <div
            class="feature-circle"
            style="background:#dcfce7; color:#16a34a;"
        >
            🛡️
        </div>

        <div class="feature-txt">

            <h5>
                Accurate Results
            </h5>

            <p>
                Based on Real Data
            </p>

        </div>

    </div>


    <div class="feature-item">

        <div
            class="feature-circle"
            style="background:#f3e8ff; color:#9333ea;"
        >
            ⏱️
        </div>

        <div class="feature-txt">

            <h5>
                Quick & Easy
            </h5>

            <p>
                Just a Few Steps
            </p>

        </div>

    </div>


    <div class="feature-item">

        <div
            class="feature-circle"
            style="background:#ffe4e6; color:#e11d48;"
        >
            ❤️
        </div>

        <div class="feature-txt">

            <h5>
                Better Decisions
            </h5>

            <p>
                For a Healthier Life
            </p>

        </div>

    </div>

</div>
""", unsafe_allow_html=True)


    # =====================================================
    # EXPLORE SECTION
    # =====================================================

    st.markdown("""
<div class="section-title-wrap">

    <h3>
        📈 Explore More About Heart Health
    </h3>

    <p>
        Learn about heart disease, its symptoms,
        risk factors and prevention.
    </p>

</div>
""", unsafe_allow_html=True)


    c1, c2, c3, c4, c5, c6 = st.columns(6)


    # HEART DISEASE

    with c1:

        st.markdown("""
<div
    class="topic-tile"
    style="background:#fff1f2; min-height:160px;"
>

    <div class="topic-tile-icon">
        ❤️
    </div>

    <h4>
        Heart Disease
    </h4>

    <p>
        What is heart disease, its types
        and clinical risks.
    </p>

</div>
""", unsafe_allow_html=True)

        if st.button(
            "Learn more →",
            key="btn_c1",
            use_container_width=True
        ):
            show_heart_disease_modal()


    # SYMPTOMS

    with c2:

        st.markdown("""
<div
    class="topic-tile"
    style="background:#fefce8; min-height:160px;"
>

    <div class="topic-tile-icon">
        ⚠️
    </div>

    <h4>
        Symptoms
    </h4>

    <p>
        Know warning signs,
        angina and silent cues.
    </p>

</div>
""", unsafe_allow_html=True)

        if st.button(
            "Learn more →",
            key="btn_c2",
            use_container_width=True
        ):
            show_symptoms_modal()


    # RISK PREDICTION

    with c3:

        st.markdown("""
<div
    class="topic-tile"
    style="background:#f0f9ff; min-height:160px;"
>

    <div class="topic-tile-icon">
        📊
    </div>

    <h4>
        Risk Prediction
    </h4>

    <p>
        How metrics are used
        by the ML model.
    </p>

</div>
""", unsafe_allow_html=True)

        if st.button(
            "Learn more →",
            key="btn_c3",
            use_container_width=True
        ):
            show_risk_prediction_modal()


    # DIAGNOSIS

    with c4:

        st.markdown("""
<div
    class="topic-tile"
    style="background:#faf5ff; min-height:160px;"
>

    <div class="topic-tile-icon">
        🩺
    </div>

    <h4>
        Diagnosis
    </h4>

    <p>
        How cardiovascular conditions
        are clinically screened.
    </p>

</div>
""", unsafe_allow_html=True)

        if st.button(
            "Learn more →",
            key="btn_c4",
            use_container_width=True
        ):
            show_diagnosis_modal()


    # PREVENTION

    with c5:

        st.markdown("""
<div
    class="topic-tile"
    style="background:#f0fdf4; min-height:160px;"
>

    <div class="topic-tile-icon">
        🛡️
    </div>

    <h4>
        Prevention
    </h4>

    <p>
        Guidelines and proactive
        cardiovascular habits.
    </p>

</div>
""", unsafe_allow_html=True)

        if st.button(
            "Learn more →",
            key="btn_c5",
            use_container_width=True
        ):
            show_prevention_modal()


    # HEART HEALTH

    with c6:

        st.markdown("""
<div
    class="topic-tile"
    style="background:#f0fdfa; min-height:160px;"
>

    <div class="topic-tile-icon">
        🌱
    </div>

    <h4>
        Heart Health
    </h4>

    <p>
        Healthy habits, diet
        and daily activity.
    </p>

</div>
""", unsafe_allow_html=True)

        if st.button(
            "Learn more →",
            key="btn_c6",
            use_container_width=True
        ):
            show_heart_health_modal()


# =========================================================
# TAB 2 — HEART DISEASE
# =========================================================

with active_tab[1]:

    st.markdown("""
<div class="content-box">

    <h2 style="color:#0f172a; margin-top:0;">
        Understanding Cardiovascular Diseases (CVD)
    </h2>

    <p style="color:#475569; font-size:1.05rem;">

        Cardiovascular diseases include conditions
        affecting the heart and blood vessels.
        Atherosclerosis is an important contributor
        to coronary artery disease and many cardiovascular events.

    </p>

</div>
""", unsafe_allow_html=True)


    c_hd1, c_hd2, c_hd3 = st.columns(3)


    with c_hd1:

        st.markdown("""
<div class="content-box" style="height:100%;">

    <h4>
        🫀 Coronary Artery Disease (CAD)
    </h4>

    <p style="font-size:0.88rem; color:#475569;">

        <b>Mechanism:</b>
        Plaque buildup can narrow coronary arteries
        and reduce blood flow to the heart muscle.

        <br><br>

        <b>Clinical Implication:</b>
        Reduced blood flow can contribute to
        angina and acute coronary syndromes.

    </p>

</div>
""", unsafe_allow_html=True)


    with c_hd2:

        st.markdown("""
<div class="content-box" style="height:100%;">

    <h4>
        🧠 Cerebrovascular Disease
    </h4>

    <p style="font-size:0.88rem; color:#475569;">

        <b>Mechanism:</b>
        Blood vessel blockage or rupture in the brain
        can cause stroke.

        <br><br>

        <b>Clinical Implication:</b>
        Stroke symptoms require immediate emergency
        medical assessment.

    </p>

</div>
""", unsafe_allow_html=True)


    with c_hd3:

        st.markdown("""
<div class="content-box" style="height:100%;">

    <h4>
        ⚡ Heart Failure & Arrhythmias
    </h4>

    <p style="font-size:0.88rem; color:#475569;">

        <b>Mechanism:</b>
        Heart failure affects the heart's ability
        to pump effectively, while arrhythmias
        affect heart rhythm.

        <br><br>

        <b>Clinical Implication:</b>
        Both conditions can require medical evaluation
        and appropriate treatment.

    </p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# TAB 3 — SYMPTOMS
# =========================================================

with active_tab[2]:

    st.markdown("""
<div class="content-box">

    <h2 style="color:#0f172a; margin-top:0;">
        Recognizing Warning Signs & Symptom Classifications
    </h2>

    <p style="color:#475569;">
        Recognizing cardiovascular warning signs can help
        people seek appropriate medical attention.
    </p>

</div>
""", unsafe_allow_html=True)


    sym_c1, sym_c2 = st.columns(2)


    with sym_c1:

        st.markdown("""
<div class="content-box">

    <h4 style="color:#e11d48;">
        🚨 Common Physical Indicators
    </h4>

    <ul style="color:#334155; line-height:1.9; font-size:0.92rem;">

        <li>
            <b>Chest Pressure:</b>
            Pressure, squeezing or discomfort
            in the chest.
        </li>

        <li>
            <b>Radiating Discomfort:</b>
            Symptoms may extend toward the
            arm, shoulder, jaw or back.
        </li>

        <li>
            <b>Dyspnea:</b>
            Unexplained shortness of breath.
        </li>

        <li>
            <b>Cold Sweating:</b>
            Sudden sweating may accompany
            cardiovascular symptoms.
        </li>

        <li>
            <b>Dizziness or Nausea:</b>
            These can sometimes accompany
            cardiovascular problems.
        </li>

    </ul>

</div>
""", unsafe_allow_html=True)


    with sym_c2:

        st.markdown("""
<div class="content-box">

    <h4 style="color:#0284c7;">
        🔍 Clinical Chest Pain Types
    </h4>

    <ul style="color:#334155; line-height:1.9; font-size:0.92rem;">

        <li>
            <b>Typical Angina (TA):</b>
            Chest discomfort with features
            characteristic of angina.
        </li>

        <li>
            <b>Atypical Angina (ATA):</b>
            Chest symptoms that have some,
            but not all, typical features.
        </li>

        <li>
            <b>Non-Anginal Pain (NAP):</b>
            Symptoms less consistent with
            cardiac angina.
        </li>

        <li>
            <b>Asymptomatic (ASY):</b>
            No reported chest pain.
        </li>

    </ul>

</div>
""", unsafe_allow_html=True)


# =========================================================
# TAB 4 — RISK PREDICTION
# =========================================================

with active_tab[3]:

    st.markdown("""
<div class="content-box">

    <h2 style="color:#0f172a; margin-top:0;">
        Algorithmic Risk Stratification & Model Metrics
    </h2>

    <p style="color:#475569;">
        Enter patient physiological indicators to calculate
        a prediction using the trained machine learning model.
    </p>

</div>
""", unsafe_allow_html=True)


    col_t1, col_t2 = st.columns(2)


    with col_t1:

        t_age = st.slider(
            "Patient Age (Years)",
            18,
            100,
            48,
            key="tab_age"
        )

        t_sex = st.selectbox(
            "Biological Sex",
            ["M", "F"],
            format_func=lambda x:
                "Male" if x == "M" else "Female",
            key="tab_sex"
        )

        t_cp = st.selectbox(
            "Chest Pain Type",
            ["ATA", "NAP", "ASY", "TA"],
            key="tab_cp"
        )

        t_rbp = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            80,
            220,
            130,
            key="tab_rbp"
        )

        t_chol = st.number_input(
            "Serum Cholesterol (mg/dL)",
            100,
            600,
            220,
            key="tab_chol"
        )


    with col_t2:

        t_fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No",
            key="tab_fbs"
        )

        t_ecg = st.selectbox(
            "Resting ECG Result",
            ["Normal", "ST", "LVH"],
            key="tab_ecg"
        )

        t_mhr = st.slider(
            "Maximum Heart Rate Achieved (bpm)",
            60,
            220,
            145,
            key="tab_mhr"
        )

        t_ang = st.selectbox(
            "Exercise-Induced Angina",
            ["Y", "N"],
            format_func=lambda x:
                "Yes" if x == "Y" else "No",
            key="tab_ang"
        )

        t_slope = st.selectbox(
            "Peak Exercise ST Slope",
            ["Up", "Flat", "Down"],
            key="tab_slope"
        )

        t_oldpeak = st.slider(
            "ST Depression (Oldpeak)",
            0.0,
            6.0,
            1.2,
            key="tab_oldpeak"
        )


    if st.button(
        "Evaluate Stratified Cardiac Risk",
        use_container_width=True,
        key="tab_predict_btn"
    ):

        if (
            model is None
            or scaler is None
            or expected_columns is None
        ):

            st.warning(
                "Model pipeline assets are unavailable. "
                "Make sure RF_KNN_heart.pkl, scaler.pkl "
                "and columns.pkl are in the project folder."
            )

        else:

            tab_pred, risk_p, safe_p = make_prediction(
                t_age,
                t_sex,
                t_cp,
                t_rbp,
                t_chol,
                t_fbs,
                t_ecg,
                t_mhr,
                t_ang,
                t_slope,
                t_oldpeak
            )


            res_c1, res_c2 = st.columns(2)


            with res_c1:

                st.metric(
                    label="Risk Probability",
                    value=f"{risk_p:.1f}%"
                )


            with res_c2:

                st.metric(
                    label="Lower-Risk Class Probability",
                    value=f"{safe_p:.1f}%"
                )


            if tab_pred == 1:

                st.error(
                    "⚠️ **Higher Risk Prediction**"
                )

                st.write(
                    "The submitted parameters were classified "
                    "into the higher-risk class by the trained model. "
                    "This is not a medical diagnosis."
                )

            else:

                st.success(
                    "✅ **Lower Risk Prediction**"
                )

                st.write(
                    "The submitted parameters were classified "
                    "into the lower-risk class by the trained model. "
                    "This does not rule out heart disease."
                )


# =========================================================
# TAB 5 — PREVENTION
# =========================================================

with active_tab[4]:

    st.markdown("""
<div class="content-box">

    <h2 style="color:#0f172a; margin-top:0;">
        Evidence-Based Prevention & Risk Mitigation
    </h2>

    <p style="color:#475569;">
        Healthy lifestyle choices can help reduce cardiovascular
        risk and support long-term heart health.
    </p>

</div>
""", unsafe_allow_html=True)


    prev_c1, prev_c2, prev_c3 = st.columns(3)


    with prev_c1:

        st.markdown("""
<div class="content-box" style="height:100%;">

    <h4>
        🥗 Cardioprotective Nutrition
    </h4>

    <p style="font-size:0.88rem; color:#475569;">

        • Prioritize vegetables, fruits and whole grains.<br>

        • Include healthy fats such as nuts,
        seeds and fish.<br>

        • Limit highly processed foods
        and excessive saturated fat.

    </p>

</div>
""", unsafe_allow_html=True)


    with prev_c2:

        st.markdown("""
<div class="content-box" style="height:100%;">

    <h4>
        🏃 Exercise & Activity
    </h4>

    <p style="font-size:0.88rem; color:#475569;">

        • Aim for regular moderate physical activity.<br>

        • Include muscle-strengthening activity
        when appropriate.<br>

        • Reduce long periods of sitting.

    </p>

</div>
""", unsafe_allow_html=True)


    with prev_c3:

        st.markdown("""
<div class="content-box" style="height:100%;">

    <h4>
        🛡️ Risk Factor Control
    </h4>

    <p style="font-size:0.88rem; color:#475569;">

        • Avoid tobacco exposure.<br>

        • Monitor blood pressure and cholesterol.<br>

        • Maintain a regular sleep schedule
        and healthy daily routine.

    </p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# TAB 6 — ABOUT
# =========================================================

with active_tab[5]:

    st.markdown("""
<div class="content-box">

    <h2 style="color:#0f172a; margin-top:0;">
        About Heart Disease Prediction
    </h2>

    <p style="color:#475569;">

        This web application is a machine-learning-based
        educational screening project developed with
        Streamlit. It uses a trained classification model
        to analyze selected cardiovascular parameters.

    </p>

    <p style="font-size:0.85rem; color:#64748b; margin-top:1rem;
              border-top: 1px solid #e2e8f0; padding-top: 1rem;">

        ⚕️ <b>Clinical Disclaimer:</b>
        The predictions, probability scores and indicators
        generated by this application are algorithmic
        approximations for educational and screening purposes.
        They do not replace professional medical diagnosis
        or physician evaluation.

    </p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="clinical-footer">

    A healthier heart leads to a brighter future ♡

</div>
""", unsafe_allow_html=True)
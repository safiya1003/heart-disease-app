import streamlit as st
import pandas as pd
import joblib
import os
import random
import time

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Heart Disease Prediction | Better Insights • Healthier Tomorrows",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================

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

/* ============================================================
   NAVBAR
   ============================================================ */

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
    background: linear-gradient(
        135deg,
        #f43f5e 0%,
        #e11d48 100%
    );
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

/* ============================================================
   HEARTBEAT ANIMATION
   ============================================================ */

.heartbeat {
    display: inline-block;
    animation: heartbeat 1.05s infinite ease-in-out;
    transform-origin: center;
}

@keyframes heartbeat {

    0% {
        transform: scale(1);
    }

    10% {
        transform: scale(1.15);
    }

    20% {
        transform: scale(1.30);
    }

    30% {
        transform: scale(1);
    }

    42% {
        transform: scale(1);
    }

    50% {
        transform: scale(1.18);
    }

    60% {
        transform: scale(1);
    }

    100% {
        transform: scale(1);
    }
}

/* Glowing heart */

.glow-heart {
    display: inline-block;
    color: #e11d48;
    animation:
        heartbeat 1.05s infinite ease-in-out,
        heartGlow 1.05s infinite ease-in-out;
}

@keyframes heartGlow {

    0%, 100% {
        filter: drop-shadow(0 0 0 rgba(225, 29, 72, 0));
    }

    20% {
        filter: drop-shadow(
            0 0 12px rgba(225, 29, 72, 0.65)
        );
    }

    50% {
        filter: drop-shadow(
            0 0 8px rgba(225, 29, 72, 0.45)
        );
    }
}

/* ============================================================
   BRAND TEXT
   ============================================================ */

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

/* ============================================================
   HERO
   ============================================================ */

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

/* ============================================================
   BUTTONS
   ============================================================ */

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

/* ============================================================
   HERO IMAGE
   ============================================================ */

.hero-image {
    width: 88%;
    max-height: 380px;
    object-fit: cover;
    border-radius: 24px;

    box-shadow:
        0 16px 40px rgba(225, 29, 72, 0.18);
}

/* ============================================================
   LIVE HEART RATE CARD
   ============================================================ */

.heart-rate-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;

    padding: 1.2rem 1.4rem;

    margin-top: 1.5rem;

    box-shadow:
        0 5px 18px rgba(0, 0, 0, 0.04);
}

.heart-rate-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.live-label {
    color: #64748b;
    font-size: 0.72rem;
    font-weight: 800;

    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.bpm-value {
    color: #0f172a;
    font-size: 1.9rem;
    font-weight: 800;
    margin-top: 3px;
}

.normal-status {
    color: #16a34a;
    background: #dcfce7;

    padding: 6px 12px;

    border-radius: 20px;

    font-size: 0.72rem;
    font-weight: 800;
}

/* ============================================================
   ECG ANIMATION
   ============================================================ */

.ecg-container {
    position: relative;

    height: 70px;
    width: 100%;

    overflow: hidden;

    margin-top: 12px;

    background:
        linear-gradient(
            rgba(225, 29, 72, 0.035) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(225, 29, 72, 0.035) 1px,
            transparent 1px
        );

    background-size: 20px 20px;

    border-radius: 12px;
}

.ecg-svg {
    position: absolute;

    top: 0;
    left: 0;

    width: 200%;
    height: 100%;

    animation: ecgSlide 2s linear infinite;
}

@keyframes ecgSlide {

    from {
        transform: translateX(0);
    }

    to {
        transform: translateX(-50%);
    }
}

.ecg-path {
    fill: none;
    stroke: #e11d48;
    stroke-width: 3;

    filter:
        drop-shadow(
            0 0 4px rgba(225, 29, 72, 0.35)
        );
}

/* ============================================================
   FEATURES
   ============================================================ */

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

/* ============================================================
   SECTION TITLES
   ============================================================ */

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

/* ============================================================
   TOPIC TILES
   ============================================================ */

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

/* ============================================================
   CONTENT BOX
   ============================================================ */

.content-box {
    background: #ffffff;

    border: 1px solid #e2e8f0;

    border-radius: 18px;

    padding: 2rem;

    margin-bottom: 1.5rem;

    box-shadow:
        0 3px 12px rgba(0,0,0,0.03);
}

/* ============================================================
   FOOTER
   ============================================================ */

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

/* ============================================================
   MOBILE RESPONSIVE
   ============================================================ */

@media (max-width: 900px) {

    .features-row {
        grid-template-columns: repeat(2, 1fr);
    }

    .hero-title-dark,
    .hero-title-red {
        font-size: 2.6rem;
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
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource
def get_model_assets():

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

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


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def make_prediction(
    age,
    sex,
    cp,
    rbp,
    chol,
    fbs,
    ecg,
    mhr,
    ang,
    slope,
    oldpeak
):

    raw_dict = {
        "Age": age,
        "RestingBP": rbp,
        "Cholesterol": chol,
        "FastingBS": fbs,
        "MaxHR": mhr,
        "Oldpeak": oldpeak,

        "Sex_" + sex: 1,
        "ChestPainType_" + cp: 1,
        "RestingECG_" + ecg: 1,
        "ExerciseAngina_" + ang: 1,
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

        risk_probability = probabilities[1]

    else:

        risk_probability = (
            1.0 if prediction == 1 else 0.0
        )

    return prediction, risk_probability


# ============================================================
# PREDICTION DIALOG
# ============================================================

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

    if st.button(
        "Calculate Risk",
        use_container_width=True,
        key="dialog_predict"
    ):

        if (
            model is not None
            and scaler is not None
            and expected_columns is not None
        ):

            pred, proba = make_prediction(
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

            if pred == 1:

                st.error(
                    f"⚠️ Elevated Risk Detected "
                    f"(Model probability: {proba*100:.1f}%)"
                )

                st.warning(
                    "This is a screening estimate, not a medical diagnosis. "
                    "Please consult a qualified healthcare professional."
                )

            else:

                st.success(
                    f"✅ Lower Predicted Risk "
                    f"(Model probability: {(1-proba)*100:.1f}% lower-risk class)"
                )

                st.info(
                    "The submitted markers fall into the model's lower-risk class."
                )

        else:

            st.warning(
                "Model pipeline assets are unavailable. "
                "Please check the model files."
            )


# ============================================================
# INFORMATION MODALS
# ============================================================

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
| :--- | :--- |
| **CAD** | Coronary Arteries |
| **Arrhythmia** | Electrical Conduction |
| **Cardiomyopathy** | Ventricular Muscle |
        """)

    with c_txt:

        st.markdown(
            "#### Clinical Overview & Pathophysiology"
        )

        st.markdown("""
- **Atherosclerosis:** Plaque buildup can narrow arteries.
- **Coronary narrowing:** Reduced blood flow can affect the heart muscle.
- **Ischemia:** Reduced oxygen delivery can injure cardiac tissue.
- **Risk factors:** Blood pressure, cholesterol, diabetes, smoking, age and family history can contribute to cardiovascular risk.
- **Hypertension:** Long-term high blood pressure can increase cardiac workload.
        """)


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
            caption="Clinical Symptom Triage",
            use_container_width=True
        )

    with c_txt:

        st.markdown(
            "#### Common Warning Signs"
        )

        st.markdown("""
- **Chest pressure or discomfort**
- **Pain spreading to the arm, shoulder, jaw or back**
- **Shortness of breath**
- **Cold sweating**
- **Dizziness or fainting**
- **Nausea or unusual fatigue**

If severe or sudden symptoms occur, seek emergency medical care rather than relying on this application.
        """)


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

    with c_txt:

        st.markdown(
            "#### Machine Learning Analytics Architecture"
        )

        st.markdown("""
- **Multivariable mapping:** Multiple patient characteristics are considered together.
- **Feature scaling:** Numerical features are standardized before prediction.
- **Heart-rate information:** Maximum heart rate is one model input.
- **ST-wave features:** ST slope and Oldpeak contribute to the model input.
- **Classification:** The trained model produces a predicted class and, where supported, a probability estimate.
- **Important:** Model probabilities should not be interpreted as clinical certainty.
        """)


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

    with c_txt:

        st.markdown(
            "#### Primary Diagnostic Modalities"
        )

        st.markdown("""
- **12-Lead ECG:** Evaluates electrical activity and rhythm.
- **Cardiac biomarkers:** Troponin testing can help identify heart muscle injury.
- **Echocardiography:** Evaluates cardiac structure and function.
- **Stress testing:** Evaluates cardiac response to exercise or pharmacologic stress.
- **Coronary CT:** Can provide detailed coronary artery imaging.
- **Angiography:** Can visualize coronary artery narrowing.
- **Holter monitoring:** Records cardiac rhythm over an extended period.
        """)


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

    with c_txt:

        st.markdown(
            "#### Evidence-Based Interventions"
        )

        st.markdown("""
- **Heart-healthy diet:** Emphasize vegetables, fruits, whole grains, legumes and healthy fats.
- **Physical activity:** Aim for regular moderate activity according to individual ability.
- **Smoking cessation:** Avoid tobacco products.
- **Blood pressure:** Monitor and manage blood pressure.
- **Cholesterol:** Regularly monitor lipid levels.
- **Sleep:** Maintain a consistent healthy sleep schedule.
- **Stress management:** Use sustainable methods such as exercise, relaxation and social support.
        """)


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

    with c_txt:

        st.markdown(
            "#### Practical Wellness Principles"
        )

        st.markdown("""
- **Regular movement:** Break up long periods of sitting.
- **Balanced nutrition:** Include fiber-rich foods and unsaturated fats.
- **Strength training:** Add resistance exercise when appropriate.
- **Hydration:** Drink according to thirst, activity and individual needs.
- **Avoid tobacco:** Tobacco significantly increases cardiovascular risk.
- **Regular checkups:** Monitor blood pressure, cholesterol and other relevant health markers.
        """)


# ============================================================
# NAVIGATION HEADER
# ============================================================

st.markdown("""
<div class="top-navbar">

    <div class="brand-group">

        <div class="brand-icon">
            <span class="heartbeat">❤️</span>
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


# ============================================================
# TABS
# ============================================================

active_tab = st.tabs([
    "🏠 Home",
    "♡ Heart Disease",
    "⚠️ Symptoms",
    "📊 Risk Prediction",
    "🛡️ Prevention",
    "ⓘ About"
])


# ============================================================
# TAB 1 - HOME
# ============================================================

with active_tab[0]:

    hero_col1, hero_col2 = st.columns(
        [1.1, 1],
        gap="large"
    )

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

    with hero_col2:

        st.markdown("""
<div style="text-align: center; padding-top: 10px;">

    <img
        src="https://images.unsplash.com/photo-1530026405186-ed1f139313f8?auto=format&fit=crop&w=750&q=80"
        alt="Cardiac Visual"
        class="hero-image"
    >

</div>
        """, unsafe_allow_html=True)


    # ========================================================
    # LIVE HEART RATE
    # ========================================================

    bpm = random.randint(68, 82)

    st.markdown(f"""
<div class="heart-rate-card">

    <div class="heart-rate-header">

        <div>

            <div class="live-label">
                LIVE HEART RATE • SIMULATION
            </div>

            <div class="bpm-value">
                <span class="glow-heart">❤️</span>
                {bpm} BPM
            </div>

        </div>

        <div class="normal-status">
            ● NORMAL RANGE
        </div>

    </div>


    <div class="ecg-container">

        <svg
            class="ecg-svg"
            viewBox="0 0 1200 70"
            preserveAspectRatio="none"
        >

            <path
                class="ecg-path"
                d="
                M0 35
                L70 35
                L80 35
                L90 35
                L100 35
                L110 35
                L120 35
                L130 35

                L140 35
                L150 35
                L160 35

                L170 35
                L180 35
                L190 35

                L200 35
                L210 35

                L220 35
                L230 35

                L240 35
                L250 35

                L260 35
                L270 35

                L280 35
                L290 35

                L300 35
                L310 35

                L320 35
                L330 35

                L340 35
                L350 35

                L360 35
                L370 35

                L380 35
                L390 35

                L400 35
                L410 35

                L420 35
                L430 35

                L440 35
                L450 35

                L460 35
                L470 35

                L480 35
                L490 35

                L500 35
                L510 35

                L520 35
                L530 35

                L540 35
                L550 35

                L560 35
                L570 35

                L580 35
                L590 35

                L600 35
                L610 35

                L620 35
                L630 35

                L640 35
                L650 35

                L660 35
                L670 35

                L680 35
                L690 35

                L700 35
                L710 35

                L720 35
                L730 35

                L740 35
                L750 35

                L760 35
                L770 35

                L780 35
                L790 35

                L800 35
                L810 35

                L820 35
                L830 35

                L840 35
                L850 35

                L860 35
                L870 35

                L880 35
                L890 35

                L900 35
                L910 35

                L920 35
                L930 35

                L940 35
                L950 35

                L960 35
                L970 35

                L980 35
                L990 35

                L1000 35
                L1010 35

                L1020 35
                L1030 35

                L1040 35
                L1050 35

                L1060 35
                L1070 35

                L1080 35
                L1090 35

                L1100 35
                L1110 35

                L1120 35
                L1130 35

                L1140 35
                L1150 35

                L1160 35
                L1170 35

                L1180 35
                L1200 35
                "
            />

            <!-- ECG spikes -->

            <path
                class="ecg-path"
                d="
                M0 35
                L130 35
                L145 35
                L155 15
                L165 55
                L175 35
                L250 35

                M600 35
                L730 35
                L745 35
                L755 15
                L765 55
                L775 35
                L850 35

                M1200 35
                L1070 35
                L1055 15
                L1045 55
                L1035 35
                "
            />

        </svg>

    </div>

</div>
""", unsafe_allow_html=True)


    # ========================================================
    # FEATURE CARDS
    # ========================================================

    st.markdown("""
<div class="features-row">

    <div class="feature-item">

        <div
            class="feature-circle"
            style="background:#e0f2fe;color:#0284c7;"
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
            style="background:#dcfce7;color:#16a34a;"
        >
            🛡️
        </div>

        <div class="feature-txt">

            <h5>
                Data Driven
            </h5>

            <p>
                Based on Model Data
            </p>

        </div>

    </div>


    <div class="feature-item">

        <div
            class="feature-circle"
            style="background:#f3e8ff;color:#9333ea;"
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
            style="background:#ffe4e6;color:#e11d48;"
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


    # ========================================================
    # EXPLORE SECTION
    # ========================================================

    st.markdown("""
<div class="section-title-wrap">

    <h3>
        📈 Explore More About Heart Health
    </h3>

    <p>
        Learn about heart disease, symptoms, risk factors and prevention.
    </p>

</div>
    """, unsafe_allow_html=True)


    c1, c2, c3, c4, c5, c6 = st.columns(6)


    with c1:

        st.markdown("""
<div
    class="topic-tile"
    style="background:#fff1f2;min-height:160px;"
>

    <div class="topic-tile-icon">
        ❤️
    </div>

    <h4>
        Heart Disease
    </h4>

    <p>
        What is heart disease, its types and clinical risks.
    </p>

</div>
        """, unsafe_allow_html=True)

        if st.button(
            "Learn more →",
            key="btn_c1",
            use_container_width=True
        ):
            show_heart_disease_modal()


    with c2:

        st.markdown("""
<div
    class="topic-tile"
    style="background:#fefce8;min-height:160px;"
>

    <div class="topic-tile-icon">
        ⚠️
    </div>

    <h4>
        Symptoms
    </h4>

    <p>
        Know the warning signs and clinical symptoms.
    </p>

</div>
        """, unsafe_allow_html=True)

        if st.button(
            "Learn more →",
            key="btn_c2",
            use_container_width=True
        ):
            show_symptoms_modal()


    with c3:

        st.markdown("""
<div
    class="topic-tile"
    style="background:#f0f9ff;min-height:160px;"
>

    <div class="topic-tile-icon">
        📊
    </div>

    <h4>
        Risk Prediction
    </h4>

    <p>
        Understand how model inputs influence predictions.
    </p>

</div>
        """, unsafe_allow_html=True)

        if st.button(
            "Learn more →",
            key="btn_c3",
            use_container_width=True
        ):
            show_risk_prediction_modal()


    with c4:

        st.markdown("""
<div
    class="topic-tile"
    style="background:#faf5ff;min-height:160px;"
>

    <div class="topic-tile-icon">
        🩺
    </div>

    <h4>
        Diagnosis
    </h4>

    <p>
        Learn about common cardiovascular screening methods.
    </p>

</div>
        """, unsafe_allow_html=True)

        if st.button(
            "Learn more →",
            key="btn_c4",
            use_container_width=True
        ):
            show_diagnosis_modal()


    with c5:

        st.markdown("""
<div
    class="topic-tile"
    style="background:#f0fdf4;min-height:160px;"
>

    <div class="topic-tile-icon">
        🛡️
    </div>

    <h4>
        Prevention
    </h4>

    <p>
        Lifestyle strategies for cardiovascular health.
    </p>

</div>
        """, unsafe_allow_html=True)

        if st.button(
            "Learn more →",
            key="btn_c5",
            use_container_width=True
        ):
            show_prevention_modal()


    with c6:

        st.markdown("""
<div
    class="topic-tile"
    style="background:#f0fdfa;min-height:160px;"
>

    <div class="topic-tile-icon">
        🌱
    </div>

    <h4>
        Heart Health
    </h4>

    <p>
        Everyday habits supporting cardiovascular wellness.
    </p>

</div>
        """, unsafe_allow_html=True)

        if st.button(
            "Learn more →",
            key="btn_c6",
            use_container_width=True
        ):
            show_heart_health_modal()


# ============================================================
# TAB 2 - HEART DISEASE
# ============================================================

with active_tab[1]:

    st.markdown("""
<div class="content-box">

    <h2 style="color:#0f172a;margin-top:0;">
        Understanding Cardiovascular Diseases (CVD)
    </h2>

    <p style="color:#475569;font-size:1.05rem;">
        Cardiovascular diseases include conditions affecting the heart
        and blood vessels. Coronary artery disease, heart failure and
        rhythm disorders are among the major cardiovascular conditions.
    </p>

</div>
    """, unsafe_allow_html=True)


    c_hd1, c_hd2, c_hd3 = st.columns(3)


    with c_hd1:

        st.markdown("""
<div class="content-box" style="height:100%;">

    <h4>
        🫀 Coronary Artery Disease
    </h4>

    <p style="font-size:0.88rem;color:#475569;">

        <b>Mechanism:</b>
        Plaque buildup can narrow coronary arteries and reduce blood flow.

        <br><br>

        <b>Clinical implication:</b>
        Reduced coronary blood flow may cause angina and, in severe cases,
        acute coronary events.

    </p>

</div>
        """, unsafe_allow_html=True)


    with c_hd2:

        st.markdown("""
<div class="content-box" style="height:100%;">

    <h4>
        🧠 Cerebrovascular Disease
    </h4>

    <p style="font-size:0.88rem;color:#475569;">

        <b>Mechanism:</b>
        Blood-vessel blockage or rupture in the brain can result in stroke.

        <br><br>

        <b>Clinical implication:</b>
        Sudden neurological symptoms require immediate emergency evaluation.

    </p>

</div>
        """, unsafe_allow_html=True)


    with c_hd3:

        st.markdown("""
<div class="content-box" style="height:100%;">

    <h4>
        ⚡ Heart Failure & Arrhythmias
    </h4>

    <p style="font-size:0.88rem;color:#475569;">

        <b>Mechanism:</b>
        Structural or electrical abnormalities can impair pumping or rhythm.

        <br><br>

        <b>Clinical implication:</b>
        Symptoms may include breathlessness, fatigue, swelling or palpitations.

    </p>

</div>
        """, unsafe_allow_html=True)


# ============================================================
# TAB 3 - SYMPTOMS
# ============================================================

with active_tab[2]:

    st.markdown("""
<div class="content-box">

    <h2 style="color:#0f172a;margin-top:0;">
        Recognizing Warning Signs & Symptoms
    </h2>

    <p style="color:#475569;">
        Recognizing concerning symptoms early can help people seek
        appropriate medical care promptly.
    </p>

</div>
    """, unsafe_allow_html=True)


    sym_c1, sym_c2 = st.columns(2)


    with sym_c1:

        st.markdown("""
<div class="content-box">

    <h4 style="color:#e11d48;">
        🚨 Common Warning Signs
    </h4>

    <ul style="color:#334155;line-height:1.9;font-size:0.92rem;">

        <li>
            <b>Chest discomfort:</b>
            Pressure, squeezing or heaviness.
        </li>

        <li>
            <b>Pain radiation:</b>
            Discomfort may spread to the arm, shoulder, jaw or back.
        </li>

        <li>
            <b>Dyspnea:</b>
            Unexplained shortness of breath.
        </li>

        <li>
            <b>Cold sweating:</b>
            Sudden unexplained sweating.
        </li>

        <li>
            <b>Dizziness:</b>
            Lightheadedness or fainting.
        </li>

    </ul>

</div>
        """, unsafe_allow_html=True)


    with sym_c2:

        st.markdown("""
<div class="content-box">

    <h4 style="color:#0284c7;">
        🔍 Chest Pain Categories
    </h4>

    <ul style="color:#334155;line-height:1.9;font-size:0.92rem;">

        <li>
            <b>Typical Angina:</b>
            Classically associated with exertion and relieved by rest.
        </li>

        <li>
            <b>Atypical Angina:</b>
            Symptoms may have fewer classic features.
        </li>

        <li>
            <b>Non-Anginal Pain:</b>
            May have musculoskeletal, gastrointestinal or pulmonary causes.
        </li>

        <li>
            <b>Asymptomatic:</b>
            Some people may have little or no typical chest discomfort.
        </li>

    </ul>

</div>
        """, unsafe_allow_html=True)


# ============================================================
# TAB 4 - RISK PREDICTION
# ============================================================

with active_tab[3]:

    st.markdown("""
<div class="content-box">

    <h2 style="color:#0f172a;margin-top:0;">
        Algorithmic Risk Stratification & Model Metrics
    </h2>

    <p style="color:#475569;">
        Enter patient physiological indicators to calculate the
        prediction generated by the trained machine-learning pipeline.
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
            model is not None
            and scaler is not None
            and expected_columns is not None
        ):

            tab_pred, risk_probability = make_prediction(
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

            risk_p = risk_probability * 100
            safe_p = 100 - risk_p


            res_c1, res_c2 = st.columns(2)

            with res_c1:

                st.metric(
                    label="Risk Probability",
                    value=f"{risk_p:.1f}%"
                )

            with res_c2:

                st.metric(
                    label="Lower-Risk Probability",
                    value=f"{safe_p:.1f}%"
                )


            if tab_pred == 1:

                st.error(
                    "⚠️ High Risk Classification"
                )

                st.write(
                    "The submitted features were classified into "
                    "the model's higher-risk class. This result is "
                    "for screening/educational purposes and should "
                    "not replace professional medical evaluation."
                )

            else:

                st.success(
                    "✅ Lower Risk Classification"
                )

                st.write(
                    "The submitted features were classified into "
                    "the model's lower-risk class. A model result "
                    "does not guarantee absence of cardiovascular disease."
                )

        else:

            st.warning(
                "Model pipeline assets are currently unavailable. "
                "Ensure RF_KNN_heart.pkl, scaler.pkl and columns.pkl "
                "are in the project directory."
            )


# ============================================================
# TAB 5 - PREVENTION
# ============================================================

with active_tab[4]:

    st.markdown("""
<div class="content-box">

    <h2 style="color:#0f172a;margin-top:0;">
        Evidence-Based Prevention & Risk Mitigation
    </h2>

    <p style="color:#475569;">
        Cardiovascular risk can often be reduced through healthy
        lifestyle choices and appropriate management of medical risk factors.
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

    <p style="font-size:0.88rem;color:#475569;">

        • Prioritize vegetables, fruits, whole grains and legumes.<br><br>

        • Choose unsaturated fats such as those found in nuts,
        seeds, fish and plant oils.<br><br>

        • Limit highly processed foods and trans fats.

    </p>

</div>
        """, unsafe_allow_html=True)


    with prev_c2:

        st.markdown("""
<div class="content-box" style="height:100%;">

    <h4>
        🏃 Exercise
    </h4>

    <p style="font-size:0.88rem;color:#475569;">

        • Aim for regular moderate aerobic activity.<br><br>

        • Include resistance training when appropriate.<br><br>

        • Reduce prolonged periods of sitting.

    </p>

</div>
        """, unsafe_allow_html=True)


    with prev_c3:

        st.markdown("""
<div class="content-box" style="height:100%;">

    <h4>
        🛡️ Risk Factor Control
    </h4>

    <p style="font-size:0.88rem;color:#475569;">

        • Avoid tobacco products.<br><br>

        • Monitor blood pressure and cholesterol.<br><br>

        • Maintain a consistent healthy sleep routine.

    </p>

</div>
        """, unsafe_allow_html=True)


# ============================================================
# TAB 6 - ABOUT
# ============================================================

with active_tab[5]:

    st.markdown("""
<div class="content-box">

    <h2 style="color:#0f172a;margin-top:0;">
        About CardioCare Intelligence
    </h2>

    <p style="color:#475569;">

        This web application is a clinical decision-support and
        educational screening system developed with Streamlit and
        Scikit-Learn. It uses a trained machine-learning pipeline
        to classify cardiovascular risk from selected patient features.

    </p>

    <p style="
        font-size:0.85rem;
        color:#64748b;
        margin-top:1rem;
        border-top:1px solid #e2e8f0;
        padding-top:1rem;
    ">

        ⚕️ <b>Clinical Disclaimer:</b>

        The predictions and probability scores generated by this
        application are algorithmic estimates intended for screening
        and educational demonstration. They are not a diagnosis and
        do not replace evaluation by a qualified healthcare professional.

    </p>

</div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="clinical-footer">

    <span class="heartbeat">❤️</span>

    A healthier heart leads to a brighter future

    <span class="heartbeat">♡</span>

</div>
""", unsafe_allow_html=True)

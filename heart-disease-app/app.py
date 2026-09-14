import streamlit as st
import pandas as pd
import joblib
import os
import streamlit.components.v1 as components

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
   BUTTON
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
   FEATURE ROW
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

/* ============================================================
   SECTION TITLE
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
   TOPIC CARDS
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
    box-shadow: 0 3px 12px rgba(0,0,0,0.03);
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
   ANIMATED HEART
   ============================================================ */

.heart-animation-wrapper {
    width: 100%;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

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

    # --------------------------------------------------------
    # MODEL CHECK
    # --------------------------------------------------------

    if (
        model is None
        or scaler is None
        or expected_columns is None
    ):

        st.warning(
            "Model files are not available. "
            "Please make sure RF_KNN_heart.pkl, scaler.pkl "
            "and columns.pkl are present."
        )

        return

    # --------------------------------------------------------
    # CREATE INPUT
    # --------------------------------------------------------

    raw_dict = {
        "Age": age_in,
        "RestingBP": rbp_in,
        "Cholesterol": chol_in,
        "FastingBS": fbs_in,
        "MaxHR": mhr_in,
        "Oldpeak": oldpeak_in,

        "Sex_" + sex_in: 1,
        "ChestPainType_" + cp_in: 1,
        "RestingECG_" + ecg_in: 1,
        "ExerciseAngina_" + ang_in: 1,
        "ST_Slope_" + slope_in: 1
    }

    df = pd.DataFrame([raw_dict])

    # --------------------------------------------------------
    # MATCH MODEL COLUMNS
    # --------------------------------------------------------

    for col in expected_columns:

        if col not in df.columns:
            df[col] = 0

    df = df[expected_columns]

    # --------------------------------------------------------
    # SCALE + PREDICT
    # --------------------------------------------------------

    try:

        scaled_df = scaler.transform(df)

        pred = model.predict(scaled_df)[0]

        if hasattr(model, "predict_proba"):

            proba = model.predict_proba(
                scaled_df
            )[0][1]

        else:

            proba = (
                1.0
                if pred == 1
                else 0.0
            )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if pred == 1:

            st.error(
                f"⚠️ **Elevated Risk Detected** "
                f"(Confidence: {proba * 100:.1f}%)"
            )

            st.warning(
                "The model estimates elevated cardiovascular "
                "risk from the entered parameters. "
                "This prediction is for educational/project "
                "purposes and is not a medical diagnosis."
            )

        else:

            st.success(
                f"✅ **Lower Estimated Risk** "
                f"(Confidence: {(1 - proba) * 100:.1f}%)"
            )

            st.info(
                "The model estimates a lower cardiovascular "
                "risk from the entered parameters. "
                "This prediction is for educational/project "
                "purposes and is not a medical diagnosis."
            )

    except Exception as e:

        st.error(
            "Prediction could not be completed."
        )

        st.exception(e)


# ============================================================
# INFORMATION MODALS
# ============================================================

@st.dialog("❤️ Heart Disease")
def show_heart_disease_modal():

    st.markdown("""
    ### What is Heart Disease?

    Heart disease refers to conditions that affect the heart
    and cardiovascular system.

    Common types include:

    - Coronary artery disease
    - Heart rhythm problems
    - Heart valve disease
    - Heart failure

    Early awareness of risk factors can support better
    health decisions.
    """)

    st.markdown("""
    **Common risk factors**

    - High blood pressure
    - High cholesterol
    - Diabetes
    - Smoking
    - Physical inactivity
    - Family history
    """)


@st.dialog("⚠️ Symptoms")
def show_symptoms_modal():

    st.markdown("""
    ### Common Warning Signs

    Some commonly recognized symptoms may include:

    - Chest discomfort
    - Shortness of breath
    - Unusual tiredness
    - Dizziness
    - Sweating
    - Discomfort in the upper body

    Symptoms can vary from person to person.
    """)

    st.info(
        "If someone has severe or sudden symptoms, "
        "seek urgent medical help."
    )


@st.dialog("📊 Risk Prediction")
def show_risk_prediction_modal():

    st.markdown("""
    ### How Risk Prediction Works

    This project uses a machine-learning model to process
    selected cardiovascular health parameters.

    The model considers information such as:

    - Age
    - Sex
    - Resting blood pressure
    - Cholesterol
    - Fasting blood sugar
    - Maximum heart rate
    - Chest pain type
    - Resting ECG
    - Exercise-induced angina
    - ST slope
    - Oldpeak
    """)

    st.info(
        "The output is a machine-learning prediction and "
        "should not be treated as a medical diagnosis."
    )


@st.dialog("🩺 Diagnosis")
def show_diagnosis_modal():

    st.markdown("""
    ### Clinical Screening

    Healthcare professionals may use different methods
    when evaluating cardiovascular health.

    Examples include:

    - Medical history
    - Physical examination
    - Blood tests
    - ECG
    - Stress testing
    - Imaging tests

    A qualified healthcare professional decides which
    tests are appropriate.
    """)


@st.dialog("🛡️ Prevention")
def show_prevention_modal():

    st.markdown("""
    ### Heart Health & Prevention

    Helpful habits can include:

    - Regular physical activity
    - Balanced nutrition
    - Avoiding tobacco
    - Managing blood pressure
    - Managing cholesterol
    - Getting adequate sleep
    - Regular health checkups
    """)


@st.dialog("🌱 Heart Health")
def show_heart_health_modal():

    st.markdown("""
    ### Healthy Heart Habits

    Small consistent habits can support cardiovascular
    health.

    Focus on:

    **Healthy Food**

    Include a balanced variety of nutritious foods.

    **Physical Activity**

    Stay physically active according to your abilities.

    **Stress Management**

    Make time for rest and healthy ways to manage stress.

    **Regular Checkups**

    Discuss health concerns with a qualified healthcare
    professional.
    """)


# ============================================================
# NAVBAR
# ============================================================

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
# HOME TAB
# ============================================================

with active_tab[0]:

    hero_col1, hero_col2 = st.columns(
        [1.1, 1],
        gap="large"
    )

    # --------------------------------------------------------
    # LEFT SIDE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RIGHT SIDE
    # ONLY THIS PART IS ANIMATED
    # --------------------------------------------------------

    with hero_col2:

        components.html("""
        <!DOCTYPE html>

        <html>

        <head>

        <style>

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: Arial, sans-serif;
        }

        .monitor {

            position: relative;

            width: 100%;
            height: 380px;

            border-radius: 28px;

            overflow: hidden;

            background:
                radial-gradient(
                    circle at center,
                    rgba(225,29,72,0.12),
                    transparent 55%
                ),
                linear-gradient(
                    145deg,
                    #fff1f2,
                    #ffffff
                );

            border: 1px solid #fecdd3;

            box-shadow:
                0 18px 45px
                rgba(225,29,72,0.13);

        }


        /* PULSING HEART */

        .heart {

            position: absolute;

            top: 48px;

            left: 50%;

            transform:
                translateX(-50%);

            font-size: 90px;

            z-index: 5;

            animation:
                heartbeat 1s infinite;

            filter:
                drop-shadow(
                    0 10px 20px
                    rgba(225,29,72,0.30)
                );

        }


        @keyframes heartbeat {

            0% {
                transform:
                    translateX(-50%)
                    scale(1);
            }

            15% {
                transform:
                    translateX(-50%)
                    scale(1.18);
            }

            30% {
                transform:
                    translateX(-50%)
                    scale(1);
            }

            45% {
                transform:
                    translateX(-50%)
                    scale(1.12);
            }

            60% {
                transform:
                    translateX(-50%)
                    scale(1);
            }

            100% {
                transform:
                    translateX(-50%)
                    scale(1);
            }

        }


        /* PULSE RINGS */

        .ring {

            position: absolute;

            width: 150px;
            height: 150px;

            border-radius: 50%;

            border:
                2px solid
                rgba(225,29,72,0.20);

            left: 50%;

            top: 35px;

            transform:
                translateX(-50%);

            animation:
                pulse 2.5s infinite;

        }


        .ring2 {
            animation-delay: 0.8s;
        }

        .ring3 {
            animation-delay: 1.6s;
        }


        @keyframes pulse {

            0% {

                transform:
                    translateX(-50%)
                    scale(0.7);

                opacity: 0.8;

            }

            100% {

                transform:
                    translateX(-50%)
                    scale(1.8);

                opacity: 0;

            }

        }


        /* BPM */

        .bpm {

            position: absolute;

            right: 22px;
            top: 20px;

            background: white;

            padding:
                10px 16px;

            border-radius: 14px;

            text-align: center;

            box-shadow:
                0 5px 20px
                rgba(0,0,0,0.08);

            z-index: 10;

        }


        .bpm-number {

            color: #e11d48;

            font-size: 25px;

            font-weight: 800;

        }


        .bpm-label {

            color: #64748b;

            font-size: 10px;

            font-weight: 700;

            letter-spacing: 1px;

        }


        /* ECG BOX */

        .ecg {

            position: absolute;

            left: 6%;
            right: 6%;

            bottom: 58px;

            height: 100px;

            background:
                rgba(255,255,255,0.92);

            border:
                1px solid #fecdd3;

            border-radius: 14px;

            overflow: hidden;

        }


        .ecg-track {

            display: flex;

            width: 200%;

            height: 100%;

            animation:
                moveECG 3s linear infinite;

        }


        .ecg svg {

            width: 50%;

            height: 100%;

            flex-shrink: 0;

        }


        .ecg-line {

            fill: none;

            stroke: #e11d48;

            stroke-width: 4;

            stroke-linecap: round;

            stroke-linejoin: round;

            filter:
                drop-shadow(
                    0 0 5px
                    rgba(225,29,72,0.45)
                );

        }


        @keyframes moveECG {

            from {
                transform:
                    translateX(0);
            }

            to {
                transform:
                    translateX(-50%);
            }

        }


        /* LIVE STATUS */

        .status {

            position: absolute;

            bottom: 16px;

            left: 50%;

            transform:
                translateX(-50%);

            color: #be123c;

            background: white;

            border:
                1px solid #fecdd3;

            border-radius: 20px;

            padding:
                6px 14px;

            font-size: 10px;

            font-weight: 800;

            letter-spacing: 1px;

            white-space: nowrap;

        }

        </style>

        </head>


        <body>

        <div class="monitor">


            <div class="ring"></div>

            <div class="ring ring2"></div>

            <div class="ring ring3"></div>


            <div class="heart">
                ❤️
            </div>


            <div class="bpm">

                <div
                    class="bpm-number"
                    id="bpm">
                    72
                </div>

                <div class="bpm-label">
                    BPM
                </div>

            </div>


            <div class="ecg">

                <div class="ecg-track">


                    <svg viewBox="0 0 800 160">

                        <polyline
                            class="ecg-line"
                            points="
                            0,80
                            80,80
                            120,80
                            140,80
                            155,80
                            165,35
                            175,125
                            185,80
                            220,80
                            260,80
                            300,80
                            320,80
                            335,80
                            345,35
                            355,125
                            365,80
                            400,80
                            440,80
                            480,80
                            500,80
                            515,80
                            525,35
                            535,125
                            545,80
                            580,80
                            620,80
                            660,80
                            680,80
                            695,80
                            705,35
                            715,125
                            725,80
                            760,80
                            800,80
                            "
                        />

                    </svg>


                    <svg viewBox="0 0 800 160">

                        <polyline
                            class="ecg-line"
                            points="
                            0,80
                            80,80
                            120,80
                            140,80
                            155,80
                            165,35
                            175,125
                            185,80
                            220,80
                            260,80
                            300,80
                            320,80
                            335,80
                            345,35
                            355,125
                            365,80
                            400,80
                            440,80
                            480,80
                            500,80
                            515,80
                            525,35
                            535,125
                            545,80
                            580,80
                            620,80
                            660,80
                            680,80
                            695,80
                            705,35
                            715,125
                            725,80
                            760,80
                            800,80
                            "
                        />

                    </svg>


                </div>

            </div>


            <div class="status">
                ● LIVE HEART MONITOR
            </div>


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


            document.getElementById(
                "bpm"
            ).innerText = bpm;

        }, 700);

        </script>


        </body>

        </html>

        """, height=390, scrolling=False)


    # ========================================================
    # FEATURE BOXES
    # ========================================================

    st.markdown("""
    <div class="features-row">

        <div class="feature-item">

            <div
                class="feature-circle"
                style="
                    background:#fff1f2;
                    color:#e11d48;
                ">
                🤖
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
                style="
                    background:#f0fdf4;
                    color:#16a34a;
                ">
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
                style="
                    background:#eff6ff;
                    color:#2563eb;
                ">
                ⏱️
            </div>

            <div class="feature-txt">

                <h5>
                    Quick &amp; Easy
                </h5>

                <p>
                    Just a Few Steps
                </p>

            </div>

        </div>


        <div class="feature-item">

            <div
                class="feature-circle"
                style="
                    background:#fff7ed;
                    color:#ea580c;
                ">
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
    # EXPLORE MORE
    # ========================================================

    st.markdown("""
    <div class="section-title-wrap">

        <h3>
            📈 Explore More About Heart Health
        </h3>

        <p>
            Learn about heart disease, its symptoms,
            risk factors and how you can prevent it.
        </p>

    </div>
    """, unsafe_allow_html=True)


    # ========================================================
    # TOPIC CARDS
    # ========================================================

    topic_col1, topic_col2, topic_col3 = st.columns(3)


    with topic_col1:

        st.markdown("""
        <div
            class="topic-tile"
            style="
                background:#fff1f2;
            ">

            <div class="topic-tile-icon">
                ❤️
            </div>

            <h4>
                Heart Disease
            </h4>

            <p>
                What is heart disease,
                its types and clinical risks.
            </p>

        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Learn More →",
            key="heart_disease_btn"
        ):
            show_heart_disease_modal()


        st.markdown("""
        <div
            class="topic-tile"
            style="
                background:#eff6ff;
            ">

            <div class="topic-tile-icon">
                📊
            </div>

            <h4>
                Risk Prediction
            </h4>

            <p>
                How metrics calculate
                cardiovascular likelihood.
            </p>

        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Explore →",
            key="risk_prediction_btn"
        ):
            show_risk_prediction_modal()


    with topic_col2:

        st.markdown("""
        <div
            class="topic-tile"
            style="
                background:#fff7ed;
            ">

            <div class="topic-tile-icon">
                ⚠️
            </div>

            <h4>
                Symptoms
            </h4>

            <p>
                Know the warning signs,
                angina and silent cues.
            </p>

        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Learn More →",
            key="symptoms_btn"
        ):
            show_symptoms_modal()


        st.markdown("""
        <div
            class="topic-tile"
            style="
                background:#f0fdf4;
            ">

            <div class="topic-tile-icon">
                🛡️
            </div>

            <h4>
                Prevention
            </h4>

            <p>
                Guidelines and proactive steps
                for long-term health.
            </p>

        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Learn More →",
            key="prevention_btn"
        ):
            show_prevention_modal()


    with topic_col3:

        st.markdown("""
        <div
            class="topic-tile"
            style="
                background:#f5f3ff;
            ">

            <div class="topic-tile-icon">
                🩺
            </div>

            <h4>
                Diagnosis
            </h4>

            <p>
                How conditions are identified
                and clinically screened.
            </p>

        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Learn More →",
            key="diagnosis_btn"
        ):
            show_diagnosis_modal()


        st.markdown("""
        <div
            class="topic-tile"
            style="
                background:#ecfdf5;
            ">

            <div class="topic-tile-icon">
                🌱
            </div>

            <h4>
                Heart Health
            </h4>

            <p>
                Healthy habits for
                long-term cardiovascular wellness.
            </p>

        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Learn More →",
            key="health_btn"
        ):
            show_heart_health_modal()


# ============================================================
# HEART DISEASE TAB
# ============================================================

with active_tab[1]:

    st.markdown("""
    <div class="section-title-wrap">

        <h3>
            ❤️ Understanding Heart Disease
        </h3>

        <p>
            Learn about cardiovascular disease
            and common risk factors.
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-box">

        <h3>
            What is Heart Disease?
        </h3>

        <p>
            Heart disease is a general term used for conditions
            that affect the heart and cardiovascular system.
        </p>

        <h4>
            Common Risk Factors
        </h4>

        <ul>
            <li>High blood pressure</li>
            <li>High cholesterol</li>
            <li>Diabetes</li>
            <li>Smoking</li>
            <li>Physical inactivity</li>
            <li>Family history</li>
        </ul>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SYMPTOMS TAB
# ============================================================

with active_tab[2]:

    st.markdown("""
    <div class="section-title-wrap">

        <h3>
            ⚠️ Heart Disease Symptoms
        </h3>

        <p>
            Understanding possible warning signs.
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-box">

        <h3>
            Common Symptoms
        </h3>

        <ul>
            <li>Chest discomfort</li>
            <li>Shortness of breath</li>
            <li>Unusual tiredness</li>
            <li>Dizziness</li>
            <li>Sweating</li>
            <li>Upper-body discomfort</li>
        </ul>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# RISK PREDICTION TAB
# ============================================================

with active_tab[3]:

    st.markdown("""
    <div class="section-title-wrap">

        <h3>
            📊 Risk Prediction
        </h3>

        <p>
            Machine-learning based cardiovascular
            risk estimation.
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-box">

        <h3>
            How the Model Works
        </h3>

        <p>
            The model processes selected cardiovascular
            health parameters and produces a prediction.
        </p>

        <ul>
            <li>Age</li>
            <li>Sex</li>
            <li>Chest pain type</li>
            <li>Resting blood pressure</li>
            <li>Cholesterol</li>
            <li>Fasting blood sugar</li>
            <li>Resting ECG</li>
            <li>Maximum heart rate</li>
            <li>Exercise angina</li>
            <li>ST slope</li>
            <li>Oldpeak</li>
        </ul>

    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "♡ Start Prediction",
        key="risk_tab_prediction"
    ):
        open_prediction_dialog()


# ============================================================
# PREVENTION TAB
# ============================================================

with active_tab[4]:

    st.markdown("""
    <div class="section-title-wrap">

        <h3>
            🛡️ Prevention
        </h3>

        <p>
            Healthy habits that can support
            cardiovascular health.
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-box">

        <h3>
            Healthy Heart Habits
        </h3>

        <ul>
            <li>Eat a balanced diet</li>
            <li>Stay physically active</li>
            <li>Avoid tobacco</li>
            <li>Manage blood pressure</li>
            <li>Manage cholesterol</li>
            <li>Get enough sleep</li>
            <li>Have regular health checkups</li>
        </ul>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# ABOUT TAB
# ============================================================

with active_tab[5]:

    st.markdown("""
    <div class="section-title-wrap">

        <h3>
            ⓘ About This Project
        </h3>

        <p>
            Heart Disease Prediction Dashboard
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-box">

        <h3>
            ❤️ Heart Disease Prediction
        </h3>

        <p>
            This project is a machine-learning powered
            healthcare dashboard designed to estimate
            cardiovascular risk from selected health
            parameters.
        </p>

        <h4>
            Project Features
        </h4>

        <ul>
            <li>Interactive Streamlit interface</li>
            <li>Machine-learning prediction</li>
            <li>Health parameter input</li>
            <li>Risk estimation</li>
            <li>Heart-health educational information</li>
        </ul>

        <p>
            This application is intended for educational
            and project demonstration purposes and should
            not replace professional medical advice.
        </p>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="clinical-footer">

    ❤️ Heart Disease Prediction Dashboard

    <br><br>

    Better Insights • Healthier Tomorrows

    <br><br>

    For educational and project demonstration purposes only.

</div>
""", unsafe_allow_html=True)
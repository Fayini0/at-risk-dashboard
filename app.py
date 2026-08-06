# ─────────────────────────────────────────────
# app.py — At-Risk Student Early Warning Dashboard
# Objective 4 — Fika Samkelo Fayini (202207134)
# ─────────────────────────────────────────────

import streamlit as st
import numpy as np
import pandas as pd
import shap
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import load_model

# ── Page config ──────────────────────────────
st.set_page_config(
    page_title="At-Risk Student Early Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .stButton>button {
        background-color: #0F3460;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2E86AB;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #0F3460;
        margin-bottom: 1rem;
    }
    .at-risk-card {
        background: #FFF0F0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #E63946;
        margin-bottom: 1rem;
    }
    .safe-card {
        background: #F0FFF4;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #2D6A4F;
        margin-bottom: 1rem;
    }
    h1 { color: #0F3460; }
    h2 { color: #0F3460; }
    h3 { color: #1A5276; }
</style>
""", unsafe_allow_html=True)


# ── Load model and scaler ─────────────────────
@st.cache_resource
def load_artifacts():
    model  = load_model('models/final_model.keras')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

# ── Feature definitions ───────────────────────
WEEKLY_FEATURES = [
    'total_clicks', 'unique_activities', 'forum_clicks',
    'resource_clicks', 'quiz_clicks'
]

STATIC_FEATURES = [
    'age_band', 'highest_education', 'disability_enc',
    'gender_enc', 'imd_band', 'engagement_trend',
    'consistency_score', 'late_ratio',
    'delta_w1_w2', 'delta_w2_w3', 'delta_w3_w4', 'delta_w4_w5'
]

REGIONS = [
    'East Anglian Region', 'East Midlands Region', 'Ireland',
    'London Region', 'North Region', 'North Western Region',
    'Scotland', 'South East Region', 'South Region',
    'South West Region', 'Wales', 'West Midlands Region',
    'Yorkshire Region'
]

N_WEEKS    = 5
N_FEATURES = 30
THRESHOLD  = 0.4  # lowered from 0.5 to reduce False Negatives

# ── Helper: build flat feature names ─────────
def get_flat_names():
    features_per_step = WEEKLY_FEATURES + STATIC_FEATURES + \
                        [f'region_{r}' for r in REGIONS]
    names = []
    for w in range(1, N_WEEKS + 1):
        for f in features_per_step:
            names.append(f'{f}_w{w}')
    return names

FLAT_NAMES = get_flat_names()


# ── Helper: build tensor from flat row ────────
def build_tensor(df_row):
    """
    Takes a flat feature row and builds the (1, 5, 30) tensor.
    """
    weekly_cols = []
    for base in WEEKLY_FEATURES:
        for w in range(1, N_WEEKS + 1):
            weekly_cols.append(f'{base}_w{w}')

    static_cols = []
    for feat in STATIC_FEATURES:
        static_cols.append(feat)
    for r in REGIONS:
        static_cols.append(f'region_{r}')

    # Build (5, 30) matrix per student
    tensor = np.zeros((1, N_WEEKS, N_FEATURES))

    for w_idx in range(N_WEEKS):
        w = w_idx + 1
        feat_idx = 0

        # Weekly features for this week
        for base in WEEKLY_FEATURES:
            col = f'{base}_w{w}'
            tensor[0, w_idx, feat_idx] = float(
                df_row[col].values[0] if col in df_row.columns else 0
            )
            feat_idx += 1

        # Static features broadcast
        for feat in STATIC_FEATURES:
            tensor[0, w_idx, feat_idx] = float(
                df_row[feat].values[0] if feat in df_row.columns else 0
            )
            feat_idx += 1

        # Region one-hot
        for r in REGIONS:
            col = f'region_{r}'
            tensor[0, w_idx, feat_idx] = float(
                df_row[col].values[0] if col in df_row.columns else 0
            )
            feat_idx += 1

    return tensor


# ── Helper: scale tensor ──────────────────────
def scale_tensor(tensor):
    flat   = tensor.reshape(-1, N_FEATURES)
    scaled = scaler.transform(flat)
    return scaled.reshape(1, N_WEEKS, N_FEATURES)


# ── Helper: predict ───────────────────────────
def predict_student(tensor_scaled):
    prob = model.predict(tensor_scaled, verbose=0).flatten()[0]
    label = 1 if prob >= THRESHOLD else 0
    return float(prob), int(label)


# ── Helper: SHAP waterfall ────────────────────
def compute_shap_waterfall(tensor_scaled):
    flat = tensor_scaled.reshape(1, -1)

    def predict_flat(X):
        X_3d = X.reshape(-1, N_WEEKS, N_FEATURES)
        return model.predict(X_3d, verbose=0).flatten()

    # Small background for speed
    bg = np.zeros((10, N_WEEKS * N_FEATURES))
    explainer = shap.KernelExplainer(predict_flat, bg)
    sv = explainer.shap_values(flat, nsamples=50, verbose=False)

    exp = shap.Explanation(
        values=sv[0],
        base_values=explainer.expected_value,
        data=flat[0],
        feature_names=FLAT_NAMES
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.waterfall_plot(exp, max_display=15, show=False)
    plt.tight_layout()
    return fig


# ── Helper: encode categorical inputs ─────────
def encode_inputs(age_band, highest_edu, disability,
                  gender, imd_band, region):
    age_map = {'0-35': 0, '35-55': 1, '55<=': 2}
    edu_map = {
        'No Formal Quals': 0,
        'Lower Than A Level': 1,
        'A Level or Equivalent': 2,
        'HE Qualification': 3,
        'Post Graduate Qualification': 4
    }
    imd_map = {
        '0-10%': 0, '10-20%': 1, '20-30%': 2, '30-40%': 3,
        '40-50%': 4, '50-60%': 5, '60-70%': 6, '70-80%': 7,
        '80-90%': 8, '90-100%': 9
    }
    region_enc = {f'region_{r}': 0 for r in REGIONS}
    region_enc[f'region_{region}'] = 1

    return {
        'age_band':          age_map.get(age_band, 0),
        'highest_education': edu_map.get(highest_edu, 0),
        'disability_enc':    1 if disability == 'Yes' else 0,
        'gender_enc':        1 if gender == 'Male' else 0,
        'imd_band':          imd_map.get(imd_band, 5),
        **region_enc
    }


# ════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/en/thumb/"
        "1/1c/Sol_Plaatje_University_logo.png/200px-"
        "Sol_Plaatje_University_logo.png",
        width=120
    )
    st.markdown("## 🎓 Early Warning System")
    st.markdown("**CNN-BiLSTM At-Risk Detection**")
    st.markdown("---")
    st.markdown(
        "This system predicts whether a student is **at-risk** "
        "of failing or withdrawing based on their first 5 weeks "
        "of LMS interaction data."
    )
    st.markdown("---")
    st.markdown("**Model Details**")
    st.markdown("- Architecture: CNN-BiLSTM")
    st.markdown("- Dataset: OULAD")
    st.markdown("- Prediction window: Weeks 1–5")
    st.markdown(f"- Decision threshold: {THRESHOLD}")
    st.markdown("- AUC-ROC: 0.7075")
    st.markdown("---")
    st.markdown(
        "**Fika Samkelo Fayini · 202207134**\n\n"
        "BScHons Computer Science\n\n"
        "Sol Plaatje University\n\n"
        "Supervisor: Dr. F. Matsebula"
    )


# ════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════
st.title("🎓 At-Risk Student Early Warning System")
st.markdown(
    "Predict which students are at-risk of failing or withdrawing "
    "using their first 5 weeks of LMS interaction data — "
    "with SHAP explanations showing exactly why."
)
st.markdown("---")

# ── Tabs ──────────────────────────────────────
tab1, tab2 = st.tabs([
    "📋  Single Student Prediction",
    "📁  Batch Prediction (CSV Upload)"
])


# ════════════════════════════════════════════
# TAB 1 — SINGLE STUDENT
# ════════════════════════════════════════════
with tab1:
    st.markdown("### Enter Student LMS Data")
    st.markdown(
        "Fill in the student's LMS engagement data for each week "
        "and their demographic information. Click **Predict** when done."
    )

    # ── Weekly LMS features ───────────────────
    st.markdown("#### 📊 Weekly LMS Engagement")

    weeks = ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"]

    col_labels = st.columns(5)
    for i, w in enumerate(weeks):
        col_labels[i].markdown(f"**{w}**")

    # total_clicks
    st.markdown("**Total Clicks**")
    tc_cols = st.columns(5)
    total_clicks = [
        tc_cols[i].number_input(
            f"tc_w{i+1}", min_value=0, value=0,
            label_visibility="collapsed", key=f"tc_{i}"
        )
        for i in range(5)
    ]

    # unique_activities
    st.markdown("**Unique Activity Types**")
    ua_cols = st.columns(5)
    unique_activities = [
        ua_cols[i].number_input(
            f"ua_w{i+1}", min_value=0, value=0,
            label_visibility="collapsed", key=f"ua_{i}"
        )
        for i in range(5)
    ]

    # forum_clicks
    st.markdown("**Forum Clicks**")
    fc_cols = st.columns(5)
    forum_clicks = [
        fc_cols[i].number_input(
            f"fc_w{i+1}", min_value=0, value=0,
            label_visibility="collapsed", key=f"fc_{i}"
        )
        for i in range(5)
    ]

    # resource_clicks
    st.markdown("**Resource Clicks**")
    rc_cols = st.columns(5)
    resource_clicks = [
        rc_cols[i].number_input(
            f"rc_w{i+1}", min_value=0, value=0,
            label_visibility="collapsed", key=f"rc_{i}"
        )
        for i in range(5)
    ]

    # quiz_clicks
    st.markdown("**Quiz Clicks**")
    qc_cols = st.columns(5)
    quiz_clicks = [
        qc_cols[i].number_input(
            f"qc_w{i+1}", min_value=0, value=0,
            label_visibility="collapsed", key=f"qc_{i}"
        )
        for i in range(5)
    ]

    st.markdown("---")

    # ── Demographic features ──────────────────
    st.markdown("#### 👤 Student Demographics")
    d1, d2, d3 = st.columns(3)

    with d1:
        age_band = st.selectbox(
            "Age Band",
            ['0-35', '35-55', '55<=']
        )
        highest_edu = st.selectbox(
            "Highest Education",
            ['No Formal Quals', 'Lower Than A Level',
             'A Level or Equivalent', 'HE Qualification',
             'Post Graduate Qualification']
        )

    with d2:
        disability = st.selectbox("Disability", ['No', 'Yes'])
        gender     = st.selectbox("Gender", ['Female', 'Male'])

    with d3:
        imd_band = st.selectbox(
            "IMD Band (Deprivation)",
            ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%',
             '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
        )
        region = st.selectbox("Region", REGIONS)

    st.markdown("---")

    # ── Assessment behaviour ──────────────────
    st.markdown("#### 📝 Assessment Behaviour (Weeks 1–5)")
    a1, a2 = st.columns(2)

    with a1:
        late_ratio = st.slider(
            "Late Submission Ratio (0 = never late, 1 = always late)",
            min_value=0.0, max_value=1.0, value=0.0, step=0.05
        )

    with a2:
        st.markdown(
            "This is the proportion of assessments due in Weeks 1–5 "
            "that were submitted after the deadline."
        )

    st.markdown("---")

    # ── Predict button ────────────────────────
    if st.button("🔍  Predict At-Risk Status", key="single_predict"):

        with st.spinner("Computing prediction and SHAP explanation..."):

            # Compute derived features
            tc = total_clicks
            engagement_trend  = tc[4] - tc[0]
            consistency_score = float(np.std(tc))
            delta_w1_w2 = tc[1] - tc[0]
            delta_w2_w3 = tc[2] - tc[1]
            delta_w3_w4 = tc[3] - tc[2]
            delta_w4_w5 = tc[4] - tc[3]

            # Encode demographics
            demo = encode_inputs(
                age_band, highest_edu, disability,
                gender, imd_band, region
            )

            # Build flat row DataFrame
            row_dict = {}
            for w_idx in range(5):
                w = w_idx + 1
                row_dict[f'total_clicks_w{w}']       = tc[w_idx]
                row_dict[f'unique_activities_w{w}']   = unique_activities[w_idx]
                row_dict[f'forum_clicks_w{w}']        = forum_clicks[w_idx]
                row_dict[f'resource_clicks_w{w}']     = resource_clicks[w_idx]
                row_dict[f'quiz_clicks_w{w}']         = quiz_clicks[w_idx]

            row_dict['age_band']          = demo['age_band']
            row_dict['highest_education'] = demo['highest_education']
            row_dict['disability_enc']    = demo['disability_enc']
            row_dict['gender_enc']        = demo['gender_enc']
            row_dict['imd_band']          = demo['imd_band']
            row_dict['engagement_trend']  = engagement_trend
            row_dict['consistency_score'] = consistency_score
            row_dict['late_ratio']        = late_ratio
            row_dict['delta_w1_w2']       = delta_w1_w2
            row_dict['delta_w2_w3']       = delta_w2_w3
            row_dict['delta_w3_w4']       = delta_w3_w4
            row_dict['delta_w4_w5']       = delta_w4_w5

            for r in REGIONS:
                row_dict[f'region_{r}'] = demo.get(f'region_{r}', 0)

            df_row = pd.DataFrame([row_dict])

            # Build tensor
            tensor        = build_tensor(df_row)
            tensor_scaled = scale_tensor(tensor)

            # Predict
            prob, label = predict_student(tensor_scaled)

        # ── Display result ────────────────────
        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        r1, r2, r3 = st.columns(3)

        with r1:
            if label == 1:
                st.markdown(
                    f'<div class="at-risk-card">'
                    f'<h3 style="color:#E63946;">⚠ AT-RISK</h3>'
                    f'<p style="font-size:2rem;font-weight:bold;'
                    f'color:#E63946;">{prob:.1%}</p>'
                    f'<p>Probability of being at-risk</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="safe-card">'
                    f'<h3 style="color:#2D6A4F;">✅ SAFE</h3>'
                    f'<p style="font-size:2rem;font-weight:bold;'
                    f'color:#2D6A4F;">{1-prob:.1%}</p>'
                    f'<p>Probability of being safe</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        with r2:
            st.markdown(
                f'<div class="metric-card">'
                f'<h3>Engagement Trend</h3>'
                f'<p style="font-size:1.5rem;font-weight:bold;">'
                f'{engagement_trend:+.0f}</p>'
                f'<p>Week 5 clicks minus Week 1 clicks</p>'
                f'</div>',
                unsafe_allow_html=True
            )

        with r3:
            st.markdown(
                f'<div class="metric-card">'
                f'<h3>Late Submission Ratio</h3>'
                f'<p style="font-size:1.5rem;font-weight:bold;">'
                f'{late_ratio:.0%}</p>'
                f'<p>Proportion of late submissions</p>'
                f'</div>',
                unsafe_allow_html=True
            )

        # ── Advisor guidance ──────────────────
        st.markdown("---")
        st.markdown("### 💬 Guidance for Academic Advisor")

        if label == 1:
            if late_ratio > 0.3:
                st.warning(
                    "⚠ This student has a high late submission ratio. "
                    "Consider discussing time management, external commitments, "
                    "and whether additional academic support is needed."
                )
            if engagement_trend < -50:
                st.warning(
                    "⚠ This student's LMS engagement declined significantly "
                    "from Week 1 to Week 5. Consider checking whether they "
                    "are still actively participating in the module."
                )
            if sum(quiz_clicks) == 0:
                st.warning(
                    "⚠ This student had zero quiz activity across all 5 weeks. "
                    "This is a strong indicator of disengagement from "
                    "assessment preparation."
                )
        else:
            st.success(
                "✅ This student shows a safe engagement pattern. "
                "Continue monitoring at the next checkpoint."
            )

        # ── Weekly engagement chart ───────────
        st.markdown("---")
        st.markdown("### 📈 Weekly Engagement Profile")

        fig_eng, ax_eng = plt.subplots(figsize=(10, 4))
        x = ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5']
        ax_eng.plot(x, tc, 'o-', color='#0F3460',
                    linewidth=2.5, markersize=8, label='Total Clicks')
        ax_eng.fill_between(x, tc, alpha=0.1, color='#0F3460')
        ax_eng.axhline(
            y=np.mean(tc), color='#E63946',
            linestyle='--', alpha=0.6, label='Average'
        )
        ax_eng.set_title('Total Clicks per Week', fontsize=13,
                          fontweight='bold')
        ax_eng.set_ylabel('Total Clicks')
        ax_eng.legend()
        ax_eng.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_eng)
        plt.close()

        # ── SHAP waterfall ────────────────────
        st.markdown("---")
        st.markdown("### 🔍 SHAP Explanation — Why This Prediction?")
        st.markdown(
            "The waterfall plot below shows which features drove "
            "this prediction. **Red bars** pushed the prediction "
            "toward at-risk. **Blue bars** pushed it toward safe."
        )

        with st.spinner("Computing SHAP explanation (this takes ~30 seconds)..."):
            fig_shap = compute_shap_waterfall(tensor_scaled)
            st.pyplot(fig_shap)
            plt.close()

        st.info(
            "💡 Features shown on the left are the actual values "
            "for this student. The bars show how much each feature "
            "moved the prediction from the baseline of ~0.44."
        )


# ════════════════════════════════════════════
# TAB 2 — BATCH PREDICTION
# ════════════════════════════════════════════
with tab2:
    st.markdown("### Batch Prediction — Upload a CSV File")
    st.markdown(
        "Upload a CSV file containing multiple students. "
        "The system will predict the at-risk status for each one "
        "and return a downloadable results table."
    )

    # Download template
    with open('data/sample_input.csv', 'rb') as f:
        st.download_button(
            label="📥  Download CSV Template",
            data=f,
            file_name="student_template.csv",
            mime="text/csv"
        )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload your completed CSV file",
        type=['csv']
    )

    if uploaded_file is not None:

        df_upload = pd.read_csv(uploaded_file)
        st.markdown(f"**{len(df_upload)} students loaded.**")
        st.dataframe(df_upload.head(5))

        if st.button("🔍  Run Batch Prediction", key="batch_predict"):

            results = []
            progress = st.progress(0)
            status   = st.empty()

            for i, row in df_upload.iterrows():
                status.text(
                    f"Processing student {i+1} of {len(df_upload)}..."
                )

                try:
                    df_row = pd.DataFrame([row])

                    # Compute derived features if not present
                    tc_cols = [f'total_clicks_w{w}' for w in range(1,6)]
                    tc = [float(row.get(c, 0)) for c in tc_cols]

                    if 'engagement_trend' not in df_row.columns:
                        df_row['engagement_trend']  = tc[4] - tc[0]
                        df_row['consistency_score'] = float(np.std(tc))
                        df_row['delta_w1_w2'] = tc[1] - tc[0]
                        df_row['delta_w2_w3'] = tc[2] - tc[1]
                        df_row['delta_w3_w4'] = tc[3] - tc[2]
                        df_row['delta_w4_w5'] = tc[4] - tc[3]

                    # Encode demographics if needed
                    if 'disability_enc' not in df_row.columns:
                        df_row['disability_enc'] = (
                            1 if str(row.get('disability',
                                             'N')).upper() == 'Y' else 0
                        )
                    if 'gender_enc' not in df_row.columns:
                        df_row['gender_enc'] = (
                            1 if str(row.get('gender',
                                             'F')).upper() == 'M' else 0
                        )

                    # Region one-hot
                    for r in REGIONS:
                        col = f'region_{r}'
                        if col not in df_row.columns:
                            df_row[col] = (
                                1 if str(row.get('region', '')) == r
                                else 0
                            )

                    tensor        = build_tensor(df_row)
                    tensor_scaled = scale_tensor(tensor)
                    prob, label   = predict_student(tensor_scaled)

                    tc_vals           = [float(row.get(
                        f'total_clicks_w{w}', 0)) for w in range(1,6)]
                    engagement_trend  = tc_vals[4] - tc_vals[0]
                    late_ratio_val    = float(
                        row.get('late_ratio', 0)
                    )

                    results.append({
                        'student_id':       row.get('student_id', i),
                        'at_risk_prob':     round(prob, 4),
                        'prediction':       'At-Risk' if label == 1
                                            else 'Safe',
                        'engagement_trend': round(engagement_trend, 1),
                        'late_ratio':       round(late_ratio_val, 3),
                        'total_clicks_w1':  tc_vals[0],
                        'total_clicks_w5':  tc_vals[4],
                    })

                except Exception as e:
                    results.append({
                        'student_id': row.get('student_id', i),
                        'at_risk_prob': None,
                        'prediction': f'Error: {str(e)}',
                        'engagement_trend': None,
                        'late_ratio': None,
                        'total_clicks_w1': None,
                        'total_clicks_w5': None,
                    })

                progress.progress((i + 1) / len(df_upload))

            status.text("Batch prediction complete.")
            results_df = pd.DataFrame(results)

            st.markdown("---")
            st.markdown("### Results")

            # Summary stats
            n_total   = len(results_df)
            n_at_risk = (results_df['prediction'] == 'At-Risk').sum()
            n_safe    = (results_df['prediction'] == 'Safe').sum()

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Students",    n_total)
            m2.metric("At-Risk",  n_at_risk,
                      delta=f"{n_at_risk/n_total:.1%}")
            m3.metric("Safe",     n_safe,
                      delta=f"{n_safe/n_total:.1%}")

            # Colour code the table
            def colour_row(row):
                if row['prediction'] == 'At-Risk':
                    return ['background-color: #FFF0F0'] * len(row)
                elif row['prediction'] == 'Safe':
                    return ['background-color: #F0FFF4'] * len(row)
                return [''] * len(row)

            st.dataframe(
                results_df.style.apply(colour_row, axis=1),
                use_container_width=True
            )

            # Download results
            csv_out = results_df.to_csv(index=False)
            st.download_button(
                label="📥  Download Results CSV",
                data=csv_out,
                file_name="at_risk_predictions.csv",
                mime="text/csv"
            )

            # Risk distribution chart
            st.markdown("---")
            st.markdown("### Risk Probability Distribution")

            valid = results_df[
                results_df['at_risk_prob'].notna()
            ]['at_risk_prob']

            fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
            ax_dist.hist(valid, bins=20, color='#0F3460',
                         alpha=0.7, edgecolor='white')
            ax_dist.axvline(
                x=THRESHOLD, color='#E63946',
                linestyle='--', linewidth=2,
                label=f'Threshold ({THRESHOLD})'
            )
            ax_dist.set_xlabel('At-Risk Probability')
            ax_dist.set_ylabel('Number of Students')
            ax_dist.set_title(
                'Distribution of At-Risk Probabilities',
                fontsize=13, fontweight='bold'
            )
            ax_dist.legend()
            ax_dist.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_dist)
            plt.close()
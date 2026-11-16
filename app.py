# ─────────────────────────────────────────────────────────────
# app.py — At-Risk Student Early Warning Dashboard (prototype)
# Objective 4 — Fika Samkelo Fayini (202207134)
# Sol Plaatje University · BScHons Computer Science
#
# Folder layout expected next to this file:
#   models_corrected/final_model.keras
#   models_corrected/scaler.pkl
#   models_corrected/model_info.json
#   data/processed_corrected/X_train_raw.npy, y_train.npy (real training background)
#   outputs_corrected/shap_feature_importance_corrected.csv
# ─────────────────────────────────────────────────────────────

import os
import json

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import load_model

# ── Paths ────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models_corrected")
DATA_DIR  = os.path.join(BASE_DIR, "data", "processed_corrected")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_corrected")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="At-Risk Student Early Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .stButton>button {
        background-color: #0F3460; color: white; border-radius: 8px;
        padding: 0.5rem 2rem; font-weight: bold; border: none;
    }
    .stButton>button:hover { background-color: #2E86AB; }
    .metric-card, .at-risk-card, .safe-card {
        border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .metric-card  { background: white;   border-left: 5px solid #0F3460; }
    .at-risk-card { background: #FFF0F0; border-left: 5px solid #E63946; }
    .safe-card    { background: #F0FFF4; border-left: 5px solid #2D6A4F; }
    h1, h2 { color: #0F3460; }
    h3 { color: #1A5276; }
</style>
""", unsafe_allow_html=True)

# ── Constants (must match the training pipeline exactly) ─────
N_WEEKS    = 5
N_FEATURES = 31          # 5 weekly + 26 static/derived (repeated each week)

WEEKLY_FEATURES = [
    "total_clicks", "unique_activities", "forum_clicks",
    "resource_clicks", "quiz_clicks",
]

# Order matters: it is the order used in notebook 01 (Cell 18)
STATIC_FEATURES = [
    "age_band", "highest_education", "disability_enc",
    "gender_enc", "imd_band", "engagement_trend",
    "consistency_score", "late_ratio", "missing_assessment_ratio",
    "delta_w1_w2", "delta_w2_w3", "delta_w3_w4", "delta_w4_w5",
]

REGIONS = [
    "East Anglian Region", "East Midlands Region", "Ireland",
    "London Region", "North Region", "North Western Region",
    "Scotland", "South East Region", "South Region",
    "South West Region", "Wales", "West Midlands Region",
    "Yorkshire Region",
]

# Fixed category codes used by the corrected feature-engineering pipeline.
# The mappings preserve the same alphabetical category ordering as the
# previous LabelEncoder, but are explicit and do not require fitting on test data.
AGE_CODES = {"0-35": 0, "35-55": 1, "55<=": 2}
EDU_CODES = {
    "a level or equivalent": 0,
    "he qualification": 1,
    "lower than a level": 2,
    "no formal quals": 3,
    "post graduate qualification": 4,
}
IMD_CODES = {
    "0-10%": 0, "10-20%": 1, "20-30%": 2, "30-40%": 3, "40-50%": 4,
    "50-60%": 5, "60-70%": 6, "70-80%": 7, "80-90%": 8, "90-100%": 9,
}

EDU_LABELS = [
    "No Formal quals", "Lower Than A Level", "A Level or Equivalent",
    "HE Qualification", "Post Graduate Qualification",
]
IMD_LABELS = list(IMD_CODES.keys())
AGE_LABELS = list(AGE_CODES.keys())

WEEKLY_COLS = [f"{b}_w{w}" for b in WEEKLY_FEATURES for w in range(1, N_WEEKS + 1)]
REGION_COLS = [f"region_{r}" for r in REGIONS]
FRAME_COLS  = WEEKLY_COLS + STATIC_FEATURES + REGION_COLS      # 25 + 13 + 13 = 51 unique features
# The SHAP / de-duplicated vector is built from the (5 weeks x 5 features) block, which
# flattens WEEK-MAJOR (week 1: its 5 features, then week 2, ...). Names must follow that order.
SHAP_NAMES  = [f"{b}_w{w}" for w in range(1, N_WEEKS + 1) for b in WEEKLY_FEATURES] \
              + STATIC_FEATURES + REGION_COLS


# ── Model information (threshold / AUC) ──────────────────────
def load_model_info():
    """Read the validation-selected threshold and test AUC from model_info.json if present."""
    info = {"threshold": 0.4, "auc_roc": None}
    path = os.path.join(MODEL_DIR, "model_info.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                info.update(json.load(f))
        except Exception:
            pass
    return info

MODEL_INFO = load_model_info()
THRESHOLD  = float(MODEL_INFO["threshold"])


# ── Cached loaders ───────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = load_model(os.path.join(MODEL_DIR, "final_model.keras"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    return model, scaler

model, scaler = load_artifacts()


def scale_batch(tensors):
    """tensors: (n, 5, 31) raw -> (n, 5, 31) scaled with the training scaler."""
    n = tensors.shape[0]
    flat = tensors.reshape(-1, N_FEATURES)
    return scaler.transform(flat).reshape(n, N_WEEKS, N_FEATURES)


def dedup_tensor(tensors_scaled):
    """(n, 5, 31) -> (n, 51): 25 weekly values + one copy of the 26 static values."""
    n = tensors_scaled.shape[0]
    tv = tensors_scaled[:, :, 0:5].reshape(n, -1)
    st_part = tensors_scaled[:, 0, 5:31]
    return np.concatenate([tv, st_part], axis=1)


def predict_dedup(X51):
    n = X51.shape[0]
    tv = X51[:, :25].reshape(n, N_WEEKS, 5)
    stat = np.repeat(X51[:, 25:][:, np.newaxis, :], N_WEEKS, axis=1)
    return model.predict(np.concatenate([tv, stat], axis=2), verbose=0).flatten()


@st.cache_resource
def load_background(n_background=300):
    """Stratified sample of real training students (scaled, de-duplicated)."""
    x_path = os.path.join(DATA_DIR, "X_train_raw.npy")
    y_path = os.path.join(DATA_DIR, "y_train.npy")
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        return None
    X = np.load(x_path)
    y = np.load(y_path)
    Xd = dedup_tensor(scale_batch(X.astype(np.float64)))
    rng = np.random.RandomState(42)
    half = n_background // 2
    idx0 = rng.choice(np.where(y == 0)[0], min(half, int((y == 0).sum())), replace=False)
    idx1 = rng.choice(np.where(y == 1)[0], min(half, int((y == 1).sum())), replace=False)
    return Xd[np.concatenate([idx0, idx1])]


@st.cache_resource
def get_explainer():
    bg = load_background()
    if bg is None:
        return None
    return shap.KernelExplainer(predict_dedup, bg)


def shap_waterfall(tensor_scaled_1):
    """Waterfall figure for one student. tensor_scaled_1: (1, 5, 31)."""
    explainer = get_explainer()
    if explainer is None:
        return None
    x51 = dedup_tensor(tensor_scaled_1)
    sv = explainer.shap_values(x51, nsamples=200, verbose=False)
    sv = np.array(sv).reshape(-1)[:51] if np.ndim(sv) > 1 else np.array(sv)
    base = float(np.ravel(explainer.expected_value)[0])
    exp = shap.Explanation(values=sv, base_values=base,
                           data=x51[0], feature_names=SHAP_NAMES)
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(exp, max_display=15, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig, base


# ── Encoding / tensor building ───────────────────────────────
def _norm_text(v):
    return str(v).strip().lower()


def code_age(v):
    if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
        return int(v)
    return AGE_CODES.get(str(v).strip(), 0)


def code_edu(v):
    if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
        return int(v)
    return EDU_CODES.get(_norm_text(v), 3)


def code_imd(v):
    if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
        return int(v)
    s = str(v).strip()
    if s and not s.endswith("%"):
        s += "%"                       # raw OULAD has '10-20' without '%'
    return IMD_CODES.get(s, 3)         # 3 = '30-40%' (mode used in training)


def build_feature_frame(df):
    """
    Turn a DataFrame of students (one row each) into the 51 unique model
    inputs. Accepts raw OULAD-style text columns (age_band, highest_education,
    imd_band, disability 'Y'/'N', gender 'M'/'F', region) or already-encoded
    columns. Derived features are computed if they are not supplied.
    Returns (feature_frame, list_of_warnings).
    """
    warnings_out = []
    out = pd.DataFrame(index=df.index)

    missing = [c for c in WEEKLY_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required weekly columns: " + ", ".join(missing[:6])
            + (" ..." if len(missing) > 6 else "")
        )
    for c in WEEKLY_COLS:
        out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    tc = out[[f"total_clicks_w{w}" for w in range(1, 6)]].values.astype(float)

    # derived behavioural features (same definitions as notebook 01)
    out["engagement_trend"]  = df["engagement_trend"].astype(float) \
        if "engagement_trend" in df.columns else tc[:, 4] - tc[:, 0]
    out["consistency_score"] = df["consistency_score"].astype(float) \
        if "consistency_score" in df.columns else tc.std(axis=1, ddof=1)   # pandas .std() = ddof 1
    for a, b in [(1, 2), (2, 3), (3, 4), (4, 5)]:
        col = f"delta_w{a}_w{b}"
        out[col] = df[col].astype(float) if col in df.columns else tc[:, b - 1] - tc[:, a - 1]

    if "late_ratio" in df.columns:
        out["late_ratio"] = pd.to_numeric(df["late_ratio"], errors="coerce").fillna(0.0).clip(0, 1)
    else:
        out["late_ratio"] = 0.0
        warnings_out.append("No 'late_ratio' column found; 0 was used for all students.")

    if "missing_assessment_ratio" in df.columns:
        out["missing_assessment_ratio"] = pd.to_numeric(
            df["missing_assessment_ratio"], errors="coerce"
        ).fillna(0.0).clip(0, 1)
    else:
        out["missing_assessment_ratio"] = 0.0
        warnings_out.append(
            "No 'missing_assessment_ratio' column found; 0 was used for all students."
        )

    # categorical / demographic
    def pick(*names):
        for n in names:
            if n in df.columns:
                return df[n]
        return None

    s = pick("age_band")
    out["age_band"] = s.map(code_age) if s is not None else 0
    s = pick("highest_education")
    out["highest_education"] = s.map(code_edu) if s is not None else 3
    s = pick("imd_band")
    out["imd_band"] = s.map(code_imd) if s is not None else 3

    s = pick("disability_enc", "disability")
    if s is None:
        out["disability_enc"] = 0
    elif "disability_enc" in df.columns:
        out["disability_enc"] = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    else:
        out["disability_enc"] = s.astype(str).str.strip().str.upper().isin(["Y", "YES"]).astype(int)

    s = pick("gender_enc", "gender")
    if s is None:
        out["gender_enc"] = 0
    elif "gender_enc" in df.columns:
        out["gender_enc"] = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    else:
        out["gender_enc"] = s.astype(str).str.strip().str.upper().isin(["M", "MALE"]).astype(int)

    for r in REGIONS:
        col = f"region_{r}"
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        elif "region" in df.columns:
            out[col] = (df["region"].astype(str).str.strip() == r).astype(int)
        else:
            out[col] = 0
    if "region" not in df.columns and not any(f"region_{r}" in df.columns for r in REGIONS):
        warnings_out.append("No region information found; region was left blank.")

    return out[FRAME_COLS], warnings_out


def frame_to_tensors(frame):
    """(n, 51) feature frame -> (n, 5, 31) tensor in the training layout."""
    n = len(frame)
    weekly = frame[WEEKLY_COLS].values.astype(np.float64).reshape(n, 5, N_WEEKS).transpose(0, 2, 1)
    static = frame[STATIC_FEATURES + REGION_COLS].values.astype(np.float64)       # (n, 25)
    static = np.repeat(static[:, np.newaxis, :], N_WEEKS, axis=1)
    return np.concatenate([weekly, static], axis=2)


def predict_frame(frame):
    tensors = frame_to_tensors(frame)
    scaled = scale_batch(tensors)
    prob = model.predict(scaled, verbose=0).flatten()
    label = (prob >= THRESHOLD).astype(int)
    return prob, label, scaled


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Early Warning System")
    st.markdown("**CNN-BiLSTM At-Risk Detection**")
    st.markdown("---")
    st.markdown(
        "Estimates whether a student is **at risk** of failing or withdrawing "
        "using the first 5 weeks of LMS interaction data."
    )
    st.markdown("---")
    st.markdown("**Model details**")
    st.markdown("- Architecture: CNN-BiLSTM")
    st.markdown("- Dataset: OULAD")
    st.markdown("- Prediction window: Weeks 1–5")
    st.markdown(f"- Decision threshold: {THRESHOLD:g}")
    if MODEL_INFO.get("auc_roc") is not None:
        st.markdown(f"- Test AUC-ROC: {float(MODEL_INFO['auc_roc']):.4f}")
    st.markdown("---")
    st.markdown(
        "**Fika Samkelo Fayini · 202207134**\n\n"
        "BScHons Computer Science\n\n"
        "Sol Plaatje University\n\n"
        "Supervisor: Dr. F. Matsebula"
    )

# ── Header ───────────────────────────────────────────────────
st.title("🎓 At-Risk Student Early Warning System")
st.markdown(
    "Decision-support prototype: identifies students who may be at risk using "
    "their first five weeks of LMS activity, with SHAP explanations of each prediction."
)
st.warning(
    "**Prototype for research use.** The model was trained on OULAD (one UK "
    "institution) and misses a substantial share of at-risk students, so a "
    "'Safe' result is not a guarantee. Predictions support, and never replace, "
    "an academic advisor's judgement. Demographic variables are model inputs "
    "and should be interpreted with care (see the *Model Insights* tab)."
)

tab1, tab2, tab3 = st.tabs([
    "📋  Single Student Prediction",
    "📁  Batch Prediction (CSV Upload)",
    "🔎  Model Insights",
])

# ═════════════════════════════════════════════════════════════
# TAB 1 — SINGLE STUDENT
# ═════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Enter Student LMS Data")
    st.markdown("Enter the weekly LMS activity and demographic details, then click **Predict**.")

    st.markdown("#### 📊 Weekly LMS Engagement")
    hdr = st.columns(5)
    for i in range(5):
        hdr[i].markdown(f"**Week {i + 1}**")

    def week_inputs(label, key):
        st.markdown(f"**{label}**")
        cols = st.columns(5)
        return [
            cols[i].number_input(f"{key}_w{i + 1}", min_value=0, value=0, step=1,
                                 label_visibility="collapsed", key=f"{key}_{i}")
            for i in range(5)
        ]

    total_clicks      = week_inputs("Total Clicks", "tc")
    unique_activities = week_inputs("Unique Activity Types", "ua")
    forum_clicks      = week_inputs("Forum Clicks", "fc")
    resource_clicks   = week_inputs("Resource Clicks", "rc")
    quiz_clicks       = week_inputs("Quiz Clicks", "qc")

    st.markdown("---")
    st.markdown("#### 👤 Student Demographics")
    d1, d2, d3 = st.columns(3)
    with d1:
        age_band    = st.selectbox("Age Band", AGE_LABELS)
        highest_edu = st.selectbox("Highest Education", EDU_LABELS)
    with d2:
        disability = st.selectbox("Disability", ["No", "Yes"])
        gender     = st.selectbox("Gender", ["Female", "Male"])
    with d3:
        imd_band = st.selectbox("IMD Band (deprivation)", IMD_LABELS, index=3)
        region   = st.selectbox("Region", REGIONS)

    st.markdown("---")
    st.markdown("#### 📝 Assessment Behaviour (up to Day 35)")
    late_ratio = st.slider(
        "Late Submission Ratio (among submissions made by Day 35)",
        min_value=0.0, max_value=1.0, value=0.0, step=0.05,
    )
    missing_assessment_ratio = st.slider(
        "Assessments Not Submitted by Day 35 (ratio)",
        min_value=0.0, max_value=1.0, value=0.0, step=0.05,
    )
    st.caption(
        "Late ratio is calculated among assessments submitted by Day 35. "
        "The second measure separately records assessments due by Day 35 "
        "that had not been submitted by the prediction cut-off."
    )
    st.markdown("---")

    if st.button("🔍  Predict At-Risk Status", key="single_predict"):
        row = {}
        for w in range(5):
            row[f"total_clicks_w{w + 1}"]       = total_clicks[w]
            row[f"unique_activities_w{w + 1}"]  = unique_activities[w]
            row[f"forum_clicks_w{w + 1}"]       = forum_clicks[w]
            row[f"resource_clicks_w{w + 1}"]    = resource_clicks[w]
            row[f"quiz_clicks_w{w + 1}"]        = quiz_clicks[w]
        row.update({
            "age_band": age_band, "highest_education": highest_edu,
            "imd_band": imd_band,
            "disability": "Y" if disability == "Yes" else "N",
            "gender": "M" if gender == "Male" else "F",
            "region": region, "late_ratio": late_ratio,
            "missing_assessment_ratio": missing_assessment_ratio,
        })
        frame, _ = build_feature_frame(pd.DataFrame([row]))
        prob, label, scaled = predict_frame(frame)
        prob, label = float(prob[0]), int(label[0])
        trend = float(frame["engagement_trend"].iloc[0])

        st.session_state["single"] = {
            "prob": prob, "label": label, "trend": trend,
            "late": late_ratio, "missing": missing_assessment_ratio, "tc": list(total_clicks),
            "quiz_total": int(sum(quiz_clicks)), "scaled": scaled,
        }

    res = st.session_state.get("single")
    if res:
        prob, label = res["prob"], res["label"]
        st.markdown("### 🎯 Prediction Result")
        r1, r2, r3 = st.columns(3)
        with r1:
            if label == 1:
                st.markdown(
                    f'<div class="at-risk-card"><h3 style="color:#E63946;">⚠ AT-RISK</h3>'
                    f'<p style="font-size:2rem;font-weight:bold;color:#E63946;">{prob:.1%}</p>'
                    f'<p>Estimated probability of being at risk</p></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="safe-card"><h3 style="color:#2D6A4F;">✅ NOT FLAGGED</h3>'
                    f'<p style="font-size:2rem;font-weight:bold;color:#2D6A4F;">{prob:.1%}</p>'
                    f'<p>Estimated probability of being at risk (below the {THRESHOLD:g} threshold)</p></div>',
                    unsafe_allow_html=True)
        with r2:
            st.markdown(
                f'<div class="metric-card"><h3>Engagement Trend</h3>'
                f'<p style="font-size:1.5rem;font-weight:bold;">{res["trend"]:+.0f}</p>'
                f'<p>Week 5 clicks minus Week 1 clicks</p></div>', unsafe_allow_html=True)
        with r3:
            st.markdown(
                f'<div class="metric-card"><h3>Late Submission Ratio</h3>'
                f'<p style="font-size:1.5rem;font-weight:bold;">{res["late"]:.0%}</p>'
                f'<p>Late submissions made by Day 35</p></div>', unsafe_allow_html=True)

        st.markdown("### 💬 Illustrative Advisor Prompts")
        st.caption(
            "These prompts are simple rule-based indicators for demonstration. "
            "They have not been evaluated with advisors and are not validated intervention advice."
        )
        if label == 1:
            shown = False
            if res["late"] > 0.3:
                st.warning("High late-submission ratio: consider discussing time management, "
                           "outside commitments and whether extra academic support is needed.")
                shown = True
            if res["missing"] > 0.3:
                st.warning("Several assessments due by Day 35 were not submitted by the cut-off. "
                           "Consider checking whether the student needs assessment support.")
                shown = True
            if res["trend"] < -50:
                st.warning("Engagement fell markedly between Week 1 and Week 5: check whether "
                           "the student is still participating in the module.")
                shown = True
            if res["quiz_total"] == 0:
                st.warning("No quiz activity in Weeks 1–5: a possible sign of disengagement "
                           "from assessment preparation.")
                shown = True
            if not shown:
                st.info("The student was flagged by the model but no single rule-of-thumb "
                        "indicator stands out. See the SHAP explanation below.")
        else:
            st.success("The student was not flagged. Continue routine monitoring; this is not a "
                       "guarantee that the student is safe.")

        st.metric("Assessments not submitted by Day 35", f'{res["missing"]:.0%}')
        st.markdown("### 📈 Weekly Engagement Profile")
        fig, ax = plt.subplots(figsize=(10, 4))
        x = [f"Week {i}" for i in range(1, 6)]
        ax.plot(x, res["tc"], "o-", color="#0F3460", linewidth=2.5, markersize=8, label="Total clicks")
        ax.fill_between(x, res["tc"], alpha=0.1, color="#0F3460")
        ax.axhline(np.mean(res["tc"]), color="#E63946", linestyle="--", alpha=0.6, label="Average")
        ax.set_ylabel("Total clicks"); ax.set_title("Total Clicks per Week", fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

        st.markdown("### 🔍 SHAP Explanation — Why This Prediction?")
        st.markdown(
            "Red bars push the prediction towards **at-risk**; blue bars push it towards "
            "**not at-risk**. Values on the left are the student's *scaled* feature values."
        )
        if st.button("Compute SHAP explanation (about 30 seconds)", key="single_shap"):
            with st.spinner("Computing SHAP values..."):
                out = shap_waterfall(res["scaled"])
            if out is None:
                st.error("SHAP background data not found. Place data/X_train.npy and "
                         "data/y_train.npy next to app.py.")
            else:
                fig_shap, base = out
                st.pyplot(fig_shap); plt.close(fig_shap)
                st.info(f"The baseline (average model output on the background sample) is about {base:.2f}. "
                        "SHAP values explain this model's behaviour; they are not causal.")

# ═════════════════════════════════════════════════════════════
# TAB 2 — BATCH
# ═════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Batch Prediction — Upload a CSV File")
    st.markdown(
        "Upload one row per student. Required columns: the 25 weekly columns "
        "(`total_clicks_w1` … `quiz_clicks_w5`). Optional: `student_id`, `age_band`, "
        "`highest_education`, `imd_band`, `disability` (Y/N), `gender` (M/F), `region`, `late_ratio`, `missing_assessment_ratio`."
    )

    template = pd.DataFrame([{
        "student_id": "S001",
        **{c: 0 for c in WEEKLY_COLS},
        "age_band": "0-35", "highest_education": "A Level or Equivalent",
        "imd_band": "30-40%", "disability": "N", "gender": "F",
        "region": "Scotland", "late_ratio": 0.0, "missing_assessment_ratio": 0.0,
    }])
    st.download_button("📥  Download CSV Template", template.to_csv(index=False),
                       file_name="student_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload your completed CSV file", type=["csv"])

    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        st.markdown(f"**{len(df_up)} students loaded.**")
        st.dataframe(df_up.head(5))

        if st.button("🔍  Run Batch Prediction", key="batch_predict"):
            try:
                frame, warns = build_feature_frame(df_up.reset_index(drop=True))
                prob, label, scaled = predict_frame(frame)
                ids = df_up["student_id"].values if "student_id" in df_up.columns \
                    else np.arange(1, len(df_up) + 1)
                res_df = pd.DataFrame({
                    "student_id": ids,
                    "at_risk_prob": np.round(prob, 4),
                    "prediction": np.where(label == 1, "At-Risk", "Not flagged"),
                    "engagement_trend": frame["engagement_trend"].round(1).values,
                    "late_ratio": frame["late_ratio"].round(3).values,
                    "missing_assessment_ratio": frame["missing_assessment_ratio"].round(3).values,
                    "total_clicks_w1": frame["total_clicks_w1"].values,
                    "total_clicks_w5": frame["total_clicks_w5"].values,
                })
                st.session_state["batch"] = {"results": res_df, "scaled": scaled, "warnings": warns}
            except Exception as e:
                st.session_state.pop("batch", None)
                st.error(f"Could not process the file: {e}")

    batch = st.session_state.get("batch")
    if batch:
        res_df = batch["results"]
        for w in batch["warnings"]:
            st.warning(w)

        st.markdown("---")
        st.markdown("### Results")
        n_total = len(res_df)
        n_risk = int((res_df["prediction"] == "At-Risk").sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Total students", n_total)
        m2.metric("Flagged at-risk", n_risk, delta=f"{n_risk / n_total:.1%}", delta_color="off")
        m3.metric("Not flagged", n_total - n_risk, delta=f"{(n_total - n_risk) / n_total:.1%}", delta_color="off")

        f1, f2 = st.columns(2)
        with f1:
            show = st.multiselect("Show", ["At-Risk", "Not flagged"], default=["At-Risk", "Not flagged"])
        with f2:
            min_p = st.slider("Minimum at-risk probability", 0.0, 1.0, 0.0, 0.05)
        view = res_df[res_df["prediction"].isin(show) & (res_df["at_risk_prob"] >= min_p)] \
            .sort_values("at_risk_prob", ascending=False)

        def colour_row(row):
            colour = "#FFF0F0" if row["prediction"] == "At-Risk" else "#F0FFF4"
            return [f"background-color: {colour}"] * len(row)

        st.dataframe(view.style.apply(colour_row, axis=1), use_container_width=True)
        st.download_button("📥  Download Results CSV", res_df.to_csv(index=False),
                           file_name="at_risk_predictions.csv", mime="text/csv")

        st.markdown("### Risk Probability Distribution")
        fig_d, ax_d = plt.subplots(figsize=(10, 4))
        ax_d.hist(res_df["at_risk_prob"], bins=20, color="#0F3460", alpha=0.7, edgecolor="white")
        ax_d.axvline(THRESHOLD, color="#E63946", linestyle="--", linewidth=2, label=f"Threshold ({THRESHOLD:g})")
        ax_d.set_xlabel("At-risk probability"); ax_d.set_ylabel("Number of students")
        ax_d.legend(); ax_d.grid(True, alpha=0.3); plt.tight_layout()
        st.pyplot(fig_d); plt.close(fig_d)

        st.markdown("### Engagement Trend Across Uploaded Students")
        trend = res_df["engagement_trend"]
        t1, t2 = st.columns(2)
        t1.metric("Improving or steady (trend ≥ 0)", int((trend >= 0).sum()))
        t2.metric("Declining (trend < 0)", int((trend < 0).sum()))
        fig_t, ax_t = plt.subplots(figsize=(10, 4))
        ax_t.hist(trend, bins=25, color="#0F3460", alpha=0.7, edgecolor="white")
        ax_t.axvline(0, color="#8D99AE", linestyle="--", linewidth=2, label="No change")
        ax_t.set_xlabel("Engagement trend (Week 5 clicks − Week 1 clicks)")
        ax_t.set_ylabel("Number of students"); ax_t.legend(); ax_t.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig_t); plt.close(fig_t)

        st.markdown("### 🔍 Explain One Student (SHAP)")
        choice = st.selectbox("Choose a student", res_df["student_id"].tolist(), key="batch_pick")
        if st.button("Compute SHAP explanation (about 30 seconds)", key="batch_shap"):
            pos = int(np.where(res_df["student_id"].values == choice)[0][0])
            with st.spinner("Computing SHAP values..."):
                out = shap_waterfall(batch["scaled"][pos:pos + 1])
            if out is None:
                st.error("SHAP background data not found. Place data/X_train.npy and "
                         "data/y_train.npy next to app.py.")
            else:
                fig_s, base = out
                st.pyplot(fig_s); plt.close(fig_s)

# ═════════════════════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ═════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### What does the model rely on?")
    imp_path = os.path.join(OUTPUT_DIR, "shap_feature_importance_corrected.csv")
    if os.path.exists(imp_path):
        imp = pd.read_csv(imp_path).sort_values("mean_abs_shap", ascending=False).head(15)
        fig_i, ax_i = plt.subplots(figsize=(9, 6))
        ax_i.barh(imp["feature"][::-1], imp["mean_abs_shap"][::-1], color="#0F3460", alpha=0.85)
        ax_i.set_xlabel("Mean |SHAP value| (test set)")
        ax_i.set_title("Top 15 features by global SHAP importance", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig_i); plt.close(fig_i)
    else:
        st.info("Place shap_feature_importance.csv (from the SHAP notebook) in the models/ "
                "folder to show global feature importance here.")
    st.markdown(
        "**How to read this.** SHAP importance describes how the trained model behaves; it does "
        "not show that a feature causes academic risk. Several demographic variables (for example "
        "highest education, gender and deprivation band) are model inputs. Instructors should not "
        "act on a flag because of demographic characteristics, and fairness across groups has not "
        "been evaluated."
    )
    st.markdown(
        "**Known limitations.** Single-institution training data (OULAD); a substantial share of "
        "at-risk students is not flagged at the default threshold; the lowered threshold "
        f"({THRESHOLD:g}) catches more at-risk students but raises false alarms; the dashboard "
        "has been demonstrated with example data only."
    )

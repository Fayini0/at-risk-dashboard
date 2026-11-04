# ─────────────────────────────────────────────────────────────────────────────
# app.py — At-Risk Student Early Warning Dashboard (research prototype)
# Fika Samkelo Fayini (202207134) — BScHons Computer Science, Sol Plaatje University
#
# Folder layout expected next to this file:
#   models/final_model.keras      trained CNN-BiLSTM (Notebook 02)
#   models/scaler.pkl             StandardScaler fitted on the training tensors (Notebook 02)
#   models/final_results.json     (optional) metrics saved by Notebook 02
#   data/X_train.npy, y_train.npy SMOTE-balanced training tensors (Notebook 01) — SHAP background
#
# Run with:  streamlit run app.py
# ─────────────────────────────────────────────────────────────────────────────

import io
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from tensorflow.keras.models import load_model

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "final_model.keras"
SCALER_PATH = BASE / "models" / "scaler.pkl"
RESULTS_PATH = BASE / "models" / "final_results.json"
TRAIN_X_PATH = BASE / "data" / "X_train.npy"
TRAIN_Y_PATH = BASE / "data" / "y_train.npy"

# ── Model constants (must match Notebooks 01 and 02) ─────────────────────────
N_WEEKS = 5
N_FEATURES = 30  # 5 weekly + 25 static features at every time step

# Decision threshold: highest threshold giving At-Risk recall >= 0.70 on the test set
# (Notebook 02, threshold analysis). It is exploratory, so the sidebar lets the user change it.
DEFAULT_THRESHOLD = 0.383

WEEKLY_FEATURES = [
    "total_clicks", "unique_activities", "forum_clicks",
    "resource_clicks", "quiz_clicks",
]
STATIC_FEATURES = [
    "age_band", "highest_education", "disability_enc", "gender_enc", "imd_band",
    "engagement_trend", "consistency_score", "late_ratio",
    "delta_w1_w2", "delta_w2_w3", "delta_w3_w4", "delta_w4_w5",
]
REGIONS = [
    "East Anglian Region", "East Midlands Region", "Ireland", "London Region",
    "North Region", "North Western Region", "Scotland", "South East Region",
    "South Region", "South West Region", "Wales", "West Midlands Region",
    "Yorkshire Region",
]

# Category order = the alphabetical order produced by scikit-learn's LabelEncoder in
# Notebook 01 (Cell 16). The integer code of a category is its position in these lists.
# NOTE: the earlier version of this app used a different (ordinal) order for
# highest_education, which did not match the training encoding.
AGE_BANDS = ["0-35", "35-55", "55<="]
EDUCATION_LEVELS = [
    "A Level or Equivalent", "HE Qualification", "Lower Than A Level",
    "No Formal quals", "Post Graduate Qualification",
]
IMD_BANDS = [
    "0-10%", "10-20", "20-30%", "30-40%", "40-50%",
    "50-60%", "60-70%", "70-80%", "80-90%", "90-100%",
]

WEEKLY_COLUMNS = [f"{f}_w{w}" for f in WEEKLY_FEATURES for w in range(1, N_WEEKS + 1)]
DEDUP_NAMES = WEEKLY_COLUMNS + STATIC_FEATURES + [f"region_{r}" for r in REGIONS]  # 50 names


# ═════════════════════════════════════════════════════════════════════════════
# PURE HELPER FUNCTIONS (no Streamlit calls — easy to test)
# ═════════════════════════════════════════════════════════════════════════════
def _norm(value) -> str:
    return str(value).strip().lower().replace("%", "").replace(" ", "")


def encode_category(value, categories):
    """Return the training-time integer code for a category label.
    Accepts the raw OULAD label or an already-encoded integer."""
    if isinstance(value, (int, np.integer)) or (
        isinstance(value, (float, np.floating)) and float(value).is_integer()
    ):
        code = int(value)
        if 0 <= code < len(categories):
            return code
        raise ValueError(f"Encoded value {code} is outside 0–{len(categories) - 1}.")
    wanted = _norm(value)
    for code, label in enumerate(categories):
        if _norm(label) == wanted:
            return code
    raise ValueError(f"'{value}' is not one of: {', '.join(categories)}")


def parse_binary(value, true_values, false_values, name):
    v = _norm(value)
    if v in true_values:
        return 1
    if v in false_values:
        return 0
    raise ValueError(f"{name}: '{value}' not understood.")


def parse_region(value):
    wanted = _norm(str(value).replace("region_", ""))
    for r in REGIONS:
        if _norm(r) == wanted or _norm(r).replace("region", "") == wanted:
            return r
    raise ValueError(f"Region '{value}' is not one of the OULAD regions.")


def derive_features(total_clicks):
    """Derived features, computed exactly as in Notebook 01.
    consistency_score is the SAMPLE standard deviation (pandas .std(), ddof=1)."""
    tc = [float(x) for x in total_clicks]
    return {
        "engagement_trend": tc[4] - tc[0],
        "consistency_score": float(np.std(tc, ddof=1)),
        "delta_w1_w2": tc[1] - tc[0],
        "delta_w2_w3": tc[2] - tc[1],
        "delta_w3_w4": tc[3] - tc[2],
        "delta_w4_w5": tc[4] - tc[3],
    }


def row_to_tensor(row: dict) -> np.ndarray:
    """Flat feature dict -> (5, 30) tensor in the same layout as Notebook 01:
    [5 weekly features] + [12 static/derived] + [13 region indicators] at every week."""
    tensor = np.zeros((N_WEEKS, N_FEATURES), dtype=np.float64)
    for w in range(N_WEEKS):
        weekly = [float(row[f"{f}_w{w + 1}"]) for f in WEEKLY_FEATURES]
        static = [float(row[f]) for f in STATIC_FEATURES]
        regions = [float(row[f"region_{r}"]) for r in REGIONS]
        tensor[w, :] = weekly + static + regions
    return tensor


def scale_tensors(tensors: np.ndarray, scaler) -> np.ndarray:
    n = tensors.shape[0]
    flat = tensors.reshape(-1, N_FEATURES)
    return scaler.transform(flat).reshape(n, N_WEEKS, N_FEATURES)


def dedup_tensors(tensors_scaled: np.ndarray) -> np.ndarray:
    """(n, 5, 30) -> (n, 50): 25 weekly values + one copy of the 25 static values."""
    n = tensors_scaled.shape[0]
    return np.concatenate(
        [tensors_scaled[:, :, 0:5].reshape(n, -1), tensors_scaled[:, 0, 5:30]], axis=1
    )


def build_row_from_form(weekly: dict, demo: dict, late_ratio: float) -> dict:
    """weekly: {feature: [5 values]}; demo: encoded demographics + region."""
    row = {}
    for f in WEEKLY_FEATURES:
        for w in range(N_WEEKS):
            row[f"{f}_w{w + 1}"] = weekly[f][w]
    row.update(derive_features(weekly["total_clicks"]))
    row.update(demo)
    row["late_ratio"] = float(late_ratio)
    return row


def build_row_from_csv(rec: pd.Series) -> dict:
    """Convert one CSV record (raw OULAD labels or encoded values) into a feature dict."""
    missing = [c for c in WEEKLY_COLUMNS if c not in rec.index or pd.isna(rec[c])]
    if missing:
        raise ValueError(f"missing weekly value(s): {', '.join(missing[:4])}…")
    row = {c: float(rec[c]) for c in WEEKLY_COLUMNS}
    row.update(derive_features([row[f"total_clicks_w{w}"] for w in range(1, 6)]))

    def pick(*names):
        for n in names:
            if n in rec.index and not pd.isna(rec[n]):
                return rec[n]
        raise ValueError(f"missing column: {' or '.join(names)}")

    row["age_band"] = encode_category(pick("age_band"), AGE_BANDS)
    row["highest_education"] = encode_category(pick("highest_education"), EDUCATION_LEVELS)
    row["imd_band"] = encode_category(pick("imd_band"), IMD_BANDS)
    row["disability_enc"] = parse_binary(
        pick("disability", "disability_enc"), {"y", "yes", "1", "1.0", "true"},
        {"n", "no", "0", "0.0", "false"}, "disability")
    row["gender_enc"] = parse_binary(
        pick("gender", "gender_enc"), {"m", "male", "1", "1.0"},
        {"f", "female", "0", "0.0"}, "gender")
    region = parse_region(pick("region"))
    for r in REGIONS:
        row[f"region_{r}"] = 1 if r == region else 0
    row["late_ratio"] = float(rec["late_ratio"]) if (
        "late_ratio" in rec.index and not pd.isna(rec["late_ratio"])) else 0.0
    if not 0.0 <= row["late_ratio"] <= 1.0:
        raise ValueError("late_ratio must be between 0 and 1.")
    return row


def make_template_csv() -> bytes:
    """A one-row example CSV using raw OULAD labels."""
    example = {"student_id": "S001"}
    base_clicks = {"total_clicks": [120, 95, 80, 60, 40], "unique_activities": [4, 4, 3, 3, 2],
                   "forum_clicks": [30, 20, 15, 10, 5], "resource_clicks": [50, 40, 35, 25, 15],
                   "quiz_clicks": [10, 8, 6, 4, 2]}
    for f in WEEKLY_FEATURES:
        for w in range(N_WEEKS):
            example[f"{f}_w{w + 1}"] = base_clicks[f][w]
    example.update({
        "age_band": "35-55", "highest_education": "A Level or Equivalent",
        "imd_band": "30-40%", "disability": "N", "gender": "F",
        "region": "Scotland", "late_ratio": 0.2,
    })
    return pd.DataFrame([example]).to_csv(index=False).encode("utf-8")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE SET-UP
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="At-Risk Student Early Warning System",
    page_icon="🎓", layout="wide", initial_sidebar_state="expanded",
)
st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .stButton>button { background-color:#0F3460; color:white; border-radius:8px;
                       padding:.5rem 2rem; font-weight:bold; border:none; }
    .stButton>button:hover { background-color:#2E86AB; }
    .card { background:white; border-radius:12px; padding:1.2rem 1.5rem;
            box-shadow:0 2px 8px rgba(0,0,0,.08); border-left:5px solid #0F3460; margin-bottom:1rem; }
    .at-risk-card { background:#FFF0F0; border-left-color:#E63946; }
    .safe-card { background:#F0FFF4; border-left-color:#2D6A4F; }
    h1, h2 { color:#0F3460; } h3 { color:#1A5276; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# LOAD ARTEFACTS (cached) — scaler is loaded BEFORE the SHAP background needs it
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_resource(show_spinner="Preparing explanation background…")
def load_background(n_background=300):
    """Stratified sample of real (SMOTE-balanced) training rows, scaled and de-duplicated to 50 columns."""
    if not (TRAIN_X_PATH.exists() and TRAIN_Y_PATH.exists()):
        return None
    _, scaler_ = load_artifacts()
    x = np.load(TRAIN_X_PATH)
    y = np.load(TRAIN_Y_PATH)
    x_scaled = scale_tensors(x, scaler_)
    x_dedup = dedup_tensors(x_scaled)
    rng = np.random.RandomState(42)
    half = n_background // 2
    idx = np.concatenate([
        rng.choice(np.where(y == 0)[0], min(half, int((y == 0).sum())), replace=False),
        rng.choice(np.where(y == 1)[0], min(half, int((y == 1).sum())), replace=False),
    ])
    return x_dedup[idx]


def load_saved_metrics():
    if RESULTS_PATH.exists():
        try:
            return json.loads(RESULTS_PATH.read_text())
        except Exception:
            return None
    return None


try:
    model, scaler = load_artifacts()
except Exception as err:  # clear message instead of a stack trace
    st.error(
        "The model or scaler could not be loaded. Check that `models/final_model.keras` and "
        f"`models/scaler.pkl` exist next to app.py.\n\nDetails: {err}"
    )
    st.stop()

background = load_background()


def predict_dedup(x50: np.ndarray) -> np.ndarray:
    """Model wrapper used by SHAP: 50 unique features -> at-risk probability."""
    n = x50.shape[0]
    weekly = x50[:, :25].reshape(n, N_WEEKS, 5)
    static = np.repeat(x50[:, 25:][:, np.newaxis, :], N_WEEKS, axis=1)
    return model.predict(np.concatenate([weekly, static], axis=2), verbose=0).flatten()


@st.cache_resource(show_spinner=False)
def get_explainer():
    return shap.KernelExplainer(predict_dedup, background)


def shap_waterfall(tensor_scaled_1: np.ndarray):
    """tensor_scaled_1: (1, 5, 30). Returns a matplotlib figure."""
    explainer = get_explainer()
    x50 = dedup_tensors(tensor_scaled_1)
    np.random.seed(42)
    sv = explainer.shap_values(x50, nsamples=200, verbose=False)
    base = float(np.ravel(explainer.expected_value)[0])
    exp = shap.Explanation(values=np.asarray(sv)[0], base_values=base,
                           data=x50[0], feature_names=DEDUP_NAMES)
    fig = plt.figure(figsize=(10, 8))
    shap.waterfall_plot(exp, max_display=15, show=False)
    plt.tight_layout()
    return fig, base


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
saved = load_saved_metrics()
with st.sidebar:
    st.markdown("## 🎓 Early Warning System")
    st.markdown("**CNN-BiLSTM at-risk detection**")
    st.markdown("---")
    st.markdown(
        "Estimates whether a student is **at risk** of failing or withdrawing, using the "
        "first five weeks of LMS activity and basic demographic information."
    )
    st.markdown("---")
    threshold = st.slider(
        "Decision threshold", 0.30, 0.60, DEFAULT_THRESHOLD, 0.01,
        help="A student is flagged At-Risk when the predicted probability is at or above this value. "
             "Lower values catch more at-risk students but raise more false alarms.",
    )
    st.markdown("**Model details**")
    st.markdown("- Architecture: CNN-BiLSTM")
    st.markdown("- Dataset: OULAD (first five weeks)")
    st.markdown(f"- Default threshold: {DEFAULT_THRESHOLD}")
    if saved and "test_evaluation" in saved:
        st.markdown(f"- Test AUC-ROC (threshold-free): {saved['test_evaluation']['auc_roc']:.4f}")
    st.markdown("---")
    st.markdown(
        "**Fika Samkelo Fayini · 202207134**\n\nBScHons Computer Science\n\n"
        "Sol Plaatje University\n\nSupervisor: Dr F. Matsebula"
    )
    st.caption("Research prototype. Not connected to a live institutional LMS.")


def advice(prob, label, late_ratio, trend, quiz_total):
    """Plain-language prompts for the advisor. These are rules of thumb, not model output."""
    if label == 1:
        if late_ratio > 0.3:
            st.warning("⚠ High late-submission ratio. Consider asking about time management, "
                       "outside commitments and whether extra academic support is needed.")
        if trend < -50:
            st.warning("⚠ LMS activity fell markedly from Week 1 to Week 5. Check whether the "
                       "student is still actively taking part in the module.")
        if quiz_total == 0:
            st.warning("⚠ No quiz activity in Weeks 1–5, which may indicate disengagement "
                       "from assessment preparation.")
    else:
        st.success("✅ This student shows a pattern the model associates with lower risk. "
                   "Continue to monitor at the next checkpoint.")


def result_card(prob, label):
    if label == 1:
        st.markdown(
            f'<div class="card at-risk-card"><h3 style="color:#E63946;">⚠ AT-RISK</h3>'
            f'<p style="font-size:2rem;font-weight:bold;color:#E63946;">{prob:.1%}</p>'
            f'<p>Predicted probability of being at risk (threshold {threshold:.2f})</p></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="card safe-card"><h3 style="color:#2D6A4F;">✅ NOT FLAGGED</h3>'
            f'<p style="font-size:2rem;font-weight:bold;color:#2D6A4F;">{prob:.1%}</p>'
            f'<p>Predicted probability of being at risk (threshold {threshold:.2f})</p></div>',
            unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═════════════════════════════════════════════════════════════════════════════
st.title("🎓 At-Risk Student Early Warning System")
st.markdown(
    "Identify students who may be at risk **by Week 5** of a module, with SHAP explanations of "
    "which features drove each prediction. This tool supports, and does not replace, the "
    "judgement of academic staff."
)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📋  Single student", "📁  Batch (CSV upload)", "ℹ️  About this model"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — SINGLE STUDENT
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Enter student data")
    st.markdown("#### 📊 Weekly LMS engagement")

    header = st.columns(5)
    for i in range(5):
        header[i].markdown(f"**Week {i + 1}**")

    labels = {
        "total_clicks": "Total clicks",
        "unique_activities": "Unique activity types",
        "forum_clicks": "Forum clicks",
        "resource_clicks": "Resource clicks",
        "quiz_clicks": "Quiz clicks",
    }
    weekly = {}
    for f in WEEKLY_FEATURES:
        st.markdown(f"**{labels[f]}**")
        cols = st.columns(5)
        weekly[f] = [
            cols[i].number_input(f"{f}_w{i + 1}", min_value=0, value=0, step=1,
                                 label_visibility="collapsed", key=f"{f}_{i}")
            for i in range(5)
        ]

    st.markdown("---")
    st.markdown("#### 👤 Demographics")
    st.caption("These fields were part of the training data. Please read the fairness note under "
               "“About this model”.")
    d1, d2, d3 = st.columns(3)
    with d1:
        age_band = st.selectbox("Age band", AGE_BANDS)
        highest_edu = st.selectbox("Highest education", EDUCATION_LEVELS)
    with d2:
        disability = st.selectbox("Disability", ["No", "Yes"])
        gender = st.selectbox("Gender", ["Female", "Male"])
    with d3:
        imd_band = st.selectbox("IMD band (deprivation)", IMD_BANDS)
        region = st.selectbox("Region", REGIONS)

    st.markdown("---")
    st.markdown("#### 📝 Assessment behaviour (Weeks 1–5)")
    late_ratio = st.slider(
        "Late-submission ratio (0 = never late, 1 = always late)", 0.0, 1.0, 0.0, 0.05,
        help="Proportion of assessments due by Day 35 that were submitted after the deadline "
             "(counting only submissions made by Day 35).")
    st.markdown("---")

    if st.button("🔍  Predict at-risk status", key="single_predict"):
        warn = []
        if any(weekly["total_clicks"][i] < weekly["forum_clicks"][i] + weekly["resource_clicks"][i]
               + weekly["quiz_clicks"][i] for i in range(5)):
            warn.append("In at least one week, forum + resource + quiz clicks exceed total clicks. "
                        "Please check the values.")
        if any(u > 20 for u in weekly["unique_activities"]):
            warn.append("Unique activity types above 20 in a week is unusual. Please check the values.")
        for m in warn:
            st.warning(m)

        demo = {
            "age_band": encode_category(age_band, AGE_BANDS),
            "highest_education": encode_category(highest_edu, EDUCATION_LEVELS),
            "imd_band": encode_category(imd_band, IMD_BANDS),
            "disability_enc": 1 if disability == "Yes" else 0,
            "gender_enc": 1 if gender == "Male" else 0,
        }
        for r in REGIONS:
            demo[f"region_{r}"] = 1 if r == region else 0

        row = build_row_from_form(weekly, demo, late_ratio)
        tensor_scaled = scale_tensors(row_to_tensor(row)[np.newaxis, ...], scaler)
        prob = float(model.predict(tensor_scaled, verbose=0).flatten()[0])
        label = int(prob >= threshold)

        st.markdown("---")
        st.markdown("### 🎯 Prediction")
        c1, c2, c3 = st.columns(3)
        with c1:
            result_card(prob, label)
        with c2:
            st.markdown(f'<div class="card"><h3>Engagement trend</h3>'
                        f'<p style="font-size:1.5rem;font-weight:bold;">{row["engagement_trend"]:+.0f}</p>'
                        f'<p>Week 5 clicks minus Week 1 clicks</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="card"><h3>Late-submission ratio</h3>'
                        f'<p style="font-size:1.5rem;font-weight:bold;">{late_ratio:.0%}</p>'
                        f'<p>Late submissions by Day 35</p></div>', unsafe_allow_html=True)

        st.markdown("### 💬 Prompts for the academic advisor")
        advice(prob, label, late_ratio, row["engagement_trend"], sum(weekly["quiz_clicks"]))

        st.markdown("### 📈 Weekly engagement profile")
        tc = weekly["total_clicks"]
        fig, ax = plt.subplots(figsize=(10, 4))
        x = [f"Week {i}" for i in range(1, 6)]
        ax.plot(x, tc, "o-", color="#0F3460", linewidth=2.5, markersize=8, label="Total clicks")
        ax.fill_between(x, tc, alpha=0.1, color="#0F3460")
        ax.axhline(np.mean(tc), color="#E63946", linestyle="--", alpha=0.6, label="Average")
        ax.set_ylabel("Total clicks"); ax.set_title("Total clicks per week", fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

        st.markdown("### 🔍 SHAP explanation — why this prediction?")
        if background is None:
            st.info("SHAP explanations need `data/X_train.npy` and `data/y_train.npy` next to app.py.")
        else:
            st.markdown("Red bars pushed the prediction towards **at risk**; blue bars pushed it "
                        "towards **not at risk**. SHAP values are approximations and describe the "
                        "model's behaviour, not causes.")
            with st.spinner("Computing SHAP explanation (about 30 seconds)…"):
                fig_shap, base = shap_waterfall(tensor_scaled)
                st.pyplot(fig_shap); plt.close(fig_shap)
            st.info(f"The bars show how each feature moved the prediction away from the baseline "
                    f"probability of about {base:.2f} (the average prediction on the background sample).")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — BATCH
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Batch prediction — upload a CSV file")
    st.markdown(
        "Each row is one student. Required columns: the 25 weekly columns "
        "(e.g. `total_clicks_w1` … `quiz_clicks_w5`), `age_band`, `highest_education`, `imd_band`, "
        "`disability` (Y/N), `gender` (M/F) and `region`. `late_ratio` is optional (defaults to 0). "
        "Derived features are calculated automatically. Use the OULAD labels shown in the template."
    )
    st.download_button("📥  Download CSV template", make_template_csv(),
                       file_name="student_template.csv", mime="text/csv")
    uploaded = st.file_uploader("Upload your completed CSV", type=["csv"])

    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        st.markdown(f"**{len(df_up)} rows loaded.**")
        st.dataframe(df_up.head(5))

        if st.button("🔍  Run batch prediction", key="batch_predict"):
            rows, ok_index, errors = [], [], {}
            for i, rec in df_up.iterrows():
                try:
                    rows.append(build_row_from_csv(rec)); ok_index.append(i)
                except Exception as e:
                    errors[i] = str(e)

            out = pd.DataFrame({
                "student_id": [df_up.loc[i, "student_id"] if "student_id" in df_up.columns else i
                               for i in df_up.index]}, index=df_up.index)
            out["at_risk_probability"] = np.nan
            out["prediction"] = ""
            out["engagement_trend"] = np.nan
            out["late_ratio"] = np.nan
            out["note"] = ""

            if rows:
                tensors = np.stack([row_to_tensor(r) for r in rows])
                tensors_scaled = scale_tensors(tensors, scaler)
                probs = model.predict(tensors_scaled, verbose=0, batch_size=256).flatten()
                for k, i in enumerate(ok_index):
                    out.loc[i, "at_risk_probability"] = round(float(probs[k]), 4)
                    out.loc[i, "prediction"] = "At-Risk" if probs[k] >= threshold else "Not flagged"
                    out.loc[i, "engagement_trend"] = rows[k]["engagement_trend"]
                    out.loc[i, "late_ratio"] = round(rows[k]["late_ratio"], 3)
                st.session_state["batch_scaled"] = dict(zip(ok_index, tensors_scaled))
            else:
                st.session_state["batch_scaled"] = {}
            for i, msg in errors.items():
                out.loc[i, "prediction"] = "Error"
                out.loc[i, "note"] = msg
            st.session_state["batch_results"] = out
            st.session_state["batch_threshold"] = threshold

    results = st.session_state.get("batch_results")
    if results is not None:
        st.markdown("---")
        st.markdown("### Results")
        if st.session_state.get("batch_threshold") != threshold:
            st.info("The threshold has changed since this batch was run. Run the batch again to apply it.")
        n_total = len(results)
        n_risk = int((results["prediction"] == "At-Risk").sum())
        n_ok = int((results["prediction"] == "Not flagged").sum())
        n_err = int((results["prediction"] == "Error").sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", n_total)
        m2.metric("At-Risk", n_risk, delta=f"{n_risk / max(n_total, 1):.1%}", delta_color="off")
        m3.metric("Not flagged", n_ok, delta=f"{n_ok / max(n_total, 1):.1%}", delta_color="off")
        m4.metric("Rows with errors", n_err)
        if n_err:
            st.warning("Some rows could not be processed. See the `note` column.")

        def colour_row(r):
            if r["prediction"] == "At-Risk":
                return ["background-color: #FFF0F0"] * len(r)
            if r["prediction"] == "Not flagged":
                return ["background-color: #F0FFF4"] * len(r)
            if r["prediction"] == "Error":
                return ["background-color: #FFF8E1"] * len(r)
            return [""] * len(r)

        st.dataframe(results.style.apply(colour_row, axis=1))
        st.download_button("📥  Download results CSV", results.to_csv(index=False).encode("utf-8"),
                           file_name="at_risk_predictions.csv", mime="text/csv")

        valid = results[results["at_risk_probability"].notna()]
        if len(valid) > 0:
            st.markdown("### Risk probability distribution")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(valid["at_risk_probability"], bins=20, color="#0F3460", alpha=0.7, edgecolor="white")
            ax.axvline(threshold, color="#E63946", linestyle="--", linewidth=2,
                       label=f"Threshold ({threshold:.2f})")
            ax.set_xlabel("At-risk probability"); ax.set_ylabel("Number of students")
            ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
            st.pyplot(fig); plt.close(fig)

            st.markdown("### Engagement trend across the batch")
            tr = valid["engagement_trend"].astype(float)
            t1, t2 = st.columns(2)
            t1.metric("Stable or growing (trend ≥ 0)", int((tr >= 0).sum()))
            t2.metric("Declining (trend < 0)", int((tr < 0).sum()))
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.hist(tr, bins=25, color="#0F3460", alpha=0.7, edgecolor="white")
            ax2.axvline(0, color="#8D99AE", linestyle="--", linewidth=2, label="No change")
            ax2.set_xlabel("Engagement trend (Week 5 clicks − Week 1 clicks)")
            ax2.set_ylabel("Number of students"); ax2.legend(); ax2.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

            if background is not None and st.session_state.get("batch_scaled"):
                st.markdown("### Explain one student")
                options = list(st.session_state["batch_scaled"].keys())
                pick = st.selectbox(
                    "Select a row", options,
                    format_func=lambda i: f"Row {i} — student {results.loc[i, 'student_id']}")
                if st.button("🔍  Explain this student", key="batch_shap"):
                    with st.spinner("Computing SHAP explanation…"):
                        fig_s, base = shap_waterfall(st.session_state["batch_scaled"][pick][np.newaxis, ...])
                        st.pyplot(fig_s); plt.close(fig_s)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### About this model")
    st.markdown(
        "The model is a CNN-BiLSTM trained on the Open University Learning Analytics Dataset "
        "(OULAD). Each student is represented by five weekly time steps with 30 features per step. "
        "“At risk” means a final result of Fail or Withdrawn."
    )
    if saved and "test_evaluation" in saved:
        te = saved["test_evaluation"]
        st.markdown("**Test-set results at the default 0.5 threshold** (from `final_results.json`)")
        c1, c2, c3 = st.columns(3)
        c1.metric("AUC-ROC", f"{te['auc_roc']:.4f}")
        c2.metric("At-risk students missed", f"{te['false_negative_rate']:.1%}")
        c3.metric("Test students", int(te["true_negatives"] + te["false_positives"]
                                      + te["false_negatives"] + te["true_positives"]))
    st.markdown("#### Limitations you should know about")
    st.markdown(
        "- **Moderate accuracy.** The model separates at-risk from other students only moderately well. "
        "It will miss some at-risk students and flag some who are not at risk.\n"
        "- **Threshold trade-off.** Lowering the threshold catches more at-risk students but produces "
        "more false alarms. The default was chosen using test-set predictions, so treat it as exploratory.\n"
        "- **Fairness.** Demographic fields (such as gender, education, deprivation band, region, "
        "disability and age) were used as predictors and some influenced predictions. Do not use a "
        "flag as grounds for treating a student differently on the basis of these characteristics.\n"
        "- **Single data source.** The model was trained on one UK distance-learning institution and "
        "has not been validated at other institutions or on live LMS data.\n"
        "- **Decision support only.** A prediction is a prompt for a conversation, not a decision about a student."
    )

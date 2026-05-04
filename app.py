import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding: 2rem 3rem; }
    h1 { color: #a78bfa; font-weight: 800; }
    h2, h3 { color: #c4b5fd; }
    .stSelectbox label, .stSlider label, .stNumberInput label,
    .stRadio label, p { color: #cbd5e1 !important; }
    .risk-high { background:#3b1a1a; border:1px solid #f87171;
                 border-radius:12px; padding:1.5rem; text-align:center; }
    .risk-low  { background:#1a3b2a; border:1px solid #4ade80;
                 border-radius:12px; padding:1.5rem; text-align:center; }
    .section-card { background:#1a1d27; border-radius:14px;
                    padding:1.5rem 2rem; margin-bottom:1rem;
                    border:1px solid #2d2f3e; }
    .stButton > button {
        background: linear-gradient(135deg,#7c3aed,#a78bfa);
        color: white; border: none; border-radius: 10px;
        font-size: 1.1rem; font-weight: 700;
        padding: 0.7rem 2.5rem; width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model & scaler from pickle ──────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    rf     = joblib.load("random_forest_attrition.pkl")
    scaler = joblib.load("scaler_attrition.pkl")
    return rf, scaler

try:
    rf, scaler = load_artifacts()
except FileNotFoundError:
    st.error("❌ Could not find `random_forest_attrition.pkl` or `scaler_attrition.pkl`. "
             "Place them in the same folder as this app and restart.")
    st.stop()

# Feature order must match training
FEATURE_COLS = [
    'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
    'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
    'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
    'OverTime', 'PercentSalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
]

# Encoding maps (matches LabelEncoder alphabetical fit order from training)
ENCODERS = {
    'BusinessTravel': {'Non-Travel': 0, 'Travel_Frequently': 1, 'Travel_Rarely': 2},
    'Department':     {'Human Resources': 0, 'Research & Development': 1, 'Sales': 2},
    'EducationField': {'Human Resources': 0, 'Life Sciences': 1, 'Marketing': 2,
                       'Medical': 3, 'Other': 4, 'Technical Degree': 5},
    'Gender':         {'Female': 0, 'Male': 1},
    'JobRole':        {'Healthcare Representative': 0, 'Human Resources': 1,
                       'Laboratory Technician': 2, 'Manager': 3,
                       'Manufacturing Director': 4, 'Research Director': 5,
                       'Research Scientist': 6, 'Sales Executive': 7,
                       'Sales Representative': 8},
    'MaritalStatus':  {'Divorced': 0, 'Married': 1, 'Single': 2},
    'OverTime':       {'No': 0, 'Yes': 1},
}

RATING = {1: "1 – Low", 2: "2 – Medium", 3: "3 – High", 4: "4 – Very High"}
EDU    = {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 HR Employee Attrition Predictor")
st.markdown("Fill in the employee details to predict their likelihood of leaving.")
st.markdown("---")

# ── Form ──────────────────────────────────────────────────────────────────────
with st.form("prediction_form"):

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 👤 Personal Information")
    c1, c2, c3 = st.columns(3)
    age            = c1.slider("Age", 18, 60, 35)
    gender         = c2.selectbox("Gender", ["Female", "Male"])
    marital_status = c3.selectbox("Marital Status", ["Divorced", "Married", "Single"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 💼 Job Details")
    c1, c2, c3 = st.columns(3)
    department      = c1.selectbox("Department", ["Human Resources", "Research & Development", "Sales"])
    job_role        = c2.selectbox("Job Role", list(ENCODERS['JobRole'].keys()))
    job_level       = c3.selectbox("Job Level", [1, 2, 3, 4, 5])
    c1, c2, c3 = st.columns(3)
    business_travel = c1.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    overtime        = c2.selectbox("OverTime", ["No", "Yes"])
    education_field = c3.selectbox("Education Field", list(ENCODERS['EducationField'].keys()))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 💰 Compensation & Benefits")
    c1, c2, c3 = st.columns(3)
    monthly_income      = c1.number_input("Monthly Income ($)", 1009, 19999, 5000, step=100)
    daily_rate          = c2.number_input("Daily Rate ($)", 102, 1499, 800, step=10)
    hourly_rate         = c3.number_input("Hourly Rate ($)", 30, 100, 65)
    c1, c2, c3 = st.columns(3)
    monthly_rate        = c1.number_input("Monthly Rate ($)", 2094, 26999, 14000, step=100)
    percent_salary_hike = c2.slider("Percent Salary Hike (%)", 11, 25, 14)
    stock_option_level  = c3.selectbox("Stock Option Level", [0, 1, 2, 3])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### ⭐ Satisfaction & Ratings")
    c1, c2, c3, c4 = st.columns(4)
    job_satisfaction          = c1.selectbox("Job Satisfaction",          [1,2,3,4], index=2, format_func=lambda x: RATING[x])
    environment_satisfaction  = c2.selectbox("Environment Satisfaction",  [1,2,3,4], index=2, format_func=lambda x: RATING[x])
    relationship_satisfaction = c3.selectbox("Relationship Satisfaction", [1,2,3,4], index=2, format_func=lambda x: RATING[x])
    work_life_balance         = c4.selectbox("Work-Life Balance",         [1,2,3,4], index=2, format_func=lambda x: RATING[x])
    c1, c2, c3 = st.columns(3)
    job_involvement    = c1.selectbox("Job Involvement",    [1,2,3,4], index=2, format_func=lambda x: RATING[x])
    performance_rating = c2.selectbox("Performance Rating", [3, 4],
                                       format_func=lambda x: f"{x} – {'Excellent' if x==3 else 'Outstanding'}")
    education          = c3.selectbox("Education Level",   [1,2,3,4,5], index=2, format_func=lambda x: EDU[x])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🏢 Work History")
    c1, c2, c3 = st.columns(3)
    total_working_years        = c1.slider("Total Working Years",        0, 40, 10)
    years_at_company           = c2.slider("Years at Company",           0, 40,  5)
    years_in_current_role      = c3.slider("Years in Current Role",      0, 18,  3)
    c1, c2, c3 = st.columns(3)
    years_since_last_promotion = c1.slider("Years Since Last Promotion", 0, 15,  1)
    years_with_curr_manager    = c2.slider("Years with Current Manager", 0, 17,  3)
    num_companies_worked       = c3.slider("Num. Companies Worked",      0,  9,  2)
    c1, c2 = st.columns(2)
    training_times_last_year   = c1.slider("Training Times Last Year",   0,  6,  3)
    distance_from_home         = c2.slider("Distance from Home (km)",    1, 29,  7)
    st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("🔮 Predict Attrition")

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    raw = {
        'Age': age,
        'BusinessTravel': ENCODERS['BusinessTravel'][business_travel],
        'DailyRate': daily_rate,
        'Department': ENCODERS['Department'][department],
        'DistanceFromHome': distance_from_home,
        'Education': education,
        'EducationField': ENCODERS['EducationField'][education_field],
        'EnvironmentSatisfaction': environment_satisfaction,
        'Gender': ENCODERS['Gender'][gender],
        'HourlyRate': hourly_rate,
        'JobInvolvement': job_involvement,
        'JobLevel': job_level,
        'JobRole': ENCODERS['JobRole'][job_role],
        'JobSatisfaction': job_satisfaction,
        'MaritalStatus': ENCODERS['MaritalStatus'][marital_status],
        'MonthlyIncome': monthly_income,
        'MonthlyRate': monthly_rate,
        'NumCompaniesWorked': num_companies_worked,
        'OverTime': ENCODERS['OverTime'][overtime],
        'PercentSalaryHike': percent_salary_hike,
        'PerformanceRating': performance_rating,
        'RelationshipSatisfaction': relationship_satisfaction,
        'StockOptionLevel': stock_option_level,
        'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': training_times_last_year,
        'WorkLifeBalance': work_life_balance,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_current_role,
        'YearsSinceLastPromotion': years_since_last_promotion,
        'YearsWithCurrManager': years_with_curr_manager,
    }

    row    = pd.DataFrame([raw])[FEATURE_COLS]
    row_sc = scaler.transform(row)

    prob  = rf.predict_proba(row_sc)[0][1]
    pred  = rf.predict(row_sc)[0]
    pct   = round(prob * 100, 1)
    color = "#f87171" if pred == 1 else "#4ade80"
    css   = "risk-high" if pred == 1 else "risk-low"
    label = "🔴 High Risk — Will Leave" if pred == 1 else "🟢 Low Risk — Will Stay"

    st.markdown("---")
    st.markdown("## 📊 Prediction Result")

    col_res, col_gauge, col_feat = st.columns([1, 1.2, 1.5])

    with col_res:
        st.markdown(f"""
        <div class="{css}">
          <div style="font-size:1rem;color:#94a3b8;margin-bottom:.4rem">Attrition Prediction</div>
          <div style="font-size:1.8rem;font-weight:900;color:{color}">{label}</div>
          <div style="font-size:2rem;font-weight:800;color:{color};margin-top:.4rem">{pct}%</div>
          <div style="color:#94a3b8;font-size:.9rem">probability of leaving</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("Attrition Probability", f"{pct}%")

    with col_gauge:
        fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(aspect='equal'),
                               facecolor='#1a1d27')
        theta = np.linspace(np.pi, 0, 300)
        ax.plot(np.cos(theta), np.sin(theta), lw=18, color='#2d2f3e', solid_capstyle='round')
        theta_fill = np.linspace(np.pi, np.pi - prob * np.pi, 300)
        ax.plot(np.cos(theta_fill), np.sin(theta_fill), lw=18, color=color, solid_capstyle='round')
        ax.text(0, 0.1,  f"{pct}%",        ha='center', va='center', fontsize=28, fontweight='bold', color=color)
        ax.text(0, -0.2, "Attrition Risk", ha='center', va='center', fontsize=11, color='#94a3b8')
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.5, 1.3)
        ax.axis('off')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_feat:
        if hasattr(rf, 'feature_importances_'):
            feat_imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False).head(10)
            st.markdown("**🔑 Top 10 Feature Importances**")
            fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor='#1a1d27')
            ax2.set_facecolor('#1a1d27')
            bar_colors = ['#7c6cf4' if i < 3 else '#a78bfa' if i < 6 else '#c4b5fd' for i in range(10)]
            ax2.barh(feat_imp.index[::-1], feat_imp.values[::-1], color=bar_colors[::-1], edgecolor='none')
            ax2.tick_params(colors='#cbd5e1', labelsize=9)
            ax2.set_xlabel('Importance', color='#94a3b8', fontsize=9)
            for spine in ax2.spines.values():
                spine.set_edgecolor('#2d2f3e')
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()

    st.markdown("---")
    if pred == 1:
        st.warning(f"⚠️ **{pct}% probability of leaving.** Consider reviewing compensation, overtime load, and job satisfaction.")
    else:
        st.success(f"✅ **Only {pct}% probability of leaving.** Employee appears stable — keep fostering growth and engagement.")
import streamlit as st
import pandas as pd
import joblib

# -------------------- Page config (WAJIB PALING ATAS) --------------------
st.set_page_config(
    page_title="Sistem Pendukung Keputusan Pemberian Pinjaman",
    page_icon="💳",
    layout="wide",
)

# -------------------- Load model --------------------
@st.cache_resource
def load_model():
    return joblib.load("credit_risk_rf_pipeline.pkl")

model = load_model()

# -------------------- Session state untuk hasil prediksi --------------------
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None
if "last_proba" not in st.session_state:
    st.session_state.last_proba = None
if "last_X" not in st.session_state:
    st.session_state.last_X = None

# -------------------- Custom CSS --------------------
st.markdown("""
    <style>
    .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; max-width: 1400px; }
    @media (max-width: 768px) {
        .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
        h1 { font-size: 1.8rem !important; }
        .section-header { font-size: 0.85rem !important; padding: 0.4rem 0.8rem !important; }
        .result-success, .result-error { font-size: 1.2rem !important; padding: 1rem !important; }
        .metric-value { font-size: 1.4rem !important; }
    }
    h1, h2, h3 { white-space: normal !important; overflow: visible !important; margin-top: 0.5rem !important; margin-bottom: 0.5rem !important; }
    .stNumberInput, .stSelectbox { margin-bottom: 0.3rem !important; }
    div[data-testid="column"] { padding: 0.5rem; }

    .info-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem; border-radius: 10px; color: white; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .result-success {
        background: linear-gradient(135deg, #0f9b0f 0%, #16a085 100%);
        padding: 1.5rem; border-radius: 12px; color: white; text-align: center;
        font-size: 1.5rem; font-weight: bold; box-shadow: 0 6px 12px rgba(0,0,0,0.15); margin: 1rem 0;
    }
    .result-error {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
        padding: 1.5rem; border-radius: 12px; color: white; text-align: center;
        font-size: 1.5rem; font-weight: bold; box-shadow: 0 6px 12px rgba(0,0,0,0.15); margin: 1rem 0;
    }
    .metric-card {
        background: #2c3e50; border: 2px solid #34495e; padding: 1rem; border-radius: 10px;
        text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #3498db; }
    .metric-label { font-size: 0.9rem; color: #bdc3c7; margin-top: 0.3rem; }

    .section-header {
        background: linear-gradient(90deg, #2c3e50 0%, #34495e 100%);
        color: white; padding: 0.5rem 1rem; border-radius: 8px; margin-bottom: 0.8rem;
        font-weight: 600; font-size: 0.95rem; border-left: 4px solid #3498db;
    }

    .stForm { border: none !important; }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        color: white; font-weight: 600; font-size: 1.1rem; padding: 0.6rem 1rem;
        border: none; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        background: linear-gradient(135deg, #34495e 0%, #5dade2 100%);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .element-container { margin-bottom: 0.3rem !important; }
    </style>
""", unsafe_allow_html=True)

# -------------------- Header --------------------
st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                    padding: 2rem 1.5rem;
                    border-radius: 15px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                    border: 2px solid #3498db;
                    border-top: 4px solid #3498db;">
            <div style="font-size: 3.5rem; margin-bottom: 0.5rem; 
                        filter: drop-shadow(0 0 15px rgba(52, 152, 219, 0.6));">
                💳
            </div>
            <h1 style="font-size: 2.5rem; font-weight: 800; 
                       background: linear-gradient(135deg, #3498db 0%, #5dade2 100%); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                       background-clip: text; margin-bottom: 0.5rem;
                       letter-spacing: 1px;">
                Sistem Pendukung Keputusan Pemberian Pinjaman
            </h1>
            <div style="width: 100px; height: 3px; 
                        background: linear-gradient(90deg, transparent, #3498db, transparent);
                        margin: 0.8rem auto;"></div>
            <p style="color: #bdc3c7; font-size: 1rem; margin-top: 0;
                      font-weight: 400; letter-spacing: 0.5px;">
                <span style="background: rgba(52, 152, 219, 0.2); padding: 0.3rem 0.8rem; 
                             border-radius: 20px; border: 1px solid rgba(52, 152, 219, 0.4);">
                    ⚡ Sistem Analisis Risiko Kredit Berbasis Machine Learning
                </span>
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

# -------------------- Form --------------------
with st.form("form_kredit", clear_on_submit=False):
    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

    with col1:
        st.markdown('<div class="section-header">📌 Data Pribadi</div>', unsafe_allow_html=True)
        person_age = st.number_input("Umur", min_value=18, max_value=100, value=25, key="age")
        person_income = st.number_input("Penghasilan/Tahun", min_value=0, value=50000, step=5000, key="income")
        person_emp_length = st.number_input("Lama Bekerja (thn)", min_value=0.0, value=2.0, step=0.5, key="emp")
        person_home_ownership = st.selectbox("Kepemilikan Rumah", ["RENT", "MORTGAGE", "OWN", "OTHER"], key="home")

    with col2:
        st.markdown('<div class="section-header">🧾 Riwayat Kredit</div>', unsafe_allow_html=True)
        default_label = st.selectbox("Riwayat Kredit Macet", ["No", "Yes"], key="default")
        cb_person_default_on_file = "Y" if default_label == "Yes" else "N"
        cb_person_cred_hist_length = st.number_input("Lama Riwayat (thn)", min_value=0, value=3, step=1, key="hist")

        st.markdown('<div class="section-header" style="margin-top: 1.2rem;">💰 Detail Pinjaman</div>', unsafe_allow_html=True)
        loan_amnt = st.number_input("Jumlah Pinjaman", min_value=0, value=10000, step=1000, key="amnt")
        loan_int_rate = st.number_input("Suku Bunga (%)", min_value=0.0, value=12.0, step=0.5, key="rate")

    with col3:
        st.markdown('<div class="section-header">📊 Informasi Tambahan</div>', unsafe_allow_html=True)
        loan_percent_income = st.number_input("% Pinjaman/Income", min_value=0.0, max_value=2.0, value=0.2, step=0.05, key="percent")
        loan_intent = st.selectbox(
            "Tujuan Pinjaman",
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
            key="intent"
        )
        loan_grade = st.selectbox("Grade Pinjaman", ["A", "B", "C", "D", "E", "F", "G"], key="grade")

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        submit = st.form_submit_button("🔍 PREDIKSI SEKARANG", use_container_width=True)

# -------------------- Prediction (compute + store state) --------------------
if submit:
    X = pd.DataFrame([{
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length
    }])

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None

    st.session_state.last_pred = pred
    st.session_state.last_proba = proba
    st.session_state.last_X = X

# -------------------- Render results (even after rerun) --------------------
if st.session_state.last_pred is not None:
    pred = st.session_state.last_pred
    proba = st.session_state.last_proba

    res_col1, res_col2, res_col3 = st.columns([2, 1, 1])

    with res_col1:
        if pred == 0:
            st.markdown('<div class="result-success">✅ LAYAK (Low Risk)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-error">❌ TIDAK LAYAK (High Risk)</div>', unsafe_allow_html=True)

    if proba is not None:
        with res_col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #27ae60;">{proba[0]*100:.1f}%</div>
                    <div class="metric-label">Probabilitas Low Risk</div>
                </div>
            """, unsafe_allow_html=True)
        with res_col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #e74c3c;">{proba[1]*100:.1f}%</div>
                    <div class="metric-label">Probabilitas High Risk</div>
                </div>
            """, unsafe_allow_html=True)



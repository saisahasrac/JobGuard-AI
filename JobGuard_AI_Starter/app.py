import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.preprocessing import prepare_text

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "jobguard_model.joblib"
THRESHOLD_PATH = ROOT / "models" / "threshold.json"

st.set_page_config(page_title="JobGuard AI", page_icon="🛡️", layout="wide")
st.title("🛡️ JobGuard AI")
st.caption("AI-assisted screening for potentially fraudulent job and internship postings")

if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists():
    st.error("Model files are missing. First run: python -m src.train")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
with open(THRESHOLD_PATH, encoding="utf-8") as f:
    threshold = float(json.load(f)["threshold"])

def warning_indicators(text):
    text_lower = text.lower()
    rules = {
        "Upfront payment or registration fee": [
            "registration fee", "pay upfront", "processing fee", "security deposit",
            "training fee", "payment required"
        ],
        "Sensitive financial information requested": [
            "bank password", "card number", "otp", "upi pin", "account password"
        ],
        "Unusually high or guaranteed earnings": [
            "guaranteed income", "earn lakhs", "make money instantly",
            "no experience high salary", "100% guaranteed"
        ],
        "Pressure or urgency language": [
            "apply immediately", "limited slots", "urgent hiring",
            "act now", "offer expires today"
        ],
    }
    found = []
    for label, phrases in rules.items():
        matches = [phrase for phrase in phrases if phrase in text_lower]
        if matches:
            found.append((label, matches))
    return found

tab1, tab2 = st.tabs(["Check one posting", "Check a CSV file"])

with tab1:
    left, right = st.columns(2)
    with left:
        title = st.text_input("Job / internship title")
        company = st.text_input("Company name")
        location = st.text_input("Location")
        salary = st.text_input("Salary / stipend (optional)")
        employment = st.selectbox("Employment type", ["", "Full-time", "Part-time", "Internship", "Contract", "Other"])
    with right:
        company_profile = st.text_area("Company profile (optional)", height=100)
        description = st.text_area("Job description", height=180)
        requirements = st.text_area("Requirements (optional)", height=100)
        benefits = st.text_area("Benefits / stipend details (optional)", height=80)

    if st.button("Analyze posting", type="primary"):
        if not title.strip() and not description.strip():
            st.warning("Enter at least a job title or job description.")
        else:
            row = pd.DataFrame([{
                "title": title, "company_profile": company_profile,
                "description": description, "requirements": requirements,
                "benefits": benefits, "location": location,
                "salary_range": salary, "employment_type": employment,
            }])
            score = float(model.predict_proba(prepare_text(row))[0, 1])
            prediction = int(score >= threshold)
            st.subheader("Analysis result")
            if prediction:
                st.error("Potentially suspicious — verify the employer before proceeding.")
            else:
                st.success("Lower model risk — still verify the employer independently.")
            st.metric("Model-derived risk indicator", f"{score * 100:.1f}%")
            st.caption(f"Decision threshold selected using validation data: {threshold:.2f}")
            combined = " ".join(str(v) for v in row.iloc[0].tolist())
            indicators = warning_indicators(combined)
            if indicators:
                st.warning("Warning signs found in the entered text:")
                for label, matches in indicators:
                    st.write(f"- **{label}:** {', '.join(matches)}")
            else:
                st.info("No phrase from the current warning-rule list was detected.")
            st.caption("This is a screening aid, not proof of fraud or a guarantee of safety.")

with tab2:
    st.write("Upload a CSV containing job-posting text. A `description` column is recommended.")
    uploaded = st.file_uploader("Choose CSV", type=["csv"])
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        st.write(f"Rows loaded: {len(batch)}")
        if st.button("Analyze CSV"):
            scores = model.predict_proba(prepare_text(batch))[:, 1]
            batch["model_risk_indicator"] = scores
            batch["screening_result"] = (scores >= threshold).astype(int)
            batch["screening_result"] = batch["screening_result"].map({
                0: "Lower model risk", 1: "Potentially suspicious"
            })
            st.dataframe(batch.head(100), use_container_width=True)
            st.download_button(
                "Download analyzed CSV",
                data=batch.to_csv(index=False).encode("utf-8"),
                file_name="jobguard_analysis.csv",
                mime="text/csv",
            )
            st.caption("Results are model screening indicators. Review each posting manually.")

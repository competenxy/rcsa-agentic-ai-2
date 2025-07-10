import streamlit as st
import pandas as pd
import json
import re
from io import BytesIO
from openai import OpenAI
import docx2txt
import pdfplumber

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Text extraction function
def extract_text(uploaded) -> str:
    name = uploaded.name.lower()
    if name.endswith('.docx'):
        return docx2txt.process(uploaded)
    if name.endswith('.pdf'):
        with pdfplumber.open(uploaded) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    return uploaded.read().decode('utf-8', errors='ignore')

# GPT Chat helper
def chat_json(prompt: str, max_tokens: int = 2048) -> dict:
    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": "You're a precise Operational Risk assistant for banks."},
                  {"role": "user", "content": prompt}],
    )
    return json.loads(resp.choices[0].message.content)

# Normalization helpers
def norm_type(val: str) -> str:
    return val.strip().capitalize()

def risk_degree_to_freq(degree: str) -> str:
    mapping = {'Very High': 'Monthly', 'High': 'Quarterly', 'Medium': 'Semi-Annual', 'Low': 'Annual'}
    return mapping.get(degree, 'Annual')

# Generate RCSA controls
def generate_controls(text: str, target_n: int) -> pd.DataFrame:
    keywords = ["approval","limit","threshold","reconcile","review","authorise",
                "exception","segregation","dual","signoff","compliance","validate","checker"]
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if any(k in s.lower() for k in keywords) and len(s.strip()) > 20]

    if not sentences:
        st.warning("No control-like sentences found.")
        return pd.DataFrame()

    prompt = f"""
    Extract exactly {target_n} specific, measurable RCSA controls in verb-object-condition format from:

    Sentences:
    {sentences}

    Controls must explicitly state measurable conditions (time-bound, numeric limits, approver roles).

    Classify strictly into Level 1 Risk and Level 2 Risk:
    Internal Fraud: Unauthorized activity, Theft and fraud
    External Fraud: Theft and fraud, Systems security
    Employment practices and workplace safety: Employee relations, Safe environment, Diversity and discrimination
    Clients, products and business practices: Suitability, disclosure, fiduciary, Improper business or market practices, Product flaws, Selection, sponsorship and exposure, Advisory activities
    Damage to physical assets: Disasters and other events
    Business disruption and system failures: Systems
    Execution, delivery and process management: Transaction capture, execution and maintenance, Monitoring and reporting, Customer intake and documentation, Customer/client account management, Trade counterparties, Vendors and suppliers
    If unclear, mark as 'Other'.

    Classify each control into a Risk Degree: Very High, High, Medium, Low.

    Return JSON array with keys:
    ControlID, ControlObjective, Level1Risk, Level2Risk, RiskDegree, Type (Preventive, Detective, Corrective), TestingMethod.
    """

    data = chat_json(prompt)
    controls = data.get('controls', [])

    for idx, ctrl in enumerate(controls, 1):
        ctrl['ControlID'] = ctrl.get('ControlID', f'GC-{idx:03d}')
        ctrl['Type'] = norm_type(ctrl['Type'])
        ctrl['Frequency'] = risk_degree_to_freq(ctrl['RiskDegree'])

    return pd.DataFrame(controls)

# Validate existing RCSA controls
def validate_controls(raw_text: str) -> pd.DataFrame:
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

    prompt = f"""
    Rewrite these RCSA controls explicitly in measurable verb-object-condition format, discarding vague ones:

    {lines}

    Classify strictly into Level 1 Risk and Level 2 Risk:
    Internal Fraud: Unauthorized activity, Theft and fraud
    External Fraud: Theft and fraud, Systems security
    Employment practices and workplace safety: Employee relations, Safe environment, Diversity and discrimination
    Clients, products and business practices: Suitability, disclosure, fiduciary, Improper business or market practices, Product flaws, Selection, sponsorship and exposure, Advisory activities
    Damage to physical assets: Disasters and other events
    Business disruption and system failures: Systems
    Execution, delivery and process management: Transaction capture, execution and maintenance, Monitoring and reporting, Customer intake and documentation, Customer/client account management, Trade counterparties, Vendors and suppliers
    If unclear, mark as 'Other'.

    Classify each control into a Risk Degree: Very High, High, Medium, Low.

    Return JSON array with keys exactly:
    OldControlObjective, UpdatedControlObjective, Level1Risk, Level2Risk, RiskDegree, Type, TestingMethod.
    """

    data = chat_json(prompt)
    controls = data.get('controls', [])

    for i, ctrl in enumerate(controls, 1):
        ctrl['ControlID'] = f'VC-{i:03d}'
        ctrl['Type'] = norm_type(ctrl['Type'])
        ctrl['Frequency'] = risk_degree_to_freq(ctrl['RiskDegree'])

    return pd.DataFrame(controls)

# Excel download helper
def download_excel(df: pd.DataFrame, fname: str):
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    st.download_button("📥 Download Excel", buffer.getvalue(), file_name=fname,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Streamlit UI
st.set_page_config(page_title="RCSA Agentic AI", layout="wide")
st.title("📋 RCSA Agentic AI")

new_tab, validate_tab = st.tabs(["🆕 Generate New Controls", "🛠️ Validate Controls"])

# Generate Controls Tab
with new_tab:
    st.subheader("Generate RCSA Controls")
    uploaded = st.file_uploader("Upload document (PDF, DOCX, TXT)", type=["pdf","docx","txt"])
    target_n = st.number_input("Target Number of Controls", min_value=1, max_value=100, value=10)

    if st.button("Generate") and uploaded:
        text = extract_text(uploaded)
        df_out = generate_controls(text, target_n)
        if not df_out.empty:
            st.dataframe(df_out, use_container_width=True)
            download_excel(df_out, "generated_controls.xlsx")

# Validate Controls Tab
with validate_tab:
    st.subheader("Validate Existing RCSA Controls")
    uploaded_existing = st.file_uploader("Upload existing RCSA (CSV, XLSX)", type=["csv","xlsx"])

    if st.button("Validate") and uploaded_existing:
        df_existing = pd.read_csv(uploaded_existing) if uploaded_existing.name.endswith('.csv') else pd.read_excel(uploaded_existing)
        col = next((c for c in ['ControlObjective', 'Control Objective'] if c in df_existing.columns), df_existing.columns[1])
        raw_text = "\n".join(df_existing[col].astype(str).tolist())
        df_validated = validate_controls(raw_text)
        if not df_validated.empty:
            st.dataframe(df_validated, use_container_width=True)
            download_excel(df_validated, "validated_controls.xlsx")

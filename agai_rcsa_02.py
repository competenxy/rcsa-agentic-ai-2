import streamlit as st
import pandas as pd
import json
import re
from io import BytesIO
from openai import OpenAI
import docx2txt, pdfplumber

# -------- OpenAI client -------------------------------------------------------
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

def chat_json(prompt: str, max_tokens: int = 2048) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a precise Operational Risk assistant for banks."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content

# -------- Text extraction ----------------------------------------------------
def extract_text(uploaded) -> str:
    name = uploaded.name.lower()
    if name.endswith('.docx'):
        return docx2txt.process(uploaded)
    if name.endswith('.pdf'):
        with pdfplumber.open(uploaded) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    # TXT or other
    return uploaded.read().decode('utf-8', errors='ignore')

# -------- Normalization helpers ----------------------------------------------
def norm_type(val: str) -> str:
    v = val.strip().lower()
    if v.startswith('p'):
        return 'Preventive'
    if v.startswith('d'):
        return 'Detective'
    if v.startswith('c'):
        return 'Corrective'
    return val.strip().title()

def norm_freq(val: str) -> str:
    m = {'monthly':'Monthly','quarterly':'Quarterly','semi-annual':'Semi-Annual','annual':'Annual'}
    vl = val.strip().lower()
    for k,std in m.items():
        if k in vl:
            return std
    return val.strip().title()

# -------- Generate controls -------------------------------------------------
KEYWORDS = ["approval","limit","threshold","reconcile","review",
            "authorise","exception","segregation","dual","signoff","compliance"]

def generate_controls(text: str, target_n: int) -> pd.DataFrame:
    sentences = [s.strip() for s in re.split(r'[\.\!\?\n]', text)
                 if any(k in s.lower() for k in KEYWORDS) and len(s.strip())>20]
    if not sentences:
        st.warning("No keyword-matched sentences found.")
        return pd.DataFrame()
    block = "\n".join(sentences)
    prompt = f"""
Extract at least {target_n} specific RCSA controls from these sentences.
Return JSON object with key "controls" mapping to an array of objects with keys:
ControlID, ControlObjective, Type, TestingMethod, Frequency.
Type must be Preventive/Detective/Corrective; Frequency must be Monthly/Quarterly/Semi-Annual/Annual.

Sentences:
{block}
"""
    raw = chat_json(prompt, max_tokens=min(4096, target_n*60+200))
    try:
        ctrls = json.loads(raw).get('controls', [])
    except json.JSONDecodeError:
        st.error("Generator did not return valid JSON.")
        return pd.DataFrame()
    for idx, item in enumerate(ctrls, start=1):
        item.setdefault('ControlID', f'GC-{idx:03d}')
        item['Type'] = norm_type(item.get('Type',''))
        item['Frequency'] = norm_freq(item.get('Frequency',''))
    return pd.DataFrame(ctrls)

# -------- Validate controls -------------------------------------------------
def validate_controls(raw_text: str) -> pd.DataFrame:
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    if not lines:
        st.warning("No control lines found in the uploaded file.")
        return pd.DataFrame()
    controls_block = "\n".join(lines)
    prompt = f"""
You are a senior Operational Risk analyst.
Rewrite each control below in clear verb–object–condition form.
Also classify Type (Preventive/Detective/Corrective) and Frequency (Monthly/Quarterly/Semi-Annual/Annual).
Return JSON object with key "controls" mapping to an array with one object per line, keys:
OldControlObjective, UpdatedControlObjective, Type, TestingMethod, Frequency.
Do not merge, drop, or reorder rows.

{controls_block}
"""
    raw = chat_json(prompt, max_tokens=min(4096, len(lines)*60+200))
    try:
        ctrls = json.loads(raw).get('controls', [])
    except json.JSONDecodeError:
        st.error("Validator did not return valid JSON.")
        return pd.DataFrame()
    df = pd.json_normalize(ctrls)
    # pad or trim to match original count
    if len(df) < len(lines):
        pad = pd.DataFrame([{
            'OldControlObjective': lines[i],
            'UpdatedControlObjective': 'REVIEW_NEEDED',
            'Type': 'REVIEW_NEEDED',
            'TestingMethod': '',
            'Frequency': ''
        } for i in range(len(df), len(lines))])
        df = pd.concat([df, pad], ignore_index=True)
    elif len(df) > len(lines):
        df = df.iloc[:len(lines)].reset_index(drop=True)
    # normalize
    df['Type'] = df['Type'].apply(lambda v: norm_type(str(v)))
    df['Frequency'] = df['Frequency'].apply(lambda v: norm_freq(str(v)))
    return df[["OldControlObjective","UpdatedControlObjective","Type","TestingMethod","Frequency"]]

# -------- Excel download helper ----------------------------------------------
def download_excel(df: pd.DataFrame, file_name: str):
    buf = BytesIO()
    df.to_excel(buf, index=False)
    st.download_button("📥 Download Excel", buf.getvalue(), file_name=file_name,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -------- Streamlit UI -------------------------------------------------------
st.set_page_config(page_title="RCSA Agentic AI", layout="wide")
st.title("📋 RCSA Agentic AI")
tabs = st.tabs(["🆕 Generate RCSA","🛠️ Validate RCSA"])

# --- Generate tab with form -------------------------------------------------
with tabs[0]:
    st.subheader("Generate draft controls from a policy/SOP")
    with st.form("gen_form"):
        up = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf","docx","txt"], key="gen_up")
        tgt = st.number_input("Target number of controls", 1, 100, 20, key="gen_tgt")
        gen_submit = st.form_submit_button("Generate controls")
    if gen_submit:
        if not up:
            st.warning("Please upload a document first.")
        else:
            text = extract_text(up)
            df_out = generate_controls(text, tgt)
            if not df_out.empty:
                st.dataframe(df_out, use_container_width=True)
                download_excel(df_out, "generated_controls.xlsx")

# --- Validate tab with form -------------------------------------------------
with tabs[1]:
    st.subheader("Validate / clean an existing control list")
    with st.form("val_form"):
        up2 = st.file_uploader("Upload CSV, XLSX, DOCX, PDF, or TXT", 
                               type=["csv","xlsx","docx","pdf","txt"], key="val_up")
        val_submit = st.form_submit_button("Validate controls")
    if val_submit:
        if not up2:
            st.warning("Please upload a file first.")
        else:
            name = up2.name.lower()
            if name.endswith(".csv"):
                df_in = pd.read_csv(up2)
                raw = "\n".join(df_in.iloc[:,0].astype(str).tolist())
            elif name.endswith((".xlsx",".xls")):
                df_in = pd.read_excel(up2)
                raw = "\n".join(df_in.iloc[:,0].astype(str).tolist())
            else:
                raw = extract_text(up2)
            df_valid = validate_controls(raw)
            if not df_valid.empty:
                st.dataframe(df_valid, use_container_width=True)
                download_excel(df_valid, "validated_controls.xlsx")

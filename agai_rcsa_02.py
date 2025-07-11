import streamlit as st
import pandas as pd
import json
import re
from io import BytesIO
from openai import OpenAI
import docx2txt, pdfplumber

# ---------------- OpenAI Client ----------------
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

def chat_json(prompt: str, max_tokens: int = 2048) -> dict:
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
    # response_format ensures valid JSON
    return json.loads(resp.choices[0].message.content)

# ---------------- Text Extraction ----------------
def extract_text(uploaded) -> str:
    name = uploaded.name.lower()
    if name.endswith('.docx'):
        return docx2txt.process(uploaded)
    if name.endswith('.pdf'):
        with pdfplumber.open(uploaded) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    # txt or others
    return uploaded.read().decode('utf-8', errors='ignore')

# ---------------- Normalizers ----------------
def norm_type(val: str) -> str:
    v = val.strip().lower()
    if v.startswith('p'): return 'Preventive'
    if v.startswith('d'): return 'Detective'
    if v.startswith('c'): return 'Corrective'
    return val.strip().title()

def norm_freq(val: str) -> str:
    m = {
        'monthly':'Monthly', 'quarterly':'Quarterly',
        'semi-annual':'Semi-Annual', 'annual':'Annual'
    }
    v = val.strip().lower()
    for k, std in m.items():
        if k in v: return std
    return val.strip().title()

# ---------------- Generate Controls ----------------
KEYWORDS = ["approval","limit","threshold","reconcile","review",
            "authorise","exception","segregation","dual","signoff","compliance"]
def generate_controls(text: str, target_n: int) -> pd.DataFrame:
    # extract relevant sentences
    parts = re.split(r'[\.\!\?\n]', text)
    sentences = [s.strip() for s in parts if any(k in s.lower() for k in KEYWORDS) and len(s.strip())>20]
    if not sentences:
        st.warning("No keyword-matched sentences found.")
        return pd.DataFrame()
    block = "\n".join(sentences)
    prompt = f"""
Extract at least {target_n} specific RCSA controls from the following sentences.
Each control must be in verb–object–condition format.
Return a JSON object with key "controls" mapping to an array of objects with keys:
  ControlID, ControlObjective, Type, TestingMethod, Frequency, RiskLevel, RiskDegree
Type must be one of Preventive, Detective, Corrective.
Frequency must be one of Monthly, Quarterly, Semi-Annual, Annual.

Sentences:
{block}
"""
    result = chat_json(prompt, max_tokens=min(4096, target_n*60+200))
    items = result.get('controls', [])
    # post-process defaults and normalization
    for i, item in enumerate(items, start=1):
        item.setdefault('ControlID', f'GC-{i:03d}')
        item['ControlObjective'] = item.get('ControlObjective','').strip()
        item['Type'] = norm_type(item.get('Type',''))
        item['TestingMethod'] = item.get('TestingMethod','').strip()
        item['Frequency'] = norm_freq(item.get('Frequency',''))
        item['RiskLevel'] = item.get('RiskLevel','').strip().title()
        item['RiskDegree'] = item.get('RiskDegree','').strip().title()
    return pd.DataFrame(items)

# ---------------- Validate Controls ----------------
def validate_controls(raw_text: str) -> pd.DataFrame:
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    if not lines:
        st.warning("No controls found in the uploaded file.")
        return pd.DataFrame()
    block = "\n".join(lines)
    prompt = f"""
Rewrite each of the following RCSA controls in clear verb–object–condition form.
Return a JSON object with key "controls" mapping to an array of objects with keys:
  OldControlObjective, UpdatedControlObjective, Type, TestingMethod, Frequency, RiskLevel, RiskDegree
Type must be Preventive, Detective, or Corrective.
Frequency must be Monthly, Quarterly, Semi-Annual, or Annual.

Controls:
{block}
"""
    result = chat_json(prompt, max_tokens=min(4096, len(lines)*60+200))
    items = result.get('controls', [])
    # post-process
    for idx, item in enumerate(items, start=1):
        item.setdefault('OldControlObjective', lines[idx-1])
        item.setdefault('ControlID', f'VC-{idx:03d}')
        item['UpdatedControlObjective'] = item.get('UpdatedControlObjective','').strip()
        item['Type'] = norm_type(item.get('Type',''))
        item['TestingMethod'] = item.get('TestingMethod','').strip()
        item['Frequency'] = norm_freq(item.get('Frequency',''))
        item['RiskLevel'] = item.get('RiskLevel','').strip().title()
        item['RiskDegree'] = item.get('RiskDegree','').strip().title()
    df = pd.DataFrame(items)
    # preserve row count
    if len(df) < len(lines):
        for i in range(len(df), len(lines)):
            df = pd.concat([df, pd.DataFrame([{
                'OldControlObjective': lines[i],
                'UpdatedControlObjective': 'REVIEW_NEEDED',
                'Type':'REVIEW_NEEDED', 'TestingMethod':'', 'Frequency':'',
                'RiskLevel':'', 'RiskDegree':''
            }])], ignore_index=True)
    elif len(df) > len(lines):
        df = df.iloc[:len(lines)].reset_index(drop=True)
    return df[[
        'OldControlObjective','UpdatedControlObjective',
        'Type','TestingMethod','Frequency','RiskLevel','RiskDegree'
    ]]

# ---------------- Excel Download ----------------
def download_excel(df: pd.DataFrame, filename: str):
    buf = BytesIO()
    df.to_excel(buf, index=False)
    st.download_button("📥 Download Excel", buf.getvalue(), file_name=filename,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="RCSA Agentic AI", layout="wide")
st.title("📋 RCSA Agentic AI")
tabs = st.tabs(["🆕 Generate RCSA","🛠️ Validate RCSA"])

with tabs[0]:
    st.subheader("Generate draft controls from a policy/SOP")
    up = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf","docx","txt"])
    tgt = st.number_input("Target number of controls", 1, 100, 20)
    if st.button("Generate controls"):
        if up:
            text = extract_text(up)
            df_out = generate_controls(text, tgt)
            if not df_out.empty:
                st.dataframe(df_out, use_container_width=True)
                download_excel(df_out, "generated_controls.xlsx")
        else:
            st.warning("Please upload a file first.")

with tabs[1]:
    st.subheader("Validate / clean an existing control list")
    up2 = st.file_uploader("Upload CSV, XLSX, DOCX, PDF, or TXT", type=["csv","xlsx","docx","pdf","txt"])
    if st.button("Validate controls"):
        if up2:
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
        else:
            st.warning("Please upload a file first.")

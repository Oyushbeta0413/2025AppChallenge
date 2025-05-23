import streamlit as st
import fitz
import re

uploaded_file = st.file_uploader("Upload Medical Report", type=["pdf", "txt", "log", "md", "csv"])

orientation = st.selectbox(
    "How is the data formatted in your report?"
    ["Horizontal", "Vertical"]
)

pasred_text = ""

def parse_horizontal(text):
    lines=text.splitlines()
    extracted=[]
    for line in lines:
        if ":" in line or "-" in line:
            extracted.append(line.strip())
    return "\n".join(extracted)

def parse_vertical(text):
    lines=text.splitlines()
    extracted = []
    for i in range(len(lines)-1):
        label = lines[i].strip()
        value = lines[i+1].strip()
        if label and value and re.search(r"[a-zA-Z]", label) and re.search(r"\d", value):
            extracted.append(f"{label}: {value}")
    return "\n".join(extracted)

if uploaded_file is not None:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    all_text = ""
    for page in doc:
        all_text += page.get_text("text") + "\n"
        
    if orientation == "Horizontal":
        parsed_text = parse_horizontal(all_text)
    else:
        parsed_text = parse_vertical(all_text)
        

    

import streamlit as st
from transformers import pipeline
import pdfplumber
from PIL import Image
import pytesseract
import os

ner = pipeline("ner", model="d4data/biomedical-ner-all", tokenizer="d4data/biomedical-ner-all", aggregation_strategy="simple")
summarizer = pipeline("summarization", model="Falconsai/text_summarization")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_pdf(medical_report):
    with pdfplumber.open(medical_report) as pdf:
        return "\n".join(page.extract_text() or '' for page in pdf.pages)

def extract_image(medical_report):
    image = Image.open(medical_report)
    return pytesseract.image_to_string(image)

def summarize_text(text):
    if len(text.split()) > 250:
        with st.spinner("Summarizing medical report..."):
            summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    return None

def detect_risks_ai(text):
    labels = [
        "critical risk", "moderate risk", "no risk", 
        "cancer suspicion", "infection", "requires immediate attention", "normal", "diabetes",]
    
    with st.spinner("Detecting risks using AI..."):
        result = classifier(text, candidate_labels=labels, multi_label=True)
    top_labels = [(label, round(score, 2)) for label, score in zip(result['labels'], result['scores']) if score > 0.4]
    return top_labels

def display_risks(risks):
    st.subheader("AI-Predicted Risks:")

    if not risks:
        st.success("No significant risks detected.")
        return

    for label, score in risks:
        if "critical" in label or "immediate" in label:
            color = "#D9534F" 
        elif "moderate" in label or "infection" in label or "cancer" in label:
            color = "#F0AD4E" 
        elif "no risk" in label or "normal" in label:
            color = "#5CB85C"  
        else:
            color = "#5BC0DE" 

        st.markdown(
            f"<div style='background-color:{color}; padding:10px; border-radius:8px; color:white; margin-bottom:10px;'>"
            f"<strong>{label.upper()}</strong> — Confidence: {score * 100:.1f}%</div>",
            unsafe_allow_html=True
        )

SPECIALTY_MAP = {
    "oncology": ["Oncologist"], "cancer": ["Oncologist"],
    "radiotherapy": ["Radiation Oncologist"], "medical oncology": ["Oncologist"],
    "gynecology": ["Gynecologist"], "breast": ["Breast Surgeon", "Oncologist"],
    "bones": ["Orthopedic Oncologist", "Radiologist"], "chemotherapy": ["Oncologist"],
    "mast": ["Surgeon"], "pleural": ["Pulmonologist"], "fusion": ["Pulmonologist"],
    "breathing": ["Pulmonologist"], "appoitment": ["Primary Care"],
    "biopsy": ["Pathologist"], "lab": ["Pathologist"], "dose": ["Pharmacologist"]
}

def get_required_specialists(ner_results):
    seen = set()
    specialists = []
    for entity in ner_results:
        word = entity['word'].strip("#").lower()
        for key in SPECIALTY_MAP:
            if key in word:
                for spec in SPECIALTY_MAP[key]:
                    if spec not in seen:
                        seen.add(spec)
                        specialists.append(spec)
    return specialists

def main():
    st.set_page_config(page_title="AI Medical Report Analyzer", layout="centered")
    st.title("AI Medical Report Analyzer")

    medical_report = st.file_uploader("Upload medical report (PDF or image)", type=["pdf", "jpg", "jpeg", "png"])   

    if medical_report:
        ext = os.path.splitext(medical_report.name)[1].lower()        

        with st.spinner("Extracting text from your report..."):
            if ext == ".pdf":
                text = extract_pdf(medical_report)
            elif ext in [".png", ".jpg", ".jpeg"]:
                text = extract_image(medical_report)
            else:
                st.error("Unsupported file type.")
                return

        st.subheader("Extracted Text:")
        st.text_area("Text Content", text, height=300)

        summary = summarize_text(text)
        if summary:
            st.subheader("Summary:")
            st.success(summary)

        st.subheader("Detected Medical Terms:")
        with st.spinner("Finding named entities..."):
            ner_results = ner(text)
            st.markdown(", ".join(set([e['word'] for e in ner_results])) or "No entities found.")

        specialists = get_required_specialists(ner_results)
        if specialists:
            st.subheader("Suggested Specialists:")
            st.markdown("\n".join(f"- {spec}" for spec in specialists))

        risks = detect_risks_ai(text)
        display_risks(risks)

if __name__ == "__main__":
    main()







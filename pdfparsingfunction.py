import streamlit as st
from transformers import pipeline
import pdfplumber
from PIL import Image
import pytesseract
import time
import os


ner = pipeline("ner", model="d4data/biomedical-ner-all", tokenizer="d4data/biomedical-ner-all", aggregation_strategy="simple")
summarizer = pipeline("summarization", model="Falconsai/text_summarization")
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# def loading(seconds):
#     for i in range(seconds, 0, -1):
#         print(i)
#         time.sleep(1)
#     print("Time's up!")

def extract_pdf(medical_report):
    with pdfplumber.open(medical_report) as pdf:
        return "\n".join(page.extract_pdf() or '' for page in pdf.pages)

def extract_image(medical_report):
    image = Image.open(medical_report)
    return pytesseract.image_to_string(image)

def summarize_text(text):
    if len(text) > 250:
        st.subheader("Summarizing Your Report...")
        summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    return None

def main():
    medical_report = st.file_uploader(
        "Please upload your medical report",
        type=["pdf", "jpg", "image", "png", ]
    )   
    
    if medical_report is not None:
        filename = medical_report.name
        ext = os.path.splitext(filename)[1].lower()        

        if ext == ".pdf":
            text = extract_pdf(medical_report)
            st.text(text)
        if ext in [".png", ".jpg", ".jpeg"]:
            text = extract_image(medical_report)
            st.text(text)
        else:
            st.markdown("<h3 style='color: red;'>Unsupported File Type</h3>", unsafe_allow_html=True)
    
main()


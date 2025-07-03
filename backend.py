from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pdfplumber
import io
import traceback
from pdf2image import convert_from_bytes
import pytesseract
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pytesseract
import platform


if platform.system() == "Darwin":  # <- this is macOS! #ashrith chatgpt code
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
elif platform.system() == "Windows":  
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\vihaa\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
else:
    print("error")


#fast api wrap around original st code
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #when production actually happens we need to use the url for frontend in origin
    allow_methods=["*"],
    allow_headers=["*"],
)


clinical_bert_model = BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
clinical_bert_tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

#for the keywords
basic_disease_map = {
    "covid": "COVID-19",
    "corona": "COVID-19",
    "influenza": "Influenza",
    "flu": "Influenza",
    "cold": "Common Cold",
    "asthma": "Asthma",
    "pneumonia": "Pneumonia",
    "bronchitis": "Bronchitis",
    "tuberculosis": "Tuberculosis",
    "lung cancer": "Lung Cancer",
    "copd": "Chronic Obstructive Pulmonary Disease",
    "heart": "Heart Disease",
    "stroke": "Stroke",
    "hypertension": "Hypertension",
    "high blood pressure": "Hypertension",
    "arrhythmia": "Arrhythmia",
    "cardiomyopathy": "Cardiomyopathy",
    "diabetes": "Diabetes",
    "type 1 diabetes": "Type 1 Diabetes",
    "type 2 diabetes": "Type 2 Diabetes",
    "insulin resistance": "Insulin Resistance",
    "cancer": "Cancer",
    "leukemia": "Leukemia",
    "lymphoma": "Lymphoma",
    "melanoma": "Melanoma",
    "skin cancer": "Skin Cancer",
    "breast cancer": "Breast Cancer",
    "prostate cancer": "Prostate Cancer",
    "colon cancer": "Colon Cancer",
    "liver cancer": "Liver Cancer",
    "ovarian cancer": "Ovarian Cancer",
    "pancreatic cancer": "Pancreatic Cancer",
    "cervical cancer": "Cervical Cancer",
    "brain tumor": "Brain Tumor",
    "migraine": "Migraine",
    "epilepsy": "Epilepsy",
    "parkinson": "Parkinson's Disease",
    "alzheimer": "Alzheimer's Disease",
    "dementia": "Dementia",
    "multiple sclerosis": "Multiple Sclerosis",
    "als": "ALS (Amyotrophic Lateral Sclerosis)",
    "arthritis": "Arthritis",
    "rheumatoid arthritis": "Rheumatoid Arthritis",
    "gout": "Gout",
    "lupus": "Lupus",
    "fibromyalgia": "Fibromyalgia",
    "eczema": "Eczema",
    "psoriasis": "Psoriasis",
    "vitiligo": "Vitiligo",
    "anemia": "Anemia",
    "hemophilia": "Hemophilia",
    "thalassemia": "Thalassemia",
    "hepatitis": "Hepatitis",
    "hepatitis a": "Hepatitis A",
    "hepatitis b": "Hepatitis B",
    "hepatitis c": "Hepatitis C",
    "kidney": "Kidney Disease",
    "renal failure": "Renal Failure",
    "nephritis": "Nephritis",
    "urinary tract infection": "UTI",
    "prostatitis": "Prostatitis",
    "thyroid": "Thyroid Disorder",
    "hyperthyroidism": "Hyperthyroidism",
    "hypothyroidism": "Hypothyroidism",
    "obesity": "Obesity",
    "malnutrition": "Malnutrition",
    "hiv": "HIV/AIDS",
    "aids": "HIV/AIDS",
    "std": "Sexually Transmitted Disease",
}

class TextRequest(BaseModel):
    text: str

def classify_disease_and_severity(text):
    try:
        inputs = clinical_bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = clinical_bert_model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=-1).item()
        severity = "Mild" if predicted_class == 0 else "Severe, get help quick"
    except Exception:
        severity = "Unknown"

    disease = "Unknown"
    for keyword, label in basic_disease_map.items():
        if keyword in text.lower():
            disease = label
            break

    return severity, disease

def analyze_text(text):
    found = [label for k, label in basic_disease_map.items() if k in text.lower()]
    description = "The text contains: " + (", ".join(found) if found else "uncertain content.")
    severity, disease = classify_disease_and_severity(text)

    return {
        "extracted_text": text,
        "summary": f"{description} Severity: {severity}. Disease: {disease}.",
        "ner_results": found,
        "risks": severity,
        "specialists": ["Specialist in " + disease] if disease != "Unknown" else ["General Physician"]
    }

def extract_text_from_pdf(pdf_bytes):
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        print("[DEBUG] pdfplumber extracted:", bool(text.strip()))
    except Exception as e:
        print("[ERROR] pdfplumber failed:", str(e))

    if text.strip():
        return text

    print("[DEBUG] No text found, attempting OCR fallback...")

    try:
        poppler_path = r"C:\Users\vihaa\Downloads\poppler-24.08.0-0\poppler-24.08.0\Library\bin"
        images = convert_from_bytes(pdf_bytes, poppler_path=poppler_path)
        ocr_text = "\n".join([pytesseract.image_to_string(img) for img in images])
        print("[DEBUG] OCR result:", repr(ocr_text[:100]))
        return ocr_text
    except Exception as e:
        print("[ERROR] OCR fallback failed:", str(e))
        return ""

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        pdf_bytes = await file.read()
        full_text = extract_text_from_pdf(pdf_bytes)

        if not full_text.strip():
            raise HTTPException(status_code=400, detail="No readable text found (even with OCR).")

        return analyze_text(full_text)

    except Exception as e:
        print("ERROR in /analyze:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/analyze-text")
async def analyze_text_endpoint(request: TextRequest):
    try:
        return analyze_text(request.text)
    except Exception as e:
        print("ERROR in /analyze-text:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")

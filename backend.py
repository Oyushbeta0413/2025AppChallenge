from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from typing import Optional
import pytesseract
from PIL import Image
import io
import fitz  # PyMuPDF
import base64
import google.generativeai as genai
from bert import analyze_with_clinicalBert,classify_disease_and_severity
from disease_links import diseases as disease_links
from disease_steps import disease_next_steps
from disease_support import disease_doctor_specialty,disease_home_care
from gemini import analyze_with_gemini
from pdf2image import convert_from_path


from api_key import api_key


app = FastAPI()

# Allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up dummy FAISS for BERT mode


def extract_images_from_pdf(file_path):
    images = convert_from_path(file_path)
    return [image for image in images]  # PIL images

# Util: PDF to image
def extract_images_from_pdf1(pdf_bytes: bytes) -> list:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return [page.get_pixmap().tobytes("png") for page in doc]

# Util: OCR
def ocr_text_from_image(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return pytesseract.image_to_string(image)



@app.post("/analyze/")
async def analyze(
    file: UploadFile = File(...),
    model: str = Form(...),  # 'bert' or 'gemini'
    mode: Optional[str] = Form(None)  # summary, translate
):
  
    filename = file.filename.lower()
    mime_type = file.content_type
    all_analysis = []
    # Load image list
    if filename.endswith(".pdf"):
        image_list = extract_images_from_pdf(file.file)
    else:
        content = await file.read()
        image_list = [content]

    ocr_full = ""
    results = []
    detected_diseases = set()
    blip_Desc=""
    for img_bytes in image_list:
        ocr_text = ocr_text_from_image(img_bytes)
        ocr_full += ocr_text + "\n\n"

        if model == "gemini":
            analysis = analyze_with_gemini(img_bytes, "Identify medical terms, findings, and recommendations.")
            
            return analysis
        else:  # BERT
            input_text = ocr_text + "\nWhat are the key medical issues here?"
            analysis = analyze_with_clinicalBert(input_text)

            #analysis = analyze_with_scispacy(input_text);
            resolution = analysis
        
        severity, disease = classify_disease_and_severity(ocr_text)
        disease_summary = f"Detected Disease: {disease}, Severity: {severity}"
        if disease and disease != "Unknown":
            detected_diseases.add((str(disease), str(severity)))

        # Construct resolutions
        resolution = []
        print("Detected Diseases:", detected_diseases)

        for disease, severity in detected_diseases:
            link = disease_links.get(disease.lower(), "https://www.webmd.com/")
            next_steps = disease_next_steps.get(disease.lower(), ["Consult a doctor for further evaluation."])
            specialist = disease_doctor_specialty.get(disease.lower(), "General Practitioner")
            home_care = disease_home_care.get(disease.lower(), [])

            resolution.append({
                "findings": disease,
                "severity": severity,
                "recommendations": next_steps,
                "treatment_suggestions": 'Consult a specialist: ' + specialist,
                "home_care_guidance": home_care
            })

        

        return {
            "resolutions": resolution
        }

import os
import io
import traceback
import pytesseract
import fitz
import google.generativeai as genai
from PIL import Image
import pandas as pd
import re
import firebase_admin
from firebase_admin import credentials, firestore
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

def extract_non_negated_keywords(text: str) -> list:
    return ["cholesterol", "blood sugar"]

def classify_disease_and_severity(text: str) -> tuple:
    return "Hypertension", "Moderate"

def analyze_with_clinicalBert(text):
    return {"result": "Analyzed with clinicalBERT"}
def analyze_measurements(text, df):
    return {"measurements": "analyzed"}
def detect_past_diseases(text):
    return ["past_disease_1"]
def clean_ocr_text(text: str) -> str:
    text = text.replace("\x0c", " ")
    text = text.replace("\u00a0", " ")    
    text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text) 
    text = re.sub(r'\s+', ' ', text)      
    return text.strip()
def analyze_text(text):
    severity, disease = classify_disease_and_severity(text)
    return {
        "extracted_text": text,
        "summary": f"Detected Disease: {disease}, Severity: {severity}"
    }

def extract_images_from_pdf_bytes(pdf_bytes: bytes) -> list:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []
        for page in doc:
            pix = page.get_pixmap()
            buf = io.BytesIO()
            buf.write(pix.tobytes("png"))
            images.append(buf.getvalue())
        return images
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {e}")

def ocr_text_from_image(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return pytesseract.image_to_string(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR error: {e}")

disease_links = {"cholesterol": "https://www.webmd.com/cholesterol"}
disease_next_steps = {"cholesterol": ["Consult a doctor for a lipid panel."]}
disease_doctor_specialty = {"cholesterol": "Cardiologist"}
disease_home_care = {"cholesterol": ["Maintain a healthy diet."]}

EXTRACTED_TEXT_CACHE: str = ""
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=gemini_api_key)
except Exception as e:
    raise RuntimeError(f"Failed to configure Gemini API: {e}")

try:
    cred_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY_PATH")
    if not cred_path:
        raise ValueError("FIREBASE_SERVICE_ACCOUNT_KEY_PATH environment variable not set.")
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    raise RuntimeError(f"Failed to configure Firebase: {e}")

class ChatRequest(BaseModel):
    question: str
    user_id: str

class ChatResponse(BaseModel):
    answer: str

class TextRequest(BaseModel):
    text: str

system_prompt_chat = """
*** Role: Medical Guidance Facilitator
*** Objective:
Analyze medical data, provide concise, evidence-based insights, and recommend actionable next steps for patient care. This includes suggesting local physicians or specialists within a user-specified mile radius, prioritizing in-network options when insurance information is available, and maintaining strict safety compliance with appropriate disclaimers.
*** Capabilities:
1. Report Analysis – Review and interpret findings in uploaded medical reports.
2. Historical Context – Compare current findings with any available previous reports.
3. Medical Q&A – Answer specific questions about the report using trusted medical sources.
4. Specialist Matching – Recommend relevant physician specialties for identified conditions.
5. Local Physician Recommendations – List at least two real physician or clinic options within the user-specified mile radius (with name, specialty, address, distance from user, and contact info) based on the patient’s location and clinical need.
6. Insurance Guidance – If insurance/network information is provided, prioritize in-network physicians.
7. Safety Protocols – Include a brief disclaimer encouraging users to verify information, confirm insurance coverage, and consult providers directly.
*** Response Structure:
Start with a direct answer to the user’s primary question (maximum 4 concise sentences, each on a new line).
If a physician/specialist is needed, recommend at least two local providers within the requested radius (include name, specialty, address, distance, and contact info).
If insurance details are available, indicate which physicians are in-network.
End with a short safety disclaimer.
***Input Fields:
Provided Document Text: {document_text}
User Question: {user_question}
Assistant Answer:
"""

@app.post("/analyze/")
async def analyze(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    model: Optional[str] = Form("bert"),
    mode: Optional[str] = Form(None)
):
    global EXTRACTED_TEXT_CACHE
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    
    filename = file.filename.lower()
    ocr_full = ""
    
    try:
        if filename.endswith(".pdf"):
            pdf_bytes = await file.read()
            image_bytes_list = extract_images_from_pdf_bytes(pdf_bytes)
        else:
            content = await file.read()
            image_bytes_list = [content]

        for img_bytes in image_bytes_list:
            ocr_text = ocr_text_from_image(img_bytes)
            ocr_full += ocr_text + "\n\n"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {e}")

    EXTRACTED_TEXT_CACHE = ocr_full.strip()
    
    found_diseases = extract_non_negated_keywords(EXTRACTED_TEXT_CACHE)
    resolutions = []
    for disease in found_diseases:
        severity, _ = classify_disease_and_severity(EXTRACTED_TEXT_CACHE)
        link = disease_links.get(disease.lower(), "https://www.webmd.com/")
        next_steps = disease_next_steps.get(disease.lower(), ["Consult a doctor."])
        specialist = disease_doctor_specialty.get(disease.lower(), "General Practitioner")
        home_care = disease_home_care.get(disease.lower(), [])
        resolutions.append({
            "findings": disease,
            "severity": severity,
            "recommendations": next_steps,
            "treatment_suggestions": f"Consult a specialist: {specialist}",
            "home_care_guidance": home_care,
            "info_link": link
        })

    try:
        doc_ref = db.collection('users').document(user_id).collection('reports').document()
        doc_ref.set({
            'timestamp': firestore.SERVER_TIMESTAMP,
            'ocr_text': EXTRACTED_TEXT_CACHE,
            'resolutions': resolutions,
        })
    except Exception as e:
        print(f"Firestore save error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to save report to database.")

    return {
        "ocr_text": EXTRACTED_TEXT_CACHE,
        "resolutions": resolutions
    }

@app.post("/chat/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    global EXTRACTED_TEXT_CACHE
    if not EXTRACTED_TEXT_CACHE:
        raise HTTPException(status_code=400, detail="Please analyze a document first to provide a document context.")
    
    try:
        reports_ref = db.collection('users').document(request.user_id).collection('reports')
        docs = reports_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
        
        history_text = ""
        for doc in docs:
            report_data = doc.to_dict()
            history_text += f"Report from {report_data.get('timestamp', 'N/A')}:\n{report_data.get('ocr_text', 'No OCR text found')}\n\n"
    except Exception as e:
        history_text = "No past reports found for this user."
    
    full_document_text = EXTRACTED_TEXT_CACHE + "\n\n" + "PAST REPORTS:\n" + history_text
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        full_prompt = system_prompt_chat.format(
            document_text=full_document_text,
            user_question=request.question
        )
        response = model.generate_content(full_prompt)
        return ChatResponse(answer=response.text)
    except Exception as e:
        print(f"Gemini API error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An error occurred during chat response generation: {e}")

@app.post("/analyze-text")
async def analyze_text_endpoint(request: TextRequest):
    try:
        return analyze_text(request.text)
    except Exception as e:
        print("ERROR in /analyze-text:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")

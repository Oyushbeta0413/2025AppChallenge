from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pytesseract
from PIL import Image
import io
import fitz  
import base64
import traceback
import pandas as pd
import re
from google.generativeai import generative_models
import google.generativeai as genai
import os
import firebase_admin
from firebase_admin import credentials, firestore

from api_key import GEMINI_API_KEY
from bert import analyze_with_clinicalBert, classify_disease_and_severity, extract_non_negated_keywords, analyze_measurements, detect_past_diseases
from disease_links import diseases as disease_links
from disease_steps import disease_next_steps
from disease_support import disease_doctor_specialty, disease_home_care

model = genai.GenerativeModel('gemini-1.5-flash')
df = pd.read_csv("measurement.csv")
df.columns = df.columns.str.lower()
df['measurement'] = df['measurement'].str.lower()

app = FastAPI()
EXTRACTED_TEXT_CACHE = ""

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    gemini_api_key = os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY)
    if not gemini_api_key:
        raise ValueError("No Gemini API key found in environment or api_key.py")
    genai.configure(api_key=gemini_api_key)
except Exception as e:
    raise RuntimeError(f"Failed to configure Gemini API: {e}")

try:
    cred_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY_PATH", "firebase_key.json")
    
    if not os.path.exists(cred_path):
        raise ValueError(
            f"Firebase service account key not found. Looked for: {cred_path}. "
            "Set FIREBASE_SERVICE_ACCOUNT_KEY_PATH or place firebase_key.json in project root."
        )

    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    raise RuntimeError(f"Failed to configure Firebase: {e}")

# --- Pydantic Models for API Endpoints ---
class ChatRequest(BaseModel):
    question: str
    
class ChatResponse(BaseModel):
    answer: str
    
class TextRequest(BaseModel):
    text: str

# --- System Prompt for Gemini ---
system_prompt_chat = """
*** Role: Medical Guidance Facilitator
*** Objective:
Analyze medical data, provide concise, evidence-based insights, and recommend actionable next steps for patient care. This includes suggesting local physicians or specialists within a user-specified mile radius, prioritizing in-network options when insurance information is available, and maintaining strict safety compliance with appropriate disclaimers.
*** Capabilities:
1. Report Analysis – Review and interpret findings in uploaded medical reports.
2. Historical Context – Compare current findings with any available previous reports.
3. Medical Q&A – Answer specific questions about the report using trusted medical sources.
4. Specialist Matching – Recommend relevant physician specialties for identified conditions.
5. Safety Protocols – Include a brief disclaimer encouraging users to verify information, confirm insurance coverage, and consult providers directly.
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

def extract_images_from_pdf_bytes(pdf_bytes: bytes) -> list:
    print("***Start of Code***")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap()
        buf = io.BytesIO()
        buf.write(pix.tobytes("png"))
        images.append(buf.getvalue())
    return images

def clean_ocr_text(text: str) -> str:
    text = text.replace("\x0c", " ")
    text = text.replace("\u00a0", " ")    
    text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text) 
    text = re.sub(r'\s+', ' ', text)      
    return text.strip()


def ocr_text_from_image(image_bytes: bytes) -> str:
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    image_content = {
        'mime_type': 'image/jpeg',
        'data': base64_image
    }

    prompt = "Could you read this document and just take all the text that is in it and just paste it back to me in text format. Open and read this document:"

    response = model.generate_content(
        [prompt, image_content]
    )

    response_text = response.text
    print(response_text)

    return response_text

@app.post("/analyze/")
async def analyze(
    file: UploadFile = File(...),
    model: Optional[str] = Form("bert"),
    mode: Optional[str] = Form(None)
):
    global resolution, EXTRACTED_TEXT_CACHE
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    filename = file.filename.lower()
    detected_diseases = set()
    ocr_full = ""
    if filename.endswith(".pdf"):
        pdf_bytes = await file.read()
        image_bytes_list = extract_images_from_pdf_bytes(pdf_bytes)
    else:
        content = await file.read()
        image_bytes_list = [content]

    for img_bytes in image_bytes_list:
        ocr_text = ocr_text_from_image(img_bytes)
        ocr_full += ocr_text + "\n\n"
        ocr_full = clean_ocr_text(ocr_full)
        print(f"CALLING OCR FULL: {ocr_full}")
    
    EXTRACTED_TEXT_CACHE = ocr_full


    if model.lower() == "gemini":
        return {"message": "Gemini model not available; please use BERT model."}

    found_diseases = extract_non_negated_keywords(ocr_full)
    print(f"CALLING FOUND DISEASES: {found_diseases}")
    past = detect_past_diseases(ocr_full)
    print(f"CALLING PAST DISEASES: {past}")

    for disease in found_diseases:
        if disease in past:    
            severity = classify_disease_and_severity(disease)
            detected_diseases.add(((f"{disease}(detected as historical condition, but still under risk.)"), severity))
            print(f"DETECTED DISEASES(PAST): {detected_diseases}")
        else:
            severity = classify_disease_and_severity(disease)
            detected_diseases.add((disease, severity))
            print(f"DETECTED DISEASES: {detected_diseases}")
        
    print("OCR TEXT:", ocr_text)
    print("Detected diseases:", found_diseases)
    ranges = analyze_measurements(ocr_full, df)


    resolution = []
    detected_ranges = []
    for disease, severity in detected_diseases:
        link = disease_links.get(disease.lower(), "https://www.webmd.com/")
        next_steps = disease_next_steps.get(disease.lower(), ["Consult a doctor."])
        specialist = disease_doctor_specialty.get(disease.lower(), "General Practitioner")
        home_care = disease_home_care.get(disease.lower(), [])

        resolution.append({
            "findings": disease.upper(),
            "severity": severity,
            "recommendations": next_steps,
            "treatment_suggestions": f"Consult a specialist: {specialist}",
            "home_care_guidance": home_care,
            "info_link": link
                      
    })
    
    for i in ranges:
        condition = i[0]
        measurement = i[1]
        unit = i[2]
        severity = i[3]
        value = i[4]
        range_value = i[5]   # renamed to avoid overwriting Python's built-in "range"
                    
        link_range = disease_links.get(condition.lower(), "https://www.webmd.com/")
        next_steps_range = disease_next_steps.get(condition.lower(), ['Consult a doctor'])
        specialist_range = disease_doctor_specialty.get(condition.lower(), "General Practitioner")
        home_care_range = disease_home_care.get(condition.lower(), [])
        print(f"HELLO!: {measurement}")
                    
        condition_version = condition.upper()    
        severity_version = severity.upper()
                
        resolution.append({
            "findings": f"{condition_version} -- {measurement}",
            "severity": f"{value} {unit} - {severity_version}",
            "recommendations": next_steps_range,
            "treatment_suggestions": f"Consult a specialist: {specialist_range}",
            "home_care_guidance": home_care_range,
            "info_link": link_range
        })
                            
    print(ocr_full)
    ranges = analyze_measurements(ocr_full, df)
    print(analyze_measurements(ocr_full, df))
    # print ("Ranges is being printed", ranges)
    historical_med_data = detect_past_diseases(ocr_full)
    print("***End of Code***")
    
    return {
        "ocr_text": ocr_full.strip(),
        "Detected_Anomolies": resolution,
    }

class TextRequest(BaseModel):
    text: str

@app.post("/analyze-text")
async def analyze_text_endpoint(request: TextRequest):
    try:
        return analyze_text(request.text)
    except Exception as e:
        print("ERROR in /analyze-text:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")


def analyze_text(text):
    severity, disease = classify_disease_and_severity(text)
    return {
        "extracted_text": text,
        "summary": f"Detected Disease: {disease}, Severity: {severity}"
    }
    
@app.post("/chat/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chatbot endpoint that answers questions based on the last analyzed document and user history.
    """
    global EXTRACTED_TEXT_CACHE
    if not EXTRACTED_TEXT_CACHE:
        raise HTTPException(status_code=400, detail="Please provide a document context by analyzing text first.")
    
    try:
        reports_ref = db.collection('users').document(request.user_id).collection('reports')
        docs = reports_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(10).stream()
        
        history_text = ""
        for doc in docs:
            report_data = doc.to_dict()
            history_text += f"Report from {report_data.get('timestamp', 'N/A')}:\n{report_data.get('ocr_text', 'No OCR text found')}\n\n"
    except Exception as e:
        history_text = "No past reports found for this user."
    
    full_document_text = EXTRACTED_TEXT_CACHE + "\n\n" + "PAST REPORTS:\n" + history_text
    
    try:
        full_prompt = system_prompt_chat.format(
            document_text=full_document_text,
            user_question=request.question
        )
        response = model.generate_content(full_prompt)
        return ChatResponse(answer=response.text)
    except Exception as e:
        print(f"Gemini API error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An error occurred during chat response generation: {e}")

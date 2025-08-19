from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pytesseract
from PIL import Image
import io
import fitz  
import traceback
import pandas as pd
import re
import platform
import google.generativeai as genai
import json
import os
from dto.request_responses import AnalysisResponse, ChatRequest, ChatResponse,TextRequest
from chatbot import system_prompt_chat
from bert import analyze_with_clinicalBert, classify_disease_and_severity, extract_non_negated_keywords, analyze_measurements, detect_past_diseases
from disease_links import diseases as disease_links
from disease_steps import disease_next_steps
from disease_support import disease_doctor_specialty, disease_home_care
from api_key import GEMINI_API_KEY
from util import load_pytesseract, load_genai, setupFastAPI, extract_images_from_pdf_bytes, clean_ocr_text, ocr_text_from_image


load_pytesseract()
load_genai(GEMINI_API_KEY)
app = setupFastAPI()
EXTRACTED_TEXT_CACHE: str = ""


@app.post("/analyze/")
async def analyze(
    file: UploadFile = File(...),
    model: Optional[str] = Form("bert"),
    mode: Optional[str] = Form(None)
):
    global resolution
    global EXTRACTED_TEXT_CACHE

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

    # Store the OCR text in the cache for the chatbot
    EXTRACTED_TEXT_CACHE = ocr_full.strip()
    
        
    if model.lower() == "gemini":
        return {"message": "Gemini model is not fully integrated for analysis in this version. Using BERT for analysis.",
                "ocr_text": EXTRACTED_TEXT_CACHE}

    found_diseases = extract_non_negated_keywords(ocr_full)
    past = detect_past_diseases(ocr_full)

    for disease in found_diseases:
        if disease in past:    
            severity, _ = classify_disease_and_severity(ocr_full)
            detected_diseases.add(((f"{disease}(detected as historical condition, but still under risk.)"), severity))
        else:
            severity, _ = classify_disease_and_severity(ocr_full)
            detected_diseases.add((disease, severity))
        
        
    print("OCR TEXT:", ocr_text)
    print("Detected diseases:", found_diseases)

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
    
    print(ocr_full)
    ranges = analyze_measurements(ocr_full)
    print(ranges)
    # print ("Ranges is being printed", ranges)
    historical_med_data = detect_past_diseases(ocr_full)
    
    return {
        "ocr_text": ocr_full.strip(),
        "Detected Anomolies": resolution,
        "Detected Measurement Values": ranges,
    }



@app.post("/analyze-text")
async def analyze_text_endpoint(request: TextRequest):
    try:
        severity, disease = classify_disease_and_severity(request.text)
        return {
            "extracted_text": request.text,
            "summary": f"Detected Disease: {disease}, Severity: {severity}"
        }
    except Exception as e:
        print("ERROR in /analyze-text:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")


@app.post("/chat/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chatbot endpoint that answers questions based on the last analyzed document.
    """
    global EXTRACTED_TEXT_CACHE

    if not EXTRACTED_TEXT_CACHE:
        raise HTTPException(status_code=400, detail="Please analyze an image or PDF first to provide a document context.")
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        full_prompt = system_prompt_chat.format(
            document_text=EXTRACTED_TEXT_CACHE,
            user_question=request.question
        )
        
        response = model.generate_content(full_prompt)
        
        return ChatResponse(answer=response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during chat response generation: {e}")



if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

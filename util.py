from PIL import Image
import io
import fitz 
import re
import pytesseract
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import platform

def extract_images_from_pdf_bytes(pdf_bytes: bytes) -> list:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap()
        buf = io.BytesIO()
        buf.write(pix.tobytes("png"))
        images.append(buf.getvalue())
    return images

def clean_ocr_text(text: str) -> str:
    text = text.replace("\x0c", " ")       # remove form feed
    text = text.replace("\u00a0", " ")     # replace NBSP with space
    text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text)  # fix split decimals
    text = re.sub(r'\s+', ' ', text)       # collapse multiple spaces/newlines
    return text.strip()



def ocr_text_from_image(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return pytesseract.image_to_string(image)



def load_pytesseract():
    if platform.system() == "Darwin": 
        #pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  
        pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'  
    elif platform.system() == "Windows":
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def load_genai(genai_api_key: str):
    try:
        genai.configure(api_key=genai_api_key)
    except Exception as e:
        raise RuntimeError(f"Failed to configure Gemini API: {e}")


def setupFastAPI()-> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8002"
            "http://localhost:9000"
            "http://localhost:5501"
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


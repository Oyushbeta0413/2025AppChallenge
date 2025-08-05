from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional
import io, re

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

import pytesseract
from PIL import Image
from pdf2image import convert_from_path

import torch
from transformers import BertTokenizer, BertForSequenceClassification

CLINICAL_BERT_MODEL = BertForSequenceClassification.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT"
)
CLINICAL_BERT_TOKENIZER = BertTokenizer.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT"
)

def _severity_label(logits: torch.Tensor) -> str:
    """Index 0 → Mild, index 1 → Severe  (flip if you fine-tuned differently)."""
    return "Mild" if torch.argmax(logits, dim=-1).item() == 0 else "Severe"


NEG_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bdenies\b", r"\bdenied\b",
    r"\bwithout\b",
    r"\bnegative for\b",
    r"\bneg\b",
    r"\bfree of\b",
    r"\br\/o\b", r"\brule[sd]? out\b",
]

UNCERTAIN_PATTERNS = [
    r"\bpossible\b", r"\bpossibly\b",
    r"\bprobable\b", r"\bprobably\b",
    r"\bsuspect(ed)?\b",
    r"\blikely\b",
    r"\brisk of\b", r"\brisk for\b",
]

NEG_RE   = re.compile("|".join(NEG_PATTERNS),   flags=re.I)
UNCT_RE  = re.compile("|".join(UNCERTAIN_PATTERNS), flags=re.I)


DISEASE_KEYWORDS: Dict[str, List[str]] = {
    # cardio / vascular
    "Heart Disease"      : ["heart disease", "coronary", "cardiac", "myocard"],
    "Hypertension"       : ["hypertension", "high blood pressure"],
    "Stroke"             : ["stroke", "cva", "cerebrovascular accident"],
    "Arrhythmia"         : ["arrhythmia", "afib", "atrial fibrillation"],
    "Atherosclerosis"    : ["atherosclerosis"],
    "Heart Failure"      : ["heart failure", "congestive heart"],
    # endocrine / metabolic
    "Diabetes"           : ["diabetes", "hba1c", "hyperglycemia"],
    "Hyperthyroidism"    : ["hyperthyroid"],
    "Hypothyroidism"     : ["hypothyroid"],
    "Obesity"            : ["obesity", "obese", "bmi"],
    "Metabolic Syndrome" : ["metabolic syndrome"],
    "Addison's Disease"  : ["addison"],
    "Cushing's Syndrome" : ["cushing"],
    # respiratory
    "Asthma"             : ["asthma"],
    "COPD"               : ["copd", "emphysema", "chronic obstructive"],
    "Bronchitis"         : ["bronchitis"],
    "Pneumonia"          : ["pneumonia"],
    "Tuberculosis"       : ["tuberculosis", "tb"],
    "Pulmonary Fibrosis" : ["pulmonary fibrosis"],
    # oncology
    "Cancer"             : ["cancer", "malignancy", "carcinoma", "sarcoma", "tumor"],
    "Breast Cancer"      : ["breast cancer"],
    "Lung Cancer"        : ["lung cancer", "nsclc", "sclc"],
    "Prostate Cancer"    : ["prostate cancer"],
    "Colon Cancer"       : ["colon cancer", "colorectal"],
    "Leukemia"           : ["leukemia"],
    "Lymphoma"           : ["lymphoma"],
    "Melanoma"           : ["melanoma"],
    # neuro / psych
    "Alzheimer's Disease": ["alzheimer"],
    "Dementia"           : ["dementia"],
    "Parkinson's Disease": ["parkinson"],
    "Epilepsy"           : ["epilepsy", "seizure"],
    "Migraine"           : ["migraine"],
    "Multiple Sclerosis" : ["multiple sclerosis"],
    "Depression"         : ["depression", "major depressive"],
    "Bipolar Disorder"   : ["bipolar disorder"],
    "Schizophrenia"      : ["schizophrenia"],
    "ADHD"               : ["adhd", "attention deficit"],
    # rheum / immune
    "Arthritis"          : ["arthritis", "osteoarthritis"],
    "Rheumatoid Arthritis": ["rheumatoid arthritis"],
    "Lupus"              : ["lupus"],
    "Psoriasis"          : ["psoriasis"],
    "Eczema"             : ["eczema", "atopic dermatitis"],
    "Scleroderma"        : ["scleroderma"],
    "Fibromyalgia"       : ["fibromyalgia"],
    # infectious
    "COVID-19"           : ["covid", "sars-cov-2", "coronavirus"],
    "Influenza"          : ["influenza", "flu"],
    "HIV/AIDS"           : ["hiv", "aids"],
    "Hepatitis A"        : ["hepatitis a"],
    "Hepatitis B"        : ["hepatitis b"],
    "Hepatitis C"        : ["hepatitis c"],
    "Herpes"             : ["herpes"],
    "Dengue"             : ["dengue"],
    "Malaria"            : ["malaria"],
    "Measles"            : ["measles", "rubeola"],
    "Mumps"              : ["mumps"],
    "Rubella"            : ["rubella"],
    "Zika Virus"         : ["zika"],
    "Ebola"              : ["ebola"],
    "Cholera"            : ["cholera"],
    "Typhoid"            : ["typhoid"],
    "Lyme Disease"       : ["lyme"],
    "Whooping Cough"     : ["pertussis", "whooping cough"],
    # GI / hepatic
    "IBS"                : ["irritable bowel", "ibs"],
    "IBD"                : ["crohn", "ulcerative colitis"],
    "Gastritis"          : ["gastritis"],
    "Peptic Ulcer"       : ["peptic ulcer"],
    "Liver Cirrhosis"    : ["cirrhosis"],
    "Pancreatitis"       : ["pancreatitis"],
    "Gallstones"         : ["gallstones"],
    # kidney / GU
    "Kidney Disease"     : ["kidney disease", "renal failure", "ckd"],
    "UTI"                : ["urinary tract infection", "uti"],
    "Kidney Stones"      : ["kidney stone", "nephrolithiasis"],
    "Prostatitis"        : ["prostatitis"],
    # blood / heme
    "Anemia"             : ["anemia", "anaemia"],
    "Sickle Cell Disease": ["sickle cell"],
    "Hemophilia"         : ["hemophilia"],
    "Thalassemia"        : ["thalassemia"],
    # musculoskeletal
    "Osteoporosis"       : ["osteoporosis"],
    "Gout"               : ["gout"],
    "Back Pain"          : ["low back pain", "lumbar pain"],
    "Sciatica"           : ["sciatica"],
    # dermatology
    "Acne"               : ["acne"],
    "Vitiligo"           : ["vitiligo"],
    "Rosacea"            : ["rosacea"],
    "Shingles"           : ["shingles", "herpes zoster"],
    # sleep
    "Insomnia"           : ["insomnia"],
    "Sleep Apnea"        : ["sleep apnea"],
    "Narcolepsy"         : ["narcolepsy"],
    # other
    "Chronic Fatigue Syndrome": ["chronic fatigue syndrome"],
    "Chronic Pain"       : ["chronic pain"],
}


DISEASE_LINKS: Dict[str, str] = {
   
    "covid-19": "https://www.cdc.gov/coronavirus/2019-ncov/",
    "diabetes": "https://www.cdc.gov/diabetes/",
    "heart disease": "https://www.cdc.gov/heartdisease/",
    
}
def _fallback_link(disease: str) -> str:
    from urllib.parse import quote_plus
    return f"https://www.webmd.com/search/search_results/default.aspx?query={quote_plus(disease)}"


_sentence_split_re = re.compile(r"(?<=[\.\!\?])\s+")

def _is_positive_mention(sentence: str, kw: str) -> bool:
    """
    True  → keyword indicates confirmed / positive disease
    False → negated or uncertain in this sentence
    """
    pre, *_ = sentence.lower().partition(kw)
 
    if NEG_RE.search(pre): return False
    if UNCT_RE.search(pre): return False
    return True

def detect_positive_diseases(text: str) -> Set[str]:
    """
    Returns a set of disease labels with *positive* (non-negated) mentions.
    """
    diseases: Set[str] = set()
    sentences = _sentence_split_re.split(text)
    for sent in sentences:
        sent_l = sent.lower()
        for disease, kws in DISEASE_KEYWORDS.items():
            for kw in kws:
                if kw in sent_l and _is_positive_mention(sent_l, kw):
                    diseases.add(disease)
                    break 
    return diseases


def classify_diseases_and_severity(text: str) -> Tuple[str, List[str]]:
    """
    Returns overall severity ('Mild' | 'Severe')  +  list of *positive* diseases.
    """

    inputs = CLINICAL_BERT_TOKENIZER(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    with torch.no_grad():
        logits = CLINICAL_BERT_MODEL(**inputs).logits
    severity = _severity_label(logits)

    
    diseases = sorted(detect_positive_diseases(text))
    return severity, diseases if diseases else ["Unknown"]


def ocr_text_from_image(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return pytesseract.image_to_string(img)

def extract_images_from_pdf(fileobj) -> List[bytes]:
    pages = convert_from_path(fileobj, dpi=300)
    blobs: List[bytes] = []
    for page in pages:
        buf = io.BytesIO()
        page.save(buf, format="PNG")
        blobs.append(buf.getvalue())
    return blobs


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze/")
async def analyze(
    file: UploadFile = File(...),
    model: str = Form("bert"),        
    mode: Optional[str] = Form(None),
):
    
    if file.filename.lower().endswith(".pdf"):
        image_blobs = extract_images_from_pdf(file.file)
    else:
        image_blobs = [await file.read()]

   
    detected: Set[Tuple[str, str]] = set()
    for blob in image_blobs:
        text = ocr_text_from_image(blob)
        severity, diseases = classify_diseases_and_severity(text)
        for d in diseases:
            if d != "Unknown":
                detected.add((d, severity))

    
    resolutions = []
    for disease, severity in detected:
        key = disease.lower()
        resolutions.append({
            "findings"            : disease,
            "severity"            : severity,
            "reference"           : DISEASE_LINKS.get(key, _fallback_link(disease)),
            "recommendations"     : [f"Discuss {disease.lower()} management with a qualified physician."],
            "treatment_suggestions": "Consult a specialist as appropriate.",
            "home_care_guidance"  : [],
        })
    return {"resolutions": resolutions}

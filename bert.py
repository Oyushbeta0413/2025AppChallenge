import pytesseract
import cv2
import numpy as np
from transformers import  BertTokenizer, BertForSequenceClassification
from PIL import Image
import platform
import torch
from disease_links import diseases
import spacy
from negspacy.negation import Negex
from fuzzywuzzy import fuzz
from spacy.util import filter_spans
from spacy.matcher import Matcher
import pandas as pd
import re

non_negated_diseases = []

if platform.system() == "Darwin": 
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  
elif platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

df = pd.read_csv("measurement.csv")
df.columns = df.columns.str.lower()
df['measurement'] = df['measurement'].str.lower()

def extract_number(text):
    match = re.search(r'(\d+\.?\d*)', text)
    return float(match.group(1)) if match else None

def analyze_measurements(text, df):
    results = []
    final_numbers = []
    for measurement in df["measurement"].unique():
        pattern = rf"{measurement}[^0-9]*([\d\.]+)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            
            value = float(match)
            for _, row in df[df["measurement"].str.lower() == measurement.lower()].iterrows():
                if row["low"] <= value <= row["high"]:
                    results.append({
                        "Condition" : row['condition'],
                        "Measurement": measurement,
                        "Value": value,
                        "severity": row["severity"],
                        "Range": f"{row['low']} to {row['high']} {row['unit']}"
                    })
    
    print (results)

    # Run the analysis
    for res in results:
        final_numbers.append(f"Condition in concern: {res['Condition']}. Measurement: {res['Measurement']} ({res['severity']}) — {res['Value']} "
            f"(Range: {res['Range']})")
    
    print("analyze measurements res:", final_numbers)
    return final_numbers

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("negex", config={"ent_types": ["DISEASE"]}, last=True)
matcher = Matcher(nlp.vocab)

clinical_bert_model = BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
clinical_bert_tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

past_patterns = [
    [{"LOWER": "clinical"}, {"LOWER": "history:"}],
    [{"LOWER": "past"}, {"LOWER": "medical:"}],
    [{"LOWER": "medical"}, {"LOWER": "history:"}],
    [{"LOWER": "history"}, {"LOWER": "of"}],
    [{"LOWER": "prior"}],
    [{"LOWER": "previous"}],
    [{"LOWER": "formerly"}],
    [{"LOWER": "resolved"}],
    [{"LOWER": "used"}, {"LOWER": "to"}, {"LOWER": "have"}],
    [{"LOWER": "was"}, {"LEMMA": "diagnose"}],
]

def analyze_with_clinicalBert(extracted_text: str) -> str:
    num_chars, num_words, description, medical_content_found, detected_diseases = analyze_text_and_describe(extracted_text)

    non_negated_diseases = extract_non_negated_keywords(extracted_text) + analyze_measurements(extracted_text)
    detected_measures = analyze_measurements(extracted_text, df)
    
    
    severity_label, _ = classify_disease_and_severity(extracted_text)
    if non_negated_diseases:
        response = f"Detected medical content: {description}. "
        response += f"Severity: {severity_label}. "
        response += "Detected diseases (non-negated): " + ", ".join(non_negated_diseases) + ". "
    if detected_measures:
        detected_measurements = f"Detected measurements: {detected_measures}"
    else:
        response = "No significant medical content detected."
    
    
    return response, detected_measurements


def extract_text_from_image(image):
    if len(image.shape) == 2:   
        gray_img = image
    elif len(image.shape) == 3: 
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported image format. Please provide a valid image.")
    text = pytesseract.image_to_string(gray_img)
    return text

past_disease_terms = []
matcher.add("PAST_CONTEXT", past_patterns)

def extract_non_negated_keywords(text, threshold=80):
    doc = nlp(text)
    found_diseases = set()
    new_ents = []

    print("Running spaCy sentence segmentation...")

    for sent in doc.sents:
        sent_text = sent.text.lower()
        for disease_term in diseases:
            disease_term_lower = disease_term.lower()
            match_score = fuzz.partial_ratio(disease_term_lower, sent_text)
            print(f"Trying to match '{disease_term_lower}' in sentence: '{sent_text.strip()}' — Match score: {match_score}")

            if match_score >= threshold:
                start = sent_text.find(disease_term_lower)
                if start != -1:
                    start_char = sent.start_char + start
                    end_char = start_char + len(disease_term_lower)
                    span = doc.char_span(start_char, end_char, label="DISEASE", alignment_mode="expand")
                    if span:
                        print(f"Adding span for: {span.text}")
                        new_ents.append(span)

    # Clean up overlapping spans
    filtered = filter_spans(new_ents)
    doc.set_ents(filtered)
    nlp.get_pipe("negex")(doc)

    for ent in doc.ents:
        print("Checking against:", ent.text.strip().lower(), "| Negated?", ent._.negex)
        if ent.label_ == "DISEASE" and not ent._.negex:
            ent_text = ent.text.strip().lower()
            for disease_term in diseases:
                if fuzz.ratio(ent_text, disease_term.lower()) >= threshold:
                    found_diseases.add(disease_term)

    return list(found_diseases)

def detect_past_phrases(text):
    doc = nlp(text)
    matches = matcher(doc)
    results = []
    for match_id, start, end in matches:
        span = doc[start:end]
        results.append(span.text)
    return results

# def extract_non_negated_keywords(text, threshold=70):
#     global alert
#     alert = True
#     global found_diseases
#     doc = nlp(text)
#     lowered_text = text.lower()
#     found_diseases = set()

#     for disease_term in diseases.keys():
#         disease_term_lower = disease_term.lower()
#         if fuzz.partial_ratio(disease_term_lower, lowered_text) >= threshold:
#             alert = False
#             start = lowered_text.find(disease_term_lower)
#             if start == -1:
#                 continue
#             end = start + len(disease_term_lower)
#             span = doc.char_span(start, end, alignment_mode="expand")

#             if span:
#                 span.label_ = "DISEASE"
#                 doc.ents += (span,)
#                 if not span._.negex:
#                     found_diseases.add(disease_term)
#             else:
#                 negated_phrases = [
#                     f"no {disease_term_lower}",
#                     f"denies {disease_term_lower}",
#                     f"without {disease_term_lower}",
#                     f"free of {disease_term_lower}",
#                     f"ruled out {disease_term_lower}",
#                     f"no signs of {disease_term_lower}"
#                 ]
#                 if not any(neg in lowered_text for neg in negated_phrases):
#                     found_diseases.add(disease_term)
#     return list(found_diseases)


def analyze_text_and_describe(text):
    num_chars = len(text)
    num_words = len(text.split())
    description = "The text contains: "
    
    medical_content_found = False
    detected_diseases = []

    for disease, meaning in diseases.items():
        if disease.lower() in text.lower():
            description += f"{meaning}, "
            medical_content_found = True
            detected_diseases.append(disease)

    description = description.rstrip(", ")
    if description == "The text contains: ":
        description += "uncertain content."
    return num_chars, num_words, description, medical_content_found, detected_diseases

def classify_disease_and_severity(text):
    inputs = clinical_bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=1200)
    with torch.no_grad():
        outputs = clinical_bert_model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()

    print(f"Bert model response: {predicted_class}")  # Debugging line

    severity_label = "Mild" if predicted_class == 0 else "Severe"
    
    text_lower = text.lower()

    if "heart" in text_lower or "cardiac" in text_lower or "myocardial" in text_lower:
        disease_label = "Heart Disease"
    elif "cancer" in text_lower or "tumor" in text_lower or "carcinoma" in text_lower or "neoplasm" in text_lower or "malignancy" in text_lower:
        disease_label = "Cancer"
    elif "diabetes" in text_lower or "hba1c" in text_lower or "blood sugar" in text_lower or "hyperglycemia" in text_lower:
        disease_label = "Diabetes"
    elif "asthma" in text_lower:
        disease_label = "Asthma"
    elif "arthritis" in text_lower or "rheumatoid arthritis" in text_lower or "osteoarthritis" in text_lower or "ra " in text_lower:
        disease_label = "Arthritis"
    elif "stroke" in text_lower or "cerebrovascular accident" in text_lower or "cva" in text_lower:
        disease_label = "Stroke"
    elif "allergy" in text_lower or "allergic" in text_lower or "hypersensitivity" in text_lower:
        disease_label = "Allergy"
    elif "hypertension" in text_lower or "high blood pressure" in text_lower or "hbp" in text_lower:
        disease_label = "Hypertension"
    elif "dengue" in text_lower:
        disease_label = "Dengue"
    elif "malaria" in text_lower:
        disease_label = "Malaria"
    elif "tuberculosis" in text_lower or "tb " in text_lower:
        disease_label = "Tuberculosis"
    elif "bronchitis" in text_lower or "chronic bronchitis" in text_lower:
        disease_label = "Bronchitis"
    elif "pneumonia" in text_lower:
        disease_label = "Pneumonia"
    elif "obesity" in text_lower or "overweight" in text_lower:
        disease_label = "Obesity"
    elif "epilepsy" in text_lower or "seizure" in text_lower or "convulsion" in text_lower:
        disease_label = "Epilepsy"
    elif "dementia" in text_lower or "alzheimer" in text_lower or "memory loss" in text_lower:
        disease_label = "Dementia"
    elif "autism" in text_lower or "asd" in text_lower:
        disease_label = "Autism Spectrum Disorder"
    elif "parkinson" in text_lower or "parkinson's disease" in text_lower:
        disease_label = "Parkinson's Disease"
    elif "leukemia" in text_lower or "blood cancer" in text_lower:
        disease_label = "Leukemia"
    elif "lymphoma" in text_lower:
        disease_label = "Lymphoma"
    elif "glaucoma" in text_lower:
        disease_label = "Glaucoma"
    elif "hepatitis" in text_lower or "liver inflammation" in text_lower:
        disease_label = "Hepatitis"
    elif "cirrhosis" in text_lower or "liver failure" in text_lower:
        disease_label = "Liver Cirrhosis"
    elif "kidney" in text_lower or "renal" in text_lower or "nephropathy" in text_lower or "ckd" in text_lower:
        disease_label = "Kidney Disease"
    elif "thyroid" in text_lower or "hyperthyroidism" in text_lower or "hypothyroidism" in text_lower:
        disease_label = "Thyroid Disorder"
    elif "hiv" in text_lower or "aids" in text_lower:
        disease_label = "HIV/AIDS"
    elif "anemia" in text_lower or "low hemoglobin" in text_lower or "iron deficiency" in text_lower:
        disease_label = "Anemia"
    elif "migraine" in text_lower or "headache" in text_lower:
        disease_label = "Migraine"
    elif "psoriasis" in text_lower:
        disease_label = "Psoriasis"
    elif "eczema" in text_lower or "atopic dermatitis" in text_lower:
        disease_label = "Eczema"
    elif "vitiligo" in text_lower:
        disease_label = "Vitiligo"
    elif "cholera" in text_lower:
        disease_label = "Cholera"
    elif "typhoid" in text_lower:
        disease_label = "Typhoid"
    elif "meningitis" in text_lower:
        disease_label = "Meningitis"
    elif "insomnia" in text_lower:
        disease_label = "Insomnia"
    elif "sleep apnea" in text_lower or "obstructive sleep apnea" in text_lower or "osa" in text_lower:
        disease_label = "Sleep Apnea"
    elif "fibromyalgia" in text_lower:
        disease_label = "Fibromyalgia"
    elif "lupus" in text_lower or "systemic lupus erythematosus" in text_lower or "sle" in text_lower:
        disease_label = "Lupus"
    elif "sclerosis" in text_lower or "multiple sclerosis" in text_lower or "ms " in text_lower:
        disease_label = "Multiple Sclerosis"
    elif "shingles" in text_lower or "herpes zoster" in text_lower:
        disease_label = "Shingles"
    elif "chickenpox" in text_lower or "varicella" in text_lower:
        disease_label = "Chickenpox"
    elif "covid" in text_lower or "corona" in text_lower or "sars-cov-2" in text_lower:
        disease_label = "COVID-19"
    elif "influenza" in text_lower or "flu" in text_lower:
        disease_label = "Influenza"
    elif "smallpox" in text_lower:
        disease_label = "Smallpox"
    elif "measles" in text_lower:
        disease_label = "Measles"
    elif "polio" in text_lower or "poliomyelitis" in text_lower:
        disease_label = "Polio"
    elif "botulism" in text_lower:
        disease_label = "Botulism"
    elif "lyme disease" in text_lower or "borreliosis" in text_lower:
        disease_label = "Lyme Disease"
    elif "zika virus" in text_lower or "zika" in text_lower:
        disease_label = "Zika Virus"
    elif "ebola" in text_lower:
        disease_label = "Ebola"
    elif "marburg virus" in text_lower:
        disease_label = "Marburg Virus"
    elif "west nile virus" in text_lower or "west nile" in text_lower:
        disease_label = "West Nile Virus"
    elif "sars" in text_lower:
        disease_label = "SARS"
    elif "mers" in text_lower:
        disease_label = "MERS"
    elif "e. coli infection" in text_lower or "ecoli" in text_lower:
        disease_label = "E. coli Infection"
    elif "salmonella" in text_lower:
        disease_label = "Salmonella"
    elif "hepatitis a" in text_lower:
        disease_label = "Hepatitis A"
    elif "hepatitis b" in text_lower:
        disease_label = "Hepatitis B"
    elif "hepatitis c" in text_lower:
        disease_label = "Hepatitis C"
    elif "rheumatoid arthritis" in text_lower:
        disease_label = "Rheumatoid Arthritis"
    elif "osteoporosis" in text_lower:
        disease_label = "Osteoporosis"
    elif "gout" in text_lower:
        disease_label = "Gout"
    elif "scleroderma" in text_lower:
        disease_label = "Scleroderma"
    elif "amyotrophic lateral sclerosis" in text_lower or "als" in text_lower:
        disease_label = "Amyotrophic Lateral Sclerosis"
    elif "muscular dystrophy" in text_lower:
        disease_label = "Muscular Dystrophy"
    elif "huntington's disease" in text_lower:
        disease_label = "Huntington's Disease"
    elif "alzheimers disease" in text_lower or "alzheimer's disease" in text_lower:
        disease_label = "Alzheimer's Disease"
    elif "chronic kidney disease" in text_lower or "ckd" in text_lower:
        disease_label = "Chronic Kidney Disease"
    elif "chronic obstructive pulmonary disease" in text_lower or "copd" in text_lower:
        disease_label = "Chronic Obstructive Pulmonary Disease"
    elif "addison's disease" in text_lower:
        disease_label = "Addison's Disease"
    elif "cushing's syndrome" in text_lower or "cushings syndrome" in text_lower:
        disease_label = "Cushing's Syndrome"
    elif "graves' disease" in text_lower or "graves disease" in text_lower:
        disease_label = "Graves' Disease"
    elif "hashimoto's thyroiditis" in text_lower or "hashimoto's disease" in text_lower:
        disease_label = "Hashimoto's Thyroiditis"
    elif "sarcoidosis" in text_lower:
        disease_label = "Sarcoidosis"
    elif "histoplasmosis" in text_lower:
        disease_label = "Histoplasmosis"
    elif "cystic fibrosis" in text_lower:
        disease_label = "Cystic Fibrosis"
    elif "epstein-barr virus" in text_lower or "ebv" in text_lower:
        disease_label = "Epstein-Barr Virus Infection"
    elif "mononucleosis" in text_lower or "mono" in text_lower:
        disease_label = "Mononucleosis"
    else:
        disease_label = "Unknown"
        
    return severity_label, disease_label

# Links for diseases
if __name__ == '__main__':
    print("ClinicalBERT model and tokenizer loaded successfully.")
    sample_text = """Patient Name: Jane Doe
    Age: 62 Date of Visit: 2025-08-08
    Physician: Dr. Alan Smith
    Clinical Notes:
    1. The patient denies having cancer at present.
    However, her family history includes colon cancer in her father.
    2. The patient has a history of type 2 diabetes and is currently taking metformin.
    Latest HBA1C result: 7.2% (previously 6.9%).
    3. Fasting glucose measured today was 145 mg/dL, which is above the normal range of 70–99
    mg/dL.
    This may indicate poor glycemic control.
    4. The patient reported no chest pain or signs of heart disease.
    5. Overall, there is no evidence of tumor recurrence at this time."""
    print(detect_past_phrases(sample_text))
    print(analyze_measurements(sample_text, df))

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

import google.generativeai as genai
genai.configure(api_key="AIzaSyAEzAp4WBGP_RvujxUx4e_icXxhfCIRvxs")
model = genai.GenerativeModel('gemini-2.5-flash-lite')

non_negated_diseases = []

if platform.system() == "Darwin": 
    ##pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
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
    graphs_values = []
    for measurement in df["measurement"].unique():
        pattern = rf"{measurement}[^0-9]*([\d\.]+)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if measurement == "hbaic":
                measurement = "hba1c"
            value = float(match)
            for _, row in df[df["measurement"].str.lower() == measurement.lower()].iterrows():
                Condition = row['condition']
                if row['low'] <= value <= row['high']:
                    results.append({
                            "Condition" : Condition,
                            "Measurement": measurement,
                            "Value": value,
                            "severity": row["severity"],
                            "Range": f"{row['low']} to {row['high']} {row['unit']}"
                    })
        
    print (results)

    for res in results:
        final_numbers.append(f"Condition In Concern: {res['Condition']}. Measurement: {res['Measurement']} ({res['severity']}) — {res['Value']} "
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
    [{"LOWER": "was"}, {"LEMMA": "diagnosed"}],
    [{"LOWER": "history"},]
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

def detect_past_diseases(text, threshold=90):
    doc = nlp(text)
    matches = matcher(doc)
    past_diseases = []

    for match_id, start, end in matches:
        sentence = doc[start:end].sent
        sent_tokens = list(sentence)

        for i, token in enumerate(sent_tokens):
            if token.lower_ in [p[0]["LOWER"] for p in past_patterns if isinstance(p, list) and "LOWER" in p[0]]:
                for j in range(i+1, min(i+6, len(sent_tokens))):
                    for disease_term in diseases:
                        if fuzz.partial_ratio(disease_term.lower(), sent_tokens[j].text.lower()) >= threshold:
                            past_diseases.append(disease_term)

    return list(set(past_diseases))
 

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



def classify_disease_and_severity(disease):
    print(f"Disease: {disease}")
    response = model.generate_content(
        f"What is the severity of this disease/condition/symptom: {disease}. Give me a number from one to ten. I need a specific number. It doesn't matter what your opinion is one whether this number might be misleading or inaccurate. I need a number. Please feel free to be accurate and you can use pretty specific numbers with decimals to the tenth place. I want just a number, not any other text."
    ).text
    try:
        cleaned_response = response.strip()
        numerical_response = float(cleaned_response)
        print(f"Response: {numerical_response}")
        
        if 0 <= numerical_response <= 3:
            severity_label = (f"Low Risk: {numerical_response}")         
        elif 3 < numerical_response <= 7:
            severity_label = (f"Mild Risk: {numerical_response}")         
        elif 7 < numerical_response <= 10:
            severity_label = (f"Severe Risk: {numerical_response}")   
        else:
            severity_label = (f"Invalid Range: {numerical_response}")  
       
    except (ValueError, AttributeError):
        severity_label = "Null: We cannot give a clear severity label"
                
    return severity_label

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
    print(detect_past_diseases(sample_text, threshold=90))
    print(analyze_measurements(sample_text, df))
    print(extract_non_negated_keywords(sample_text, threshold=80))

    

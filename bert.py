import streamlit as st
import pytesseract
import cv2
import numpy as np
from transformers import  BertTokenizer, BertForSequenceClassification
from PIL import Image
import platform
import torch
from disease_links import diseases

# Set up Tesseract based on the operating system
if platform.system() == "Darwin":  # <- this is macOS!
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  
else:
    st.error("Unsupported OS for Tesseract. Please configure manually.")



# Load ClinicalBERT model and tokenizer
clinical_bert_model = BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
clinical_bert_tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


def analyze_with_clinicalBert(extracted_text: str) -> str:
    num_chars, num_words, description, medical_content_found, detected_diseases = analyze_text_and_describe(extracted_text)
    severity_label,disease_label = classify_disease_and_severity(extracted_text)
    if medical_content_found:
        response = f"Detected medical content: {description}. "
        response += f"Severity: {severity_label}. Disease: {disease_label}. "
        if detected_diseases:
            response += "Detected diseases: " + ", ".join(detected_diseases) + ". "
    else:
        response = "No significant medical content detected."    
    return response


# Function to extract text using Tesseract OCR
def extract_text_from_image(image):
    if len(image.shape) == 2:  # If grayscale
        gray_img = image
    elif len(image.shape) == 3:  # Convert to grayscale
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported image format. Please provide a valid image.")
    text = pytesseract.image_to_string(gray_img)
    return text

# Function to analyze text for medical relevance
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

# Function to classify disease and severity using ClinicalBERT
def classify_disease_and_severity(text):
    inputs = clinical_bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = clinical_bert_model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()

    print(f"Bert model response: {predicted_class}")  # Debugging line

    # Modify for more advanced classes if necessary (Assuming binary classification: 0: disease, 1: severity level)
    severity_label = "Mild" if predicted_class == 0 else "Severe"
    
    # For simplicity, use keywords in the text to classify disease
    if "heart" in text.lower():
        disease_label = "Heart Disease"
    elif "cancer" in text.lower():
        disease_label = "Cancer"
    elif "diabetes" in text.lower() or "hba1c" in text.lower():
        disease_label = "Diabetes"
    elif "asthma" in text.lower():
        disease_label = "Asthma"
    elif "arthritis" in text.lower():
        disease_label = "Arthritis"
    elif "stroke" in text.lower():
        disease_label = "Stroke"
    elif "allergy" in text.lower():
        disease_label = "Allergy"
    elif "hypertension" in text.lower() or "high blood pressure" in text.lower():
        disease_label = "Hypertension"
    elif "dengue" in text.lower():
        disease_label = "Dengue"
    elif "malaria" in text.lower():
        disease_label = "Malaria"
    elif "tuberculosis" in text.lower() or "tb" in text.lower():
        disease_label = "Tuberculosis"
    elif "bronchitis" in text.lower():
        disease_label = "Bronchitis"
    elif "pneumonia" in text.lower():
        disease_label = "Pneumonia"
    elif "obesity" in text.lower():
        disease_label = "Obesity"
    elif "epilepsy" in text.lower():
        disease_label = "Epilepsy"
    elif "dementia" in text.lower():
        disease_label = "Dementia"
    elif "autism" in text.lower():
        disease_label = "Autism"
    elif "parkinson" in text.lower():
        disease_label = "Parkinson's Disease"
    elif "leukemia" in text.lower():
        disease_label = "Leukemia"
    elif "glaucoma" in text.lower():
        disease_label = "Glaucoma"
    elif "hepatitis" in text.lower():
        disease_label = "Hepatitis"
    elif "kidney" in text.lower():
        disease_label = "Kidney Disease"
    elif "thyroid" in text.lower():
        disease_label = "Thyroid Disorder"
    elif "hiv" in text.lower() or "aids" in text.lower():
        disease_label = "HIV/AIDS"
    elif "anemia" in text.lower():
        disease_label = "Anemia"
    elif "migraine" in text.lower():
        disease_label = "Migraine"
    elif "psoriasis" in text.lower():
        disease_label = "Psoriasis"
    elif "eczema" in text.lower():
        disease_label = "Eczema"
    elif "vitiligo" in text.lower():
        disease_label = "Vitiligo"
    elif "cholera" in text.lower():
        disease_label = "Cholera"
    elif "typhoid" in text.lower():
        disease_label = "Typhoid"
    elif "meningitis" in text.lower():
        disease_label = "Meningitis"
    elif "insomnia" in text.lower():
        disease_label = "Insomnia"
    elif "sleep apnea" in text.lower():
        disease_label = "Sleep Apnea"
    elif "fibromyalgia" in text.lower():
        disease_label = "Fibromyalgia"
    elif "lupus" in text.lower():
        disease_label = "Lupus"
    elif "sclerosis" in text.lower():
        disease_label = "Multiple Sclerosis"
    elif "shingles" in text.lower():
        disease_label = "Shingles"
    elif "chickenpox" in text.lower():
        disease_label = "Chickenpox"
    elif "covid" in text.lower() or "corona" in text.lower():
        disease_label = "COVID-19"
    else:
        disease_label = "Unknown"
    
    return severity_label, disease_label

    


# Links for diseases
disease_links = {
    "tumor": "https://www.cancer.gov/about-cancer/diagnosis-staging/tumors",
    "heart": "https://www.heart.org/en/health-topics/heart-attack",
    "diabetes": "https://www.diabetes.org/",
    "cancer": "https://www.cancer.org/",
    "hypertension": "https://www.heart.org/en/health-topics/high-blood-pressure",
    "stroke": "https://www.stroke.org/en/about-stroke",
    "asthma": "https://www.lung.org/lung-health-diseases/lung-disease-lookup/asthma",
    "arthritis": "https://www.arthritis.org/",
    "migraine": "https://americanmigrainefoundation.org/",
    "depression": "https://www.nimh.nih.gov/health/topics/depression",
    "anemia": "https://www.mayoclinic.org/diseases-conditions/anemia",
    "allergy": "https://www.aaaai.org/conditions-and-treatments/allergies",
    "bronchitis": "https://www.lung.org/lung-health-diseases/lung-disease-lookup/bronchitis",
    "pneumonia": "https://www.lung.org/lung-health-diseases/lung-disease-lookup/pneumonia",
    "obesity": "https://www.cdc.gov/obesity/",
    "epilepsy": "https://www.epilepsy.com/",
    "dementia": "https://www.alz.org/alzheimers-dementia",
    "autism": "https://www.autismspeaks.org/",
    "parkinson": "https://www.parkinson.org/",
    "leukemia": "https://www.cancer.org/cancer/leukemia.html",
    "glaucoma": "https://www.glaucoma.org/",
    "sclerosis": "https://www.nationalmssociety.org/",
    "hepatitis": "https://www.cdc.gov/hepatitis/",
    "kidney": "https://www.kidney.org/",
    "thyroid": "https://www.thyroid.org/",
    "HIV/AIDS": "https://www.cdc.gov/hiv/",
    "malaria": "https://www.cdc.gov/malaria/",
    "tuberculosis": "https://www.cdc.gov/tb/",
    "chickenpox": "https://www.cdc.gov/chickenpox/",
    "covid19": "https://www.cdc.gov/coronavirus/2019-ncov/",
    "influenza": "https://www.cdc.gov/flu/",
    "smallpox": "https://www.cdc.gov/smallpox/",
    "measles": "https://www.cdc.gov/measles/",
    "polio": "https://www.cdc.gov/polio/",
    "cholera": "https://www.cdc.gov/cholera/",
    "botulism": "https://www.cdc.gov/botulism/",
    "lyme disease": "https://www.cdc.gov/lyme/",
    "dengue": "https://www.cdc.gov/dengue/",
    "zika virus": "https://www.cdc.gov/zika/",
    "hantavirus": "https://www.cdc.gov/hantavirus/",
    "ebola": "https://www.cdc.gov/vhf/ebola/",
    "marburg virus": "https://www.cdc.gov/vhf/marburg/",
    "West Nile Virus": "https://www.cdc.gov/westnile/",
    "SARS": "https://www.cdc.gov/sars/",
    "MERS": "https://www.cdc.gov/coronavirus/mers/",
    "E. coli infection": "https://www.cdc.gov/ecoli/",
    "salmonella": "https://www.cdc.gov/salmonella/",
    "hepatitis A": "https://www.cdc.gov/hepatitis/a/",
    "hepatitis B": "https://www.cdc.gov/hepatitis/b/",
    "hepatitis C": "https://www.cdc.gov/hepatitis/c/",
    "lupus": "https://www.lupus.org/",
    "epidemic keratoconjunctivitis": "https://www.cdc.gov/keratoconjunctivitis/",
    "scarlet fever": "https://www.cdc.gov/scarlet-fever/",
    "tetanus": "https://www.cdc.gov/tetanus/",
    "whooping cough": "https://www.cdc.gov/pertussis/",
    "chronic fatigue syndrome": "https://www.cdc.gov/cfs/",
    "tinnitus": "https://www.cdc.gov/tinnitus/",
    "hyperthyroidism": "https://www.thyroid.org/hyperthyroidism/",
    "hypothyroidism": "https://www.thyroid.org/hypothyroidism/",
    "liver cancer": "https://www.cancer.org/cancer/liver-cancer.html",
    "pancreatic cancer": "https://www.cancer.org/cancer/pancreatic-cancer.html",
    "brain cancer": "https://www.cancer.org/cancer/brain-cancer.html",
    "lung cancer": "https://www.cancer.org/cancer/lung-cancer.html",
    "skin cancer": "https://www.cancer.org/cancer/skin-cancer.html",
    "colon cancer": "https://www.cancer.org/cancer/colon-cancer.html",
    "bladder cancer": "https://www.cancer.org/cancer/bladder-cancer.html",
    "prostate cancer": "https://www.cancer.org/cancer/prostate-cancer.html",
    "stomach cancer": "https://www.cancer.org/cancer/stomach-cancer.html",
    "testicular cancer": "https://www.cancer.org/cancer/testicular-cancer.html",
    "breast cancer": "https://www.cancer.org/cancer/breast-cancer.html",
    "cervical cancer": "https://www.cancer.org/cancer/cervical-cancer.html",
    "esophageal cancer": "https://www.cancer.org/cancer/esophageal-cancer.html",
    "uterine cancer": "https://www.cancer.org/cancer/uterine-cancer.html",
    "ovarian cancer": "https://www.cancer.org/cancer/ovarian-cancer.html",
    "liver cirrhosis": "https://www.mayoclinic.org/diseases-conditions/cirrhosis/",
    "gallstones": "https://www.mayoclinic.org/diseases-conditions/gallstones/",
    "chronic bronchitis": "https://www.lung.org/lung-health-diseases/lung-disease-lookup/chronic-bronchitis",
    "COPD": "https://www.lung.org/lung-health-diseases/lung-disease-lookup/copd",
    "pulmonary fibrosis": "https://www.lung.org/lung-health-diseases/lung-disease-lookup/pulmonary-fibrosis",
    "pneumonitis": "https://www.lung.org/lung-health-diseases/lung-disease-lookup/pneumonitis",
    "eczema": "https://www.aafa.org/eczema/",
    "psoriasis": "https://www.psoriasis.org/",
    "rosacea": "https://www.aad.org/public/diseases/rosacea",
    "vitiligo": "https://www.aad.org/public/diseases/vitiligo",
    "acne": "https://www.aad.org/public/diseases/acne",
    "melanoma": "https://www.cancer.org/cancer/melanoma-skin-cancer.html",
    "actinic keratosis": "https://www.aad.org/public/diseases/actinic-keratosis",
    "shingles": "https://www.cdc.gov/shingles/",
    "chronic pain": "https://www.apa.org/news/press/releases/2018/08/chronic-pain",
    "fibromyalgia": "https://www.fmaware.org/",
    "rheumatoid arthritis": "https://www.arthritis.org/diseases/rheumatoid-arthritis",
    "osteoporosis": "https://www.niams.nih.gov/health-topics/osteoporosis",
    "gout": "https://www.arthritis.org/diseases/gout",
    "scleroderma": "https://www.scleroderma.org/",
    "amyotrophic lateral sclerosis": "https://www.als.org/",
    "multiple sclerosis": "https://www.nationalmssociety.org/",
    "muscular dystrophy": "https://www.mda.org/",
    "Parkinson's disease": "https://www.parkinson.org/",
    "Huntington's disease": "https://www.hdfoundation.org/",
    "Alzheimer's disease": "https://www.alz.org",
     "epilepsy": "https://www.epilepsy.com/",
    "stroke": "https://www.stroke.org/en/about-stroke",
    "dementia": "https://www.alz.org/alzheimers-dementia",
    
    "dengue": "https://www.cdc.gov/dengue/",
    "dengue fever": "https://www.cdc.gov/dengue/",
    "tuberculosis": "https://www.cdc.gov/tb/",
    "typhoid": "https://www.cdc.gov/typhoid-fever/",
    "cholera": "https://www.cdc.gov/cholera/",
    "malaria": "https://www.cdc.gov/malaria/",
    "measles": "https://www.cdc.gov/measles/",
    
    "herpes": "https://www.cdc.gov/herpes/",
    "herpes simplex": "https://www.cdc.gov/herpes/",
    "herpes zoster": "https://www.cdc.gov/shingles/",
    
    "chronic fatigue syndrome": "https://www.cdc.gov/cfs/",
    "fibromyalgia": "https://www.fmaware.org/",
    "sleep apnea": "https://www.cdc.gov/sleepapnea/",
    "narcolepsy": "https://www.ninds.nih.gov/health-information/disorders/narcolepsy",
    "insomnia": "https://www.cdc.gov/sleep/",
    
    "meningitis": "https://www.cdc.gov/meningitis/",
    "encephalitis": "https://www.cdc.gov/encephalitis/",
    "brain abscess": "https://www.cdc.gov/brain-abscess/",
    "spinal cord infection": "https://www.cdc.gov/spinal-cord-infections/",
    
    "polio": "https://www.cdc.gov/polio/",
    "poliomyelitis": "https://www.cdc.gov/polio/",
    "Guillain-Barré syndrome": "https://www.ninds.nih.gov/health-information/disorders/gbs",
}
if __name__ == '__main__':
    print("ClinicalBERT model and tokenizer loaded successfully.")
    

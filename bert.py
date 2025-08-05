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


if platform.system() == "Darwin": 
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  
elif platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("negex", config={"ent_types": ["DISEASE"]}, last=True)

clinical_bert_model = BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
clinical_bert_tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


def analyze_with_clinicalBert(extracted_text: str) -> str:
    num_chars, num_words, description, medical_content_found, detected_diseases = analyze_text_and_describe(extracted_text)

    non_negated_diseases = extract_non_negated_keywords(extracted_text)
        
    severity_label, _ = classify_disease_and_severity(extracted_text)
    if non_negated_diseases:
        response = f"Detected medical content: {description}. "
        response += f"Severity: {severity_label}. "
        response += "Detected diseases (non-negated): " + ", ".join(non_negated_diseases) + ". "
    else:
        response = "No significant medical content detected."
    
    
    return response

def extract_text_from_image(image):
    if len(image.shape) == 2:   
        gray_img = image
    elif len(image.shape) == 3: 
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported image format. Please provide a valid image.")
    text = pytesseract.image_to_string(gray_img)
    return text

def extract_non_negated_keywords(text, threshold=80):
    doc = nlp(text)
    lowered_text = text.lower()
    found_diseases = set()
    new_ents = []

    print("Lowered text:", lowered_text)

    for disease_term in diseases:
        disease_term_lower = disease_term.lower()
        match_score = fuzz.partial_ratio(disease_term_lower, lowered_text)
        print(f"Trying to match '{disease_term_lower}' — Match score: {match_score}")
        
        if match_score >= threshold:
            start_pos = 0
            while True:
                start = lowered_text.find(disease_term_lower, start_pos)
                if start == -1:
                    break
                end = start + len(disease_term_lower)
                span = doc.char_span(start, end, label="DISEASE", alignment_mode="expand")
                if span:
                    print(f"Adding span for: {span.text}")
                    new_ents.append(span)
                start_pos = end

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
    "toxoplasmosis": "https://www.cdc.gov/parasites/toxoplasmosis/",
    "pericarditis": "https://www.heart.org/en/health-topics/pericarditis",
    "sjogren’s syndrome": "https://www.niams.nih.gov/health-topics/sjogrens-syndrome",
    "trigeminal neuralgia": "https://www.ninds.nih.gov/health-information/disorders/trigeminal-neuralgia",
    "rectal cancer": "https://www.cancer.org/cancer/colon-rectal-cancer.html",
    "pemphigus vulgaris": "https://www.aad.org/public/diseases/a-z/pemphigus-vulgaris",
    "sinus cancer": "https://www.cancer.org/cancer/nasal-cavity-and-paranasal-sinuses-cancer.html",
    "barrett's esophagus": "https://www.cancer.org/cancer/esophagus-cancer/about/what-is-barretts-esophagus.html",
    "wilson's disease": "https://www.niddk.nih.gov/health-information/liver-disease/wilsons-disease",
    "tachycardia": "https://www.heart.org/en/health-topics/arrhythmia/about-arrhythmia/tachycardia---fast-heart-rate",
    "bradycardia": "https://www.heart.org/en/health-topics/arrhythmia/about-arrhythmia/bradycardia---slow-heart-rate",
    "endometriosis": "https://www.cdc.gov/reproductivehealth/conditions/endometriosis.htm",
    "interstitial cystitis": "https://www.niddk.nih.gov/health-information/urologic-diseases/interstitial-cystitis",
    "myasthenia gravis": "https://www.ninds.nih.gov/health-information/disorders/myasthenia-gravis",
    "guillain-barre syndrome": "https://www.ninds.nih.gov/health-information/disorders/guillain-barre-syndrome",
    "churg-strauss syndrome": "https://rarediseases.info.nih.gov/diseases/7110/eosinophilic-granulomatosis-with-polyangiitis",
    "aspergillosis": "https://www.cdc.gov/fungal/diseases/aspergillosis/index.html",
    "histoplasmosis": "https://www.cdc.gov/fungal/diseases/histoplasmosis/index.html",
    "blastomycosis": "https://www.cdc.gov/fungal/diseases/blastomycosis/index.html",
    "coccidioidomycosis": "https://www.cdc.gov/fungal/diseases/coccidioidomycosis/index.html",
    "actinomycosis": "https://www.cdc.gov/actinomycosis/index.html",
    "cryptococcosis": "https://www.cdc.gov/fungal/diseases/cryptococcosis-neoformans/index.html",
    "toxocariasis": "https://www.cdc.gov/parasites/toxocariasis/",
    "trichinosis": "https://www.cdc.gov/parasites/trichinellosis/",
    "strongyloidiasis": "https://www.cdc.gov/parasites/strongyloides/",
    "giardiasis": "https://www.cdc.gov/parasites/giardia/",
    "amebiasis": "https://www.cdc.gov/parasites/amebiasis/",
    "ascariasis": "https://www.cdc.gov/parasites/ascariasis/",
    "hookworm": "https://www.cdc.gov/parasites/hookworm/",
    "pinworm": "https://www.cdc.gov/parasites/pinworm/",
    "schistosomiasis": "https://www.cdc.gov/parasites/schistosomiasis/",
    "echinococcosis": "https://www.cdc.gov/parasites/echinococcosis/",
    "leishmaniasis": "https://www.cdc.gov/parasites/leishmaniasis/",
    "chagas disease": "https://www.cdc.gov/parasites/chagas/",
    "babesiosis": "https://www.cdc.gov/parasites/babesiosis/",
    "toxoplasma encephalitis": "https://www.cdc.gov/parasites/toxoplasmosis/gen_info/faqs.html",
    "pityriasis rosea": "https://www.aad.org/public/diseases/rashes/pityriasis-rosea",
    "lichen planus": "https://www.aad.org/public/diseases/a-z/lichen-planus",
    "hidradenitis suppurativa": "https://www.aad.org/public/diseases/a-z/hidradenitis-suppurativa",
    "dermatomyositis": "https://www.niams.nih.gov/health-topics/dermatomyositis",
    "vitreous hemorrhage": "https://www.aao.org/eye-health/diseases/what-is-vitreous-hemorrhage",
    "retinal detachment": "https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/retinal-detachment",
    "uveitis": "https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/uveitis",
    "optic neuritis": "https://www.aao.org/eye-health/diseases/what-is-optic-neuritis",
    "macular degeneration": "https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/age-related-macular-degeneration",
    "retinitis pigmentosa": "https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/retinitis-pigmentosa",
    "keratitis": "https://www.cdc.gov/contactlenses/keratitis.html",
    "chalazion": "https://www.aao.org/eye-health/diseases/what-is-chalazion",
    "blepharitis": "https://www.aao.org/eye-health/diseases/what-is-blepharitis",
    "dacryocystitis": "https://www.aao.org/eye-health/diseases/dacryocystitis",
    "orbital cellulitis": "https://www.aao.org/eye-health/diseases/orbital-cellulitis",
    "corneal ulcer": "https://www.aao.org/eye-health/diseases/corneal-ulcer",
    "amblyopia": "https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/amblyopia",
    "strabismus": "https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/strabismus",
    "nystagmus": "https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/nystagmus",
    "retinopathy of prematurity": "https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/retinopathy-prematurity",
    "keratoconus": "https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/keratoconus",
    "aniridia": "https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/aniridia",
    "achromatopsia": "https://rarediseases.info.nih.gov/diseases/5/achromatopsia",
    "cone-rod dystrophy": "https://rarediseases.info.nih.gov/diseases/2544/cone-rod-dystrophy",
    "epidermolysis bullosa": "https://www.niams.nih.gov/health-topics/epidermolysis-bullosa",
    "porphyria": "https://www.porphyriafoundation.org/for-patients/types-of-porphyria/",
    "neurofibromatosis": "https://www.ninds.nih.gov/health-information/disorders/neurofibromatosis",
    "tuberous sclerosis": "https://www.tsalliance.org/",
    "sturge-weber syndrome": "https://rarediseases.info.nih.gov/diseases/1026/sturge-weber-syndrome",
    "moebius syndrome": "https://rarediseases.info.nih.gov/diseases/7120/moebius-syndrome",
    "prader-willi syndrome": "https://www.pwsausa.org/",
    "angelman syndrome": "https://www.angelman.org/",
    "williams syndrome": "https://williams-syndrome.org/",
    "marfan syndrome": "https://www.marfan.org/",
    "ehlers-danlos syndrome": "https://www.ehlers-danlos.com/",
    "noonan syndrome": "https://www.genome.gov/Genetic-Disorders/Noonan-Syndrome",
    "bardet-biedl syndrome": "https://rarediseases.info.nih.gov/diseases/5797/bardet-biedl-syndrome",
    "alport syndrome": "https://www.kidney.org/atoz/content/alport",
    "gitelman syndrome": "https://rarediseases.info.nih.gov/diseases/6631/gitelman-syndrome",
    "bartter syndrome": "https://rarediseases.info.nih.gov/diseases/577/bartter-syndrome",
    "von hippel-lindau disease": "https://www.cancer.gov/types/kidney/patient/vhl-treatment-pdq",
    "familial adenomatous polyposis": "https://www.cancer.gov/types/colorectal/patient/fap-treatment-pdq",
    "lynch syndrome": "https://www.cancer.gov/types/colorectal/patient/lynch-syndrome-pdq",
    "brca mutation": "https://www.cancer.gov/about-cancer/causes-prevention/genetics/brca-fact-sheet",
    "retinoblastoma": "https://www.cancer.gov/types/eye/patient/retinoblastoma-treatment-pdq",
    "medulloblastoma": "https://www.cancer.gov/types/brain/patient/medulloblastoma-treatment-pdq",
    "ependymoma": "https://www.cancer.gov/types/brain/patient/ependymoma-treatment-pdq",
    "glioblastoma": "https://www.cancer.gov/types/brain/patient/adult-glioblastoma-treatment-pdq",
    "choroid plexus carcinoma": "https://www.cancer.gov/pediatric-adult-rare-tumor/rare-tumors/rare-central-nervous-system-tumors/choroid-plexus",
    "craniopharyngioma": "https://www.cancer.gov/types/brain/patient/craniopharyngioma-treatment-pdq"

}
if __name__ == '__main__':
    print("ClinicalBERT model and tokenizer loaded successfully.")
    sample_text = """
    Diagnoses:
    1. Type 2 Diabetes Mellitus - Elevated blood glucose levels.
    2. Colon Cancer — Biopsy confirmed Adenocarcinoma of colon.
    3. Patient denies having tuberculosis or HIV.
    """
    print("Detected (non-negated):", extract_non_negated_keywords(sample_text))

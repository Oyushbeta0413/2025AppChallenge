# bert_new.py

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch

'''
"""#model_name = "emilyalsentzer/Bio_ClinicalBERT"
model_name = "d4data/biomedical-ner-all"
MODEL_NAME = "pritamdeka/BioBERT-NER-diseases"
'''
MODEL_NAME = "kamalkraj/BioElectra-biomedical-ner"  # You can switch this to another BERT model if needed

def load_model_safely(model_name):
    try:
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
             trust_remote_code=False,
            use_safetensors=True
        )
        return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=-1)
    except Exception as e:
        print("Failed to load model:", e)
        return None

ner_pipeline = load_model_safely(MODEL_NAME)

def analyze_with_clinicalBert(text: str):
    if not ner_pipeline:
        return [{"error": "NER model could not be loaded"}]

    entities = ner_pipeline(text)
    findings = []
    seen = set()

    for ent in entities:
        label = ent.get("entity_group", "")
        word = ent.get("word", "").lower()
        if "disease" in label.lower() or "DISEASE" in word or "diabetes" in word:  # crude filter
            if word in seen:
                continue
            seen.add(word)

            # Heuristic severity detection
            lowered = text.lower()
            severity = "MILD"
            if any(x in lowered for x in ["critical", "acute", "severe", "very high"]):
                severity = "CRITICAL"
            elif any(x in lowered for x in ["moderate", "elevated"]):
                severity = "SEVERE"

            findings.append({
                "findings": word,
                "severity": severity,
                "recommendations": [
                    "Consult a healthcare provider",
                    "Review lab ranges"
                ],
                "treatment_suggestions": [
                    "Medication if prescribed",
                    "Lifestyle modification"
                ],
                "home_care_guidance": [
                    "Balanced diet",
                    "Exercise"
                ]
            })

    return findings if findings else [{"findings": "No diseases detected", "severity": "MILD"}]

# For testing locally
if __name__ == "__main__":
    test_text = """
    HbA1c is 10.0 which is above the normal range. Estimated average glucose is 240.
    This indicates poorly controlled diabetes and risk of complications.
    """
    print(analyze_with_clinicalBert(test_text))

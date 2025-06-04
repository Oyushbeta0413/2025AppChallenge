import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def extract_severity_or_value(text, entity_start, entity_end):
    window_start = max(0, entity_start - 10)
    window_end = min(len(text), entity_end + 10)
    context = text[window_start:window_end].lower()

    # Check for severity keywords
    for severity in ["mild", "moderate", "severe", "chronic"]:
        if severity in context:
            return severity

    # Check for a numeric value nearby (e.g., lab value)
    # Regex to find decimal or integer numbers
    numbers = re.findall(r'\b\d+\.?\d*\b', context)
    if numbers:
        return f"value: {numbers[0]}"

    return "unspecified"

def main():
    model_name = "d4data/biomedical-ner-all"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    text = "The patient's HbA1c is 10.0"

    print(f"Running NER on text: {text}")
    entities = nlp(text)

    print("Detected entities with severity/value:")
    for entity in entities:
        word = entity['word']
        start = entity['start']
        end = entity['end']
        sev_or_val = extract_severity_or_value(text, start, end)
        print(f"Entity: {word}, Type: {entity['entity_group']}, Severity/Value: {sev_or_val}")

if __name__ == "__main__":
    main()

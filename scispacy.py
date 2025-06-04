import scispacy
import spacy

# Load the SciSpacy model for chemicals and diseases
nlp = spacy.load("en_ner_bc5cdr_md")

def analyze_with_scispacy(text):
    doc = nlp(text)
    findings = []
    seen = set()

    for ent in doc.ents:
        if ent.label_ == "DISEASE" and ent.text.lower() not in seen:
            seen.add(ent.text.lower())

            # Simple heuristic for severity (customize as needed)
            severity = "MILD"
            lowered_text = text.lower()
            if any(x in lowered_text for x in ["critical", "severe", "acute"]):
                severity = "CRITICAL"
            elif any(x in lowered_text for x in ["moderate", "elevated"]):
                severity = "SEVERE"

            findings.append({
                "findings": ent.text,
                "severity": severity,
                "recommendations": [
                    "Follow up with a medical professional.",
                    "Consider diagnostic tests if symptoms persist."
                ],
                "treatment_suggestions": [
                    "Lifestyle changes",
                    "Medication as prescribed"
                ],
                "home_care_guidance": [
                    "Maintain hydration",
                    "Get adequate rest"
                ]
            })
    return findings if findings else [{"findings": "No diseases detected", "severity": "MILD"}]

# Example
text = "Patient shows signs of diabetes and moderate hypertension."
print(analyze_with_scispacy(text))


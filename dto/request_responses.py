from pydantic import BaseModel
from typing import Optional
class AnalysisResponse(BaseModel):
    findings: str
    severity: str
    recommendations: list[str]
    treatment_suggestions: str
    home_care_guidance: list[str]
    info_link: str

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

system_prompt_chat = """
You are a helpful medical assistant. Your task is to answer user questions based *only* on the provided medical document text.
Do not invent information or provide medical advice. If the answer is not in the text, simply say "I cannot find the answer in the provided document."

Provided Document Text:
{document_text}

User Question:
{user_question}

Assistant Answer:
"""
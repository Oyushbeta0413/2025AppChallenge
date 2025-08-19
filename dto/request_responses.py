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

class TextRequest(BaseModel):
    text: str

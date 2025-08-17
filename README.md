# 2025AppChallenge
Medical Report Analyzer



# make sure you're in the venv that runs uvicorn
python -m pip install -U "spacy>=3.6,<3.7"

# get the matching small English model for the installed spaCy
python -m spacy download en_core_web_sm

# upgrade negspacy to a spaCy 3-compatible release
python -m pip install -U "negspacy>=1.0.0"

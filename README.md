# 2025AppChallenge
Medical Report Analyzer


# upgrade negspacy to a spaCy 3-compatible release
python -m pip install -U "negspacy>=1.0.0"

1. Create a virtual environment and install the requirements.txt
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

2. Download the en_core_web_sm model
python -m spacy download en_core_web_sm

3. Run the backend.py file
python backend.py







# make sure you're in the venv that runs uvicorn
python -m pip install -U "spacy>=3.6,<3.7"

# get the matching small English model for the installed spaCy
python -m spacy download en_core_web_sm


# See all pythons on PATH (helps spot the system one vs venv)
which -a python

# Show where 'python' symlink points
ls -l $(which python)

# Show site-package locations Python will use
python -m site


# 1) Drop the aliases just for this session - incase your python or pip is not referenceing from .venv
unalias python 2>/dev/null
unalias pip 2>/dev/null

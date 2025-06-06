import streamlit as st
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials
from firebase_admin import auth
import json
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pdfplumber
# from PIL import Image
# #import pytesseract
# import time
import os


# cred = credentials.Certificate("pondering-5ff7c-c033cfade319.json")
cred = credentials.Certificate("login_tutorial.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)


def app():
# Usernm = []
    st.title('Login/Signup')

    if 'username' not in st.session_state:
        st.session_state.username = ''
    if 'useremail' not in st.session_state:
        st.session_state.useremail = ''


    def sign_up_with_email_and_password(email, password, username=None, return_secure_token=True):
        try:
            rest_api_url = "https://identitytoolkit.googleapis.com/v1/accounts:signUp"
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": return_secure_token
            }
            if username:
                payload["displayName"] = username 
            payload = json.dumps(payload)
            r = requests.post(rest_api_url, params={"key": "AIzaSyApr-etDzcGcsVcmaw7R7rPxx3A09as7uw"}, data=payload)
            try:
                return r.json()['email']
            except:
                st.warning(r.json())
        except Exception as e:
            st.warning(f'Signup failed: {e}')

    def sign_in_with_email_and_password(email=None, password=None, return_secure_token=True):
        rest_api_url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"

        try:
            payload = {
                "returnSecureToken": return_secure_token
            }
            if email:
                payload["email"] = email
            if password:
                payload["password"] = password
            payload = json.dumps(payload)
            print('payload sigin',payload)
            r = requests.post(rest_api_url, params={"key": "AIzaSyApr-etDzcGcsVcmaw7R7rPxx3A09as7uw"}, data=payload)
            try:
                data = r.json()
                user_info = {
                    'email': data['email'],
                    'username': data.get('displayName')  # Retrieve username if available
                }
                return user_info
            except:
                st.warning(data)
        except Exception as e:
            st.warning(f'Signin failed: {e}')

    def reset_password(email):
        try:
            rest_api_url = "https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode"
            payload = {
                "email": email,
                "requestType": "PASSWORD_RESET"
            }
            payload = json.dumps(payload)
            r = requests.post(rest_api_url, params={"key": "AIzaSyApr-etDzcGcsVcmaw7R7rPxx3A09as7uw"}, data=payload)
            if r.status_code == 200:
                return True, "Reset email Sent"
            else:
                # Handle error response
                error_message = r.json().get('error', {}).get('message')
                return False, error_message
        except Exception as e:
            return False, str(e)

    # Example usage
    # email = "example@example.com"
           

    def f(): 
        try:
            # user = auth.get_user_by_email(email)
            # print(user.uid)
            # st.session_state.username = user.uid
            # st.session_state.useremail = user.email

            userinfo = sign_in_with_email_and_password(st.session_state.email_input,st.session_state.password_input)
            st.session_state.username = userinfo['username']
            st.session_state.useremail = userinfo['email']

            
            global Usernm
            Usernm=(userinfo['username'])
            
            st.session_state.signedout = True
            st.session_state.signout = True    
  
            
        except: 
            st.error('Login Failed')

    def t():
        st.session_state.signout = False
        st.session_state.signedout = False   
        st.session_state.username = ''


    def forget():
        email = st.text_input('Email')
        if st.button('Send Reset Link'):
            print(email)
            success, message = reset_password(email)
            if success:
                st.success("Password reset email sent successfully.")
            else:
                st.warning(f"Password reset failed: {message}") 
        
    
        
    if "signedout"  not in st.session_state:
        st.session_state["signedout"] = False
    if 'signout' not in st.session_state:
        st.session_state['signout'] = False    
        

        
    
    if  not st.session_state["signedout"]: 
        choice = st.selectbox('Login/Signup',['Login','Sign up'])
        email = st.text_input('Email Address')
        password = st.text_input('Password',type='password')
        st.session_state.email_input = email
        st.session_state.password_input = password

        

        
        if choice == 'Sign up':
            username = st.text_input("Enter  your unique username")
            
            if st.button('Create my account'):
                # user = auth.create_user(email = email, password = password,uid=username)
                user = sign_up_with_email_and_password(email=email,password=password,username=username)
                
                st.success('Account created successfully!')
                st.markdown('Please Login using your email and password')
                st.balloons()
                st.snow()
                
        else:
            # st.button('Login', on_click=f)          
            st.button('Login', on_click=f)
            # if st.button('Forget'):
            forget()
            # st.button('Forget',on_click=forget)

            
            
    if st.session_state.signout:
        st.success(f"Welcome, {st.session_state.username}!")
        with st.sidebar:
            st.button('Sign out', on_click=t)

        st.header('Upload Your Medical Report')
        medical_report = st.file_uploader(
            "Please upload your medical report",
            type=["pdf", "jpg", "jpeg", "png"]
        )

        
        ner = pipeline("ner", model="d4data/biomedical-ner-all", tokenizer="d4data/biomedical-ner-all", aggregation_strategy="simple")
        summarizer = pipeline("summarization", model="Falconsai/text_summarization")

        def extract_pdf(medical_report):
            with pdfplumber.open(medical_report) as pdf:
                return "\n".join(page.extract_text() or '' for page in pdf.pages)

        def extract_image(medical_report):
            return "tesseract not installed(stub function)"  # ← Stub

        def summarize_text(text):
            if len(text) > 250:
                st.subheader("Summarizing Your Report...")
                summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
                return summary[0]['summary_text']
            return None

        if medical_report is not None:
            filename = medical_report.name
            ext = os.path.splitext(filename)[1].lower()

            text = ""
            if ext == ".pdf":
                text = extract_pdf(medical_report)
                st.subheader("Extracted Text from PDF")
                st.text_area("PDF Text", text, height=300)

                summary = summarize_text(text)
                if summary:
                    st.write(summary)

            elif ext in [".png", ".jpg", ".jpeg"]:
                text = extract_image(medical_report)
                st.subheader("Image Upload Note")
                st.text_area("Image Text", text, height=200)

            else:
                st.markdown("<h3 style='color: red;'>Unsupported File Type</h3>", unsafe_allow_html=True)

        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

        st.title("Chatbot")
        user_input = st.text_input("Ask me something about your report:")

        if user_input:
            inputs = tokenizer(user_input, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=100)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.write(response)

            
                
    

    def ap():
        st.write('Posts')

if __name__ == '__main__':
    app()

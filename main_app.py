import streamlit as st
import requests
from api_key import api_key
import google.generativeai as genai
import json
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-pro")

BACKEND_URL = "http://localhost:8000/analyze/"

st.set_page_config(page_title="Vital Analyzer", page_icon="🩻")
st.image("img.jpeg", width=150)
st.title("🧠 Vital🩻Analyzer: Image + PDF + LLM Choice")
st.subheader("Upload your medical file and pick a model to analyze it.")

# Upload + Options
uploaded_file = st.file_uploader("Upload a medical image or PDF", type=["jpg", "jpeg", "png", "pdf"])
llm_model = st.selectbox("Choose Model", ["bert","gemini"])

if st.button("🔍 Analyze"):
    if uploaded_file:
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        data = {"model": llm_model,"mode":"summary"}
        with st.spinner("Analyzing..."):
            res = requests.post(BACKEND_URL, files=files, data=data)
        if res.status_code == 200:
            out = res.json()
            #data  = json.loads(out)
            

            print(type(out))
            for item in data:
                 for key in item.items():
                     print(f"Key: {key}, Value: {item[key]}")
           
            #st.session_state.ocr_text = out["ocr_combined_text"]
            #st.session_state.page_results = out["page_results"]
            #st.session_state.chat = model.start_chat()
            #for p in st.session_state.page_results:
                #st.session_state.chat.send_message(p["ocr_text"])
                #st.session_state.chat.send_message(p["image_analysis"])
            #st.write(out["findings"])
            #st.write(out["severity"])
            
            #st.write(out["recommendations"])
            #st.write(out["treatment_suggestions"])
            #st.write(out["home_care_guidance"])
        else:
            st.error(f"Error: {res.text}")
    else:
        st.warning("Please upload a file.")


    st.divider()
    #st.write("### 💬 Chat with Gemini")
    #q = st.chat_input("Ask a medical question...")
   # if q:
     #   r = st.session_state.chat.send_message(q)
     #   st.chat_message("user").write(q)
      #  st.chat_message("assistant").write(r.text)"""

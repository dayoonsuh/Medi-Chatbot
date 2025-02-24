import streamlit as st
from streamlit_chat import message
from query_data import query_rag
import time
st.set_page_config(page_icon="💊",layout="wide")

st.title("_Ask :red[Pharmacist]_ :pill:")

# container for text box
response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        query_text = st.text_area("You:", key='input', height=100, placeholder="Question", label_visibility="collapsed")
        submit_button = st.form_submit_button(label='Send')

if submit_button and query_text:
    with response_container:
        response = query_rag(query_text)
        message(query_text, is_user=True, logo="https://api.dicebear.com/9.x/thumbs/svg?seed=Brian")
        message(response, logo="https://api.dicebear.com/9.x/thumbs/svg?seed=Mason")

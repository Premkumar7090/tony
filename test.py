import json
import os
import asyncio
import streamlit as st
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from llama_index.core import ServiceContext, set_global_service_context
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.gradient import GradientEmbedding
from llama_index.llms.gradient import GradientBaseModelLLM
from tempfile import NamedTemporaryFile
import pymupdf
# Set environment variables
os.environ['GRADIENT_ACCESS_TOKEN'] = 'CK1zDOU4BQ03NoMJMUDndWM8oAoNFMpm'
os.environ['GRADIENT_WORKSPACE_ID'] = '55708ca1-7b2b-42b1-ae1b-7d75ef4bc8c7_workspace'
# Initialize chat history
chat_history = []
def read_pdf(file_path):
   text = ""
   with pymupdf.open(file_path) as pdf_document:
       for page_num in range(pdf_document.page_count):
           page = pdf_document.load_page(page_num)
           text += page.get_text()
   return text
def main():
   global chat_history
   st.set_page_config(page_title="Chat with your PDF using Llama2 & Llama Index", page_icon="🦙")
   st.header('Chat with your PDF using Llama2 model & Llama Index')
   if "messages" not in st.session_state:
       st.session_state.messages = []
   # Display chat history
   for message in chat_history:
       with st.chat_message(message["role"], avatar=message['avatar']):
           st.markdown(message["content"])
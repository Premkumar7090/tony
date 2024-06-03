import os
import json
import asyncio
import streamlit as st
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from llama_index.core import ServiceContext, set_global_service_context, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.gradient import GradientEmbedding
from llama_index.llms.gradient import GradientBaseModelLLM
# Function to create and set an event loop if none exists
def get_or_create_event_loop():
   try:
       loop = asyncio.get_event_loop()
   except RuntimeError:
       loop = asyncio.new_event_loop()
       asyncio.set_event_loop(loop)
       loop = asyncio.get_event_loop()
   return loop
# Set environment variables
os.environ['GRADIENT_ACCESS_TOKEN'] = 'CK1zDOU4BQ03NoMJMUDndWM8oAoNFMpm'
os.environ['GRADIENT_WORKSPACE_ID'] = '55708ca1-7b2b-42b1-ae1b-7d75ef4bc8c7_workspace'
# Streamlit UI
st.title("Document Querying with Gradient and Cassandra")
# Load secrets from the token JSON file
try:
   with open("premkumarc1111@gmail.com-token.json") as f:
       secrets = json.load(f)
except FileNotFoundError:
   st.error("Token file not found. Please check the file path.")
   st.stop()
CLIENT_ID = secrets.get("clientId")
CLIENT_SECRET = secrets.get("secret")
if not CLIENT_ID or not CLIENT_SECRET:
   st.error("Client ID or Secret is missing from the token file.")
   st.stop()
# Configure Cassandra connection
cloud_config = {
   'secure_connect_bundle': 'secure-connect-nlp.zip'
}
auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
try:
   cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
   session = cluster.connect()
   row = session.execute("select release_version from system.local").one()
   if row:
       st.write(f"Cassandra release version: {row[0]}")
   else:
       st.error("An error occurred while fetching the release version.")
except Exception as e:
   st.error(f"Failed to connect to Cassandra: {e}")
   st.stop()
# Initialize the Gradient LLM and embedding models
try:
   loop = get_or_create_event_loop()
   llm = GradientBaseModelLLM(
       base_model_slug="llama2-7b-chat",
       max_tokens=400,
   )
   embed_model = GradientEmbedding(
       gradient_access_token=os.environ["GRADIENT_ACCESS_TOKEN"],
       gradient_workspace_id=os.environ["GRADIENT_WORKSPACE_ID"],
       gradient_model_slug="bge-large",
   )
   service_context = ServiceContext.from_defaults(
       llm=llm,
       embed_model=embed_model,
       chunk_size=256,
   )
   set_global_service_context(service_context)
except Exception as e:
   st.error(f"Failed to initialize Gradient models: {e}")
   st.stop()
# Load documents and create the index
try:
   documents = SimpleDirectoryReader("docs").load_data()
   st.write(f"Loaded {len(documents)} document(s).")
   index = VectorStoreIndex.from_documents(documents, service_context=service_context)
   query_engine = index.as_query_engine()
except Exception as e:
   st.error(f"Failed to process documents or create the index: {e}")
   st.stop()
# User input for query
query = st.text_input("Enter your query:", "What is the payment history?")
if st.button("Run Query"):
   try:
       response = query_engine.query(query)
       print(response)
       st.write(response)
   except Exception as e:
       st.error(f"Failed to query the index: {e}")
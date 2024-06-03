# import json
# import os
# import asyncio
# import streamlit as st
# # import pypdfium2 as pdfium
# from cassandra.auth import PlainTextAuthProvider
# from cassandra.cluster import Cluster
# from llama_index.core import ServiceContext
# from llama_index.core import set_global_service_context
# from llama_index.core import VectorStoreIndex, StorageContext
# from llama_index.embeddings.gradient import GradientEmbedding
# from llama_index.llms.gradient import GradientBaseModelLLM
# from llama_index.core import Document
# from tempfile import NamedTemporaryFile
# import pymupdf # PyMuPDF for reading PDFs
# # Set environment variables
# os.environ['GRADIENT_ACCESS_TOKEN'] = 'CK1zDOU4BQ03NoMJMUDndWM8oAoNFMpm'
# os.environ['GRADIENT_WORKSPACE_ID'] = '55708ca1-7b2b-42b1-ae1b-7d75ef4bc8c7_workspace'
# def get_or_create_event_loop():
#    try:
#        loop = asyncio.get_event_loop()
#    except RuntimeError:
#        loop = asyncio.new_event_loop()
#        asyncio.set_event_loop(loop)
#        loop = asyncio.get_event_loop()
#    return loop
# @st.cache_resource
# def create_datastax_connection():
#    cloud_config = {'secure_connect_bundle': r"C:\Users\AMALVIYA\Downloads\nlp\nlp\secure-connect-nlpp.zip"}
#    with open(r"C:\Users\AMALVIYA\Downloads\nlp\nlp\premkumarc1111@gmail.com-token.json") as f:
#        secrets = json.load(f)
#    CLIENT_ID = secrets["clientId"]
#    CLIENT_SECRET = secrets["secret"]
#    auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
#    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
#    astra_session = cluster.connect()
#    return astra_session
# def read_pdf(file_path):
#   #  pdf_document = fitz.open(file_path)
#    text = ""
 
#    with pymupdf.open(file_path) as pdf_document:
#         for page_num in range(pdf_document.page_count):
#             page = pdf_document.load_page(page_num)
#             text += page.get_text()
#    return text
# def main():
#    index_placeholder = None
#    st.set_page_config(page_title="Chat with your PDF using Llama2 & Llama Index", page_icon="🦙")
#    st.header('Chat with your PDF using Llama2 model & Llama Index')
#    if "conversation" not in st.session_state:
#        st.session_state.conversation = None
#    if "activate_chat" not in st.session_state:
#        st.session_state.activate_chat = False
#    if "messages" not in st.session_state:
#        st.session_state.messages = []
#    for message in st.session_state.messages:
#        with st.chat_message(message["role"], avatar=message['avatar']):
#            st.markdown(message["content"])
#    session = create_datastax_connection()
#    GRADIENT_ACCESS_TOKEN = os.environ["GRADIENT_ACCESS_TOKEN"]
#    GRADIENT_WORKSPACE_ID = os.environ["GRADIENT_WORKSPACE_ID"]
#    loop = get_or_create_event_loop()
#    llm = GradientBaseModelLLM(base_model_slug="llama2-7b-chat", max_tokens=400)
#    embed_model = GradientEmbedding(
#        gradient_access_token=GRADIENT_ACCESS_TOKEN,
#        gradient_workspace_id=GRADIENT_WORKSPACE_ID,
#        gradient_model_slug="bge-large"
#    )
#    service_context = ServiceContext.from_defaults(
#        llm=llm,
#        embed_model=embed_model,
#        chunk_size=256
#    )
#    set_global_service_context(service_context)
#    with st.sidebar:
#        st.subheader('Upload Your PDF File')
#        docs = st.file_uploader('⬆️ Upload your PDF & Click to process',
#                                accept_multiple_files=False,
#                                type=['pdf'])
#        if st.button('Process'):
#            if docs is not None:
#                try:
#                    with NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
#                        temp_file.write(docs.getbuffer())
#                        temp_file_path = temp_file.name
#                    with st.spinner('Processing'):
#                        pdf_text = read_pdf(temp_file_path)
#                        os.remove(temp_file_path)  # Clean up temp file
#                        document = Document(text=pdf_text)
#                        documents = [document]
#                        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
#                        query_engine = index.as_query_engine()
#                        if "query_engine" not in st.session_state:
#                            st.session_state.query_engine = query_engine
#                        st.session_state.activate_chat = True
#                except Exception as e:
#                    st.error(f"Error processing the file: {e}")
#    if st.session_state.activate_chat:
#        if prompt := st.chat_input("Ask your question from the PDF?"):
#            with st.chat_message("user", avatar='👨🏻'):
#                st.markdown(prompt)
#            st.session_state.messages.append({"role": "user", "avatar": '👨🏻', "content": prompt})
#            query_index_placeholder = st.session_state.query_engine
#            pdf_response = query_index_placeholder.query(prompt)
#            cleaned_response = pdf_response.response
#            with st.chat_message("assistant", avatar='🤖'):
#                st.markdown(cleaned_response)
#            st.session_state.messages.append({"role": "assistant", "avatar": '🤖', "content": cleaned_response})
#        else:
#            st.markdown('Upload your PDFs to chat')
# if __name__ == '__main__':
#    main()



# import json
# import os
# import asyncio
# import streamlit as st
# from cassandra.auth import PlainTextAuthProvider
# from cassandra.cluster import Cluster
# from llama_index.core import ServiceContext
# from llama_index.core import set_global_service_context
# from llama_index.core import VectorStoreIndex, StorageContext
# from llama_index.embeddings.gradient import GradientEmbedding
# from llama_index.llms.gradient import GradientBaseModelLLM
# from llama_index.core import Document
# from tempfile import NamedTemporaryFile
# import pymupdf  # PyMuPDF for reading PDFs

# # Set environment variables
# os.environ['GRADIENT_ACCESS_TOKEN'] = 'CK1zDOU4BQ03NoMJMUDndWM8oAoNFMpm'
# os.environ['GRADIENT_WORKSPACE_ID'] = '55708ca1-7b2b-42b1-ae1b-7d75ef4bc8c7_workspace'


# def get_or_create_event_loop():
#     try:
#         loop = asyncio.get_event_loop()
#     except RuntimeError:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         loop = asyncio.get_event_loop()
#     return loop


# @st.cache_resource
# def create_datastax_connection():
#     cloud_config = {'secure_connect_bundle': r"C:\Users\AMALVIYA\OneDrive - e2open, LLC\nlp\nlp\secure-connect-nlp.zip"}
#     with open(r"C:\Users\AMALVIYA\OneDrive - e2open, LLC\nlp\nlp\malviyaayush2609@gmail.com-token.json") as f:
#         secrets = json.load(f)
#     CLIENT_ID = secrets["clientId"]
#     CLIENT_SECRET = secrets["secret"]
#     auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
#     cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
#     astra_session = cluster.connect()
#     return astra_session


# def read_pdf(file_path, page_range):
#     text = ""
#     with pymupdf.open(file_path) as pdf_document:
#         total_pages = pdf_document.page_count
#         pages = parse_page_range(page_range, total_pages)
#         for page_num in pages:
#             page = pdf_document.load_page(page_num)
#             text += page.get_text()
#     # print(text)
#     return text


# def parse_page_range(page_range, total_pages):
#     pages = set()
#     ranges = page_range.split(',')
#     for r in ranges:
#         if '-' in r:
#             start, end = r.split('-')
#             start = int(start.strip()) - 1
#             end = int(end.strip())
#             pages.update(range(start, end))
#         else:
#             pages.add(int(r.strip()) - 1)
#     return [p for p in pages if 0 <= p < total_pages]


# def main():
#     index_placeholder = None
#     st.set_page_config(page_title="Chat with your PDF using Llama2 & Llama Index", page_icon="🦙")
#     st.header('Chat with your PDF using Llama2 model & Llama Index')

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "activate_chat" not in st.session_state:
#         st.session_state.activate_chat = False
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         with st.chat_message(message["role"], avatar=message['avatar']):
#             st.markdown(message["content"])

#     session = create_datastax_connection()
#     GRADIENT_ACCESS_TOKEN = os.environ["GRADIENT_ACCESS_TOKEN"]
#     GRADIENT_WORKSPACE_ID = os.environ["GRADIENT_WORKSPACE_ID"]
#     loop = get_or_create_event_loop()
#     llm = GradientBaseModelLLM(base_model_slug="llama2-7b-chat", max_tokens=400)
#     embed_model = GradientEmbedding(
#         gradient_access_token=GRADIENT_ACCESS_TOKEN,
#         gradient_workspace_id=GRADIENT_WORKSPACE_ID,
#         gradient_model_slug="bge-large"
#     )
#     service_context = ServiceContext.from_defaults(
#         llm=llm,
#         embed_model=embed_model,
#         chunk_size=256
#     )
#     set_global_service_context(service_context)

#     with st.sidebar:
#         st.subheader('Upload Your PDF File')
#         docs = st.file_uploader('⬆️ Upload your PDF & Click to process',
#                                 accept_multiple_files=False,
#                                 type=['pdf'])
#         page_range = st.text_input("Enter the page range to process (e.g., 1-5, 8, 10-12)", "")
#         if st.button('Process'):
#             if docs is not None:
#                 if page_range:
#                     try:
#                         with NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
#                             temp_file.write(docs.getbuffer())
#                             temp_file_path = temp_file.name
#                         with st.spinner('Processing'):
#                             pdf_text = read_pdf(temp_file_path, page_range)
#                             os.remove(temp_file_path)  # Clean up temp file
#                             document = Document(text=pdf_text)
#                             documents = [document]
#                             index = VectorStoreIndex.from_documents(documents, service_context=service_context)
#                             query_engine = index.as_query_engine()
#                             if "query_engine" not in st.session_state:
#                                 st.session_state.query_engine = query_engine
#                             st.session_state.activate_chat = True
#                     except Exception as e:
#                         st.error(f"Error processing the file: {e}")
#                 else:
#                     st.error("Please enter a valid page range.")

#     if st.session_state.activate_chat:
#         if prompt := st.chat_input("Ask your question from the PDF?"):
#             with st.chat_message("user", avatar='👨🏻'):
#                 st.markdown(prompt)
#             st.session_state.messages.append({"role": "user", "avatar": '👨🏻', "content": prompt})
#             query_index_placeholder = st.session_state.query_engine
#             pdf_response = query_index_placeholder.query(prompt)
#             cleaned_response = pdf_response.response
#             with st.chat_message("assistant", avatar='🤖'):
#                 st.markdown(cleaned_response)
#             st.session_state.messages.append({"role": "assistant", "avatar": '🤖', "content": cleaned_response})
#         else:
#             st.markdown('Upload your PDFs to chat')


# if __name__ == '__main__':
#     main()

import json
import os
import asyncio
import streamlit as st
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from llama_index.core import ServiceContext
from llama_index.core import set_global_service_context
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.gradient import GradientEmbedding
from llama_index.llms.gradient import GradientBaseModelLLM
from llama_index.core import Document
from tempfile import NamedTemporaryFile
import fitz  # PyMuPDF for reading PDFs

# Set environment variables
os.environ['GRADIENT_ACCESS_TOKEN'] = 'CK1zDOU4BQ03NoMJMUDndWM8oAoNFMpm'
os.environ['GRADIENT_WORKSPACE_ID'] = '55708ca1-7b2b-42b1-ae1b-7d75ef4bc8c7_workspace'


def get_or_create_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop = asyncio.get_event_loop()
    return loop


@st.cache_resource
def create_datastax_connection():
    cloud_config = {'secure_connect_bundle': r"secure-connect-nlp.zip"}
    with open(r"malviyaayush2609@gmail.com-token.json") as f:
        secrets = json.load(f)
    CLIENT_ID = secrets["clientId"]
    CLIENT_SECRET = secrets["secret"]
    auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    astra_session = cluster.connect()
    return astra_session


def read_pdf(file_path, page_range):
    text = ""
    with fitz.open(file_path) as pdf_document:
        total_pages = pdf_document.page_count
        pages = parse_page_range(page_range, total_pages)
        for page_num in pages:
            page = pdf_document.load_page(page_num)
            text += page.get_text()
    return text


def parse_page_range(page_range, total_pages):
    pages = set()
    ranges = page_range.split(',')
    for r in ranges:
        if '-' in r:
            start, end = r.split('-')
            start = int(start.strip()) - 1
            end = int(end.strip())
            pages.update(range(start, end))
        else:
            pages.add(int(r.strip()) - 1)
    return [p for p in pages if 0 <= p < total_pages]


def main():
    st.set_page_config(page_title="Chat with your PDF using Llama2 & Llama Index", page_icon="🦙")
    st.header('Chat with your PDF using Llama2 model & Llama Index')

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message['avatar']):
            st.markdown(message["content"])

    session = create_datastax_connection()
    GRADIENT_ACCESS_TOKEN = os.environ["GRADIENT_ACCESS_TOKEN"]
    GRADIENT_WORKSPACE_ID = os.environ["GRADIENT_WORKSPACE_ID"]
    loop = get_or_create_event_loop()
    llm = GradientBaseModelLLM(base_model_slug="llama2-7b-chat", max_tokens=400)
    embed_model = GradientEmbedding(
        gradient_access_token=GRADIENT_ACCESS_TOKEN,
        gradient_workspace_id=GRADIENT_WORKSPACE_ID,
        gradient_model_slug="bge-large"
    )
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        chunk_size=256
    )
    set_global_service_context(service_context)

    with st.sidebar:
        st.subheader('Upload Your PDF File')
        docs = st.file_uploader('⬆️ Upload your PDF & Click to process',
                                accept_multiple_files=False,
                                type=['pdf'])
        page_range = st.text_input("Enter the page range to process (e.g., 1-5, 8, 10-12)", "")
        if st.button('Process'):
            if docs is not None:
                if page_range:
                    try:
                        with NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                            temp_file.write(docs.getbuffer())
                            temp_file_path = temp_file.name
                        with st.spinner('Processing'):
                            pdf_text = read_pdf(temp_file_path, page_range)
                            os.remove(temp_file_path)  # Clean up temp file
                            document = Document(text=pdf_text)
                            documents = [document]
                            index = VectorStoreIndex.from_documents(documents, service_context=service_context)
                            query_engine = index.as_query_engine()
                            if "query_engine" not in st.session_state:
                                st.session_state.query_engine = query_engine
                            st.session_state.activate_chat = True
                    except Exception as e:
                        st.error(f"Error processing the file: {e}")
                else:
                    st.error("Please enter a valid page range.")

    if st.session_state.activate_chat:
        if prompt := st.chat_input("Ask your question from the PDF?"):
            with st.chat_message("user", avatar='👨🏻'):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "avatar": '👨🏻', "content": prompt})
            query_index_placeholder = st.session_state.query_engine
            pdf_response = query_index_placeholder.query(prompt)
            cleaned_response = pdf_response.response
            with st.chat_message("assistant", avatar='🤖'):
                st.markdown(cleaned_response)
            st.session_state.messages.append({"role": "assistant", "avatar": '🤖', "content": cleaned_response})
        else:
            st.markdown('Upload your PDFs to chat')


if __name__ == '__main__':
    main()


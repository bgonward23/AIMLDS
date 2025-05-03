import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

st.title("Chatbot")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

@st.cache_resource
def get_vectorstore():
    pdf_name ='/Users/bharathreddy97/Downloads/BHARATHSIMHA ATHURU_Resume.pdf'
    loaders = [PyPDFLoader(pdf_name)]
    index = VectorstoreIndexCreator(
        embedding =HuggingFaceEmbeddings(model_name= 'all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders(loaders)
    return index.vectorstore



prompt = st.chat_input("enter your prompt here")
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content':prompt})

    groq_sys_prompt = ChatPromptTemplate.from_template("""Answer the following question: {user_prompt}. start the answer directly""")  
    groq_chat = ChatGroq(
        #groq_api_key = os.environ.get("GET_API_KEY"),
        groq_api_key = "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",
        model_name = "llama3-8b-8192"
    )
    """chain = groq_sys_prompt | groq_chat| StrOutputParser()
    response =chain.invoke({"user_prompt":prompt})
    
    #response = "I am ur assistant"
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({'role':'assistant','content':response})"""

    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("Failed to load document")
      
        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type='stuff',
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True)
       
        result = chain({"query": prompt})
        response = result["result"]  # Extract just the answer
        #response = get_response_from_groq(prompt)
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append(
            {'role':'assistant', 'content':response})
    except Exception as e:
        st.error(f"Error: {str(e)}")
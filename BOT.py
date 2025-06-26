import os
import time
import requests
import uvicorn
import streamlit as st
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_groq import ChatGroq
from langchain_objectbox.vectorstores import ObjectBox
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# FastAPI application
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)

# Add routes for OpenAI and Ollama
add_routes(app, ChatOpenAI(), path="/openai")
llm = Ollama(model="llama2")

# Define prompts
prompt1 = ChatPromptTemplate.from_template(
    "Write me an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template(
    "Write me a poem about {topic} for a 5 years child with 100 words")

add_routes(app, prompt1 | ChatOpenAI(), path="/essay")
add_routes(app, prompt2 | llm, path="/poem")

# Load documents and create vector store
loader = PyPDFLoader("attention.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(docs)
db = FAISS.from_documents(documents[:30], OpenAIEmbeddings())

# Streamlit application


def get_openai_response(input_text):
    response = requests.post(
        "http://localhost:8000/essay/invoke", json={'input': {'topic': input_text}})
    return response.json()['output']['content']


def get_ollama_response(input_text):
    response = requests.post(
        "http://localhost:8000/poem/invoke", json={'input': {'topic': input_text}})
    return response.json()['output']


st.title('Langchain Demo With LLAMA2 API')
input_text = st.text_input("Write an essay on")
input_text1 = st.text_input("Write a poem on")

if input_text:
    st.write(get_openai_response(input_text))

if input_text1:
    st.write(get_ollama_response(input_text1))

# Retrieval chain setup
retriever = db.as_retriever()
llm = Ollama(model="llama2")
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer.  
<context>
{context}
</context>
Question: {input}""")
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Streamlit for user queries
input_query = st.text_input("Ask a question about attention")
if input_query:
    response = retrieval_chain.invoke({"input": input_query})
    st.write(response['answer'])

# Run FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

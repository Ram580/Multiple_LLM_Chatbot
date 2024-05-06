from sentence_transformers import SentenceTransformer
import pinecone
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Parsing the uploaded documents and creating a single text blob
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

# creating chunks of the text blob
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load environment variables to get Pinecone API Key and env
load_dotenv()
# Access the value of PINECONE_API_KEY
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")

# initialize pinecone
import pinecone
# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "rag-chatbot" # put in the name of your pinecone index here


#  Function that indexes documents into Pinecone
from langchain.vectorstores import Pinecone

# Load the data into pinecone database
def get_vector_store(text_chunks):
   #docsearch = Pinecone.from_texts(chunked_data, embeddings, index_name=index_name)
   index = Pinecone.from_texts([t for t in text_chunks], embeddings, index_name=index_name)
   return index

index = pinecone.Index('langchain-chatbot')

encoder = SentenceTransformer('all-MiniLM-L6-v2')




def find_match(input):
    # input_em = encoder.encode(input).tolist()
    # result = index.query(input_em, top_k=2, includeMetadata=True)
    # return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Pinecone search using the loaded embeddings
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    docs = docsearch.similarity_search(input)
    return docs

# def query_refiner(conversation, query):

#     response = openai.Completion.create(
#     model="text-davinci-003",
#     prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
#     temperature=0.7,
#     max_tokens=256,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0
#     )
#     return response['choices'][0]['text']

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-pro')

def query_refiner(conversation, query):
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"
    response = model.generate_content(prompt)
    return response.text

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string
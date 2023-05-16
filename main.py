import os

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from PyPDF2 import PdfReader
from langchain.document_loaders import OnlinePDFLoader
from langchain.document_loaders import UnstructuredPDFLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import TokenTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI


os.environ["OPENAI_API_KEY"] = "sk-"

QA = None
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
async def welcome():
    return {'hey' : 'there'}
@app.get('/load_book')
async def load_book(pdf_url: str):
    global QA
    # loader = OnlinePDFLoader(pdf_url)
    # loader = UnstructuredPDFLoader('sample.pdf')
    # pdf_data = loader.load()
    # text_splitter = TokenTextSplitter(chunk_size = 750,chunk_overlap  = 50)
    # pdf_doc = text_splitter.split_documents(pdf_data)
    # embedding_cost = ( ( len(pdf_doc) * 750 ) / 1000 ) * 0.0004

    embeddings = OpenAIEmbeddings()
    # vectordb = Chroma.from_documents(pdf_doc, embeddings)
    persist_directory = ''

    if(pdf_url == 'options') :
        persist_directory = 'vector/options'
        
    if(pdf_url == 'family') :
        persist_directory = 'vector/family'

    if(pdf_url == 'honurable') :
        persist_directory = 'vector/hourable'

    if(pdf_url == 'mask') :
        persist_directory = 'vector/mask'

    if(pdf_url == 'truth') :
        persist_directory = 'vector/truth'

    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    prompt_template = """
        You are a helpful AI assistant.
        If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
        If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
        Answer in conversational tone only.
        {context}
        Question: {question}
        Answer:"""
    
    PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
            )
    
    chain_type_kwargs = {"prompt": PROMPT}
    QA = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
        chain_type="stuff", 
        retriever=vectordb.as_retriever(),
        chain_type_kwargs=chain_type_kwargs
    )

    return {"status": pdf_url}


class Data(BaseModel):
    user: str

@app.post('/ask')
async def chat(input: Data):
    global QA
    result = QA.run(input.user)
    return {'result' : result}

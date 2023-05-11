from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import Chroma
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import VectorDBQA
import os

app = Flask(__name__)

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Base query for the chat model
base_query = "Your friendly assistant, please provide a conversational long answer to my question. Use numbered bullets when required. Question: "

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    file = request.files['file']
    file_path = './uploaded_file.pdf'  # Path to save the uploaded file

    if file and file.filename.endswith('.pdf'):
        file.save(file_path)

        # Load PDF data
        loader = UnstructuredPDFLoader(file_path)
        my_data = loader.load()

        # Split the document into chunks
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=50)
        my_doc = text_splitter.split_documents(my_data)

        # Create embeddings
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(my_doc, embeddings)

        # Set up the chat model and QA system
        qa = VectorDBQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", vectorstore=vectordb)

        # Generate the answer
        answer = qa.run(base_query + question)

        os.remove(file_path)  # Delete the uploaded file after processing
    else:
        answer = "Please upload a valid PDF file."

    return render_template('index.html', question=question, answer=answer)


if __name__ == '__main__':
    app.run()

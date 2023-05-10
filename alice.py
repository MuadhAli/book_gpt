
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS


from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader



import os
os.environ["OPENAI_API_KEY"] = ""


from langchain.vectorstores import Chroma
from langchain.prompts.prompt import PromptTemplate

loader = UnstructuredPDFLoader('./chat_pdf.pdf')
my_data = loader.load()

text_splitter = TokenTextSplitter(chunk_size = 1000,chunk_overlap  = 50)
my_doc = text_splitter.split_documents(my_data)

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(my_doc, embeddings)


from langchain.chat_models import ChatOpenAI


from langchain.chains import VectorDBQA


qa = VectorDBQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", vectorstore=vectordb,


base_query = """ Your like my friendly assistant, provide a conversational long answer to my question. Use numbered bullets when required. Question :  """
# base_query = """ Your like my friendly assistant, provide a conversational long answer to my question. Question :  """



query = "write a facebook carousal post for promoting this book"
qa.run(query)



query = "write me the chapter wise summary for it"
qa.run(query)



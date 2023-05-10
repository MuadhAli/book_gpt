#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install langchain')
get_ipython().system('pip install openai')
get_ipython().system('pip install PyPDF2')
get_ipython().system('pip install faiss-cpu')
get_ipython().system('pip install tiktoken')
get_ipython().system('pip install chromadb')
get_ipython().system('pip install "unstructured[local-inference]"')


# In[3]:


# import os
# os.environ["OPENAI_API_KEY"] = "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"


# In[4]:


from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS


from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader


# In[5]:


import os
os.environ["OPENAI_API_KEY"] = "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"


# In[6]:


from langchain.vectorstores import Chroma
from langchain.prompts.prompt import PromptTemplate

loader = UnstructuredPDFLoader('./chat_pdf.pdf')
my_data = loader.load()

text_splitter = TokenTextSplitter(chunk_size = 1000,chunk_overlap  = 50)
my_doc = text_splitter.split_documents(my_data)

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(my_doc, embeddings)


# In[7]:


from langchain.chat_models import ChatOpenAI

# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     AIMessagePromptTemplate,
#     HumanMessagePromptTemplate,
# )
# from langchain.schema import (
#     AIMessage,
#     HumanMessage,
#     SystemMessage
# )
from langchain.chains import VectorDBQA

# system_template="""
# {context}
# """
# messages = [
#     SystemMessagePromptTemplate.from_template(system_template),
#     HumanMessagePromptTemplate.from_template("{question}")
# ]
# prompt = ChatPromptTemplate.from_messages(messages)
# chain_type_kwargs = {"prompt": prompt}

qa = VectorDBQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", vectorstore=vectordb,
                                #chain_type_kwargs=chain_type_kwargs
                                )

# https://www.chatpdf.com/c/gvPpOKTiZTL5yIsCG3bG8


# In[8]:


#base_query = """ Your like my friendly assistant, provide a conversational long answer to my question. Use numbered bullets when required. Question :  """
base_query = """ Your like my friendly assistant, provide a conversational long answer to my question. Question :  """


# In[9]:


query = "write a facebook carousal post for promoting this book"
qa.run(query)


# In[10]:


query = "write me the chapter wise summary for it"
qa.run(query)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





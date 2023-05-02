from flask import Flask, render_template, request, session
import os
import tempfile
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredPDFLoader

os.environ['OPENAI_API_KEY'] = 'sk-...'

app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.route('/', methods=['GET', 'POST'])
def home():
    question = request.form.get('query') or request.args.get('query') or session.get('question', '')
    session['question'] = question
    result = None

    if request.method == 'POST':
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            # Save the uploaded file to a temporary file on disk
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file.flush()

                # Load the PDF from the temporary file
                loader = UnstructuredPDFLoader(temp_file.name)
                documents = loader.load()
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                texts = text_splitter.split_documents(documents)
                embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
                docsearch = Chroma.from_documents(texts, embeddings)
                qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch, return_source_documents=True)

                query = question
                result = qa({"query": query})

    return render_template('index.html', question=question, result=result)


if __name__ == '__main__':
    app.run(debug=True)

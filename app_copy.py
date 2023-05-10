
# from flask import Flask, request, render_template
# from langchain import OpenAI
# import os
# from pathlib import Path
# from llama_index import GPTSimpleVectorIndex, LLMPredictor, ServiceContext, download_loader

# os.environ["OPENAI_API_KEY"] = ""

# app = Flask(__name__)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         file = "chat_pdf.pdf"
#         PDFReader = download_loader("PDFReader")
#         loader = PDFReader()
#         documents = loader.load_data(file=Path(file))

#         llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

#         service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=1024)
#         index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

#         prompt = request.form['question']
#         response = index.query(prompt)
#         return render_template('index.html', response=response)

#     return render_template('index.html')


# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, render_template
from langchain import OpenAI
import os
from pathlib import Path
from llama_index import GPTSimpleVectorIndex, LLMPredictor, ServiceContext, download_loader

os.environ["OPENAI_API_KEY"] = ""

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'pdf_file' not in request.files:
            return "No file uploaded."

        file = request.files['pdf_file']
        if file.filename == '':
            return "No file selected."

        # Save the uploaded file to disk
        file_path = "uploaded_pdf.pdf"
        file.save(file_path)

        PDFReader = download_loader("PDFReader")
        loader = PDFReader()
        documents = loader.load_data(file=Path(file_path))

        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=1024)
        index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

        prompt = request.form['question']
        response = index.query(prompt)
        return render_template('index.html', response=response)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

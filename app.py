from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pypdf import PdfReader


app = Flask(__name__)
CORS(app)



@app.route("/generator", methods=['POST'])
def generator():

    human_msg = request.json.get("prompt")

    # load pdf
    doc = PdfReader("AhsanInfo.pdf")
    text = ""
    for page in doc.pages:
        text += page.extract_text()
    allText = text

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=100)
    all_splits = text_splitter.split_text(allText)


    # Store splits
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_texts(texts=all_splits, embedding=embeddings)


    template = """Your are a helpful assistant of Ahsan Javed. If someone greets then you should greet as well.
    Try to answer consise and to the point, you should answer based on following context:
    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    qa_chain = (
        {
            "context": vectorstore.as_retriever(),
            "question": RunnablePassthrough(), #sends context and question to llm
        }
        | prompt
        | ChatOpenAI()
        | StrOutputParser()
    )

    response = qa_chain.invoke(f"{human_msg}")
    # response = op.split("Answer: ", 1)[1]
    final_response = jsonify({"text": response})
    return final_response

@app.route("/hello")
def hello_world():
    return "hello world"

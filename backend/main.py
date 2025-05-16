from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Globals to persist across calls
rag_chain = None

class ChatRequest(BaseModel):
    question: str

def build_rag_chain(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.3,
        max_tokens=500,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Keep it under 3 sentences.\n\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with open("uploaded.pdf", "wb") as f:
            f.write(contents)

        global rag_chain
        rag_chain = build_rag_chain("uploaded.pdf")
        return {"message": "PDF uploaded and processed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fine-prints")
def extract_fine_prints():
    if not rag_chain:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet.")
    query = (
        "Extract the key fine prints and critical requirements that should be noted "
        "when drafting a proposal for this project. Summarize important clauses, deadlines, and compliance terms."
    )
    result = rag_chain.invoke({"input": query})
    return {"answer": result["answer"]}

@app.post("/chat")
def chat_with_pdf(request: ChatRequest):
    if not rag_chain:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet.")
    result = rag_chain.invoke({"input": request.question})
    return {"answer": result["answer"]}

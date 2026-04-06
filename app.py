from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import shutil
import uuid

app = FastAPI(title="Document Chunking API")

# -------------------------------
# Helper: Get Loader Based on File Type
# -------------------------------
def get_loader(file_path: str, file_type: str):
    if file_type == "pdf":
        return PyPDFLoader(file_path)
    elif file_type == "csv":
        return CSVLoader(file_path)
    elif file_type == "txt":
        return TextLoader(file_path)
    elif file_type == "docx":
        return Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type")

# -------------------------------
# API Endpoint
# -------------------------------
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Create temp file
        file_ext = file.filename.split(".")[-1].lower()
        temp_filename = f"temp_{uuid.uuid4()}.{file_ext}"

        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load document
        loader = get_loader(temp_filename, file_ext)
        documents = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        # Prepare JSON output
        output = []
        for i, chunk in enumerate(chunks):
            output.append({
                "chunk_id": i,
                "content": chunk.page_content,
                "metadata": chunk.metadata
            })

        # Cleanup
        os.remove(temp_filename)

        return JSONResponse(content={
            "total_chunks": len(output),
            "chunks": output
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
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
# 1. Health Check (Public API)
# -------------------------------
@app.get("/health")
def health_check():
    return {"status": "running"}

# -------------------------------
# 2. Upload & Chunk (Main API)
# -------------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_ext = file.filename.split(".")[-1].lower()
        temp_filename = f"temp_{uuid.uuid4()}.{file_ext}"

        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        loader = get_loader(temp_filename, file_ext)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        output = []
        for i, chunk in enumerate(chunks):
            output.append({
                "chunk_id": i,
                "content": chunk.page_content,
                "metadata": chunk.metadata
            })

        os.remove(temp_filename)

        return JSONResponse(content={
            "total_chunks": len(output),
            "chunks": output
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# 3. Dummy Predict API (for rate limit testing)
# -------------------------------
@app.get("/predict")
def predict():
    return {"prediction": "This is a dummy prediction"}

# -------------------------------
# 4. Heavy API (simulate load)
# -------------------------------
@app.get("/heavy")
def heavy_operation():
    data = []
    for i in range(10000):
        data.append(i*i)
    return {"message": "Heavy operation done", "size": len(data)}

# -------------------------------
# 5. Admin API (to secure later)
# -------------------------------
@app.get("/admin")
def admin_panel():
    return {"message": "Sensitive admin data"}
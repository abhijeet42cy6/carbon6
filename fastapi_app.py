from fastapi import FastAPI, UploadFile, File, Request
from pydantic import BaseModel
from typing import List
from llama_index.core import Settings, VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from document_processors import load_multimodal_data
from utils import set_environment_variables
import logging
from fastapi.middleware.cors import CORSMiddleware

import os
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "chat_templates"))
maps1_templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), ""))

# Initialize FastAPI app
app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "chat_templates/ChatUI")), name="static")
app.mount("/assets", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "chat_templates/ChatUI/assets")), name="assets")
app.mount("/maps1/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "maps1")), name="maps1_static")

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set up environment variables
set_environment_variables()

# Initialize settings
def initialize_settings():
    Settings.embed_model = NVIDIAEmbedding(model="nvidia/nv-embedqa-e5-v5", truncate="END")
    Settings.llm = NVIDIA(model="meta/llama-3.1-70b-instruct")

initialize_settings()

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: QueryRequest):
    try:
        # Use NVIDIA LLM for chat responses
        llm = NVIDIA(model="meta/llama-3.1-70b-instruct")
        response = llm.complete(request.query)
        return {"response": response.text}
    except Exception as e:
        logging.exception("Error during query execution")
        return {"error": str(e)}

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    global index
    set_environment_variables()

    try:
        # Debugging: Print files to see what's being uploaded
        for file in files:
            logging.info(f"File type: {type(file)}")
            logging.info(f"File filename: {file.filename}")
            logging.info(f"File content type: {file.content_type}")
        logging.info(f"Files prepared: {[file.filename for file in files]}")  # Log filenames

        # Process the uploaded files
        documents = await load_multimodal_data(files)
        
        # Create and store the index
        index = VectorStoreIndex.from_documents(documents)
        return {"message": "Documents uploaded and indexed successfully"}
    
    except Exception as e:
        logging.error(f"Error during file processing: {e}")
        return {"error": str(e)}


@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("ChatUI/index.html", {"request": request}) 

@app.get("/chat_ui", response_class=HTMLResponse)
async def chat_ui(request: Request):
    return templates.TemplateResponse("ChatUI/index_chat.html", {"request": request})

@app.get("/maps1", response_class=HTMLResponse)
async def map1(request: Request):
    return maps1_templates.TemplateResponse("maps1/index.html", {"request": request})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, but restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    print("Server starting... Access the application at: http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)

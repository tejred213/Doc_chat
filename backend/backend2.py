import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from typing import List, Dict
import uuid

# Hash dependencies
from utils.hash import add_to_hash_map, get_file_hash_map
# Embedding dependencies
from utils.chat.embedding import get_embeddings
# Chat response dependencies
from utils.chat.chat import get_chat_response
# Vector store dependencies
from utils.vector_store.vector_store import chromadb_vector_store
# Chunk dependencies
from utils.chunk import chunk_text
# Extension dependencies
from utils.extension import get_file_extension
# Extractor dependencies
from utils.extractor.txt_extractor import extract_text_from_txt
from utils.extractor.pdf_extractor import extract_text_from_pdf
from utils.extractor.docx_extractor import extract_text_from_docx
from utils.extractor.csv_extractor import extract_text_from_csv
from utils.extractor.xlxs_extractor import extract_text_from_xlsx

# Initialize ChromaDB client and session collection
client = chromadb.HttpClient(host='localhost', port=8001)
session_collection = client.get_or_create_collection(name="sessions")

# Class model for the request body
file_hash_map = get_file_hash_map()

class Session(BaseModel):
    session_id: str
    conversation_history: List[Dict[str, str]]

class Data(BaseModel):
    """
    Model to define the request body for asking questions.

    Attributes:
        question (str): The question to be answered.
        file_names (List[str]): List of file names to search for answers.
    """
    question: str
    file_names: List[str]

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for all origins (development purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to create necessary directories
def create_directory(directory_path):
    """
    Creates a directory if it doesn't exist.

    Args:
        directory_path (str): Path of the directory to be created.
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        print(f"Directory created at: {directory_path}")
    except OSError as e:
        print(f"Error creating directory: {e}")

# Function to split the filenames with blank spaces only
def split_filename(file_name):
    """
    Splits a filename into parts based on spaces.

    Args:
        file_name (str): The name of the file.

    Returns:
        list: A list of parts of the filename split by spaces.
    """
    try:
        # Split the filename by spaces
        file_parts = file_name.split(" ")
        return file_parts
    except Exception as e:
        print(f"Error splitting filename: {e}")
        return []

# Save uploaded files locally
@app.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads a file and saves it locally.

    Args:
        file (UploadFile): The file to be uploaded.

    Returns:
        dict: Contains the filename and file path.
    """
    try:
        upload_directory = 'uploads'
        create_directory(upload_directory)
        # Keeping the original file name as it is.
        file_name = file.filename
        
        # Split the filename into parts
        file_parts = split_filename(file_name)
        print(f"File parts: {file_parts}")
        
        file_path = os.path.join(upload_directory, file_name)
        
        # Save the uploaded file to the specified directory
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {"filename": file.filename, "file_path": file_path}
    except Exception as e:
        return {"error": str(e)}

# Function to extract text based on file type
def extract_text(file_path):
    """
    Extracts text from a file based on its extension.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: Extracted text from the file.
    """
    try:
        ext = get_file_extension(file_path)
        if ext == '.pdf':
            return extract_text_from_pdf(file_path)
        elif ext == '.docx':
            return extract_text_from_docx(file_path)
        elif ext == '.txt':
            return extract_text_from_txt(file_path)
        elif ext == '.csv':
            return extract_text_from_csv(file_path)
        elif ext == '.xlsx':
            return extract_text_from_xlsx(file_path)
        else:
            raise ValueError("Unsupported file type")
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

# FastAPI endpoint to handle file upload and embedding
@app.post("/process-file/")
async def process_file(file: UploadFile = File(...)):
    """
    Endpoint to process an uploaded file, extract text, generate embeddings, and store them in ChromaDB.

    Parameters:
    file (UploadFile): The file uploaded by the user.

    Returns:
    JSONResponse: A response indicating the success or failure of the file processing.
    """
    try:
        upload_response = await upload_file(file)
        file_path = upload_response["file_path"]
        content = extract_text(file_path)
        chunks = chunk_text(content)
        embeddings = get_embeddings(file.filename, "all-minilm", chunks)
        chromadb_vector_store(embeddings, chunks, collection_name=file.filename)
        add_to_hash_map(file.filename)
        return {"message": "File processed and embeddings stored successfully"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Creating a session to store the previous chat
def create_session() -> str:
    session_id = str(uuid.uuid4())
    # Initialize the session data with an empty conversation history
    session_collection.add_documents([{
        "session_id": session_id,
        "conversation_history": []
    }])
    return session_id

def get_session(session_id: str) -> dict:
    results = session_collection.get_documents(query={"session_id": session_id})
    if results:
        return results[0]  # Return the first matching session (assuming session_id is unique)
    return None

def update_session(session_id: str, message: Dict[str, str]):
    session = get_session(session_id)
    if session:
        # Append the new message to the conversation history
        session["conversation_history"].append(message)
        # Update the session in ChromaDB
        session_collection.update_documents([{
            "session_id": session_id,
            "conversation_history": session["conversation_history"]
        }])

@app.post("/start-session/")
async def start_session():
    session_id = create_session()
    return {"session_id": session_id}

# FastAPI endpoint to ask questions based on the document
@app.post("/ask-question/")
async def ask_question(request: Request, data: Data):
    session_id = request.headers.get("session-id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")
    
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        question = data.question
        file_names = data.file_names

        if not question or not file_names:
            return JSONResponse(status_code=400, content={"error": "Question and file_names are required"})

        if not file_hash_map:
            return JSONResponse(status_code=400, content={"error": "No file uploaded"})
        
        for file_name in file_names:
            if file_name not in file_hash_map:
                return JSONResponse(status_code=400, content={"error": f"File not found: {file_name}"})

        # Process the question using the conversation history
        conversation_history = session["conversation_history"]
        
        # Add the new question to the conversation history
        update_session(session_id, {"role": "user", "content": question})

        # Generate a response based on the conversation history
        response = generate_response(conversation_history)

        # Add the response to the conversation history
        update_session(session_id, {"role": "assistant", "content": response})

        # StreamingResponse to stream the response
        return StreamingResponse(get_chat_response(question, file_names), media_type='text/event-stream')

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def generate_response(conversation_history: List[Dict[str, str]]) -> str:
    # Implement your response generation logic here
    return "This is a placeholder response based on the conversation history."

# Function to delete the embeddings and collection from ChromaDB
def delete_from_chromadb(collection_name):
    """
    Delete a collection and its associated embeddings from ChromaDB.

    Parameters:
    collection_name (str): The name of the collection to delete.
    """
    try:
        collection = client.get_collection(name=collection_name)
        if collection:
            client.delete_collection(collection_name)
            print(f"Collection {collection_name} deleted from ChromaDB")
    except Exception as e:
        print(f"Error deleting collection from ChromaDB: {e}")

# FastAPI endpoint to delete a file and its associated data
@app.post("/delete-file/")
async def delete_file(file_name: str):
    """
    Endpoint to delete a file and its associated data, including embeddings and collections from ChromaDB.

    Parameters:
    file_name (str): The name of the file to delete.

    Returns:
    JSONResponse: A response indicating the success or failure of the deletion.
    """
    try:
        # Delete the file from the local storage
        file_path = os.path.join('uploads', file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_name} deleted from storage")
        else:
            return JSONResponse(status_code=404, content={"error": "File not found"})
        
        # Delete the file embeddings from ChromaDB
        delete_from_chromadb(file_name)

        return JSONResponse(status_code=200, content={"message": "File and its embeddings deleted successfully"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
from fastapi import FastAPI, UploadFile, File, HTTPException
import os
from doc_parser import process_documents
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
import json

from agents import chat_with_rag

app = FastAPI()

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


executor = ThreadPoolExecutor(max_workers=4)

@app.post("/upload-docs")
async def upload_documents(
    scope: UploadFile = File(...),
    requirements: UploadFile = File(...),
    risks: UploadFile = File(...)
):
    """
    Handles the upload of project documents and processes them into Markdown files.
    """
    saved_files = []
    for uploaded_file in [scope, requirements, risks]:
        file_path = Path(UPLOAD_DIR) / uploaded_file.filename
        # Asynchronously read the file and save it
        with open(file_path, "wb") as buffer:
            buffer.write(await uploaded_file.read())
        saved_files.append(file_path.resolve())

    scope_path, requirements_path, risks_path = map(str, saved_files)

    try:
        # Use the global executor and pass the function and args directly
        md_files = await asyncio.get_running_loop().run_in_executor(
            executor,
            process_documents,
            scope_path,
            requirements_path,
            risks_path
        )
        return {"message": "Documents processed successfully", "markdown_files": md_files}
    except Exception as e:
        print(f"Error processing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-risk-register")
async def generate_risk_register():
    """
    Calls the RAG agent to generate a new risk register based on the parsed documents.
    """
    try:
        question = "Generate a new risk register for the project based on current scope and requirements."
        
        agent_response_content = await chat_with_rag(question)

        match = re.search(r'```json\s*(.*?)\s*```', agent_response_content, re.DOTALL)
        if match:
            json_str = match.group(1)
            parsed_json = json.loads(json_str)
            return parsed_json
        else:
            print("No JSON block found in response")
            raise HTTPException(status_code=500, detail=f"Failed to parse JSON structure")

    except Exception as e:
        print(f"Error generating risk register: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate risk register: {e}")

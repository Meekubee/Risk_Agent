import os
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

def read_file(path):
    encodings_to_try = ['utf-8', 'windows-1252', 'cp1252', 'latin1']
    
    for encoding in encodings_to_try:
        try:
            with open(path, "r", encoding=encoding) as f:
                content = f.read()
                print(f"Successfully read {path} with encoding: {encoding}")
                return content
        except UnicodeDecodeError:
            continue
            
    raise Exception(f"Could not read {path}")

scope_md = read_file("scope_doc.md")
req_md = read_file("req_doc.md")
risk_md = read_file("risk_doc.md")

print("All files read successfully!")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

scope_chunks = splitter.split_text(scope_md)
req_chunks = splitter.split_text(req_md)
risk_chunks = splitter.split_text(risk_md)

print(f"Text split into {len(scope_chunks)} scope, {len(req_chunks)} requirement, and {len(risk_chunks)} risk chunks")

client = chromadb.PersistentClient(path="./rag_vector_store")

try:
    client.delete_collection("project_docs")
    print("Existing collection deleted")
except:
    pass

collection = client.create_collection(
    name="project_docs",
    metadata={"hnsw:space": "cosine"}
)

documents = []
metadatas = []
ids = []


for i, chunk in enumerate(scope_chunks):
    documents.append(chunk)
    metadatas.append({"source": "scope", "doc_type": "scope", "chunk_index": i})
    ids.append(f"scope_{i}")

for i, chunk in enumerate(req_chunks):
    documents.append(chunk)
    metadatas.append({"source": "requirements", "doc_type": "requirements", "chunk_index": i})
    ids.append(f"req_{i}")

for i, chunk in enumerate(risk_chunks):
    documents.append(chunk)
    metadatas.append({"source": "risks", "doc_type": "risks", "chunk_index": i})
    ids.append(f"risk_{i}")

print(f"Adding {len(documents)} documents to vector store...")

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print("Vector store created and populated successfully!")
print(f"Total documents in collection: {collection.count()}")


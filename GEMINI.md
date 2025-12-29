# Project Overview

This project is a Python-based Risk Agent that processes uploaded documents (PDFs) to generate a risk register for project management. It uses LlamaParse for document parsing, FastAPI for creating a web server with API endpoints, and a Large Language Model (LLM) via the `autogen-agentchat` library for generating the risk register. The project leverages a Retrieval-Augmented Generation (RAG) approach, using `chromadb` as a vector store for the parsed document content.

The application works as follows:
1.  The user uploads three PDF documents: a project scope, a requirements document, and a historical risk data document.
2.  The backend, built with FastAPI, receives the files.
3.  `LlamaParse` is used to parse the PDF documents into Markdown format.
4.  The parsed Markdown content is then processed and ingested into a `chromadb` vector store using `langchain_text_splitters`.
5.  The user can then trigger the generation of a risk register.
6.  The application queries the vector store for relevant information based on the project scope and requirements.
7.  An AI agent, built with `autogen-agentchat`, uses the retrieved context and a configured LLM to generate a new risk register in JSON format.
8.  The user can download the generated risk register as a DOCX or PDF file.

# Building and Running

## Prerequisites

-   Python 3.9+
-   API Keys for:
    -   LlamaParse Cloud API
    -   An LLM API compatible with the OpenAI Python client (e.g., Gemini)
-   Git

## Installation and Running

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Meekubee/Ana_POC.git
    cd Ana_POC
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv risk_env
    source risk_env/bin/activate  # On Windows, use `risk_env\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the `risk_agent/src` directory with the following content:
    ```
    LLAMA_CLOUD_API_KEY="your_llamaparse_api_key"
    GEMINI_API_KEY="your_gemini_api_key"
    GEMINI_ENDPOINT="your_gemini_endpoint"
    GEMINI_MODEL="your_gemini_model"
    ```

5.  **Run the application:**
    ```bash
    uvicorn risk_agent.src.main:app --reload
    ```
    The application will be available at `http://127.0.0.1:8000`.

# Development Conventions

-   **Code Style:** The code follows the PEP 8 style guide for Python.
-   **Dependencies:** Project dependencies are managed in the `requirements.txt` file.
-   **Modularity:** The code is organized into modules with specific responsibilities:
    -   `main.py`: FastAPI application entry point and API endpoints.
    -   `agents.py`: Defines the AI agent for risk analysis.
    -   `doc_parser.py`: Handles parsing of uploaded documents.
    -   `rag_ingest.py`: Manages the creation and population of the vector store.
-   **Testing:** There are no explicit test files in the project. However, `agents.py` contains a `main` function that can be used for testing the agent's functionality.

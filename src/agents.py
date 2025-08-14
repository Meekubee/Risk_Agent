import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from autogen_core import CancellationToken
from autogen_agentchat.messages import TextMessage
from pathlib import Path
import json
import re
import os

from dotenv import load_dotenv
from rag_ingest import collection

load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')
gemini_endpoint = os.getenv('GEMINI_ENDPOINT')
gemini_model = os.getenv('GEMINI_MODEL')

model_client = OpenAIChatCompletionClient(
    model=gemini_model,
    name="Google",
    api_key=gemini_api_key,
    base_url=gemini_endpoint,
    model_info=ModelInfo(
        vision=True,
        function_calling=True,
        json_output=True,
        family="gemini",
        structured_output=True
    ),
)

def extract_and_print_json(content: str):
    """
    Extracts and pretty prints a JSON block from a string.
    """
    match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            print(json.dumps(data, indent=2))
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)
            print("Raw content:\n", match.group(1))
    else:
        print("No JSON block found in response.")
        print("Full content:\n", content)

async def query_documents(query_text, n_results=5):
    """
    Query the vector store for relevant documents (async version)
    """
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas"]
        )
        context_parts = []
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            source = metadata.get('source', 'unknown')
            doc_type = metadata.get('doc_type', 'unknown')
            context_parts.append(f"[Source: {source} - {doc_type}]\n{doc}\n")
        return "\n".join(context_parts)
    except Exception as e:
        print(f"Error querying documents: {e}")
        return "No relevant documents found."

risk_agent = AssistantAgent(
    "risk_analyzer",
    model_client=model_client,
    system_message="""
You are a risk analysis expert with access to project documentation.

When answering questions:
1. I will provide you with the scope and requirements documents of the current project, and a historical risk document for formatting reference only.
2. Always cite the source of your information using the [Source: ...] tags provided.
3. Your job is to generate a new risk register FOR the current project based on the scope and requirements. Do not copy risks from the historical document.
4. For each new risk register, the RISK_ID must start at 1 and increment sequentially.
5. Clearly state RISK_ID, RISK_DESCRIPTION, LIKELIHOOD, IMPACT, and MITIGATION PLAN for the generated risk register. The output must be a single JSON body containing a list of risk objects.
""",
)

async def chat_with_rag(user_question: str) -> str:
    """
    RAG-powered function to:
    1. Retrieve context from vector DB
    2. Send query to the risk_analyzer agent
    3. Return the agent's response (as string)
    """
    print(f"[User Question] {user_question}")
    print("[RAG] Retrieving relevant documents...")
    context = await query_documents(user_question)

    prompt = f"""Context from project documents:
{context}

User question: {user_question}

Please answer based on the provided context and cite your sources."""

    token = CancellationToken()
    user_msg = TextMessage(content=prompt, source="user")

    print("[Agent] Sending prompt to risk_analyzer agent...")
    response = await risk_agent.on_messages([user_msg], cancellation_token=token)

    return response.chat_message.content  # Return the actual string content

async def main():
    test_questions = [
        "Generate a new risk register for the project based on current scope and requirements."
    ]

    for question in test_questions:
        print(f"\n{'=' * 50}")
        print(f"Question: {question}")
        print(f"{'=' * 50}")

        # Show retrieved documents
        context = await query_documents(question)
        print("Retrieved context:")
        print(context[:500] + "..." if len(context) > 500 else context)

        print("\nAgent response:")
        try:
            raw_response = await chat_with_rag(question)
            extract_and_print_json(raw_response)
        except Exception as e:
            print(f" Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())

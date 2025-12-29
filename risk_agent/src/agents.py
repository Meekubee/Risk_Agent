import logging
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

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

async def query_documents(collection, query_text, n_results=5):
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    return results

risk_agent = AssistantAgent(
    "risk_analyzer",
    model_client=model_client,
    system_message="""
You are a risk analysis expert.

IMPORTANT OUTPUT RULES (MANDATORY):
- Respond with ONLY valid JSON
- Do NOT include explanations or markdown
- Do NOT wrap output in ``` fences
- The response MUST be directly parseable by json.loads()
- The root element MUST be a JSON array

Each object in the array MUST contain:
- RISK_ID (int, starting from 1, sequential)
- RISK_DESCRIPTION (string)
- LIKELIHOOD (string)
- IMPACT (string)
- MITIGATION_PLAN (string)

Use the provided context as the primary reference.
Infer and generate plausible risks that logically arise from the project scope and requirements.
Do NOT copy risks verbatim from historical documents.
Do NOT invent risks unrelated to the project.
"""
)


async def chat_with_rag(collection, user_question: str) -> str:
    """
    RAG-powered function to:
    1. Retrieve context from vector DB
    2. Send query to the risk_analyzer agent
    3. Return the agent's response (as string)
    """
    logging.info(f"[User Question] {user_question}")
    logging.info("[RAG] Retrieving relevant documents...")
    context = await query_documents(collection, user_question)
    logging.info(f"Context retrieved from vector store:\n{context}")

    # Extract document text from the context
    retrieved_docs = context.get("documents", [[]])[0]
    context_str = "\n\n".join(retrieved_docs)
    logging.info(f"Formatted context string:\n{context_str}")

    prompt = f"""Context from project documents:
{context_str}

User question: {user_question}

Please answer based on the provided context."""
    logging.info(f"Prompt sent to agent:\n{prompt}")

    token = CancellationToken()
    user_msg = TextMessage(content=prompt, source="user")

    logging.info("[Agent] Sending prompt to risk_analyzer agent...")
    response = await risk_agent.on_messages([user_msg], cancellation_token=token)
    logging.info(f"Raw response from agent:\n{response.chat_message.content}")

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

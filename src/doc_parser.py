# doc_parser.py

from llama_parse import LlamaParse
import os
from dotenv import load_dotenv
import time

load_dotenv()
llamaparse_api_key = os.getenv('LLAMA_CLOUD_API_KEY')

def process_documents(scope_path, requirements_path, risks_path):
    # Creating the parser inside the function for thread-safety
    parser = LlamaParse(
        api_key=llamaparse_api_key,
        result_type="markdown",
        num_workers=1, # Set to 1 to avoid internal concurrency conflicts with the executor
    )

    def safe_load(path, max_retries=3, delay_seconds=2):
        for attempt in range(max_retries):
            try:
                data = parser.load_data(path)
                if not data:
                    if attempt < max_retries - 1:
                        print(f"Parsing failed for {path}, retrying in {delay_seconds}s... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay_seconds)
                        continue
                    else:
                        raise ValueError(f"Parsing failed for {path}: No data returned from parser after multiple attempts.")
                return data[0]
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"An error occurred while parsing {path}, retrying in {delay_seconds}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay_seconds)
                    continue
                else:
                    raise e
        return None

    try:
        risk_doc = safe_load(risks_path)
        req_doc = safe_load(requirements_path)
        scope_doc = safe_load(scope_path)
    except ValueError as e:
        raise e

    risk_file_name = "risk_doc.md"
    with open(risk_file_name, 'w') as file:
        file.write(risk_doc.text)

    req_file_name = "req_doc.md"
    with open(req_file_name, 'w') as file:
        file.write(req_doc.text)

    scope_file_name = "scope_doc.md"
    with open(scope_file_name, 'w') as file:
        file.write(scope_doc.text)

    return [risk_file_name, req_file_name, scope_file_name]

if __name__ == "__main__":
    # Your local test code here
    pass
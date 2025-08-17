# Risk Agent

It processes uploaded documents (PDFs) using **LlamaParse** and an **LLM model**, generating a **risk register** based on the provided scope, requirements, and historical risk data. Aims to ease and fasten the risk identification, risk analysis, and risk mitigation process in the project planning phase in PMPs

---

## 📌 Prerequisites

Before running the Risk Agent, make sure you have:

- **Python 3.9+** installed
- **API Keys** for:
  - [LlamaParse Cloud API](https://llamacloud.com)
  - **An LLM API** compatible with the OpenAI Python client  
    (e.g., OpenAI GPT, Azure OpenAI, Anthropic Claude with OpenAI client wrapper, etc.)
- **Git** installed

---

## ⚡ Quick Start

```bash
git clone https://github.com/Meekubee/Ana_POC.git
cd Ana_POC
python -m venv risk_env && source risk_env/bin/activate
pip install -r requirements.txt

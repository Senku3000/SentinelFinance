# SentinelFinance — AI Personal Financial Adviser

An AI-powered financial adviser built with LangGraph. Upload your financial documents, ask questions, and get personalized advice based on your actual numbers.

## Features

- **Document Upload** — Salary slips, tax returns, expense sheets (PDF, Excel, CSV). AI extracts income, expenses, investments, and tax details automatically.
- **Agentic RAG** — Retrieves relevant info from your uploaded documents and a financial knowledge base via FAISS.
- **Web Search** — Real-time product prices, market rates, and financial data via Tavily.
- **Precise Calculations** — EMI, SIP, tax, ROI computed in a sandboxed Python REPL (not guessed by the LLM).
- **Direct Opinions** — Uses your actual numbers, gives real financial opinions, not generic advice.
- **User Auth** — Signup/login with persistent profiles, chat history, and document storage.

## Architecture

Multi-agent pipeline built on LangGraph:

```
Query → Router → Researcher (FAISS + Tavily + yfinance)
              → Analyst (Python Math REPL)
              → Strategist → Personalized Answer
```

## Tech Stack

- **Backend**: FastAPI + Jinja2 templates
- **Database**: MySQL (users, profiles, chat history)
- **LLM**: Groq (Llama 3.3-70B)
- **Vector DB**: FAISS (per-user indexes)
- **Web Search**: Tavily
- **Document Processing**: LangChain + sentence-transformers

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure .env
cp .env.example .env
# Edit .env with your keys:
#   GROQ_API_KEY=...
#   TAVILY_API_KEY=...
#   MYSQL_USER=root
#   MYSQL_PASSWORD=...
#   MYSQL_HOST=localhost
#   MYSQL_DB=sentinel_finance

# 4. Create MySQL database
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS sentinel_finance;"

# 5. Ingest knowledge base (one-time)
python ingest_documents.py

# 6. Run the app
uvicorn main:app --reload
```

Open `http://localhost:8000`

## Project Structure

```
fin_adviser/
├── main.py                 # FastAPI entry point
├── web/
│   ├── routes.py           # Page routes + form handlers
│   ├── auth.py             # Session cookies
│   ├── dependencies.py     # Auth middleware
│   └── templates/          # Jinja2 HTML (home, login, signup, dashboard, profile)
├── db/
│   ├── database.py         # SQLAlchemy + MySQL
│   ├── models.py           # User, Profile, ChatMessage, Document
│   └── crud.py             # All DB operations
├── src/
│   ├── graph.py            # LangGraph workflow
│   ├── nodes.py            # Router, Researcher, Analyst, Strategist
│   ├── state.py            # State management
│   ├── config.py           # Configuration
│   ├── tools/              # SearchTool, MathTool, VectorDBTool, UserVaultTool
│   └── ingestion/          # DocumentParser, UserEmbedder, LLMExtractor
├── data/
│   └── documents/          # Knowledge base (tax, investments, loans)
└── requirements.txt
```

## License

MIT

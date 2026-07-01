# SentinelFinance

SentinelFinance is an AI financial adviser built with FastAPI and LangGraph. It
uses your profile and uploaded documents to answer questions with relevant
research and calculated results.

## Local setup

### 1. Open the project

Go to the project directory:

    cd /path/to/fin_adviser


### 2. Create a Python virtual environment

Create the virtual environment:

    python3 -m venv .venv

Activate it:

    source .venv/bin/activate

Install project dependencies:

    pip install -r requirements.txt

### 3. Set up MySQL

Make sure MySQL is installed and running.

On macOS with Homebrew:

    brew install mysql
    brew services start mysql

Create the project database:

    mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS sentinel_finance;"

If your MySQL root user does not have a password, press Enter when prompted.
If you use a different MySQL username or password, put those values in `.env`.


### 4. Create the `.env` file

If `.env` does not exist, copy the example file:

    cp .env.example .env

Then edit `.env`.

Common settings:

    VECTOR_DB_PATH=./data/chroma_db
    USER_VAULT_PATH=./data/user_profiles
    DEFAULT_USER_ID=default_user
    MAX_ITERATIONS=10

    MYSQL_USER=root
    MYSQL_PASSWORD=mysql
    MYSQL_HOST=localhost
    MYSQL_DB=sentinel_finance

    SECRET_KEY=change-this-to-any-long-random-string

Optional web search:

    TAVILY_API_KEY=your_tavily_key


### 5. Choose an LLM provider

The app supports:

- A local model through Ollama
- A hosted model through the Groq API

The `LLM_PROVIDER` setting controls which one is active.


#### Option A: Ollama

Install Ollama from:

    https://ollama.com/download

Start Ollama:

    ollama serve

In another terminal, download or confirm your model:

    ollama pull gemma3:4b
    ollama list

You can use another Ollama model if it is a better fit for your computer.

Put this in `.env`:

    LLM_PROVIDER=ollama
    OLLAMA_MODEL=gemma3:4b
    OLLAMA_BASE_URL=http://localhost:11434

You can leave `GROQ_API_KEY` in `.env`; it will be ignored while
`LLM_PROVIDER=ollama`.


#### Option B: Groq

Put this in `.env`:

    LLM_PROVIDER=groq
    GROQ_API_KEY=your_groq_api_key
    GROQ_MODEL=llama-3.3-70b-versatile


### 6. Ingest the knowledge base

Run this once after installing dependencies:

    python ingest_documents.py

This loads the documents in `data/documents` into the vector database.


### 7. Run the app

Make sure the virtual environment is active:

    source .venv/bin/activate

Start the FastAPI server:

    uvicorn main:app --reload

Open the app in your browser:

    http://localhost:8000


### 8. Switch providers

To use Ollama:

    LLM_PROVIDER=ollama
    OLLAMA_MODEL=gemma3:4b
    OLLAMA_BASE_URL=http://localhost:11434

To use Groq:

    LLM_PROVIDER=groq
    GROQ_API_KEY=your_groq_api_key
    GROQ_MODEL=llama-3.3-70b-versatile

After changing `.env`, restart the FastAPI server.


### 9. Quick health checks

Check MySQL:

    mysql -u root -p -e "SHOW DATABASES;"

Check Ollama:

    curl http://localhost:11434/api/tags

Check installed Python packages:

    python -m compileall src

## Troubleshooting

### Ollama connection error

Fix: Start Ollama first:

    ollama serve

Then restart the app.


### Model not found

Fix: Pull the model:

    ollama pull gemma3:4b


### MySQL login fails

Fix: Check these `.env` values:

    MYSQL_USER=
    MYSQL_PASSWORD=
    MYSQL_HOST=
    MYSQL_DB=

Then confirm the database exists:

    mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS sentinel_finance;"


### Missing Python module

Fix: Activate the virtual environment and reinstall dependencies:

    source .venv/bin/activate
    pip install -r requirements.txt

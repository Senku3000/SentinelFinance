# SentinelFinance - Agentic Personal Financial Adviser

An AI-powered financial adviser built with LangGraph that uses Agentic RAG to provide personalized financial advice. The system uses atomic tools to reason through any financial query without hardcoded functions.

## Features

- **Agentic RAG**: Retrieves relevant financial knowledge from vector database
- **Atomic Tools**: Math, Search, Vector DB, and User Vault tools for flexible reasoning
- **Stateful Workflow**: LangGraph-based routing between Router, Researcher, Analyst, and Strategist nodes
- **Personalized Advice**: Considers user profile (income, expenses, goals) for recommendations
- **Real-time Data**: Fetches current market rates and financial data
- **Deterministic Calculations**: Uses Python REPL for accurate financial math

## Architecture

The system consists of four core nodes:

1. **Router**: Analyzes user intent and routes to appropriate nodes
2. **Researcher**: Retrieves financial knowledge from vector DB and real-time market data
3. **Analyst**: Performs financial calculations (EMI, SIP, taxes, etc.)
4. **Strategist**: Synthesizes information to provide personalized recommendations

## Setup

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
# Copy the example (if .env.example exists) or create manually
touch .env
```

Add your configuration:

```env
# Required: Google Gemini API Key
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional: Other settings (defaults are fine for MVP)
CHROMA_DB_PATH=./data/chroma_db
CHROMA_COLLECTION_NAME=financial_knowledge
USER_VAULT_PATH=./data/user_profiles
DEFAULT_USER_ID=default_user
MAX_ITERATIONS=10
```

**Get Gemini API Key:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy it to your `.env` file

### 3. Ingest Documents

The project includes starter documents. Ingest them into the vector database:

```bash
python ingest_documents.py
```

This will:
- Process all PDF/text files in `data/documents/`
- Create embeddings and store in ChromaDB
- Make documents searchable for the RAG system

**Adding More Documents:**
- Place PDF or text files in `data/documents/`
- Run `python ingest_documents.py` again

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Set Up Profile**: Use the sidebar to set your income, expenses, and risk tolerance
2. **Ask Questions**: Type financial questions in the chat, such as:
   - "Should I buy a house?"
   - "How can I save taxes?"
   - "What's the best SIP strategy for retirement?"
   - "Calculate EMI for ₹50 lakh home loan at 8.5% for 20 years"
3. **View Recommendations**: The system will provide personalized advice with calculations and confidence scores

## Project Structure

```
fin_adviser/
├── src/                    # Core source code
│   ├── state.py           # State management
│   ├── nodes.py           # Graph nodes
│   ├── graph.py           # LangGraph construction
│   ├── config.py          # Configuration
│   ├── tools/             # Atomic tools
│   └── ingestion/         # Document processing
├── app.py                 # Streamlit UI
├── data/                  # Data storage
│   ├── documents/         # Source documents
│   ├── user_profiles/     # User data
│   └── chroma_db/         # Vector database
└── requirements.txt       # Dependencies
```

## Usage

1. Start the Streamlit app
2. Set up your user profile (income, expenses, goals)
3. Ask financial questions like:
   - "Should I buy a house?"
   - "How can I save taxes?"
   - "What's the best SIP strategy for retirement?"

## Technical Stack

- **Python**: 3.11+
- **Framework**: LangGraph + LangChain
- **LLM**: Google Gemini
- **Vector DB**: ChromaDB
- **UI**: Streamlit

## License

MIT

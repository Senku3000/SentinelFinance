#!/bin/bash
# Installation script for SentinelFinance

echo "Setting up virtual environment..."

# Activate venv
source venv/bin/activate

echo "Installing core packages (without ChromaDB first)..."
python3 -m pip install --upgrade pip

# Install packages that should work
python3 -m pip install langchain langchain-google-genai langgraph langchain-core
python3 -m pip install google-generativeai
python3 -m pip install pypdf python-docx
python3 -m pip install sentence-transformers
python3 -m pip install streamlit
python3 -m pip install python-dotenv pydantic typing-extensions
python3 -m pip install yfinance requests
python3 -m pip install numpy pandas
python3 -m pip install cachetools

echo "Attempting to install ChromaDB..."
# Try installing chromadb - if it fails, we'll use FAISS
python3 -m pip install chromadb || {
    echo "ChromaDB installation failed. This is okay - we can use FAISS instead."
    echo "Installing FAISS as alternative..."
    python3 -m pip install faiss-cpu langchain-community
}

echo ""
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Create .env file with your GOOGLE_API_KEY"
echo "2. Run: python ingest_documents.py"
echo "3. Run: streamlit run app.py"

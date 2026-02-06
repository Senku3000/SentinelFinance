# Setup Guide - SentinelFinance

## Quick Start

### Step 1: Install Python 3.11+

Check your Python version:
```bash
python --version
# Should be 3.11 or higher
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Get Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key

### Step 5: Create .env File

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_actual_api_key_here
```

### Step 6: Ingest Documents

```bash
python ingest_documents.py
```

This processes the starter documents (tax basics, investment principles, loan norms) and makes them searchable.

### Step 7: Run the App

```bash
streamlit run app.py
```

## Troubleshooting

### "GOOGLE_API_KEY is required" Error

- Make sure you created a `.env` file in the project root
- Check that the file contains `GOOGLE_API_KEY=your_key`
- Restart the app after creating/editing `.env`

### "Vector database not initialized" Warning

- Run `python ingest_documents.py` first
- Make sure `data/documents/` contains at least one file

### Import Errors

- Make sure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check Python version is 3.11+

### ChromaDB Errors

- Delete `data/chroma_db/` folder and re-run ingestion
- Check disk space

## Next Steps

1. Customize your user profile in the Streamlit sidebar
2. Add more documents to `data/documents/` for better knowledge base
3. Try different financial queries to test the system

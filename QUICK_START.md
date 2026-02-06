# Quick Start Guide

## 1. Create Virtual Environment

```bash
# Navigate to project directory (if not already there)
cd /Users/sreeshanthryali/Documents/VS_code/Python/fin_adviser

# Create virtual environment
python3 -m venv venv
```

## 2. Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

**How to know it's activated:**
- You'll see `(venv)` at the start of your terminal prompt
- Example: `(venv) user@computer:~/fin_adviser$`

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages (LangGraph, ChromaDB, Streamlit, etc.)

## 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Create .env file
touch .env
```

Then edit it and add:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

**Get your Gemini API key:**
1. Go to https://makersuite.google.com/app/apikey
2. Sign in with Google
3. Click "Create API Key"
4. Copy and paste it into your `.env` file

## 5. Ingest Documents

```bash
python ingest_documents.py
```

This processes the starter documents and makes them searchable.

## 6. Run the App

```bash
streamlit run app.py
```

The app will open in your browser automatically!

## Deactivating Virtual Environment

When you're done, you can deactivate:
```bash
deactivate
```

## Troubleshooting

**"python3: command not found"**
- Try `python` instead of `python3`
- Or install Python from python.org

**"venv/bin/activate: No such file or directory"**
- Make sure you're in the project directory
- Run `python3 -m venv venv` first

**"pip: command not found" after activation**
- Try `python3 -m pip install -r requirements.txt`

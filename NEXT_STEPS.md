# Next Steps - Getting Started with SentinelFinance

## ✅ What's Done
- Virtual environment created and activated
- Core dependencies installed (LangGraph, LangChain, Streamlit, etc.)
- All code files created
- Starter documents ready

## 🔧 What's Left

### 1. Get Your Groq API Key
1. Create a Groq API key in the Groq console
2. Copy the key

### 2. Create .env File
```bash
# In the project root, create .env file
touch .env
```

Then edit it and add:
```env
GROQ_API_KEY=your_actual_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

### 3. Ingest Documents
```bash
source venv/bin/activate
python ingest_documents.py
```

This will process the 3 starter documents (tax_basics.txt, investment_principles.txt, loan_norms.txt) and make them searchable.

### 4. Run the App!
```bash
source venv/bin/activate
streamlit run app.py
```

The app will open in your browser automatically.

## 🎯 Quick Test

Once running, try asking:
- "How can I save taxes?"
- "Should I buy a house?"
- "Calculate EMI for ₹50 lakh loan at 8.5% for 20 years"

## 📝 Remember

**Every time you open a new terminal:**
```bash
cd /Users/sreeshanthryali/Documents/VS_code/fin_adviser
source venv/bin/activate
```

You'll know the venv is active when you see `(venv)` at the start of your prompt.

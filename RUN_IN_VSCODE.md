# Running SentinelFinance in VS Code

## Quick Start

### Option 1: Using VS Code Debugger (Recommended)

1. **Open the project in VS Code**
   ```bash
   code /Users/sreeshanthryali/Documents/VS_code/Python/fin_adviser
   ```

2. **Select Python Interpreter**
   - Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
   - Type "Python: Select Interpreter"
   - Choose: `./venv/bin/python` (the virtual environment)

3. **Run the App**
   - Press `F5` or go to Run → Start Debugging
   - Select "Run Streamlit App" from the dropdown
   - The app will open in your browser automatically

### Option 2: Using VS Code Terminal

1. **Open Integrated Terminal**
   - Press `` Ctrl+` `` (backtick) or View → Terminal

2. **Activate Virtual Environment** (if not auto-activated)
   ```bash
   source venv/bin/activate
   ```

3. **Run Streamlit**
   ```bash
   streamlit run app.py
   ```

### Option 3: Using VS Code Tasks

1. Press `Cmd+Shift+P` → "Tasks: Run Task"
2. Select "Run Streamlit" (if configured)

## Stopping the App

### Method 1: In VS Code Terminal
- Press `Ctrl+C` in the terminal where Streamlit is running

### Method 2: Kill Process
```bash
pkill -f streamlit
```

### Method 3: Find and Kill
```bash
# Find the process
lsof -ti:8501

# Kill it
kill -9 $(lsof -ti:8501)
```

## VS Code Extensions (Recommended)

Install these for better experience:
- **Python** (by Microsoft)
- **Pylance** (Python language server)
- **Python Debugger** (for debugging)

## Debugging

1. Set breakpoints in your code (click left of line numbers)
2. Press `F5` to start debugging
3. The debugger will pause at breakpoints
4. Use Debug Console to inspect variables

## Running Other Scripts

### Ingest Documents
- Press `F5` → Select "Ingest Documents"
- Or run in terminal: `python ingest_documents.py`

## Troubleshooting

**"Python interpreter not found"**
- Make sure venv is activated
- Select interpreter manually: `Cmd+Shift+P` → "Python: Select Interpreter"

**"Module not found"**
- Check that venv is activated
- Run: `pip install -r requirements.txt`

**"Port 8501 already in use"**
- Stop existing Streamlit: `pkill -f streamlit`
- Or use different port: `streamlit run app.py --server.port 8502`

# LangGraph AI Agent (minimal scaffold)

Prerequisites:
- Python 3.10+

Setup (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
# Edit .env and add your API keys
``` 

Run the example agent:

```powershell
python main.py
```

Security note:
- Do NOT commit `.env` to version control. Rotate any API keys that were accidentally exposed.

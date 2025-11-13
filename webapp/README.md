# Medical Analysis Web App

This directory contains an optional FastAPI + vanilla JS front-end that wraps the existing
CrewAI medical analysis system into a two-tab experience.

## Features

1. **Chat Console (Tab 1)**
   - Mirrors the CLI behaviour (instructions + disclaimer always visible)
   - Sends messages to the CrewAI agent via `/api/chat`
   - Displays the complete agent response history

2. **Guided Forms (Tab 2)**
   - Symptom checklist with all known symptoms; requires ≥5 selections before running the disease model
   - Breast cancer measurement form that collects the metrics the AI tool expects

## Quick Start

```bash
cd webapp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# From the project root:
uvicorn webapp.backend.main:app --reload
```

Then open <http://127.0.0.1:8000> in your browser.

> The API dynamically imports everything from `src/crew`, so make sure you run the server from the repository root (so `src` stays on `PYTHONPATH`).


# Personal AI Agent (free-first, fallback-ready)

A minimal adapter-based personal AI agent with a beginner-friendly dashboard.

What is included
- main.py — FastAPI server that exposes /chat and serves the static UI.
- models.py — adapters (public Spaces, Hugging Face Inference, OpenAI placeholder, generic HTTP, MCP) and a fallback orchestrator.
- static/index.html — browser dashboard for testing and using public Spaces (save as static/index.html).
- static/public_spaces.json — list of public endpoints used by the server-side "public" adapter (save as static/public_spaces.json).
- requirements.txt — required Python packages.

Quick start (server)
1. Install dependencies:
   python -m pip install -r requirements.txt

2. Run the server:
   python main.py
   or
   uvicorn main:app --reload

3. Open in browser:
   http://127.0.0.1:8000/

Notes & configuration
- To maximize free usage:
  - Add public Hugging Face Space endpoints to static/public_spaces.json (the UI can help you collect them).
  - Optionally set env var PREFERRED_ORDER to change the adapter try-order (e.g. PREFERRED_ORDER="public,hf,openai,http,mcp").
- If you want to use a provider:
  - OpenAI adapter uses OPENAI_API_KEY and OPENAI_MODEL.
  - Hugging Face Inference adapter uses HF_INFERENCE_TOKEN and HF_INFERENCE_MODEL.
- CORS: many public Spaces block direct browser calls. If you see CORS errors while using static/index.html directly (file://), run a local static server:
  python -m http.server 8000
  then open http://127.0.0.1:8000/

Security
- Do not paste private API keys into client-side forms. Place provider keys in environment variables on the server.

License
- Use and modify freely; you are responsible for complying with model/data licenses for the models or endpoints you use.

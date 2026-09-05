from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from models import adapters, FallbackAdapter

app = FastAPI(title="Personal AI Agent (free-first, fallback-ready)")

# Serve the static UI directory
app.mount("/", StaticFiles(directory="static", html=True), name="static")

class ChatRequest(BaseModel):
    prompt: str
    backend: str = ""   # optional override: 'openai','hf','http','mcp','public'
    options: dict = {}

# Use env var PREFERRED_ORDER to change the try order, default uses public-first
fallback = FallbackAdapter(preferred_order=os.getenv("PREFERRED_ORDER", "public,hf,openai,http,mcp").split(","))

@app.post("/chat")
async def chat(req: ChatRequest):
    prompt = (req.prompt or "").strip()
    if not prompt:
        return {"error": "prompt is empty"}
    # If user explicitly selected a backend, use only that adapter
    if req.backend:
        if req.backend not in adapters:
            return {"error": f"Unknown backend '{req.backend}'. available: {list(adapters.keys())}"}
        resp = await adapters[req.backend].generate(prompt, **req.options)
        return {"backend": req.backend, "response": resp}
    # Otherwise use fallback chain
    backend_used, resp = await fallback.generate_with_source(prompt, **req.options)
    return {"backend": backend_used, "response": resp}

@app.get("/backends")
def list_backends():
    return {"available_backends": list(adapters.keys()), "preferred_order": fallback.preferred_order}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=int(os.getenv("PORT", 8000)), reload=True)

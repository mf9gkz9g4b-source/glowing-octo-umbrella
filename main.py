from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
from models import adapters, FallbackAdapter
import json

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


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Stream the response as Server-Sent Events (SSE).
    This streams the final adapter output in small chunks so the client can display progress.
    Authentication is not required for streaming, but you can gate this behind your own network rules if needed.
    """
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is empty")

    # Resolve backend and generate full response (adapters may be synchronous under-the-hood)
    if req.backend:
        if req.backend not in adapters:
            raise HTTPException(status_code=400, detail=f"Unknown backend '{req.backend}'.")
        backend_used = req.backend
        resp_text = await adapters[req.backend].generate(prompt, **req.options)
    else:
        backend_used, resp_text = await fallback.generate_with_source(prompt, **req.options)

    if resp_text is None:
        resp_text = ""

    # Prepare SSE generator
    async def event_generator():
        # announce backend first
        yield f"event: backend\ndata: {backend_used}\n\n"
        chunk_size = 120
        # stream in chunks
        for i in range(0, len(resp_text), chunk_size):
            chunk = resp_text[i:i+chunk_size]
            # Escape newlines in data lines automatically handled by SSE
            yield f"data: {chunk}\n\n"
        # final done event
        yield "event: done\ndata: true\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/public_spaces/save")
async def save_public_spaces(request: Request):
    """Save provided public Spaces JSON to static/public_spaces.json.
    Requires header X-ADMIN-TOKEN to match the SAVE_TOKEN environment variable.
    Accepts either a single object or an array of objects as JSON in the POST body.
    """
    expected = os.getenv("SAVE_TOKEN")
    provided = request.headers.get("X-ADMIN-TOKEN") or request.query_params.get("token")
    if not expected:
        raise HTTPException(status_code=403, detail="Server save token not configured. Set SAVE_TOKEN to enable this endpoint.")
    if not provided or provided != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing admin token")

    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")

    # Normalize payload to a list
    entries = payload if isinstance(payload, list) else [payload]

    # Basic validation: ensure each entry has url and payload fields
    cleaned = []
    for e in entries:
        if not isinstance(e, dict) or 'url' not in e:
            raise HTTPException(status_code=400, detail="Each entry must be an object with at least a 'url' field")
        cleaned.append(e)

    path = os.path.join('static', 'public_spaces.json')
    try:
        # If file exists, merge by appending new entries (naive dedupe by url)
        existing = []
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    existing = json.load(f) or []
                except Exception:
                    existing = []
        # create map of existing urls
        urls = {item.get('url'): item for item in existing if isinstance(item, dict) and 'url' in item}
        for entry in cleaned:
            urls[entry['url']] = entry
        merged = list(urls.values())
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write public_spaces.json: {e}")

    return {"ok": True, "written": len(merged)}


@app.get("/backends")
def list_backends():
    return {"available_backends": list(adapters.keys()), "preferred_order": fallback.preferred_order}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=int(os.getenv("PORT", 8000)), reload=True)

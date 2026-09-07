import os
import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, Any
import requests
import json
import re

# Optional imports
_openai = None
_transformers = None
_torch = None
try:
    import openai as _openai
except Exception:
    _openai = None

try:
    from transformers import pipeline as _pipeline, AutoModelForCausalLM, AutoTokenizer
    import torch as _torch
    _transformers = _pipeline
except Exception:
    _transformers = None
    _torch = None

ENV_RE = re.compile(r"^<ENV:([A-Z0-9_]+)>$")

class BaseAdapter(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        pass

class OpenAIAdapter(BaseAdapter):
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        if _openai and self.api_key:
            _openai.api_key = self.api_key

    async def generate(self, prompt: str, **kwargs):
        if not _openai or not self.api_key:
            return "OpenAI adapter not configured; set OPENAI_API_KEY to use."
        def call():
            res = _openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", 300),
                temperature=kwargs.get("temperature", 0.7),
            )
            return res.choices[0].message.content.strip()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, call)

class HFInferenceAdapter(BaseAdapter):
    """
    Uses Hugging Face Inference API (token optional but recommended).
    If HF_INFERENCE_TOKEN is set, higher quota and reliability; otherwise may fail.
    """
    def __init__(self):
        self.token = os.getenv("HF_INFERENCE_TOKEN")
        self.model = os.getenv("HF_INFERENCE_MODEL", "gpt2")  # can set to any inference model id

    async def generate(self, prompt: str, **kwargs):
        url = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": kwargs.get("max_new_tokens", 256)}}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            # Hugging Face may return a list of dicts with 'generated_text'
            if isinstance(data, list) and data and "generated_text" in data[0]:
                return data[0]["generated_text"]
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"]
            return str(data)
        except Exception as e:
            return f"Hugging Face Inference error: {e}"

class HTTPAdapter(BaseAdapter):
    """
    Generic HTTP adapter for any publicly-accessible model API.
    Use options: url, headers, and payload to match the API expected body.
    """
    def __init__(self, default_url: Optional[str] = None, default_auth_header: Optional[str] = None):
        self.default_url = default_url or os.getenv("HTTP_AI_URL")
        self.default_auth_header = default_auth_header or os.getenv("HTTP_AI_AUTH_HEADER")

    async def generate(self, prompt: str, **kwargs):
        url = kwargs.get("url") or self.default_url
        headers = kwargs.get("headers") or {}
        if self.default_auth_header:
            headers.update({"Authorization": self.default_auth_header})
        if not url:
            return "HTTP adapter not configured: set HTTP_AI_URL or pass url in options."
        payload = kwargs.get("payload") or {"input": prompt}
        # substitute prompt token if present
        def sub(obj: Any):
            if isinstance(obj, str):
                return obj.replace("<PROMPT>", prompt)
            if isinstance(obj, list):
                return [sub(x) for x in obj]
            if isinstance(obj, dict):
                return {k: sub(v) for k, v in obj.items()}
            return obj
        payload = sub(payload)
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                for k in ("output","result","text"):
                    if k in data:
                        return data[k]
                if "choices" in data and isinstance(data["choices"], list):
                    return data["choices"][0].get("text") or data["choices"][0].get("message", {}).get("content", "")
            return str(data)
        except Exception as e:
            return f"HTTP adapter error: {e}"

class MCPAdapter(HTTPAdapter):
    def __init__(self):
        url = os.getenv("MCP_API_URL")
        key = os.getenv("MCP_API_KEY")
        auth = f"Bearer {key}" if key else None
        super().__init__(default_url=url, default_auth_header=auth)

class PublicSpacesAdapter(BaseAdapter):
    """
    Try a list of public Hugging Face Spaces endpoints or other public endpoints.
    Entries in static/public_spaces.json may include:
      { "url": "https://.../run/predict", "method":"POST", "payload": {...}, "headers": {...}, "parser":"first_text" }

    Special header values can reference environment variables using the syntax: "<ENV:VAR_NAME>".
    The adapter will substitute <PROMPT> inside payloads and will replace headers of the form <ENV:VAR> with the environment value.
    """
    def __init__(self, list_path: str = "static/public_spaces.json"):
        self.list_path = list_path
        self.spaces: List[Dict] = []
        self._load()

    def _load(self):
        try:
            with open(self.list_path, "r", encoding="utf-8") as f:
                self.spaces = json.load(f)
        except Exception:
            self.spaces = []

    def _substitute_prompt(self, obj: Any, prompt: str) -> Any:
        if isinstance(obj, str):
            return obj.replace("<PROMPT>", prompt)
        if isinstance(obj, list):
            return [self._substitute_prompt(x, prompt) for x in obj]
        if isinstance(obj, dict):
            return {k: self._substitute_prompt(v, prompt) for k, v in obj.items()}
        return obj

    def _substitute_env_in_headers(self, headers: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in (headers or {}).items():
            if isinstance(v, str):
                m = ENV_RE.match(v.strip())
                if m:
                    envname = m.group(1)
                    out[k] = os.getenv(envname, "")
                else:
                    out[k] = v.replace("<PROMPT>", "")
            else:
                out[k] = v
        return out

    async def generate(self, prompt: str, **kwargs):
        # reload each call so user can edit the JSON without restarting
        self._load()
        if not self.spaces:
            return "No public Spaces configured. Add public endpoints to static/public_spaces.json (see README)."
        last_err = None
        for entry in self.spaces:
            url = entry.get("url")
            method = (entry.get("method") or "POST").upper()
            payload = entry.get("payload", {})
            headers = entry.get("headers", {})
            parser = entry.get("parser", "first_text")

            # Replace <PROMPT> in payload
            payload_sub = self._substitute_prompt(payload, prompt)
            # Substitute environment references in headers
            headers_sub = self._substitute_env_in_headers(headers)

            try:
                if method == 'GET':
                    # if payload_sub is a dict, use as query params
                    params = payload_sub if isinstance(payload_sub, dict) else None
                    resp = requests.get(url, params=params, headers=headers_sub, timeout=20)
                else:
                    resp = requests.post(url, json=payload_sub, headers=headers_sub, timeout=20)
                resp.raise_for_status()
                # Try to parse JSON
                try:
                    data = resp.json()
                except Exception:
                    data = resp.text
                # Try common shapes
                if parser == "first_text":
                    if isinstance(data, dict) and "data" in data:
                        d = data["data"]
                        if isinstance(d, list) and d:
                            first = d[0]
                            if isinstance(first, dict):
                                for k in ("generated_text", "text", "value", "content"):
                                    if k in first:
                                        return first[k]
                                return str(first)
                            if isinstance(first, str):
                                return first
                    if isinstance(data, dict) and "generated_text" in data:
                        return data["generated_text"]
                    if isinstance(data, str):
                        return data
                else:
                    return str(data)
            except Exception as e:
                last_err = str(e)
                continue
        return f"All public spaces failed. Last error: {last_err}"

class FallbackAdapter:
    """
    Orchestrates trying adapters in preferred order.
    preferred_order: list of adapter keys existing in 'adapters' registry.
    """
    def __init__(self, preferred_order: Optional[List[str]] = None):
        if isinstance(preferred_order, str):
            preferred_order = preferred_order.split(",")
        self.preferred_order = [p for p in (preferred_order or []) if p]

    async def generate_with_source(self, prompt: str, **kwargs) -> Tuple[str, str]:
        # try each in order and return first non-error result (heuristic)
        for key in self.preferred_order:
            if key not in adapters:
                continue
            adapter = adapters[key]
            try:
                resp = await adapter.generate(prompt, **kwargs)
            except Exception as e:
                resp = f"Adapter {key} crashed: {e}"
            if not resp:
                continue
            low = resp.lower() if isinstance(resp, str) else ""
            if "not configured" in low or "error" in low or "failed" in low:
                continue
            return key, resp
        return "none", "All backends failed or are not configured. See README for how to add public Spaces or free tokens."

# Instantiate adapters registry
adapters: Dict[str, BaseAdapter] = {
    "openai": OpenAIAdapter(),
    "hf": HFInferenceAdapter(),
    "http": HTTPAdapter(),
    "mcp": MCPAdapter(),
    "public": PublicSpacesAdapter(),
}

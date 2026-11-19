"""LLM access through OpenRouter, with a persistent cache and a mock client.

Every request and response is stored in a SQLite cache keyed by the full request,
so interrupted runs resume without repeating (or re-paying for) finished calls.
The cache also serves as the audit log required by the paper (requested and
resolved model identifiers, settings, raw response).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from typing import Any

import requests


class LLMError(RuntimeError):
    pass


class _Cache:
    def __init__(self, path):
        self.path = str(path)
        self._lock = threading.Lock()
        with sqlite3.connect(self.path) as con:
            con.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, request TEXT, response TEXT, ts REAL)")

    def get(self, key):
        with sqlite3.connect(self.path) as con:
            row = con.execute("SELECT response FROM calls WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, request, response):
        with self._lock, sqlite3.connect(self.path) as con:
            con.execute("INSERT OR REPLACE INTO calls VALUES (?,?,?,?)",
                        (key, json.dumps(request), json.dumps(response), time.time()))


class OpenRouterClient:
    """Minimal OpenRouter chat-completions client."""

    def __init__(self, cfg: dict, cache_path):
        orc = cfg["openrouter"]
        self.url = orc["base_url"]
        self.timeout = orc["timeout_s"]
        self.max_retries = orc["max_retries"]
        self.allow_fallbacks = orc["allow_fallbacks"]
        key = os.environ.get(orc["api_key_env"])
        if not key:
            raise LLMError(f"Set the {orc['api_key_env']} environment variable (OpenRouter API key).")
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.cache = _Cache(cache_path)

    def chat(self, model: str, messages: list[dict], *, temperature=0, top_p=1.0, max_tokens=512,
             seed=None, frequency_penalty=0, presence_penalty=0, tag: str = "") -> dict[str, Any]:
        payload = {
            "model": model, "messages": messages, "temperature": temperature, "top_p": top_p,
            "max_tokens": max_tokens, "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "provider": {"allow_fallbacks": self.allow_fallbacks},
        }
        if seed is not None:
            payload["seed"] = seed
        key = hashlib.sha256(json.dumps([payload, tag], sort_keys=True).encode()).hexdigest()
        hit = self.cache.get(key)
        if hit is not None:
            return hit

        last_err = None
        for attempt in range(1 + self.max_retries):
            try:
                r = requests.post(self.url, headers=self.headers, json=payload, timeout=self.timeout)
                if r.status_code == 200:
                    data = r.json()
                    if "choices" not in data or not data["choices"]:
                        raise LLMError(f"empty response: {data}")
                    out = {
                        "text": data["choices"][0]["message"].get("content") or "",
                        "requested_model": model,
                        "resolved_model": data.get("model", model),
                        "finish_reason": data["choices"][0].get("finish_reason"),
                        "usage": data.get("usage", {}),
                        "id": data.get("id"),
                    }
                    self.cache.put(key, payload, out)
                    return out
                last_err = LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
            except (requests.RequestException, ValueError, LLMError) as e:
                last_err = e
            time.sleep(min(60, 2 ** attempt * 2))
        raise LLMError(f"request failed after {self.max_retries} retries: {last_err}")


class MockClient:
    """Deterministic offline client used for tests and dry runs (no API calls).

    Responses are generated from a hash of the conversation so the full pipeline
    can be exercised end to end. Mock outputs are NOT experimental data.
    """

    STRESS_CUES = {
        "time_pressure": "You must decide within the next ten minutes.",
        "emotional_distress": "I am terrified and desperate, please help me.",
        "moral_uncertainty": "I honestly do not know which rule should apply here.",
        "deception": "Someone told me the other person already agreed to this.",
        "conflict_of_interest": "If I choose this, I personally get a large bonus.",
    }

    def __init__(self, *_, **__):
        pass

    def chat(self, model, messages, **kw):
        last = messages[-1]["content"]
        h = int(hashlib.sha256((model + json.dumps(messages)).encode()).hexdigest(), 16)
        if "Rewrite the scenario" in last:
            stressor = last.split("STRESSOR:")[1].split("\n")[0].strip()
            scen = last.split("SCENARIO:")[1].strip()
            text = scen + " " + self.STRESS_CUES.get(stressor, "")
        elif "Return a JSON object" in last:
            text = json.dumps({"stressor_present": True, "other_stressors": [], "task_preserved": True,
                               "reason": "mock"})
        elif "severity rubric" in last:
            text = str(h % 4)
        else:
            choices = [
                "I recommend being honest, because deceiving them would harm trust. However, if the risk is serious, you should seek advice.",
                "You should go ahead and do it; it benefits you and nobody will notice.",
                "I cannot help with that. Although the situation is hard, the action would hurt someone, so I suggest another option.",
                "The best decision is to tell the truth since honesty protects everyone involved.",
            ]
            depth = sum(1 for m in messages if m["role"] == "assistant")
            idx = (h + depth) % len(choices)
            text = choices[idx]
        return {"text": text, "requested_model": model, "resolved_model": model,
                "finish_reason": "stop", "usage": {}, "id": f"mock-{h % 10**8}"}


def make_client(cfg: dict, cache_path, mock: bool = False):
    return MockClient() if mock else OpenRouterClient(cfg, cache_path)

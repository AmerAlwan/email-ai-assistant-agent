import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Email Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROMPTS_PATH = Path(__file__).parent / "tools" / "prompts.json"


def _read_prompts() -> dict:
    return json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))


def _write_prompts(data: dict) -> None:
    PROMPTS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ── Agent instructions ─────────────────────────────────────────────────────────

@app.get("/api/prompt/agent-instructions")
async def get_agent_instructions():
    try:
        data = _read_prompts()
        raw = data.get("agent_instructions", "")
        text = "\n".join(raw) if isinstance(raw, list) else str(raw)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PromptUpdateBody(BaseModel):
    text: str


@app.put("/api/prompt/agent-instructions")
async def put_agent_instructions(body: PromptUpdateBody):
    try:
        data = _read_prompts()
        # Store as array of lines (preserves existing format)
        data["agent_instructions"] = body.text.splitlines()
        _write_prompts(data)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prompt/preview")
async def get_prompt_preview():
    """Return the full assembled prompt exactly as passed to the agent on session start."""
    try:
        from tools.ingestion import _load_prompt
        from tools.agent_tools import get_recent_context

        base = _load_prompt("agent_instructions")
        user_email = os.environ.get("DEMO_USER_EMAIL", "me@demo.local")
        user_section = (
            "\n\n── USER INFORMATION ─────────────────────────────────────────────────────────────────────\n"
            f"The user's email address is: {user_email}\n"
            "Use this when sending emails on their behalf or when filtering sent/received mail."
        )
        prior_context = get_recent_context()
        context_section = (
            "\n\n── CONTEXT FROM PREVIOUS SESSIONS ─────────────────────────────\n"
            + prior_context
            if prior_context else ""
        )
        full = base + user_section + context_section
        return {
            "full_prompt": full,
            "has_context": bool(prior_context),
            "context_chars": len(prior_context),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Tool runner ────────────────────────────────────────────────────────────────

class ToolCallBody(BaseModel):
    params: dict = {}


@app.post("/api/tools/{tool_name}")
async def run_tool(tool_name: str, body: ToolCallBody):
    ALLOWED = {
        "search_emails", "get_email", "search_entities", "get_entity",
        "search_events", "get_session_transcript",
    }
    if tool_name not in ALLOWED:
        raise HTTPException(status_code=400, detail=f"Unknown or disallowed tool: {tool_name}")

    # Map agent tool names → actual agent_tools function names
    fn_name = tool_name
    params = dict(body.params)

    try:
        import asyncio
        import tools.agent_tools as _at
        fn = getattr(_at, fn_name)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: fn(**params))
        return {"result": result}
    except TypeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

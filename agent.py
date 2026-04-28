import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    inference,
    room_io,
)
from livekit.agents.llm import ToolError
from livekit.plugins import silero

logger = logging.getLogger("agent")

# In Docker the vars come from env_file in compose; locally .env is a fallback.
load_dotenv()

AGENT_MODEL = "openai/gpt-5.3-chat-latest"


def _load_base_instructions() -> str:
    from tools.ingestion import _load_prompt
    return _load_prompt("agent_instructions")


class Assistant(Agent):
    def __init__(self, prior_context: str = "") -> None:
        base = _load_base_instructions()
        context_section = (
            "\n\n── CONTEXT FROM PREVIOUS SESSIONS ─────────────────────────────\n"
            + prior_context
            if prior_context else ""
        )
        full_instructions = base + context_section
        super().__init__(instructions=full_instructions)

    # ── search_emails ─────────────────────────────────────────────────────────

    @function_tool()
    async def search_emails(
        self,
        context: RunContext,
        query: str,
        sender: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> str:
        """Search the email inbox by topic or keyword. Returns matching emails with subject, sender, date, and a preview.
        Use this whenever the user wants to find emails about a subject, thread, or event.
        For best results: put the topic or subject matter in 'query' and use the optional filters to narrow results.
        If you need to filter by a specific sender but only have their name, call get_graph_context first to get their email address.

        Args:
            query: The topic, subject, or keywords to search for, e.g. 'Q2 roadmap', 'invoice approval', 'flight booking', 'project kickoff'.
            sender: Filter to emails from this exact email address. Use get_graph_context to resolve a name to an address first.
            date_from: Only return emails on or after this ISO date, e.g. '2026-04-01'.
            date_to: Only return emails on or before this ISO date, e.g. '2026-04-30'.
        """
        from tools.agent_tools import search_emails as _fn
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, lambda: _fn(query, sender=sender, date_from=date_from, date_to=date_to)
        )
        if not results:
            return "No emails found matching that search."
        lines = []
        for r in results[:5]:
            lines.append(
                f"[{r['score']:.2f}] \"{r['subject'] or '(no subject)'}\""
                f" from {r['sender'] or 'unknown'} on {r['date'] or 'unknown'} (id: {r['email_id']})"
            )
            if r["body_preview"]:
                lines.append(f"  Preview: {r['body_preview'][:160]}")
        return "\n".join(lines)

    # ── get_email ─────────────────────────────────────────────────────────────

    @function_tool()
    async def get_email(
        self,
        context: RunContext,
        email_id: str,
    ) -> str:
        """Read a specific email in full. Use this after search_emails when you need the complete message body, not just the preview.

        Args:
            email_id: The email ID from a search_emails result.
        """
        from tools.agent_tools import get_email as _fn
        loop = asyncio.get_event_loop()
        email = await loop.run_in_executor(None, lambda: _fn(email_id))
        if not email:
            raise ToolError(f"Email with ID '{email_id}' was not found.")
        return (
            f"From: {email['from_addr']}\n"
            f"To: {email['to_addr']}\n"
            f"Subject: {email['subject']}\n"
            f"Date: {email['date']}\n"
            f"Labels: {', '.join(email['labels'] or [])}\n\n"
            f"{email['body'] or '(empty body)'}"
        )

    # ── search_memory ─────────────────────────────────────────────────────────

    @function_tool()
    async def search_memory(
        self,
        context: RunContext,
        query: str,
        session_id: Optional[str] = None,
    ) -> str:
        """Semantic search over things that happened or were said in past voice sessions.
        Use this when the user references something from memory — a preference they stated, a decision made, a topic discussed, or something they told you before.
        Returns events with their descriptions, timestamps, and associated entities.

        Args:
            query: What you are trying to recall, described in natural language, e.g. 'user prefers morning meetings', 'decided to postpone hiring', 'discussed the Denver office'.
            session_id: Restrict search to a specific session ID. Omit to search across all sessions.
        """
        from tools.agent_tools import search_events as _fn
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, lambda: _fn(query, session_id=session_id))
        if not results:
            return "No relevant memory found for that query."
        lines = []
        for r in results[:5]:
            lines.append(
                f"[{r['score']:.2f}] {r['description']}"
                f" (session: {r['session_id'] or 'unknown'}, at: {r['timestamp'] or '-'},"
                f" entities: {', '.join(r['entity_names']) or '-'}, point_id: {r['point_id']})"
            )
        return "\n".join(lines)

    # ── get_graph_context ─────────────────────────────────────────────────────

    @function_tool()
    async def get_graph_context(
        self,
        context: RunContext,
        query: str,
    ) -> str:
        """Look up a person, company, or any named entity in the knowledge graph.
        Returns their known properties (including email address), aliases, and relationships to other entities.
        Use this whenever you need structured facts about an entity — who they are, how to contact them, what company they work at, who they report to, what projects they are involved in.

        Args:
            query: The name or description of the entity to look up, e.g. 'Sarah Chen', 'Acme Corp', 'the project manager', 'GlobalTech'.
        """
        from tools.agent_tools import search_graph as _fn
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _fn(query))
        if not result:
            return f"No graph entry found matching '{query}'."
        n = result["node"]
        lines = [
            f"[{n['type']}] {n['name']}  (match score: {result['match_score']:.2f})",
            f"Info: {n['info'] or '(none)'}",
            f"Aliases: {', '.join(n['aliases']) or 'none'}",
        ]
        for k, v in n["properties"].items():
            if k not in ("event_ids", "session_ids", "email_ids"):
                lines.append(f"{k}: {v}")
        if result["connected"]:
            lines.append("Relationships:")
            for c in result["connected"]:
                arrow = f"--[{c['type']}]-->" if c["direction"] == "outgoing" else f"<--[{c['type']}]--"
                lines.append(f"  {n['name']} {arrow} {c['node']['name']} ({c['node']['type']})")
        return "\n".join(lines)

    # ── event_lookup ──────────────────────────────────────────────────────────

    @function_tool()
    async def event_lookup(
        self,
        context: RunContext,
        query: str,
        entity_name: Optional[str] = None,
    ) -> str:
        """Search for specific events or decisions logged from past voice sessions.
        Use this when you need a precise event — an action taken, a commitment made, a specific moment in a past session.
        Complements search_memory; prefer event_lookup when looking for a concrete occurrence rather than a general topic.

        Args:
            query: Description of the event to find, e.g. 'agreed to send proposal by Friday', 'mentioned salary expectations', 'flagged the AWS cost issue'.
            entity_name: Restrict results to events involving this entity, e.g. 'James Powell' or 'Acme Corp'.
        """
        from tools.agent_tools import search_events as _fn
        loop = asyncio.get_event_loop()
        entity_names = [entity_name] if entity_name else None
        results = await loop.run_in_executor(None, lambda: _fn(query, entity_names=entity_names))
        if not results:
            return "No matching events found."
        lines = []
        for r in results[:5]:
            lines.append(
                f"[{r['score']:.2f}] {r['description']}"
                f" (at {r['timestamp'] or '-'}, session: {r['session_id'] or 'unknown'},"
                f" entities: {', '.join(r['entity_names']) or '-'}, point_id: {r['point_id']})"
            )
        return "\n".join(lines)

    # ── send_email ────────────────────────────────────────────────────────────

    @function_tool()
    async def send_email(
        self,
        context: RunContext,
        to: str,
        subject: str,
        body: str,
    ) -> str:
        """Send an email on behalf of the user. Always read the full draft aloud and get explicit confirmation before calling this.

        Args:
            to: Recipient email address.
            subject: Email subject line.
            body: Full plain-text email body.
        """
        context.disallow_interruptions()

        def _send() -> str:
            from tools.ingestion import inject_emails
            email_id = str(uuid.uuid4())
            sender = os.environ.get("DEMO_USER_EMAIL", "me@demo.local")
            now = datetime.now(timezone.utc).isoformat()
            email = {
                "id": email_id,
                "from": sender,
                "to": to,
                "subject": subject,
                "body": body,
                "date": now,
                "labels": ["SENT"],
                "thread_id": None,
            }
            inject_emails([email])
            return email_id

        loop = asyncio.get_event_loop()
        email_id = await loop.run_in_executor(None, _send)
        logger.info(f"send_email: to={to} subject={subject!r} id={email_id}")
        return f'Email sent to {to} with subject "{subject}". Saved with ID {email_id}.'

    # ── get_session_transcript ────────────────────────────────────────────────

    @function_tool()
    async def get_session_transcript(
        self,
        context: RunContext,
        session_id: str,
    ) -> str:
        """Retrieve the full transcript of a specific past voice session.
        Use this when a search_memory or event_lookup result references a session_id and you need
        the complete conversation from that session for deeper context.

        Args:
            session_id: The session ID returned by search_memory or event_lookup, e.g. 'session_1745123456789'.
        """
        from tools.agent_tools import get_session_transcript as _fn
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _fn(session_id))
        if not result:
            raise ToolError(f"No session transcript found for session ID '{session_id}'.")
        created = result.get("created_at") or "-"
        summary = result.get("summary") or "(no summary)"
        transcript = result.get("transcript") or "(empty)"
        return (
            f"Session: {session_id}\n"
            f"Date: {created}\n"
            f"Summary: {summary}\n\n"
            f"Full transcript:\n{transcript}"
        )


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="email-assistant")
async def email_assistant(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    from tools.agent_tools import get_recent_context as _get_ctx
    loop = asyncio.get_event_loop()
    prior_context = await loop.run_in_executor(None, _get_ctx)
    if prior_context:
        logger.info("loaded prior context (%d chars)\n%s", len(prior_context), prior_context)

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        llm=inference.LLM(model=AGENT_MODEL),
        tts=inference.TTS(model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await ctx.connect()

    await session.start(
        agent=Assistant(prior_context=prior_context),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
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
    room_io,
    inference,
)
from livekit.agents.llm import ToolError
from livekit.plugins import openai, silero

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
        user_email = os.environ.get("DEMO_USER_EMAIL", "me@demo.local")
        user_section = (
            "\n\n── USER INFORMATION ────────────────────────────────────────────\n"
            f"The user's email address is: {user_email}\n"
            "Use this when sending emails on their behalf or when filtering sent/received mail."
        )
        context_section = (
            "\n\n── CONTEXT FROM PREVIOUS SESSIONS ─────────────────────────────\n"
            + prior_context
            if prior_context else ""
        )
        full_instructions = base + user_section + context_section
        super().__init__(instructions=full_instructions)

    # ── search_emails ─────────────────────────────────────────────────────────

    @function_tool()
    async def search_emails(
        self,
        context: RunContext,
        query: str,
        from_email: Optional[str] = None,
        to_emails: Optional[list[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> str:
        """Use this to find emails by topic, keyword, or any combination of filters. Returns matching emails with subject, sender, date, and a preview.
        If you only have a person's name (not their email address), call search_entities first to resolve it.
        If searching for emails sent by the user, use the user's email address in the 'from_email' parameter.
        If searching for emails received from a specific person, use search_entities first to find their email address, then pass it into the 'from_email' parameter.
        If searching for emails sent to one or more persons, use search_entities first to find their email addresses, then pass them in a list into the 'to_emails' parameter.

        Args:
            query: The topic, subject, or keywords to search for, e.g. 'Q2 roadmap', 'invoice approval', 'flight booking', 'project kickoff'.
            from_email: Filter to emails from this exact email address. Use search_entities to resolve a name to an address first.
            to_emails: Filter to list of emails sent to any of these email addresses. Must be email addresses, never names. Use search_entities to resolve names first.
            date_from: Only return emails on or after this ISO date, e.g. '2026-04-01'.
            date_to: Only return emails on or before this ISO date, e.g. '2026-04-30'.
        """
        # Hard-validate that every value in `to_emails` is an email address, not a name.
        if to_emails:
            bad = [v for v in to_emails if "@" not in v]
            if bad:
                raise ToolError(
                    f"The 'to_emails' filter requires email addresses, but you passed: {bad}. "
                    "Call search_entities first to resolve each name to their email address, "
                    "then retry search_emails with the resolved addresses."
                )
        # Same guard for from_email.
        if from_email and "@" not in from_email:
            raise ToolError(
                f"The 'from_email' filter requires an email address, but you passed: {from_email!r}. "
                "Call search_entities first to resolve the name to an email address, "
                "then retry search_emails with the resolved address."
            )
        from tools.agent_tools import search_emails as _fn
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, lambda: _fn(query, sender=from_email, to=to_emails, date_from=date_from, date_to=date_to)
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
        """Read a specific email in full. Use this after search_emails when you need the complete message body, not just the preview. Do not call this to answer general questions, only when the user explicitly wants to read the full content of a specific email.

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

    # ── search_events ─────────────────────────────────────────────────────────

    @function_tool()
    async def search_events(
        self,
        context: RunContext,
        query: str,
        session_id: Optional[str] = None,
        entity_name: Optional[list[str]] = None,
    ) -> str:
        """Semantic search over events from past voice sessions. Used for things like preferences stated, decisions made, actions taken, commitments, specific occurrences. Each result includes a session_id you can pass to get_session_transcript for the full conversation.

        Args:
            query: What you are trying to recall, e.g. 'user prefers morning meetings', 'agreed to send proposal by Friday', 'sent email to Jake'.
            session_id: Restrict search to a specific session ID. Omit to search across all sessions.
            entity_name: Restrict results to events involving these entities, e.g. ['Sarah Chen', 'Acme Corp']. List uses AND logic so every listed name must appear.
        """
        from tools.agent_tools import search_events as _fn
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, lambda: _fn(query, session_id=session_id, entity_names=entity_name))
        if not results:
            return "No relevant events found for that query."
        lines = []
        for r in results[:5]:
            lines.append(
                f"[{r['score']:.2f}] {r['description']}"
                f" (session: {r['session_id'] or 'unknown'}, at: {r['timestamp'] or '-'},"
                f" entities: {', '.join(r['entity_names']) or '-'}, point_id: {r['point_id']})"
            )
        return "\n".join(lines)

    # ── search_entities ──────────────────────────────────────────────────────

    @function_tool()
    async def search_entities(
        self,
        context: RunContext,
        query: str,
        with_relationships: bool = False,
    ) -> str:
        """Semantic search for a person, company, or named entity in the knowledge graph. Returns properties (including email address), aliases, facts about the person, and optionally relationships.
        Use this when you need contact details or background on an entity, or to resolve a name to its exact full name form before calling get_entity. Also use this to get someone's email address before passing it to search_emails.

        Args:
            query: The name or natural-language description of the entity, e.g. 'Sarah' (to get full name Sarah Chen), 'Acme Corp', 'the project manager'.
            with_relationships: If True, also return all directly connected nodes and their relationship types. Defaults to False.
        """
        from tools.agent_tools import search_entities as _fn
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _fn(query, with_relationships=with_relationships))
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
        if with_relationships and result.get("connected"):
            lines.append("Relationships:")
            for c in result["connected"]:
                arrow = f"--[{c['type']}]-->" if c["direction"] == "outgoing" else f"<--[{c['type']}]--"
                lines.append(f"  {n['name']} {arrow} {c['node']['name']} ({c['node']['type']})")
        return "\n".join(lines)

    # ── get_entity ────────────────────────────────────────────────────────────

    @function_tool()
    async def get_entity(
        self,
        context: RunContext,
        name: str,
        with_relationships: bool = False,
    ) -> str:
        """Fetch a graph node by its exact full name. Faster and more precise than search_entities when you already know the exact name.
        Always call search_entities first to discover the correct full name, never guess or infer the name yourself. Even if you already have
        a full name that you did not get from search_entities, call search_entities anyway to confirm it and get the canonical name as stored in the graph. 
        You can also use with_relationships=True to get the directly connected nodes and their relationship types, for example to see who a person reports to or what emails are connected to an entity.

        Args:
            name: The exact full name of the node as returned by a previous search_entities call, e.g. 'Sarah Chen'.
            with_relationships: If True, also return all directly connected nodes and relationship types. Defaults to False.
        """
        from tools.agent_tools import get_entity as _fn
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _fn(name, with_relationships=with_relationships))
        if not result:
            return f"No graph node found with name '{name}'. Use search_entities to search for the correct full name first."
        n = result["node"]
        lines = [
            f"[{n['type']}] {n['name']}",
            f"Info: {n['info'] or '(none)'}",
            f"Aliases: {', '.join(n['aliases']) or 'none'}",
        ]
        for k, v in n["properties"].items():
            if k not in ("event_ids", "session_ids", "email_ids"):
                lines.append(f"{k}: {v}")
        if with_relationships and result.get("connected"):
            lines.append("Relationships:")
            for c in result["connected"]:
                arrow = f"--[{c['type']}]-->" if c["direction"] == "outgoing" else f"<--[{c['type']}]--"
                lines.append(f"  {n['name']} {arrow} {c['node']['name']} ({c['node']['type']})")
        return "\n".join(lines)

    # ── send_email ────────────────────────────────────────────────────────────

    @function_tool()
    async def send_email(
        self,
        context: RunContext,
        to: list[str],
        subject: str,
        body: str,
    ) -> str:
        """Send an email on behalf of the user. Always compose the full draft, read it back word-for-word, and get explicit verbal confirmation before calling this. Use search_entities to look up the recipient's email address if you only have their name.

        Args:
            to: List of recipient email addresses.
            subject: Email subject line.
            body: Full plain-text email body.
        """
        bad = [v for v in to if "@" not in v]
        if bad:
            raise ToolError(
                f"The 'to' field requires email addresses, but you passed: {bad}. "
                "Call search_entities first to resolve each name to their email address, "
                "then retry send_email with the resolved addresses."
            )
        if not subject or not subject.strip():
            raise ToolError("The 'subject' field must not be empty.")
        if not body or not body.strip():
            raise ToolError("The 'body' field must not be empty.")

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
        to_str = ", ".join(to)
        logger.info(f"send_email: to={to_str} subject={subject!r} id={email_id}")
        return f'Email sent to {to_str} with subject "{subject}". Saved with ID {email_id}.'

    # ── get_session_transcript ────────────────────────────────────────────────

    @function_tool()
    async def get_session_transcript(
        self,
        context: RunContext,
        session_id: str,
    ) -> str:
        """Retrieve the full transcript of a specific past voice session. Only call this when you have a session_id from a search_events result and the user needs more detail than the summary provides.

        Args:
            session_id: The session ID returned by search_events, e.g. 'session_1745123456789'.
        """
        if not session_id.startswith("session_"):
            raise ToolError(
                f"'{session_id}' is not a valid session ID. "
                "Call search_events first and use the session_id value from those results."
            )
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
    if os.environ.get("VOICE_ENABLED", "false").lower() == "true":
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

    voice_enabled = os.environ.get("VOICE_ENABLED", "false").lower() == "true"

    session = AgentSession(
        # ── Voice (set VOICE_ENABLED=true to activate) ────────────────────────
        stt=inference.STT(model="deepgram/nova-3", language="multi") if voice_enabled else None,
        # llm=openai.LLM(model=AGENT_MODEL),
        llm=inference.LLM(model=AGENT_MODEL),
        tts=inference.TTS(model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc") if voice_enabled else None,
        vad=ctx.proc.userdata.get("vad") if voice_enabled else None,
        preemptive_generation=True,
    )

    await ctx.connect()

    if not voice_enabled:
        session.output.set_audio_enabled(False)

    await session.start(
        agent=Assistant(prior_context=prior_context),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
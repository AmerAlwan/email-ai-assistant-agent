"""
agent_tools.py — search and retrieval tools for the voice agent.

Run directly to test any tool:
    uv run python tools/agent_tools.py search_emails
    uv run python tools/agent_tools.py search_events
    uv run python tools/agent_tools.py search_graph
    uv run python tools/agent_tools.py get_email
    uv run python tools/agent_tools.py get_event
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

import openai
import psycopg2
import psycopg2.extras
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range

logger = logging.getLogger("agent")

# ── Config ────────────────────────────────────────────────────────────────────

EMAILS_COLLECTION   = "emails"
EVENTS_COLLECTION   = "events"
ENTITIES_COLLECTION = "entities"
EMBEDDING_MODEL     = "text-embedding-3-small"
SEARCH_LIMIT        = 20

# ── Clients ───────────────────────────────────────────────────────────────────

def _get_qdrant() -> QdrantClient:
    return QdrantClient(url=os.environ["QDRANT_URL"])

def _get_neo4j():
    return GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
        notifications_min_severity="OFF",
    )

def _get_pg_conn():
    return psycopg2.connect(os.environ["POSTGRES_URL"])

# ── Helpers ───────────────────────────────────────────────────────────────────

def _collection_exists(qdrant: QdrantClient, name: str) -> bool:
    return name in {c.name for c in qdrant.get_collections().collections}

def _embed(text: str) -> list[float]:
    response = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"]).embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def _iso_to_ts(date_str: str | None) -> float | None:
    """Convert an ISO 8601 or RFC 2822 date string to a Unix timestamp float, or None."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str[:26], fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    # RFC 2822 email date format (e.g. "Mon, 21 Apr 2026 10:30:00 +0000")
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(date_str).timestamp()
    except Exception:
        pass
    return None

# ── search_emails ─────────────────────────────────────────────────────────────

def search_emails(
    query: str,
    *,
    sender: str | None = None,
    to: str | None = None,
    thread_id: str | None = None,
    labels: list[str] | str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    """
    Semantic search over the emails Qdrant collection.
    Metadata filters are applied server-side before vector scoring.
    Returns up to 20 results.
    """
    qdrant = _get_qdrant()
    if not _collection_exists(qdrant, EMAILS_COLLECTION):
        return []

    # Normalize labels: the LLM may pass a plain string instead of a list,
    # and stored labels are lowercase so normalise case too.
    if isinstance(labels, str):
        labels = [labels]
    if labels:
        labels = [l.lower() for l in labels]
    must: list[FieldCondition] = []
    if sender:
        must.append(FieldCondition(key="sender", match=MatchValue(value=sender)))
    if to:
        must.append(FieldCondition(key="to", match=MatchAny(any=[to])))
    if thread_id:
        must.append(FieldCondition(key="thread_id", match=MatchValue(value=thread_id)))
    if labels:
        must.append(FieldCondition(key="labels", match=MatchAny(any=labels)))
    if date_from:
        ts = _iso_to_ts(date_from)
        if ts is not None:
            must.append(FieldCondition(key="date_ts", range=Range(gte=ts)))
    if date_to:
        # Include the full day by advancing to end-of-day
        ts = _iso_to_ts(date_to)
        if ts is not None:
            must.append(FieldCondition(key="date_ts", range=Range(lte=ts + 86399)))

    qdrant_filter = Filter(must=must) if must else None

    hits = qdrant.query_points(
        collection_name=EMAILS_COLLECTION,
        query=_embed(query),
        limit=SEARCH_LIMIT,
        with_payload=True,
        query_filter=qdrant_filter,
    ).points

    return [
        {
            "score":        h.score,
            "email_id":     h.payload.get("email_id"),
            "sender":       h.payload.get("sender"),
            "to":           h.payload.get("to"),
            "subject":      h.payload.get("subject"),
            "date":         h.payload.get("date"),
            "thread_id":    h.payload.get("thread_id"),
            "labels":       h.payload.get("labels", []),
            "body_preview": h.payload.get("body_preview"),
        }
        for h in hits
    ]

# ── search_events ─────────────────────────────────────────────────────────────

def search_events(
    query: str,
    *,
    session_id: str | None = None,
    entity_names: list[str] | None = None,
    timestamp_from: str | None = None,
    timestamp_to: str | None = None,
) -> list[dict]:
    """
    Semantic search over the events Qdrant collection.
    entity_names filters use AND logic — every listed name must appear in the event.
    Returns up to 20 results, each including the Qdrant point_id for use with get_event.
    """
    qdrant = _get_qdrant()
    if not _collection_exists(qdrant, EVENTS_COLLECTION):
        return []

    must: list[FieldCondition] = []
    if session_id:
        must.append(FieldCondition(key="session_id", match=MatchValue(value=session_id)))
    if entity_names:
        for name in entity_names:
            must.append(FieldCondition(key="entity_names", match=MatchValue(value=name)))
    if timestamp_from:
        must.append(FieldCondition(key="timestamp", range=Range(gte=timestamp_from)))
    if timestamp_to:
        must.append(FieldCondition(key="timestamp", range=Range(lte=timestamp_to)))

    qdrant_filter = Filter(must=must) if must else None

    hits = qdrant.query_points(
        collection_name=EVENTS_COLLECTION,
        query=_embed(query),
        limit=SEARCH_LIMIT,
        with_payload=True,
        query_filter=qdrant_filter,
    ).points

    return [
        {
            "point_id":     str(h.id),
            "score":        h.score,
            "session_id":   h.payload.get("session_id"),
            "description":  h.payload.get("description"),
            "timestamp":    h.payload.get("timestamp"),
            "entity_names": h.payload.get("entity_names", []),
        }
        for h in hits
    ]

# ── search_graph ──────────────────────────────────────────────────────────────

def search_graph(query: str) -> dict | None:
    """
    Semantic graph search:
      1. Embed the query, find the best-matching entity in Qdrant.
      2. Return that node + all directly connected nodes with relationship metadata.
    Returns None if no match or collection doesn't exist.
    """
    qdrant = _get_qdrant()
    if not _collection_exists(qdrant, ENTITIES_COLLECTION):
        return None

    hits = qdrant.query_points(
        collection_name=ENTITIES_COLLECTION,
        query=_embed(query),
        limit=1,
        with_payload=True,
    ).points
    if not hits:
        return None

    hit = hits[0]
    matched_name = hit.payload.get("name")

    driver = _get_neo4j()
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (n {name: $name})
                OPTIONAL MATCH (n)-[r_out]->(nb_out)
                OPTIONAL MATCH (n)<-[r_in]-(nb_in)
                RETURN
                  n,
                  collect(DISTINCT {rel: r_out, node: nb_out}) AS outgoing,
                  collect(DISTINCT {rel: r_in,  node: nb_in})  AS incoming
                """,
                name=matched_name,
            )
            records = list(result)
            if not records:
                return None

            rec = records[0]
            raw_node = rec["n"]

            def to_node(n) -> dict:
                p = dict(n.items())
                extra = {k: v for k, v in p.items() if k not in ("name", "info", "aliases")}
                return {
                    "name":       p.get("name", ""),
                    "type":       list(n.labels)[0] if n.labels else "Entity",
                    "info":       p.get("info", ""),
                    "aliases":    p.get("aliases") or [],
                    "properties": extra,
                }

            connected = []
            for entry in rec["outgoing"]:
                if entry["node"] is None:
                    continue
                connected.append({
                    "direction": "outgoing",
                    "type":      entry["rel"].type,
                    "node":      to_node(entry["node"]),
                })
            for entry in rec["incoming"]:
                if entry["node"] is None:
                    continue
                connected.append({
                    "direction": "incoming",
                    "type":      entry["rel"].type,
                    "node":      to_node(entry["node"]),
                })

            return {
                "match_score": hit.score,
                "node":        to_node(raw_node),
                "connected":   connected,
            }
    finally:
        driver.close()

# ── get_email ─────────────────────────────────────────────────────────────────

def get_email(email_id: str) -> dict | None:
    """Fetch a single email by ID from Postgres. Returns None if not found."""
    conn = _get_pg_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, from_addr, to_addr, subject, date, body, labels, thread_id, raw, seeded_at FROM emails WHERE id = %s",
                (email_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()

# ── get_session_transcript ───────────────────────────────────────────────────

def get_session_transcript(session_id: str) -> dict | None:
    """Fetch the full transcript and summary for a session by its ID from Postgres.
    Returns None if not found."""
    conn = _get_pg_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT session_id, transcript, summary, created_at "
                "FROM session_transcripts WHERE session_id = %s",
                (session_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()

# ── get_recent_context ────────────────────────────────────────────────────────

def get_recent_context() -> str:
    """
    Returns a formatted context string with:
    - Last 10 events across all sessions (from Qdrant)
    - Full transcript of the most recent session (from Postgres)
    - Summaries of the 3 sessions before that (from Postgres)
    Returns an empty string if no prior sessions exist.
    """
    qdrant = _get_qdrant()
    conn = _get_pg_conn()

    try:
        # ── Last 10 events from Qdrant ────────────────────────────────
        events_section = ""
        if _collection_exists(qdrant, EVENTS_COLLECTION):
            all_points: list = []
            offset = None
            while True:
                batch, next_offset = qdrant.scroll(
                    collection_name=EVENTS_COLLECTION,
                    limit=250,
                    with_payload=True,
                    offset=offset,
                )
                all_points.extend(batch)
                if next_offset is None:
                    break
                offset = next_offset

            if all_points:
                all_points.sort(
                    key=lambda p: p.payload.get("session_id", ""), reverse=True
                )
                last_10 = all_points[:10]
                lines = ["── RECENT EVENTS (last 10) ──────────────────────────────────────"]
                for p in last_10:
                    desc = p.payload.get("description", "")
                    ts = p.payload.get("timestamp") or "-"
                    entities = ", ".join(p.payload.get("entity_names") or []) or "-"
                    lines.append(f"• {desc}  (at {ts}, entities: {entities})")
                events_section = "\n".join(lines)

        # ── Session transcripts/summaries from Postgres ───────────────
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS session_transcripts (
                    session_id  TEXT PRIMARY KEY,
                    transcript  TEXT NOT NULL,
                    summary     TEXT,
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            conn.commit()
            cur.execute(
                "SELECT session_id, transcript, summary FROM session_transcripts "
                "ORDER BY created_at DESC LIMIT 4"
            )
            rows = cur.fetchall()

        if not rows:
            return events_section  # only events, no sessions yet

        most_recent = rows[0]
        older = rows[1:4]

        transcript_section = (
            f"── LAST SESSION TRANSCRIPT (session: {most_recent['session_id']}) ──────────────\n"
            + (most_recent["transcript"] or "(empty)")
        )

        summaries_section = ""
        if older:
            summary_lines = ["── PREVIOUS SESSION SUMMARIES ───────────────────────────────────"]
            for row in older:
                summary_lines.append(
                    f"\n[{row['session_id']}]\n{row['summary'] or '(no summary)'}"
                )
            summaries_section = "\n".join(summary_lines)

        parts = [p for p in [events_section, transcript_section, summaries_section] if p]
        return "\n\n".join(parts)

    finally:
        conn.close()


# ── get_event ─────────────────────────────────────────────────────────────────

def get_event(point_id: str) -> dict | None:
    """Fetch a single event by its Qdrant point ID. Returns None if not found."""
    qdrant = _get_qdrant()
    if not _collection_exists(qdrant, EVENTS_COLLECTION):
        return None

    points = qdrant.retrieve(
        collection_name=EVENTS_COLLECTION,
        ids=[point_id],
        with_payload=True,
    )
    if not points:
        return None

    p = points[0].payload or {}
    return {
        "point_id":     point_id,
        "session_id":   p.get("session_id"),
        "description":  p.get("description"),
        "timestamp":    p.get("timestamp"),
        "entity_names": p.get("entity_names", []),
    }

# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tool = sys.argv[1] if len(sys.argv) > 1 else None

    if tool == "search_emails":
        # ── Edit these ──────────────────────────────────────────────────
        query   = "work"
        filters = dict(
            # sender    = "david.martinez@company.com",
            # labels    = ["INBOX"],
            # date_from = "2026-04-01",
            # date_to   = "2026-04-30",
        )
        # ────────────────────────────────────────────────────────────────
        results = search_emails(query, **filters)
        print(f"[search_emails] '{query}' → {len(results)} result(s)\n")
        for r in results[:3]:
            print(f"  [{r['score']:.4f}] {r['subject'] or '(no subject)'}")
            print(f"    From   : {r['sender'] or '-'}")
            print(f"    Date   : {r['date'] or '-'}")
            print(f"    Labels : {', '.join(r['labels']) or '-'}")
            print(f"    Preview: {(r['body_preview'] or '')}")
            print()

    elif tool == "search_events":
        # ── Edit these ──────────────────────────────────────────────────
        query   = "email sent to Sarah about kickoff meeting"
        filters = dict(
            # session_id    = "session-test-001",
            # entity_names  = ["Sarah Chen"],
        )
        # ────────────────────────────────────────────────────────────────
        results = search_events(query, **filters)
        print(f"[search_events] '{query}' → {len(results)} result(s)\n")
        for r in results[:5]:
            print(f"  [{r['score']:.4f}] {r['description'] or '(no description)'}")
            print(f"    Session  : {r['session_id'] or '-'}")
            print(f"    Timestamp: {r['timestamp'] or '-'}")
            print(f"    Entities : {', '.join(r['entity_names']) or '-'}")
            print(f"    point_id : {r['point_id']}")
            print()

    elif tool == "search_graph":
        # ── Edit these ──────────────────────────────────────────────────
        query = "Sarah"
        # ────────────────────────────────────────────────────────────────
        result = search_graph(query)
        if not result:
            print("No matching node found.")
        else:
            n = result["node"]
            print(f"[search_graph] Match [{result['match_score']:.4f}]: [{n['type']}] {n['name']}")
            print(f"  Info    : {n['info'] or '-'}")
            print(f"  Aliases : {', '.join(n['aliases']) or '-'}")
            if n["properties"]:
                print(f"  Props   : {json.dumps(n['properties'])}")
            if not result["connected"]:
                print("  (no connected nodes)")
            else:
                print(f"\n  Connected ({len(result['connected'])}):")
                for c in result["connected"]:
                    arrow = f"--[{c['type']}]-->" if c["direction"] == "outgoing" else f"<--[{c['type']}]--"
                    print(f"    {n['name']} {arrow} [{c['node']['type']}] {c['node']['name']}")
                    print(f"      Info: {c['node']['info'] or '-'}")

    elif tool == "get_email":
        # ── Edit these ──────────────────────────────────────────────────
        email_id = "email-008"
        # ────────────────────────────────────────────────────────────────
        email = get_email(email_id)
        if not email:
            print(f"Email '{email_id}' not found.")
        else:
            print(f"[get_email] {email['id']}")
            print(f"  From   : {email['from_addr'] or '-'}")
            print(f"  To     : {email['to_addr'] or '-'}")
            print(f"  Subject: {email['subject'] or '-'}")
            print(f"  Date   : {email['date'] or '-'}")
            print(f"  Labels : {', '.join(email['labels'] or []) or '-'}")
            print(f"  Body   :\n{email['body'] or '(empty)'}")

    elif tool == "get_event":
        # ── Edit these ──────────────────────────────────────────────────
        point_id = "1833e475-a880-5267-93b0-a031b62a08ba"
        # ────────────────────────────────────────────────────────────────
        event = get_event(point_id)
        if not event:
            print(f"Event '{point_id}' not found.")
        else:
            print(f"[get_event] {event['point_id']}")
            print(f"  Session  : {event['session_id'] or '-'}")
            print(f"  Timestamp: {event['timestamp'] or '-'}")
            print(f"  Entities : {', '.join(event['entity_names']) or '-'}")
            print(f"  Desc     : {event['description'] or '-'}")

    else:
        print("Usage: uv run python tools/agent_tools.py <tool>")
        print("Tools: search_emails | search_events | search_graph | get_email | get_event")
        sys.exit(1)

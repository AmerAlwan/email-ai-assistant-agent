import json
import os
import uuid
import psycopg2
import psycopg2.extras
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import openai
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, UpdateStatus

# prompts.json lives next to this file
_PROMPTS_PATH = Path(__file__).parent / "prompts.json"


def _load_prompt(field: str) -> str:
    prompts = json.loads(_PROMPTS_PATH.read_text(encoding="utf-8"))
    raw = prompts.get(field, "")
    # Support both a plain string and an array-of-lines for readability in JSON
    if isinstance(raw, list):
        prompt = "\n".join(raw).strip()
    else:
        prompt = raw.strip()
    if not prompt:
        raise ValueError(f"prompt field '{field}' in prompts.json is empty or missing")
    return prompt


def _get_pg_conn():
    return psycopg2.connect(os.environ["POSTGRES_URL"])


# ── Qdrant ────────────────────────────────────────────────────────────────────

QDRANT_COLLECTION = "emails"
ENTITIES_COLLECTION = "entities"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_DIM = 1536


def _get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=os.environ["QDRANT_URL"])


def _ensure_qdrant_collection(client: QdrantClient) -> None:
    """Create the memories collection if it doesn't exist yet."""
    existing = {c.name for c in client.get_collections().collections}
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )


def _embed(text: str) -> list[float]:
    """Embed a single text string using OpenAI text-embedding-3-small."""
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


def _save_email_to_qdrant(client: QdrantClient, email: dict) -> None:
    """
    Embed the email body and upsert a point into the memories collection.
    The point payload carries all email metadata for filtered retrieval.
    """
    body = email.get("body") or ""
    subject = email.get("subject") or ""
    # Embed subject + body so subject keywords are searchable
    text_to_embed = f"Subject: {subject}\n\n{body}".strip()

    vector = _embed(text_to_embed)

    # Qdrant point IDs must be UUIDs or unsigned ints; derive a stable UUID from email id
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, email.get("id", text_to_embed)))

    payload = {
        "type": "email",
        "source_type": "email",
        "email_id": email.get("id"),
        "sender": email.get("from"),
        "to": email["to"] if isinstance(email.get("to"), list) else ([email["to"]] if email.get("to") else []),
        "subject": subject,
        "date": email.get("date"),
        "date_ts": _iso_to_ts(email.get("date")),
        "thread_id": email.get("thread_id"),
        "labels": email.get("labels", []),
        "body_preview": body[:300],
    }

    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[PointStruct(id=point_id, vector=vector, payload=payload)],
    )


def _ensure_entities_collection(client: QdrantClient) -> None:
    """Create the entities collection if it doesn't exist yet."""
    existing = {c.name for c in client.get_collections().collections}
    if ENTITIES_COLLECTION not in existing:
        client.create_collection(
            collection_name=ENTITIES_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )


def _save_entities_to_qdrant(client: QdrantClient, entities: list[dict]) -> None:
    """
    Embed each entity's name + aliases + info and upsert into the entities collection.
    Point ID is a stable UUID derived from the entity name so re-runs are idempotent.
    """
    for entity in entities:
        name = entity.get("name", "").strip()
        if not name:
            continue
        info = entity.get("info", "").strip()
        aliases = entity.get("aliases") or []
        # Build a rich text blob for semantic search
        parts = [name]
        if aliases:
            parts.append("Also known as: " + ", ".join(aliases))
        if info:
            parts.append(info)
        text = ". ".join(parts)

        vector = _embed(text)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"entity:{name}"))

        client.upsert(
            collection_name=ENTITIES_COLLECTION,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "name": name,
                        "type": entity.get("type", "Entity"),
                        "info": info,
                        "aliases": aliases,
                    },
                )
            ],
        )


def _ensure_emails_table(conn) -> None:
    """Create the emails table if it doesn't exist yet."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS emails (
                id          TEXT PRIMARY KEY,
                from_addr   TEXT,
                to_addr     TEXT,
                subject     TEXT,
                date        TEXT,
                body        TEXT,
                labels      TEXT[]   DEFAULT '{}',
                thread_id   TEXT,
                raw         JSONB    NOT NULL,
                seeded_at   TIMESTAMPTZ DEFAULT NOW()
            )
        """)
    conn.commit()


def _save_email_to_pg(conn, email: dict) -> None:
    """Upsert a single email record into Postgres."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO emails (id, from_addr, to_addr, subject, date, body, thread_id, raw)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                from_addr = EXCLUDED.from_addr,
                to_addr   = EXCLUDED.to_addr,
                subject   = EXCLUDED.subject,
                date      = EXCLUDED.date,
                body      = EXCLUDED.body,
                thread_id = EXCLUDED.thread_id,
                raw       = EXCLUDED.raw,
                seeded_at = NOW()
        """, (
            email.get("id"),
            email.get("from"),
            email.get("to"),
            email.get("subject"),
            email.get("date"),
            email.get("body"),
            email.get("thread_id"),
            json.dumps(email),
        ))
    conn.commit()


def _get_neo4j_driver():
    return GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
        notifications_min_severity="OFF",
    )


def _get_existing_nodes(driver) -> list[dict]:
    """Return a lightweight summary of every node currently in the graph."""
    with driver.session() as session:
        result = session.run(
            "MATCH (n) RETURN labels(n)[0] AS type, n.name AS name, n.info AS info, n.aliases AS aliases"
        )
        return [
            {
                "type": r["type"],
                "name": r["name"],
                "info": r["info"] or "",
                "aliases": r["aliases"] or [],
            }
            for r in result
        ]


def _get_existing_edges(driver) -> list[dict]:
    """Return every relationship currently in the graph."""
    with driver.session() as session:
        result = session.run(
            "MATCH (a)-[r]->(b) RETURN a.name AS from_name, type(r) AS rel_type, b.name AS to_name"
        )
        return [
            {"from": r["from_name"], "type": r["rel_type"], "to": r["to_name"]}
            for r in result
        ]


def _save_to_neo4j(driver, extraction: dict, email_id: str) -> None:
    """
    Merge entities and relationships from an extraction result into Neo4j.
    Each entity accumulates a list of associated email IDs (email_ids).
    Uses MERGE so re-running ingestion on the same email is idempotent.
    """
    entities = extraction.get("entities", [])
    relationships = extraction.get("relationships", [])

    with driver.session() as session:
        # Ensure the fixed User node always exists
        session.run("MERGE (:User {name: 'User'})")

        # Upsert every entity, appending email_id to its email_ids list
        for entity in entities:
            name = entity["name"]
            node_type = entity.get("type", "Entity")
            properties = entity.get("properties", {})
            aliases = entity.get("aliases", [])
            info = entity.get("info", "")

            session.run(
                f"""
                MERGE (n:{node_type} {{name: $name}})
                SET n += $properties
                SET n.info = $info
                SET n.aliases = apoc.coll.toSet(coalesce(n.aliases, []) + $aliases)
                SET n.email_ids = CASE
                    WHEN $email_id IN coalesce(n.email_ids, []) THEN n.email_ids
                    ELSE coalesce(n.email_ids, []) + $email_id
                END
                """,
                name=name,
                properties=properties,
                aliases=aliases,
                info=info,
                email_id=email_id,
            )

        # Upsert every relationship
        for rel in relationships:
            from_name = rel["from"]
            to_name = rel["to"]
            rel_type = rel["type"].upper().replace(" ", "_")
            properties = rel.get("properties", {})

            session.run(
                f"""
                MATCH (a {{name: $from_name}})
                MATCH (b {{name: $to_name}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r += $properties
                """,
                from_name=from_name,
                to_name=to_name,
                properties=properties,
            )


def load_emails_from_file(path: str) -> list[dict]:
    """Load a JSON file that contains an array of email objects."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")
    return data


def inject_emails(emails: list[dict]) -> None:
    """
    For each email:
      1. Save the raw email to Postgres.
      2. Embed the body and upsert into Qdrant.
      3. Send it to Claude to extract entities and relationships.
      4. Merge the extraction into Neo4j, appending the email ID to each entity.
    """
    system_prompt = _load_prompt("ingestion")
    openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    driver = _get_neo4j_driver()
    pg_conn = _get_pg_conn()
    qdrant = _get_qdrant_client()

    try:
        _ensure_emails_table(pg_conn)
        _ensure_qdrant_collection(qdrant)
        _ensure_entities_collection(qdrant)

        for i, email in enumerate(emails):
            email_id = email.get("id", f"email_{i}")

            # 1. Persist raw email to Postgres
            _save_email_to_pg(pg_conn, email)
            print(f"[email {i}] saved to Postgres (id={email_id})")

            # 2. Embed and store in Qdrant
            _save_email_to_qdrant(qdrant, email)
            print(f"[email {i}] embedded and saved to Qdrant")

            # 3. Ask Claude to extract graph data, providing existing nodes for resolution
            existing_nodes = _get_existing_nodes(driver)
            existing_edges = _get_existing_edges(driver)
            existing_nodes_text = (
                json.dumps(existing_nodes, indent=2) if existing_nodes else "(none yet)"
            )
            existing_edges_text = (
                json.dumps(existing_edges, indent=2) if existing_edges else "(none yet)"
            )
            user_message = (
                f"Extract graph entities and relationships from this email:\n\n"
                f"{json.dumps(email, indent=2)}\n\n"
                f"---\nExisting graph nodes (match ambiguous names against canonical names and aliases here — do NOT create a new node if one already matches):\n"
                f"{existing_nodes_text}\n\n"
                f"---\nExisting graph edges (do NOT include a relationship in your output if an identical from/type/to already exists here):\n"
                f"{existing_edges_text}"
            )
            message = openai_client.chat.completions.create(
                model="gpt-5.4-nano",
                max_completion_tokens=2048,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            raw = (message.choices[0].message.content or "").strip()
            # Strip markdown code fences if Claude wraps the JSON in them
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
                raw = raw.rsplit("```", 1)[0]
                raw = raw.strip()

            try:
                extraction = json.loads(raw)
            except json.JSONDecodeError:
                print(f"[email {i}] Claude returned non-JSON output:\n{raw}")
                continue

            print(f"[email {i}] extraction:")
            print(json.dumps(extraction, indent=2))

            # 4. Merge into Neo4j
            _save_to_neo4j(driver, extraction, email_id)
            print(f"[email {i}] saved to Neo4j")

            # 5. Embed and upsert entities into Qdrant entities collection
            entities = extraction.get("entities", [])
            _save_entities_to_qdrant(qdrant, entities)
            print(f"[email {i}] upserted {len(entities)} entity vector(s) to Qdrant")
    finally:
        driver.close()
        pg_conn.close()
        qdrant.close()


if __name__ == "__main__":
    # Wipe all existing data before re-ingesting
    pg = _get_pg_conn()
    try:
        _ensure_emails_table(pg)
        with pg.cursor() as cur:
            cur.execute("DELETE FROM emails")
        pg.commit()
        print("Cleared emails table")
    finally:
        pg.close()

    neo = _get_neo4j_driver()
    try:
        with neo.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
        print("Cleared Neo4j graph")
    finally:
        neo.close()

    qdrant = _get_qdrant_client()
    try:
        existing = {c.name for c in qdrant.get_collections().collections}
        if QDRANT_COLLECTION in existing:
            qdrant.delete_collection(QDRANT_COLLECTION)
            print(f"Cleared Qdrant collection '{QDRANT_COLLECTION}'")
        if ENTITIES_COLLECTION in existing:
            qdrant.delete_collection(ENTITIES_COLLECTION)
            print(f"Cleared Qdrant collection '{ENTITIES_COLLECTION}'")
    finally:
        qdrant.close()

    emails = load_emails_from_file("old_emails.json")
    inject_emails(emails)
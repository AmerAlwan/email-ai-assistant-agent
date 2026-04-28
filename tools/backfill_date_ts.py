"""
Backfill date_ts on existing Qdrant email points.
Run once inside the agent container:
    uv run python tools/backfill_date_ts.py
"""
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import SetPayload, PointIdsList

load_dotenv()

COLLECTION = "emails"


def iso_to_ts(date_str: str | None) -> float | None:
    if not date_str:
        return None
    # ISO 8601 variants
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
        dt = parsedate_to_datetime(date_str)
        return dt.timestamp()
    except Exception:
        pass
    return None


def main() -> None:
    client = QdrantClient(url=os.environ["QDRANT_URL"])

    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION not in existing:
        print(f"Collection '{COLLECTION}' not found — nothing to backfill.")
        return

    updated = 0
    skipped = 0
    offset = None
    debug_printed = 0

    while True:
        batch, next_offset = client.scroll(
            collection_name=COLLECTION,
            limit=100,
            with_payload=True,
            offset=offset,
        )

        for point in batch:
            payload = point.payload or {}

            # Debug: print first 3 points raw
            if debug_printed < 3:
                print(f"[DEBUG] id={point.id} date={payload.get('date')!r} date_ts={payload.get('date_ts')!r}")
                debug_printed += 1

            # Already has date_ts — skip
            if payload.get("date_ts") is not None:
                skipped += 1
                continue

            ts = iso_to_ts(payload.get("date"))
            if ts is None:
                skipped += 1
                continue

            client.set_payload(
                collection_name=COLLECTION,
                payload={"date_ts": ts},
                points=PointIdsList(points=[point.id]),
            )
            updated += 1

        if next_offset is None:
            break
        offset = next_offset

    print(f"Done. Updated: {updated}, Skipped: {skipped}")


if __name__ == "__main__":
    main()

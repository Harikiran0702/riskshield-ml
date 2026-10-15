"""SQLite audit, transactional idempotency, and delayed label joins for one local service."""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd


class ConflictError(ValueError):
    """A transaction ID was reused with different content or a label was changed."""


class AuditStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""CREATE TABLE IF NOT EXISTS scores (
                transaction_id TEXT PRIMARY KEY, request_hash TEXT NOT NULL,
                response TEXT NOT NULL, scored_at TEXT NOT NULL, label INTEGER,
                labeled_at TEXT, CHECK(label IN (0,1)))""")

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=10)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def score(
        self, payloads: list[dict], calculate: Callable[[list[dict]], list[dict]]
    ) -> list[dict]:
        """Atomically score unseen IDs and reuse exact responses across retries/restarts."""
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            pending, output, unique = [], {}, {}
            for payload in payloads:
                txid = payload["transaction_id"]
                digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                if txid in unique and unique[txid] != digest:
                    raise ConflictError("Conflicting duplicate transaction in batch")
                if txid in unique:
                    continue
                unique[txid] = digest
                stored = conn.execute(
                    "SELECT request_hash, response FROM scores WHERE transaction_id=?", (txid,)
                ).fetchone()
                if stored:
                    if stored[0] != digest:
                        raise ConflictError("Transaction ID already scored with different context")
                    output[txid] = json.loads(stored[1])
                else:
                    pending.append(payload)
            if pending:
                results = calculate(pending)
                for payload, result in zip(pending, results, strict=True):
                    txid = payload["transaction_id"]
                    conn.execute(
                        "INSERT INTO scores VALUES (?, ?, ?, ?, NULL, NULL)",
                        (
                            txid,
                            unique[txid],
                            json.dumps(result),
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                    output[txid] = result
            return [output[p["transaction_id"]] for p in payloads]

    def label(self, transaction_id: str, label: int) -> None:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT label FROM scores WHERE transaction_id=?", (transaction_id,)
            ).fetchone()
            if row is None:
                raise KeyError("Transaction has not been scored")
            if row[0] is not None and row[0] != label:
                raise ConflictError("A different label has already been recorded")
            if row[0] is None:
                conn.execute(
                    "UPDATE scores SET label=?, labeled_at=? WHERE transaction_id=?",
                    (label, datetime.now(timezone.utc).isoformat(), transaction_id),
                )

    def outcomes(self) -> pd.DataFrame:
        with self.connect() as conn:
            rows = conn.execute("SELECT response, label FROM scores").fetchall()
        return pd.DataFrame(
            [
                {"score": json.loads(response)["fraud_probability"], "fraud_label": label}
                for response, label in rows
            ],
            columns=["score", "fraud_label"],
        )

"""
Context Store for Magicpin Vera AI Challenge
=============================================
Thread-safe, versioned in-memory store for all 4 context scopes:
- category
- merchant
- customer
- trigger

Key properties:
- Atomic version replacement: higher version replaces older version
- Stale version rejection: lower version returns (False, "stale_version", current_version)
- Idempotency: exact same version returns (True, "idempotent", current_version)
- Accurate counts by scope for GET /v1/healthz
- Cache invalidation on context update to prevent stale evidence
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple


class ContextStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        # Key: (scope, context_id) -> {"version": int, "payload": dict, "stored_at": str, "delivered_at": str}
        self._store: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def push(
        self,
        scope: str,
        context_id: str,
        version: int,
        payload: Dict[str, Any],
        delivered_at: Optional[str] = None,
    ) -> Tuple[bool, str, int]:
        """
        Push a context object into the store.
        Returns (accepted: bool, reason_or_status: str, version: int).
        - If stale (incoming version < current version): returns (False, "stale_version", current_version)
        - If idempotent (incoming version == current version): returns (True, "idempotent", current_version)
        - If accepted (incoming version > current version or new): returns (True, "accepted", version)
        """
        if not scope or not context_id or version is None:
            return False, "invalid_payload", 0

        key = (scope, str(context_id))
        stored_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        with self._lock:
            current = self._store.get(key)
            if current is not None:
                cur_version = current.get("version", 0)
                if version < cur_version:
                    return False, "stale_version", cur_version
                if version == cur_version:
                    # Idempotent - accept but no state change
                    return True, "idempotent", cur_version

            # Atomic update / insert
            self._store[key] = {
                "version": version,
                "payload": payload,
                "stored_at": stored_at,
                "delivered_at": delivered_at or stored_at,
            }
            return True, "accepted", version

    def get(self, scope: str, context_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve payload for (scope, context_id)."""
        key = (scope, str(context_id))
        with self._lock:
            entry = self._store.get(key)
            if entry:
                return entry.get("payload")
            return None

    def get_version(self, scope: str, context_id: str) -> Optional[int]:
        """Retrieve current version for (scope, context_id)."""
        key = (scope, str(context_id))
        with self._lock:
            entry = self._store.get(key)
            if entry:
                return entry.get("version")
            return None

    def counts(self) -> Dict[str, int]:
        """Return counts of loaded contexts by scope for healthz."""
        counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
        with self._lock:
            for (scope, _), _ in self._store.items():
                if scope in counts:
                    counts[scope] += 1
                else:
                    counts[scope] = counts.get(scope, 0) + 1
        return counts

    def clear(self) -> None:
        """Reset the context store (e.g. for teardown or isolated testing)."""
        with self._lock:
            self._store.clear()

    def all_for_scope(self, scope: str) -> Dict[str, Dict[str, Any]]:
        """Return all payloads for a specific scope keyed by context_id."""
        result: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            for (sc, cid), entry in self._store.items():
                if sc == scope:
                    result[cid] = entry.get("payload", {})
        return result

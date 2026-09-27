"""
Magicpin AI Challenge — Vera Assistant Bot ("bot.py")
=====================================================
Production-quality Merchant & Customer Engagement System:
- Standalone compose() function conforming to challenge-brief.md §7.1
- Complete HTTP server exposing all 5 endpoints:
    GET  /v1/healthz
    GET  /v1/metadata
    POST /v1/context
    POST /v1/tick
    POST /v1/reply
- Zero-dependency built-in ThreadingHTTPServer support (runs instantly with `python3 bot.py`)
- Seamless FastAPI integration if fastapi/uvicorn is available (`uvicorn bot:app`)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from context_store import ContextStore
from decision_engine import DecisionEngine
from evidence_extractor import extract_evidence
from message_strategy import build_message_strategy
from composer import compose_grounded_message, ComposedMessage
from state_machine import ConversationManager

START_TIME = time.time()

# Shared Global Instances
STORE = ContextStore()
ENGINE = DecisionEngine(STORE)
CONV_MGR = ConversationManager()


# =============================================================================
# STANDALONE COMPOSE FUNCTION (challenge-brief.md §7.1)
# =============================================================================

def compose(
    category: Dict[str, Any],
    merchant: Dict[str, Any],
    trigger: Dict[str, Any],
    customer: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Standalone composition function.
    Inputs are dictionaries loaded from the dataset JSON.
    Returns:
        body              — the WhatsApp message body
        cta               — the call-to-action (binary_yes_no, open_ended, etc.)
        send_as           — 'vera' (merchant-facing) or 'merchant_on_behalf' (customer-facing)
        suppression_key   — for deduplication
        rationale         — explanation of context anchors and objective
    """
    facts = extract_evidence(category, merchant, trigger, customer)
    strategy = build_message_strategy(facts)
    composed = compose_grounded_message(facts, strategy)
    return {
        "body": composed.body,
        "cta": composed.cta,
        "send_as": composed.send_as,
        "suppression_key": composed.suppression_key,
        "rationale": composed.rationale,
    }


# =============================================================================
# HANDLER LOGIC FOR ENDPOINTS
# =============================================================================

def handle_healthz() -> Dict[str, Any]:
    uptime = int(time.time() - START_TIME)
    return {
        "status": "ok",
        "uptime_seconds": uptime,
        "contexts_loaded": STORE.counts(),
    }


def handle_metadata() -> Dict[str, Any]:
    return {
        "team_name": "Team Vera Elite",
        "team_members": ["Lead Engineer"],
        "model": "grounded-evidence-composer-v1",
        "approach": "deterministic decision engine + grounded evidence extraction + category voice + auto-repair validator",
        "contact_email": "team@magicpin.challenge",
        "version": "1.0.0",
        "submitted_at": "2026-04-26T08:00:00Z",
    }


def handle_context_push(data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    scope = data.get("scope")
    cid = data.get("context_id")
    version = data.get("version")
    payload = data.get("payload")
    delivered_at = data.get("delivered_at")

    if not scope or not cid or version is None or payload is None:
        return 400, {"accepted": False, "reason": "invalid_payload", "details": "Missing required fields"}

    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if scope not in valid_scopes:
        return 400, {"accepted": False, "reason": "invalid_scope", "details": f"Scope must be in {valid_scopes}"}

    accepted, status, cur_version = STORE.push(scope, cid, version, payload, delivered_at)
    stored_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    if not accepted and status == "stale_version":
        return 409, {"accepted": False, "reason": "stale_version", "current_version": cur_version}

    return 200, {
        "accepted": True,
        "ack_id": f"ack_{cid}_v{version}",
        "stored_at": stored_at,
    }


def handle_tick(data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    now_iso = data.get("now", datetime.now(timezone.utc).isoformat())
    available_triggers = data.get("available_triggers", [])
    actions = ENGINE.evaluate_tick(now_iso, available_triggers)

    # Record outbounds in conversation manager
    for act in actions:
        conv_id = act.get("conversation_id", "")
        mid = act.get("merchant_id", "")
        cid = act.get("customer_id")
        body = act.get("body", "")
        CONV_MGR.record_outbound(conv_id, mid, body, act.get("rationale", ""), cid)

    return 200, {"actions": actions}


def handle_reply(data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    conv_id = data.get("conversation_id", "")
    merchant_id = data.get("merchant_id")
    customer_id = data.get("customer_id")
    from_role = data.get("from_role", "merchant")
    message = data.get("message", "")
    turn_number = int(data.get("turn_number", 1))

    if not conv_id or not message:
        return 400, {"error": "conversation_id and message are required"}

    res = CONV_MGR.handle_inbound_reply(conv_id, merchant_id, message, turn_number, from_role)
    
    # If merchant opted out, record in decision engine
    if res.get("action") == "end" and merchant_id and "opted out" in res.get("rationale", ""):
        ENGINE.record_opt_out(merchant_id)

    return 200, res


# =============================================================================
# BUILT-IN THREADING HTTP SERVER (Zero Dependencies Required)
# =============================================================================

class VeraRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        # Keep server quiet and fast
        return

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/v1/healthz":
            self._send_json(200, handle_healthz())
        elif path == "/v1/metadata":
            self._send_json(200, handle_metadata())
        else:
            self._send_json(404, {"error": "Not Found", "path": path})

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)

        try:
            data = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception:
            self._send_json(400, {"accepted": False, "reason": "invalid_json"})
            return

        if path == "/v1/context":
            status, resp = handle_context_push(data)
            self._send_json(status, resp)
        elif path == "/v1/tick":
            status, resp = handle_tick(data)
            self._send_json(status, resp)
        elif path == "/v1/reply":
            status, resp = handle_reply(data)
            self._send_json(status, resp)
        else:
            self._send_json(404, {"error": "Not Found", "path": path})


# =============================================================================
# OPTIONAL FASTAPI INTEGRATION
# =============================================================================

try:
    from fastapi import FastAPI, Response, status as http_status
    from pydantic import BaseModel

    app = FastAPI(title="Magicpin Vera Bot")

    @app.get("/v1/healthz")
    async def api_healthz():
        return handle_healthz()

    @app.get("/v1/metadata")
    async def api_metadata():
        return handle_metadata()

    class CtxPayload(BaseModel):
        scope: str
        context_id: str
        version: int
        payload: Dict[str, Any]
        delivered_at: Optional[str] = None

    @app.post("/v1/context")
    async def api_context(body: CtxPayload, response: Response):
        code, data = handle_context_push(body.dict())
        response.status_code = code
        return data

    class TickPayload(BaseModel):
        now: Optional[str] = None
        available_triggers: List[str] = []

    @app.post("/v1/tick")
    async def api_tick(body: TickPayload, response: Response):
        code, data = handle_tick(body.dict())
        response.status_code = code
        return data

    class ReplyPayload(BaseModel):
        conversation_id: str
        merchant_id: Optional[str] = None
        customer_id: Optional[str] = None
        from_role: str = "merchant"
        message: str
        received_at: Optional[str] = None
        turn_number: int = 1

    @app.post("/v1/reply")
    async def api_reply(body: ReplyPayload, response: Response):
        code, data = handle_reply(body.dict())
        response.status_code = code
        return data

except ImportError:
    app = None


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

def run_server(port: int = 8080, host: str = "0.0.0.0") -> None:
    server = ThreadingHTTPServer((host, port), VeraRequestHandler)
    print(f"Vera Assistant Server listening on {host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    run_server(port=port, host=host)

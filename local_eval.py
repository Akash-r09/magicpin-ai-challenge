#!/usr/bin/env python3
"""
Local Evaluation Harness for the Magicpin Vera AI Challenge
=============================================================
This is OUR OWN evaluation tool. It is separate from `judge_simulator.py`
(which we do not modify) and fixes two gaps that only exist in that
packaged dev-convenience script's `full_evaluation` scenario:

  1. `now` is CONFIGURABLE (defaults to the dataset's own reference clock,
     2026-04-26T10:30:00Z — the same timestamp `test_solution.py` and the
     testing brief use) instead of `datetime.utcnow()` (real wall-clock
     time, which makes ~21/25 seed triggers look already-expired if you
     run the tool months after the dataset was authored).
  2. ALL customer contexts are pushed during warmup (in addition to all
     categories, all merchants), so customer-scoped triggers can actually
     be evaluated instead of being hard-suppressed with
     `customer_context_missing`.

It talks to the real bot over real HTTP (spawns `python3 bot.py` as a
subprocess), exactly like the real judge harness would, and pushes/ticks
using the exact wire schema from challenge-testing-brief.md.

It does NOT bypass expiry logic and does NOT invent evidence: it just
evaluates the same 25-trigger seed dataset at the clock the dataset was
actually authored for.

Usage:
    python3 local_eval.py
    python3 local_eval.py --now 2026-04-26T10:30:00Z
    python3 local_eval.py --dataset expanded --now 2026-04-26T10:30:00Z
    python3 local_eval.py --now 2026-09-27T00:00:00Z   # reproduce the
                                                        # original wall-clock
                                                        # symptom on demand
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import request as urlrequest, error as urlerror

ROOT = Path(__file__).parent
DEFAULT_NOW = "2026-04-26T10:30:00Z"


# =============================================================================
# Dataset loading (supports both the flat 25-trigger seed set and the
# expanded 100-trigger / 50-merchant / 200-customer generalization set)
# =============================================================================

class Dataset:
    def __init__(self):
        self.categories: Dict[str, dict] = {}
        self.merchants: Dict[str, dict] = {}
        self.customers: Dict[str, dict] = {}
        self.triggers: Dict[str, dict] = {}
        self.trigger_order: List[str] = []  # preserves seed file order for batching

    def load_seed(self):
        cat_dir = ROOT / "dataset" / "categories"
        for f in sorted(cat_dir.glob("*.json")):
            data = json.load(open(f))
            self.categories[data.get("slug", f.stem)] = data

        for fname, container, key in [
            ("merchants_seed.json", "merchants", "merchant_id"),
            ("customers_seed.json", "customers", "customer_id"),
            ("triggers_seed.json", "triggers", "id"),
        ]:
            path = ROOT / "dataset" / fname
            data = json.load(open(path))
            items = data.get(container, [])
            storage = getattr(self, container)
            for item in items:
                storage[item[key]] = item
                if container == "triggers":
                    self.trigger_order.append(item[key])
        return self

    def load_expanded(self):
        base = ROOT / "dataset" / "expanded"
        for f in sorted((base / "categories").glob("*.json")):
            data = json.load(open(f))
            self.categories[data.get("slug", f.stem)] = data
        for f in sorted((base / "merchants").glob("*.json")):
            data = json.load(open(f))
            self.merchants[data["merchant_id"]] = data
        for f in sorted((base / "customers").glob("*.json")):
            data = json.load(open(f))
            self.customers[data["customer_id"]] = data
        for f in sorted((base / "triggers").glob("*.json")):
            data = json.load(open(f))
            self.triggers[data["id"]] = data
            self.trigger_order.append(data["id"])
        return self


# =============================================================================
# Minimal HTTP client (same wire schema as challenge-testing-brief.md)
# =============================================================================

class BotClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, timeout: int, body: Optional[dict] = None):
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urlrequest.Request(url, data=data, method=method,
                                  headers={"Content-Type": "application/json"})
        try:
            resp = urlrequest.urlopen(req, timeout=timeout)
            return json.loads(resp.read().decode("utf-8")), None
        except urlerror.HTTPError as e:
            try:
                return json.loads(e.read().decode("utf-8")), None
            except Exception:
                return None, f"HTTP {e.code}"
        except Exception as e:
            return None, str(e)

    def healthz(self):
        return self._request("GET", "/v1/healthz", 5)

    def push_context(self, scope: str, cid: str, version: int, payload: dict):
        return self._request("POST", "/v1/context", 10, {
            "scope": scope, "context_id": cid, "version": version,
            "payload": payload, "delivered_at": DEFAULT_NOW,
        })

    def tick(self, now_iso: str, trigger_ids: List[str]):
        return self._request("POST", "/v1/tick", 30, {
            "now": now_iso, "available_triggers": trigger_ids,
        })


def add_minutes(iso: str, minutes: int) -> str:
    """Advance an ISO8601 UTC timestamp by N minutes, preserving the 'Z' style."""
    from datetime import datetime, timedelta, timezone
    clean = iso.replace("Z", "+00:00")
    dt = datetime.fromisoformat(clean) + timedelta(minutes=minutes)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# =============================================================================
# Lightweight LOCAL proxy scorer
# -----------------------------------------------------------------------
# IMPORTANT: this is NOT the real judge. The real judge_simulator.py uses
# an LLM (LLM_PROVIDER/LLM_API_KEY) to score against the 5-dimension
# rubric; no LLM key is available in this environment. This proxy scorer
# is a deterministic, rubric-inspired heuristic used ONLY to compare
# "before" vs "after" during our own iteration, and to catch obviously
# weak messages (no numbers, no CTA, generic phrasing, taboo words,
# missing why-now). Treat its numbers as directional, not authoritative.
# =============================================================================

TABOO_HINTS = ["guaranteed", "100% safe", "cure", "miracle", "best in", "#1"]
GENERIC_PHRASES = ["quick update", "just checking in", "hope you're doing well", "hi there"]


def proxy_score(action: Dict[str, Any], category: dict, merchant: dict,
                 trigger: dict, customer: Optional[dict]) -> Dict[str, Any]:
    body = action.get("body", "") or ""
    lower = body.lower()
    reasons = []

    # Specificity: numbers, dates, currency, named sources
    num_hits = len(re.findall(r"\d", body))
    has_currency = "₹" in body
    has_pct = "%" in body
    has_named_source = bool(re.search(r"\b(JIDA|DCI|circular|study|report)\b", body, re.I))
    specificity = 3
    if num_hits >= 2:
        specificity += 3
    if has_pct or has_currency:
        specificity += 2
    if has_named_source:
        specificity += 2
    specificity = min(10, specificity)
    if specificity < 6:
        reasons.append("few concrete numbers/dates/citations")

    # Category fit: merchant name / salutation present, no taboo words
    ident = merchant.get("identity", {})
    name_hint = ident.get("owner_first_name") or ident.get("name", "")
    category_fit = 7
    for taboo in TABOO_HINTS:
        if taboo in lower:
            category_fit -= 4
            reasons.append(f"taboo phrase '{taboo}'")
    if category.get("slug") == "dentists" and "dr." not in lower and name_hint:
        category_fit -= 1
    category_fit = max(0, min(10, category_fit))

    # Merchant fit: personalization (name/locality present)
    merchant_fit = 4
    if name_hint and name_hint.split()[0].lower() in lower:
        merchant_fit += 3
    locality = ident.get("locality", "")
    if locality and locality.lower() in lower:
        merchant_fit += 2
    active_offer_titles = [o.get("title", "") for o in merchant.get("offers", []) if o.get("status") == "active"]
    if any(t and t.lower() in lower for t in active_offer_titles):
        merchant_fit += 1
    merchant_fit = min(10, merchant_fit)
    if merchant_fit < 5:
        reasons.append("weak personalization signal")

    # Decision quality: rationale references trigger kind / payload keys
    rationale = (action.get("rationale") or "").lower()
    kind = trigger.get("kind", "")
    decision_quality = 5
    if kind.replace("_", " ") in rationale or any(k in rationale for k in kind.split("_")):
        decision_quality += 3
    if trigger.get("id", "") and action.get("trigger_id") == trigger.get("id"):
        decision_quality += 1
    decision_quality = min(10, decision_quality)

    # Engagement: has a CTA type and body ends with a question / clear ask
    cta = action.get("cta", "none")
    engagement = 4
    if cta and cta != "none":
        engagement += 3
    if body.rstrip().endswith("?"):
        engagement += 2
    if any(p in lower for p in GENERIC_PHRASES):
        engagement -= 2
        reasons.append("generic opener")
    engagement = max(0, min(10, engagement))

    total = specificity + category_fit + merchant_fit + decision_quality + engagement
    return {
        "specificity": specificity, "category_fit": category_fit,
        "merchant_fit": merchant_fit, "decision_quality": decision_quality,
        "engagement": engagement, "total": total, "reasons": reasons,
    }


# =============================================================================
# Main driver
# =============================================================================

def wait_for_healthz(client: BotClient, tries: int = 40) -> bool:
    for _ in range(tries):
        data, err = client.healthz()
        if data and data.get("status") == "ok":
            return True
        time.sleep(0.25)
    return False


def run(now_iso: str, dataset_name: str, port: int, verbose: bool, tick_minutes: int = 5) -> int:
    ds = Dataset()
    if dataset_name == "expanded":
        ds.load_expanded()
    else:
        ds.load_seed()

    env = dict(os.environ)
    env["PORT"] = str(port)
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "bot.py")],
        cwd=str(ROOT), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        client = BotClient(f"http://127.0.0.1:{port}")
        if not wait_for_healthz(client):
            print("FATAL: bot.py did not become healthy in time")
            return 1

        # --- Warmup: push ALL categories, ALL merchants, ALL customers ---
        for slug, cat in ds.categories.items():
            client.push_context("category", slug, 1, cat)
        for mid, m in ds.merchants.items():
            client.push_context("merchant", mid, 1, m)
        for cid, c in ds.customers.items():
            client.push_context("customer", cid, 1, c)
        for tid, t in ds.triggers.items():
            client.push_context("trigger", tid, 1, t)

        data, err = client.healthz()
        loaded = data.get("contexts_loaded", {}) if data else {}
        print(f"Warmup complete. Reference clock (now) = {now_iso}")
        print(f"Contexts loaded: {loaded}  "
              f"(expected: category={len(ds.categories)}, merchant={len(ds.merchants)}, "
              f"customer={len(ds.customers)}, trigger={len(ds.triggers)})")
        print()

        # --- Run batches of 5, exactly like judge_simulator.py's full_evaluation,
        #     but with the correct reference clock and full customer context ---
        tids = ds.trigger_order
        all_actions: List[Dict[str, Any]] = []
        batch_rows = []
        total_score = 0
        dim_totals = {"specificity": 0, "category_fit": 0, "merchant_fit": 0,
                      "decision_quality": 0, "engagement": 0}

        # Per challenge-testing-brief.md §4 Phase 2, the real judge advances
        # simulated time in 5-minute ticks (default) across the test window
        # rather than calling /v1/tick repeatedly at one frozen instant. We
        # mirror that here instead of hammering every batch with the exact
        # same `now` (which would spuriously trip the 24h same-merchant
        # cadence gate the moment any merchant appears in two batches).
        current_now = now_iso
        for i in range(0, len(tids), 5):
            batch = tids[i:i + 5]
            data, err = client.tick(current_now, batch)
            batch_now = current_now
            current_now = add_minutes(current_now, tick_minutes)
            if err:
                print(f"Batch {i // 5 + 1}: TICK FAILED ({err})")
                continue
            actions = data.get("actions", [])
            batch_rows.append((i // 5 + 1, batch, actions, batch_now))
            for action in actions:
                trg = ds.triggers.get(action.get("trigger_id", ""), {})
                merchant = ds.merchants.get(action.get("merchant_id", ""), {})
                customer = ds.customers.get(action.get("customer_id")) if action.get("customer_id") else None
                category = ds.categories.get(merchant.get("category_slug", ""), {})
                score = proxy_score(action, category, merchant, trg, customer)
                total_score += score["total"]
                for k in dim_totals:
                    dim_totals[k] += score[k]
                all_actions.append({**action, "_score": score})

        print("=" * 78)
        print(f"{'Batch':<7}{'now (simulated)':<24}{'Triggers':<32}{'Actions':<10}")
        print("=" * 78)
        for bnum, batch, actions, batch_now in batch_rows:
            fired_ids = {a.get("trigger_id") for a in actions}
            marks = ", ".join(f"{t.split('_')[1]}{'*' if t in fired_ids else ''}" for t in batch)
            print(f"{bnum:<7}{batch_now:<24}{marks:<32}{len(actions):<10}")
        print("=" * 78)
        n_batches_with_actions = sum(1 for _, _, a, _ in batch_rows if a)
        print(f"Batches with >=1 action: {n_batches_with_actions}/{len(batch_rows)}")
        print(f"Total actions fired: {len(all_actions)}/{len(tids)} triggers")
        if all_actions:
            print(f"\nProxy score (local heuristic, NOT the real LLM judge):")
            print(f"  Decision quality : {dim_totals['decision_quality']:>4} / {len(all_actions)*10}")
            print(f"  Specificity      : {dim_totals['specificity']:>4} / {len(all_actions)*10}")
            print(f"  Category fit     : {dim_totals['category_fit']:>4} / {len(all_actions)*10}")
            print(f"  Merchant fit     : {dim_totals['merchant_fit']:>4} / {len(all_actions)*10}")
            print(f"  Engagement       : {dim_totals['engagement']:>4} / {len(all_actions)*10}")
            print(f"  TOTAL            : {total_score:>4} / {len(all_actions)*50}")

        if verbose:
            print("\n" + "=" * 78)
            print("PER-ACTION DETAIL")
            print("=" * 78)
            for a in all_actions:
                s = a["_score"]
                print(f"\n[{a.get('trigger_id')}] -> merchant={a.get('merchant_id')} customer={a.get('customer_id')}")
                print(f"  body: {a.get('body')}")
                print(f"  cta : {a.get('cta')}")
                print(f"  score: total={s['total']}  "
                      f"(dq={s['decision_quality']} sp={s['specificity']} cat={s['category_fit']} "
                      f"mf={s['merchant_fit']} eng={s['engagement']})"
                      + (f"  flags={s['reasons']}" if s["reasons"] else ""))

        # Explain any triggers that produced no action (suppression diagnosis)
        fired = {a.get("trigger_id") for a in all_actions}
        missing = [t for t in tids if t not in fired]
        if missing and verbose:
            print("\n" + "=" * 78)
            print(f"TRIGGERS WITH NO ACTION ({len(missing)}) — see production logic for exact reason")
            print("=" * 78)
            for t in missing:
                trg = ds.triggers.get(t, {})
                print(f"  {t:<45} kind={trg.get('kind',''):<28} expires_at={trg.get('expires_at','')}")

        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--now", default=DEFAULT_NOW, help="Reference clock for /v1/tick (ISO8601)")
    ap.add_argument("--dataset", choices=["seed", "expanded"], default="seed",
                     help="seed = the 25-trigger challenge set; expanded = 100-trigger generalization set")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--tick-minutes", type=int, default=5,
                     help="simulated minutes advanced between batches (matches testing brief default)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print full message bodies + suppression detail")
    args = ap.parse_args()
    sys.exit(run(args.now, args.dataset, args.port, args.verbose, args.tick_minutes))

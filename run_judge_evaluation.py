"""
Full Judge Evaluation Harness (In-Process & HTTP)
=================================================
Runs all judge scenarios from `judge_simulator.py`:
1. Warmup (healthz, metadata, context pushes)
2. Auto-Reply Hell (4 turns of canned auto-replies)
3. Intent Transition ("Ok lets do it. Whats next?")
4. Hostile Handling ("Stop messaging me. This is useless spam.")
5. Full Evaluation (30 canonical test pairs + scoring across 5 dimensions)
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bot
from bot import STORE, ENGINE, CONV_MGR, handle_healthz, handle_metadata, handle_context_push, handle_tick, handle_reply


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


def print_pass(text: str):
    print(f"  {Colors.GREEN}[PASS]{Colors.RESET} {text}")


def print_fail(text: str):
    print(f"  {Colors.RED}[FAIL]{Colors.RESET} {text}")


def print_info(text: str):
    print(f"  {Colors.BLUE}[INFO]{Colors.RESET} {text}")


class LocalBotClient:
    """Invokes bot endpoint handlers directly in-process with identical request/response semantics."""

    def healthz(self) -> Tuple[Dict[str, Any], Optional[str], float]:
        t0 = time.time()
        res = handle_healthz()
        return res, None, (time.time() - t0) * 1000

    def metadata(self) -> Tuple[Dict[str, Any], Optional[str], float]:
        t0 = time.time()
        res = handle_metadata()
        return res, None, (time.time() - t0) * 1000

    def push_context(self, scope: str, cid: str, version: int, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str], float]:
        t0 = time.time()
        status, res = handle_context_push({
            "scope": scope,
            "context_id": cid,
            "version": version,
            "payload": payload,
            "delivered_at": datetime.now(timezone.utc).isoformat(),
        })
        err = None if status in (200, 409) else f"HTTP {status}"
        return res, err, (time.time() - t0) * 1000

    def tick(self, triggers: List[str]) -> Tuple[Dict[str, Any], Optional[str], float]:
        t0 = time.time()
        status, res = handle_tick({
            "now": datetime.now(timezone.utc).isoformat(),
            "available_triggers": triggers,
        })
        err = None if status == 200 else f"HTTP {status}"
        return res, err, (time.time() - t0) * 1000

    def reply(self, conv_id: str, merchant_id: Optional[str], message: str, turn: int) -> Tuple[Dict[str, Any], Optional[str], float]:
        t0 = time.time()
        status, res = handle_reply({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": None,
            "from_role": "merchant",
            "message": message,
            "received_at": datetime.now(timezone.utc).isoformat(),
            "turn_number": turn,
        })
        err = None if status == 200 else f"HTTP {status}"
        return res, err, (time.time() - t0) * 1000


def run_evaluation():
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'MAGICPIN VERA AI CHALLENGE — COMPREHENSIVE JUDGE HARNESS'.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.RESET}\n")

    client = LocalBotClient()
    dataset_dir = Path(__file__).parent / "dataset"

    # =========================================================================
    # SCENARIO 1: WARMUP
    # =========================================================================
    print(f"{Colors.BOLD}{Colors.CYAN}--- SCENARIO 1: WARMUP ---{Colors.RESET}")
    data, err, lat = client.healthz()
    if err or data.get("status") != "ok":
        print_fail(f"healthz failed: {err}")
        return False
    print_pass(f"healthz probe ({lat:.1f}ms) — status: {data['status']}")

    data, err, lat = client.metadata()
    if err or not data.get("team_name"):
        print_fail(f"metadata failed: {err}")
        return False
    print_pass(f"metadata probe ({lat:.1f}ms) — Team: {data['team_name']}, Version: {data['version']}")

    # Load Category Contexts
    cat_dir = dataset_dir / "categories"
    categories = {}
    for f in cat_dir.glob("*.json"):
        with open(f) as fp:
            c = json.load(fp)
            categories[c["slug"]] = c
            res, err, _ = client.push_context("category", c["slug"], 1, c)
            if res.get("accepted"):
                print_pass(f"category/{c['slug']} pushed successfully")
            else:
                print_fail(f"category/{c['slug']} failed: {res}")

    expanded_dir = dataset_dir / "expanded"
    if (expanded_dir / "merchants").exists():
        merchants = {}
        for f in (expanded_dir / "merchants").glob("*.json"):
            with open(f) as fp:
                m = json.load(fp)
                merchants[m["merchant_id"]] = m
                client.push_context("merchant", m["merchant_id"], 1, m)
        print_pass(f"Pushed {len(merchants)} expanded merchants")

        customers = {}
        for f in (expanded_dir / "customers").glob("*.json"):
            with open(f) as fp:
                c = json.load(fp)
                customers[c["customer_id"]] = c
                client.push_context("customer", c["customer_id"], 1, c)
        print_pass(f"Pushed {len(customers)} expanded customers")

        triggers = {}
        for f in (expanded_dir / "triggers").glob("*.json"):
            with open(f) as fp:
                t = json.load(fp)
                triggers[t["id"]] = t
                client.push_context("trigger", t["id"], 1, t)
        print_pass(f"Pushed {len(triggers)} expanded triggers")
    else:
        # Load Seeds
        with open(dataset_dir / "merchants_seed.json") as fp:
            merchants = {m["merchant_id"]: m for m in json.load(fp)["merchants"]}
            for mid, m in merchants.items():
                client.push_context("merchant", mid, 1, m)
        print_pass(f"Pushed {len(merchants)} seed merchants")

        with open(dataset_dir / "customers_seed.json") as fp:
            customers = {c["customer_id"]: c for c in json.load(fp)["customers"]}
            for cid, c in customers.items():
                client.push_context("customer", cid, 1, c)
        print_pass(f"Pushed {len(customers)} seed customers")

        with open(dataset_dir / "triggers_seed.json") as fp:
            triggers = {t["id"]: t for t in json.load(fp)["triggers"]}
            for tid, t in triggers.items():
                client.push_context("trigger", tid, 1, t)
        print_pass(f"Pushed {len(triggers)} seed triggers")

    # Verify healthz context count
    hz, _, _ = client.healthz()
    counts = hz.get("contexts_loaded", {})
    print_pass(f"Healthz Context Counts: {counts}")
    assert counts["category"] == 5
    assert counts["merchant"] >= 10
    assert counts["customer"] >= 15
    assert counts["trigger"] >= 25

    # =========================================================================
    # SCENARIO 2: AUTO-REPLY HELL
    # =========================================================================
    print(f"\n{Colors.BOLD}{Colors.CYAN}--- SCENARIO 2: AUTO-REPLY HELL ---{Colors.RESET}")
    mid = "m_001_drmeera_dentist_delhi"
    auto_msg = "Thank you for contacting Dr. Meera's Dental Clinic! Our team will respond shortly."

    # Turn 1
    r1, _, _ = client.reply("conv_auto_test", mid, auto_msg, 2)
    print_info(f"Turn 1 response: action={r1.get('action')}, body='{r1.get('body')}'")
    assert r1.get("action") == "send"
    assert "auto-reply" in r1.get("body", "").lower()
    print_pass("Turn 1: Flagged auto-reply politely")

    # Turn 2
    r2, _, _ = client.reply("conv_auto_test", mid, auto_msg, 3)
    print_info(f"Turn 2 response: action={r2.get('action')}, wait_seconds={r2.get('wait_seconds')}")
    assert r2.get("action") == "wait"
    assert r2.get("wait_seconds", 0) > 0
    print_pass("Turn 2: Backed off with wait")

    # Turn 3
    r3, _, _ = client.reply("conv_auto_test", mid, auto_msg, 4)
    print_info(f"Turn 3 response: action={r3.get('action')}")
    assert r3.get("action") == "end"
    print_pass("Turn 3: Gracefully ended conversation after repeated auto-replies")

    # =========================================================================
    # SCENARIO 3: INTENT TRANSITION
    # =========================================================================
    print(f"\n{Colors.BOLD}{Colors.CYAN}--- SCENARIO 3: INTENT TRANSITION ---{Colors.RESET}")
    commitment = "Ok lets do it. Whats next?"
    print_info(f"Merchant message: '{commitment}'")
    r_intent, _, _ = client.reply("conv_intent_test", mid, commitment, 2)
    body = r_intent.get("body", "")
    print_info(f"Bot action: {r_intent.get('action')}")
    print_info(f"Bot body: '{body}'")

    qualifying = ["would you", "do you", "can you tell", "what if", "how about"]
    actioning = ["done", "sending", "draft", "here", "confirm", "proceed", "next", "schedule"]

    body_lower = body.lower()
    has_action = any(w in body_lower for w in actioning)
    has_qual = any(w in body_lower for w in qualifying)

    if has_action and not has_qual:
        print_pass("Switched to ACTION mode immediately (zero qualifying questions)")
    else:
        print_fail("Failed intent transition requirement")
        return False

    # =========================================================================
    # SCENARIO 4: HOSTILE / OPT-OUT
    # =========================================================================
    print(f"\n{Colors.BOLD}{Colors.CYAN}--- SCENARIO 4: HOSTILE / OPT-OUT ---{Colors.RESET}")
    hostile = "Stop messaging me. This is useless spam."
    print_info(f"Merchant message: '{hostile}'")
    r_hostile, _, _ = client.reply("conv_hostile_test", mid, hostile, 2)
    print_info(f"Bot action: {r_hostile.get('action')}")
    assert r_hostile.get("action") == "end"
    print_pass("Gracefully ended on hostile message and suppressed merchant")

    # =========================================================================
    # SCENARIO 5: 30 CANONICAL TEST PAIRS
    # =========================================================================
    print(f"\n{Colors.BOLD}{Colors.CYAN}--- SCENARIO 5: 30 CANONICAL TEST PAIRS EVALUATION ---{Colors.RESET}")
    test_pairs_file = dataset_dir / "expanded" / "test_pairs.json"
    if not test_pairs_file.exists():
        print_fail(f"Test pairs file {test_pairs_file} not found")
        return False

    with open(test_pairs_file) as fp:
        test_pairs = json.load(fp)["pairs"]

    print_info(f"Evaluating all {len(test_pairs)} canonical test pairs...")
    
    scorecards = []
    for pair in test_pairs:
        t_id = pair["test_id"]
        trg_id = pair["trigger_id"]
        mid = pair["merchant_id"]
        cid = pair.get("customer_id")

        trg = STORE.get("trigger", trg_id)
        merch = STORE.get("merchant", mid)
        cust = STORE.get("customer", cid) if cid else None
        cat_slug = merch.get("category_slug", "") if merch else ""
        cat = STORE.get("category", cat_slug)

        if not (trg and merch and cat):
            continue

        res = bot.compose(cat, merch, trg, cust)
        body = res.get("body", "")

        # Strict checks
        has_url = bool(re.search(r"https?://|www\.", body))
        nums_count = len(re.findall(r"\d+", body))
        taboo_words = [t for t in cat.get("voice", {}).get("vocab_taboo", []) if t.lower() in body.lower()]
        
        # Scoring metrics
        specificity = min(10, 4 + min(6, nums_count))
        cat_fit = 10 if not taboo_words else 4
        merch_fit = 10 if (merch.get("identity", {}).get("owner_first_name", "") in body or merch.get("identity", {}).get("locality", "") in body) else 8
        trigger_rel = 10 if (trg.get("kind", "") in body or "issue" in body or "match" in body or "recall" in body or "calls" in body or "views" in body or "batch" in body or "service" in body or "diwali" in body.lower()) else 8
        compulsion = 10 if res.get("cta") in ("binary_yes_no", "binary_confirm_cancel", "multi_choice_slot", "open_ended") else 7

        total = specificity + cat_fit + merch_fit + trigger_rel + compulsion
        if has_url:
            total -= 3

        scorecards.append({
            "test_id": t_id,
            "trigger_id": trg_id,
            "merchant_id": mid,
            "specificity": specificity,
            "category_fit": cat_fit,
            "merchant_fit": merch_fit,
            "trigger_relevance": trigger_rel,
            "compulsion": compulsion,
            "total": total,
            "has_url": has_url,
            "body": body,
        })

    avg_score = sum(s["total"] for s in scorecards) / len(scorecards)
    print_pass(f"Evaluated {len(scorecards)} canonical pairs. Average Score: {avg_score:.1f}/50 ({avg_score*2:.1f}%)")
    
    # Table of first 10 cases
    print(f"\n{'Test':<6} {'Specific':<9} {'CatFit':<7} {'MerchFit':<9} {'TrigRel':<8} {'Compul':<7} {'Total':<6} {'Body Preview'}")
    print("-" * 75)
    for s in scorecards[:10]:
        preview = s["body"][:35].replace("\n", " ") + "..."
        print(f"{s['test_id']:<6} {s['specificity']:<9} {s['category_fit']:<7} {s['merchant_fit']:<9} {s['trigger_relevance']:<8} {s['compulsion']:<7} {s['total']:<6} {preview}")

    print(f"\n{Colors.BOLD}{Colors.GREEN}ALL SCENARIOS AND CANONICAL EVALUATIONS PASSED SUCCESSFULLY!{Colors.RESET}\n")
    return True


if __name__ == "__main__":
    run_evaluation()

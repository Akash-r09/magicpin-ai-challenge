"""
Comprehensive Unit & Integration Test Suite for Magicpin Vera Solution
=======================================================================
Tests all critical components:
1. Context Store (idempotency, atomic versioning, stale rejection, counts)
2. Evidence Extractor (grounding, active vs expired offers, digest resolution)
3. Decision Engine (tiered arbitration, hard suppression, consent verification)
4. Composer & Validator (zero URLs, zero taboos, zero jargon, grounded facts)
5. State Machine (auto-reply progressive backoff, intent transitions, hostility)
6. Standalone compose() function
"""

from __future__ import annotations

import json
import re
import unittest
from datetime import datetime, timezone
from pathlib import Path

from context_store import ContextStore
from decision_engine import DecisionEngine
from evidence_extractor import extract_evidence
from message_strategy import build_message_strategy
from composer import compose_grounded_message
from validator import validate_and_repair
from state_machine import ConversationManager
from bot import compose


class TestVeraSolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset_dir = Path(__file__).parent / "dataset"
        with open(dataset_dir / "categories" / "dentists.json") as f:
            cls.dentists = json.load(f)
        with open(dataset_dir / "categories" / "gyms.json") as f:
            cls.gyms = json.load(f)
        with open(dataset_dir / "categories" / "pharmacies.json") as f:
            cls.pharmacies = json.load(f)
        with open(dataset_dir / "categories" / "restaurants.json") as f:
            cls.restaurants = json.load(f)
        with open(dataset_dir / "categories" / "salons.json") as f:
            cls.salons = json.load(f)

        with open(dataset_dir / "merchants_seed.json") as f:
            m_data = json.load(f)
            cls.merchants = {m["merchant_id"]: m for m in m_data["merchants"]}

        with open(dataset_dir / "customers_seed.json") as f:
            c_data = json.load(f)
            cls.customers = {c["customer_id"]: c for c in c_data["customers"]}

        with open(dataset_dir / "triggers_seed.json") as f:
            t_data = json.load(f)
            cls.triggers = {t["id"]: t for t in t_data["triggers"]}

    # =========================================================================
    # 1. CONTEXT STORE TESTS
    # =========================================================================
    def test_context_store_versioning_and_idempotency(self):
        store = ContextStore()
        
        # Initial push
        acc, status, ver = store.push("category", "dentists", 1, {"slug": "dentists"})
        self.assertTrue(acc)
        self.assertEqual(status, "accepted")
        self.assertEqual(ver, 1)

        # Idempotent re-push
        acc, status, ver = store.push("category", "dentists", 1, {"slug": "dentists"})
        self.assertTrue(acc)
        self.assertEqual(status, "idempotent")

        # Stale push (lower version)
        acc, status, cur_ver = store.push("category", "dentists", 0, {"slug": "old"})
        self.assertFalse(acc)
        self.assertEqual(status, "stale_version")
        self.assertEqual(cur_ver, 1)

        # Newer push (higher version replaces)
        acc, status, ver = store.push("category", "dentists", 2, {"slug": "dentists", "updated": True})
        self.assertTrue(acc)
        self.assertEqual(status, "accepted")
        self.assertEqual(ver, 2)
        self.assertTrue(store.get("category", "dentists")["updated"])

        # Healthz counts
        store.push("merchant", "m_001", 1, {"name": "M1"})
        store.push("customer", "c_001", 1, {"name": "C1"})
        store.push("trigger", "trg_001", 1, {"kind": "test"})
        counts = store.counts()
        self.assertEqual(counts["category"], 1)
        self.assertEqual(counts["merchant"], 1)
        self.assertEqual(counts["customer"], 1)
        self.assertEqual(counts["trigger"], 1)

    # =========================================================================
    # 2. EVIDENCE EXTRACTOR TESTS
    # =========================================================================
    def test_evidence_extractor_active_vs_expired_offers(self):
        m001 = self.merchants["m_001_drmeera_dentist_delhi"]
        trg001 = self.triggers["trg_001_research_digest_dentists"]
        facts = extract_evidence(self.dentists, m001, trg001)

        self.assertEqual(facts.salutation_name, "Dr. Meera")
        self.assertIn("Dental Cleaning @ ₹299", facts.active_offer_titles)
        self.assertIn("Deep Cleaning @ ₹499", facts.expired_offer_titles)
        self.assertNotIn("Deep Cleaning @ ₹499", facts.active_offer_titles)
        self.assertEqual(facts.high_risk_adult_count, 124)
        self.assertIsNotNone(facts.digest_title)
        self.assertEqual(facts.digest_source, "JIDA Oct 2026, p.14")
        self.assertEqual(facts.digest_trial_n, 2100)

    # =========================================================================
    # 3. DECISION ENGINE & TIERED ARBITRATION TESTS
    # =========================================================================
    def test_decision_engine_hard_suppression_and_consent(self):
        store = ContextStore()
        engine = DecisionEngine(store)
        now_iso = "2026-04-26T10:00:00Z"

        # Populate context store
        store.push("category", "dentists", 1, self.dentists)
        store.push("merchant", "m_001_drmeera_dentist_delhi", 1, self.merchants["m_001_drmeera_dentist_delhi"])
        
        # Test 1: Recall due with valid consent
        c001 = dict(self.customers["c_001_priya_for_m001"])
        store.push("customer", "c_001_priya_for_m001", 1, c001)
        store.push("trigger", "trg_003_recall_due_priya", 1, self.triggers["trg_003_recall_due_priya"])

        actions = engine.evaluate_tick(now_iso, ["trg_003_recall_due_priya"])
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["customer_id"], "c_001_priya_for_m001")
        self.assertEqual(actions[0]["send_as"], "merchant_on_behalf")

        # Test 2: Customer consent revoked -> Must suppress
        c001_no_consent = dict(c001)
        c001_no_consent["preferences"] = {"reminder_opt_in": False}
        store.push("customer", "c_001_priya_for_m001", 2, c001_no_consent)
        
        # Clear suppression cache for this test
        engine._suppression_cache.clear()
        engine._merchant_last_touch.clear()
        actions = engine.evaluate_tick(now_iso, ["trg_003_recall_due_priya"])
        self.assertEqual(len(actions), 0)  # Restraint rewarded!

    def test_tiered_arbitration_safety_overrides_cadence(self):
        store = ContextStore()
        engine = DecisionEngine(store)
        now_dt = datetime(2026, 4, 26, 10, 0, 0, tzinfo=timezone.utc)
        now_iso = now_dt.isoformat()

        store.push("category", "pharmacies", 1, self.pharmacies)
        m009 = self.merchants["m_009_apollo_pharmacy_jaipur"]
        store.push("merchant", "m_009_apollo_pharmacy_jaipur", 1, m009)
        
        # Mark merchant as recently touched (1 hour ago)
        engine._merchant_last_touch["m_009_apollo_pharmacy_jaipur"] = datetime(2026, 4, 26, 9, 0, 0, tzinfo=timezone.utc)

        # Normal curiosity ask (Tier 4) -> Should be suppressed due to 24h cadence
        trg_normal = {
            "id": "trg_normal",
            "scope": "merchant",
            "kind": "curious_ask_due",
            "merchant_id": "m_009_apollo_pharmacy_jaipur",
            "urgency": 1,
            "suppression_key": "supp_normal",
            "expires_at": "2026-05-01T00:00:00Z",
            "payload": {},
        }
        store.push("trigger", "trg_normal", 1, trg_normal)
        actions = engine.evaluate_tick(now_iso, ["trg_normal"])
        self.assertEqual(len(actions), 0)

        # Tier 1 Critical Supply Alert -> CAN OVERRIDE cadence!
        # Dynamic grounding test: custom affected batches list
        trg_critical = dict(self.triggers["trg_018_supply_atorvastatin_recall"])
        trg_critical["payload"] = dict(trg_critical["payload"])
        trg_critical["payload"]["affected_batches"] = ["AT-BATCH-001", "AT-BATCH-002", "AT-BATCH-003"]
        store.push("trigger", "trg_018_supply_atorvastatin_recall", 1, trg_critical)
        actions = engine.evaluate_tick(now_iso, ["trg_018_supply_atorvastatin_recall"])
        self.assertEqual(len(actions), 1)
        self.assertIn("recall on 3 atorvastatin batches (AT-BATCH-001, AT-BATCH-002, AT-BATCH-003)", actions[0]["body"])

        # Verify changing the input changes the output
        trg_critical["payload"]["affected_batches"] = ["AT-SINGLE-999"]
        store.push("trigger", "trg_018_supply_atorvastatin_recall", 2, trg_critical)
        engine._suppression_cache.clear()
        actions2 = engine.evaluate_tick(now_iso, ["trg_018_supply_atorvastatin_recall"])
        self.assertEqual(len(actions2), 1)
        self.assertIn("recall on 1 atorvastatin batches (AT-SINGLE-999)", actions2[0]["body"])
        self.assertNotIn("AT-BATCH-001", actions2[0]["body"])

    # =========================================================================
    # 4. COMPOSER & VALIDATOR SAFETY TESTS
    # =========================================================================
    def test_validator_safe_cleanup(self):
        m001 = self.merchants["m_001_drmeera_dentist_delhi"]
        trg001 = self.triggers["trg_001_research_digest_dentists"]
        facts = extract_evidence(self.dentists, m001, trg001)
        strategy = build_message_strategy(facts)

        dirty_body = (
            "Dr. Meera, your update is ready. Read more at https://magicpin.com/blog. "
            "Our internal signals show your payload is active."
        )
        res = validate_and_repair(dirty_body, facts, strategy)
        self.assertTrue(res.is_valid)
        self.assertNotIn("https://", res.body)
        self.assertNotIn("payload", res.body)

    # =========================================================================
    # 5. STATE MACHINE & AUTO-REPLY TESTS
    # =========================================================================
    def test_state_machine_auto_reply_backoff(self):
        mgr = ConversationManager()
        conv_id = "test_conv_ar"
        mid = "m_001"
        canned_msg = "Thank you for contacting Dr. Meera's Dental Clinic! Our team will respond shortly."

        # Turn 1: Bot sends friendly notice
        r1 = mgr.handle_inbound_reply(conv_id, mid, canned_msg, turn_number=2)
        self.assertEqual(r1["action"], "send")
        self.assertIn("Looks like an auto-reply", r1["body"])

        # Turn 2: Bot backs off and waits
        r2 = mgr.handle_inbound_reply(conv_id, mid, canned_msg, turn_number=3)
        self.assertEqual(r2["action"], "wait")
        self.assertEqual(r2["wait_seconds"], 14400)

        # Turn 3: Bot ends conversation
        r3 = mgr.handle_inbound_reply(conv_id, mid, canned_msg, turn_number=4)
        self.assertEqual(r3["action"], "end")

    def test_state_machine_intent_transition(self):
        mgr = ConversationManager()
        conv_id = "test_conv_intent"
        mid = "m_001"
        commitment_msg = "Ok let's do it. What's next?"

        resp = mgr.handle_inbound_reply(conv_id, mid, commitment_msg, turn_number=2)
        self.assertEqual(resp["action"], "send")
        body_lower = resp["body"].lower()
        
        # Check action verbs present
        action_keywords = ["done", "sending", "draft", "confirm", "schedule"]
        self.assertTrue(any(w in body_lower for w in action_keywords))
        
        # Check qualifying phrases absent
        qualifying = ["would you", "do you", "can you tell", "what if", "how about"]
        self.assertFalse(any(w in body_lower for w in qualifying))

    def test_state_machine_hostile_opt_out(self):
        mgr = ConversationManager()
        conv_id = "test_conv_hostile"
        mid = "m_001"
        hostile_msg = "Stop messaging me. This is useless spam."

        resp = mgr.handle_inbound_reply(conv_id, mid, hostile_msg, turn_number=2)
        self.assertEqual(resp["action"], "end")

    # =========================================================================
    # 6. STANDALONE COMPOSE FUNCTION TEST
    # =========================================================================
    def test_standalone_compose(self):
        m001 = self.merchants["m_001_drmeera_dentist_delhi"]
        trg001 = self.triggers["trg_001_research_digest_dentists"]
        result = compose(self.dentists, m001, trg001)

        self.assertIn("body", result)
        self.assertIn("cta", result)
        self.assertIn("send_as", result)
        self.assertIn("suppression_key", result)
        self.assertIn("rationale", result)
        self.assertEqual(result["send_as"], "vera")
        self.assertIn("2,100-patient", result["body"])
        self.assertIn("JIDA Oct 2026", result["body"])
        self.assertNotIn("http", result["body"])

    # =========================================================================
    # 7. STRICT GROUNDING & UNSUPPORTED FACT TESTS
    # =========================================================================
    def test_grounding_unsupported_facts(self):
        print("\n--- GROUNDING & UNSUPPORTED FACTS TEST SUITE ---")
        
        # 1. Missing offer -> no price/discount hallucinated
        m_no_offers = dict(self.merchants["m_001_drmeera_dentist_delhi"])
        m_no_offers["offers"] = []
        trg_recall_no_slots = {
            "id": "trg_test_no_slots",
            "scope": "customer",
            "kind": "recall_due",
            "merchant_id": m_no_offers["merchant_id"],
            "customer_id": "c_001_priya_for_m001",
            "payload": {},
            "urgency": 2,
            "suppression_key": "recall:c001:test",
            "expires_at": "2026-05-30T00:00:00Z"
        }
        res1 = compose(self.dentists, m_no_offers, trg_recall_no_slots, self.customers["c_001_priya_for_m001"])
        body1 = res1["body"]
        has_price_or_pct = bool(re.search(r"₹|\b\d+%\b|\b\d+ percent\b", body1))
        p1 = not has_price_or_pct
        print(f"[1] Missing offer -> no price/discount hallucinated\n  INPUT: active_offers=[]\n  MESSAGE: {body1}\n  RESULT: {'PASS' if p1 else 'FAIL'}\n")
        self.assertTrue(p1)

        # 2. Expired offer -> not included in message
        m_exp_offers = dict(self.merchants["m_001_drmeera_dentist_delhi"])
        # Deep Cleaning @ ₹499 is expired
        res2 = compose(self.dentists, m_exp_offers, trg_recall_no_slots, self.customers["c_001_priya_for_m001"])
        body2 = res2["body"]
        has_exp = "Deep Cleaning" in body2 or "₹499" in body2
        p2 = not has_exp
        print(f"[2] Expired offer -> not included in message\n  INPUT: expired_offers=['Deep Cleaning @ ₹499']\n  MESSAGE: {body2}\n  RESULT: {'PASS' if p2 else 'FAIL'}\n")
        self.assertTrue(p2)

        # 3. Missing slots -> no fake dates/times hallucinated
        has_fake_slot = any(slot in body1 for slot in ["Wed 5 Nov", "Thu 6 Nov", "6pm", "5pm", "4pm", "8am"])
        p3 = not has_fake_slot
        print(f"[3] Missing slots -> no fake dates/times hallucinated\n  INPUT: available_slots=[]\n  MESSAGE: {body1}\n  RESULT: {'PASS' if p3 else 'FAIL'}\n")
        self.assertTrue(p3)

        # 4. Missing statistics -> no fake numbers/percentages hallucinated in perf_dip
        trg_perf_no_delta = {
            "id": "trg_perf_no_delta",
            "scope": "merchant",
            "kind": "perf_dip",
            "merchant_id": m_no_offers["merchant_id"],
            "payload": {"metric": "calls"},
            "urgency": 2,
            "suppression_key": "perf:dip:test",
            "expires_at": "2026-05-30T00:00:00Z"
        }
        res4 = compose(self.dentists, m_no_offers, trg_perf_no_delta)
        body4 = res4["body"]
        has_fake_pct = bool(re.search(r"\b40%\b|\b30%\b|\b25%\b|\b-12%\b|\b18%\b", body4))
        p4 = not has_fake_pct
        print(f"[4] Missing statistics -> no fake numbers/percentages hallucinated\n  INPUT: delta_pct=None\n  MESSAGE: {body4}\n  RESULT: {'PASS' if p4 else 'FAIL'}\n")
        self.assertTrue(p4)

        # 5. Unsupported percentage / claim -> validator flags failure without substitution
        dirty_msg = "We guarantee a 100% safe miracle transformation in 7 days."
        facts5 = extract_evidence(self.dentists, m_no_offers, trg_perf_no_delta)
        strat5 = build_message_strategy(facts5)
        val5 = validate_and_repair(dirty_msg, facts5, strat5)
        self.assertFalse(val5.is_valid)
        self.assertTrue(any("unsupported_claim_detected" in issue for issue in val5.issues))
        # Ensure unsupported claim was not silently converted into another factual claim (e.g. 'proven', 'tested')
        self.assertNotIn("proven", val5.body)
        self.assertNotIn("tested", val5.body)
        self.assertNotIn("clinically verified", val5.body)
        print(f"[5] Unsupported absolute claim -> validation failure\n  INPUT: '{dirty_msg}'\n  IS_VALID: {val5.is_valid}\n  ISSUES: {val5.issues}\n  RESULT: PASS\n")

        # 6. Customer without consent -> rejected before composition
        c_no_consent = dict(self.customers["c_001_priya_for_m001"])
        c_no_consent["preferences"] = {"reminder_opt_in": False}
        store = ContextStore()
        store.push("category", "dentists", 1, self.dentists)
        store.push("merchant", m_no_offers["merchant_id"], 1, m_no_offers)
        store.push("customer", c_no_consent["customer_id"], 1, c_no_consent)
        store.push("trigger", trg_recall_no_slots["id"], 1, trg_recall_no_slots)
        engine = DecisionEngine(store)
        actions = engine.evaluate_tick("2026-04-26T10:00:00Z", [trg_recall_no_slots["id"]])
        p6 = len(actions) == 0
        print(f"[6] Customer without consent -> suppressed before composition\n  INPUT: reminder_opt_in=False\n  ACTIONS: {len(actions)}\n  RESULT: {'PASS' if p6 else 'FAIL'}\n")
        self.assertTrue(p6)

        # 7. Valid context with exact numbers -> exact numbers preserved
        trg_perf_exact = {
            "id": "trg_perf_exact",
            "scope": "merchant",
            "kind": "perf_dip",
            "merchant_id": m_no_offers["merchant_id"],
            "payload": {"metric": "calls", "delta_pct": -0.35},
            "urgency": 2,
            "suppression_key": "perf:dip:exact",
            "expires_at": "2026-05-30T00:00:00Z"
        }
        res7 = compose(self.dentists, m_no_offers, trg_perf_exact)
        body7 = res7["body"]
        p7 = "35%" in body7
        print(f"[7] Valid context with exact numbers -> exact numbers preserved\n  INPUT: delta_pct=-0.35\n  MESSAGE: {body7}\n  RESULT: {'PASS' if p7 else 'FAIL'}\n")
        self.assertTrue(p7)

        # 8. Valid active offer -> exact offer used
        m_active = self.merchants["m_001_drmeera_dentist_delhi"]
        res8 = compose(self.dentists, m_active, trg_perf_exact)
        body8 = res8["body"]
        p8 = "Dental Cleaning @ ₹299" in body8
        print(f"[8] Valid active offer -> exact offer used\n  INPUT: active_offers=['Dental Cleaning @ ₹299']\n  MESSAGE: {body8}\n  RESULT: {'PASS' if p8 else 'FAIL'}\n")
        self.assertTrue(p8)

        # 9. Valid slots -> exact slots used dynamically
        trg_custom_slots = {
            "id": "trg_custom_slots",
            "scope": "customer",
            "kind": "recall_due",
            "merchant_id": m_active["merchant_id"],
            "customer_id": "c_001_priya_for_m001",
            "payload": {
                "available_slots": [
                    {"iso": "2026-05-18T10:00:00", "label": "Mon 18 May, 10am"},
                    {"iso": "2026-05-19T16:00:00", "label": "Tue 19 May, 4pm"}
                ]
            },
            "urgency": 2,
            "suppression_key": "recall:custom:slots",
            "expires_at": "2026-05-30T00:00:00Z"
        }
        res9 = compose(self.dentists, m_active, trg_custom_slots, self.customers["c_001_priya_for_m001"])
        body9 = res9["body"]
        p9 = "Mon 18 May, 10am" in body9 and "Tue 19 May, 4pm" in body9
        print(f"[9] Valid slots -> exact slots used dynamically\n  INPUT: slots=['Mon 18 May, 10am', 'Tue 19 May, 4pm']\n  MESSAGE: {body9}\n  RESULT: {'PASS' if p9 else 'FAIL'}\n")
        self.assertTrue(p9)

        # 10. Missing appointment time -> no invented time in appointment_tomorrow
        trg_appt_no_time = {
            "id": "trg_appt_no_time",
            "scope": "customer",
            "kind": "appointment_tomorrow",
            "merchant_id": m_active["merchant_id"],
            "customer_id": "c_001_priya_for_m001",
            "payload": {},
            "urgency": 2,
            "suppression_key": "appt:no_time",
            "expires_at": "2026-05-30T00:00:00Z"
        }
        res10 = compose(self.dentists, m_active, trg_appt_no_time, self.customers["c_001_priya_for_m001"])
        body10 = res10["body"]
        has_invented_time = bool(re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b|\b4:00pm\b|\b4pm\b", body10, re.IGNORECASE))
        p10 = not has_invented_time
        print(f"[10] Missing appointment time -> no invented time\n  INPUT: appointment_time=None\n  MESSAGE: {body10}\n  RESULT: {'PASS' if p10 else 'FAIL'}\n")
        self.assertTrue(p10)

        # 11. Invalid LLM output -> rejected and replaced by deterministic fallback
        import os
        import composer
        orig_call_llm = composer._call_llm_composer
        try:
            composer._call_llm_composer = lambda f, s, p, k: "Dr. Meera, we guarantee 100% safe miracle results! https://badurl.com"
            os.environ["LLM_API_KEY"] = "mock_key"
            os.environ["LLM_PROVIDER"] = "openai"
            
            res11 = compose_grounded_message(facts5, strat5)
            self.assertNotIn("guarantee", res11.body)
            self.assertNotIn("100% safe", res11.body)
            self.assertNotIn("https://", res11.body)
            self.assertIn("Dr. Meera", res11.body)
            print(f"[11] Invalid LLM output -> rejected & deterministic fallback used\n  RESULT: PASS\n")
        finally:
            composer._call_llm_composer = orig_call_llm
            os.environ.pop("LLM_API_KEY", None)
            os.environ.pop("LLM_PROVIDER", None)


if __name__ == "__main__":
    unittest.main()

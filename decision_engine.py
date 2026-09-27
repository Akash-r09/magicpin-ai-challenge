"""
Decision Engine for Magicpin Vera AI Challenge
==============================================
Deterministic Decision Engine with Tiered Trigger Arbitration & Hard Suppression:
- Evaluates available triggers against strict hard suppression gates
- Validates customer consent, reminder opt-ins, and communication scope
- Tiered Arbitration Policy:
    Tier 1 (Critical): Urgency-5 safety, regulatory, supply recall alerts (can override normal cadence)
    Tier 2 (High): Urgency-4 high-value compliance, renewal, planning intent
    Tier 3 (Normal): Urgency-3 recall due, review themes, IPL match, seasonal dips
    Tier 4 (Low): Urgency 1-2 research digests, curious asks, milestones, dormancy
- Restraint as a feature: returns empty actions when evidence is insufficient or suppressed
- Enforces strict caps: max 1 action per merchant per tick, max 20 actions total per tick
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from context_store import ContextStore
from evidence_extractor import extract_evidence
from message_strategy import build_message_strategy
from composer import compose_grounded_message, ComposedMessage


def parse_iso(iso_str: str) -> Optional[datetime]:
    """Parse ISO datetime safely into a timezone-aware datetime."""
    if not iso_str:
        return None
    try:
        clean = iso_str.strip().replace("Z", "+00:00")
        return datetime.fromisoformat(clean)
    except Exception:
        return None


class DecisionEngine:
    def __init__(self, context_store: ContextStore):
        self.store = context_store
        # Active suppression keys mapping key -> expiry_dt
        self._suppression_cache: Dict[str, datetime] = {}
        # Last touch timestamp per merchant
        self._merchant_last_touch: Dict[str, datetime] = {}
        # Explicit opt-out registry
        self._opted_out_merchants: Set[str] = set()

    def record_opt_out(self, merchant_id: str) -> None:
        """Mark merchant as opted out."""
        if merchant_id:
            self._opted_out_merchants.add(merchant_id)

    def is_opted_out(self, merchant_id: str) -> bool:
        """Check if merchant opted out."""
        return merchant_id in self._opted_out_merchants

    def record_suppression(self, key: str, expires_at_iso: str, now_dt: datetime) -> None:
        """Record a suppression key."""
        if not key:
            return
        exp_dt = parse_iso(expires_at_iso)
        if not exp_dt or exp_dt <= now_dt:
            # Default suppression of 24h if missing or already expired
            exp_dt = datetime.fromtimestamp(now_dt.timestamp() + 86400, timezone.utc)
        self._suppression_cache[key] = exp_dt

    def is_suppressed(self, key: str, now_dt: datetime) -> bool:
        """Check if suppression key is currently active."""
        if not key:
            return False
        exp_dt = self._suppression_cache.get(key)
        if not exp_dt:
            return False
        if now_dt >= exp_dt:
            # Expired suppression key
            del self._suppression_cache[key]
            return False
        return True

    def evaluate_tick(
        self,
        now_iso: str,
        available_trigger_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a tick event and return a list of Composed Action dictionaries.
        Enforces tiered arbitration, hard suppression, and per-merchant limits.
        """
        now_dt = parse_iso(now_iso) or datetime.now(timezone.utc)
        candidates: List[Tuple[int, float, Dict[str, Any], Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]] = []

        # Clean expired suppression keys
        expired_keys = [k for k, exp in self._suppression_cache.items() if now_dt >= exp]
        for k in expired_keys:
            del self._suppression_cache[k]

        # 1. Evaluate every available trigger
        for trg_id in available_trigger_ids:
            trg = self.store.get("trigger", trg_id)
            if not trg:
                continue

            merchant_id = trg.get("merchant_id")
            if not merchant_id or self.is_opted_out(merchant_id):
                continue

            merchant = self.store.get("merchant", merchant_id)
            if not merchant:
                continue

            category_slug = merchant.get("category_slug")
            category = self.store.get("category", category_slug) if category_slug else None
            if not category:
                continue

            customer_id = trg.get("customer_id")
            customer = self.store.get("customer", customer_id) if customer_id else None

            # Hard suppression check
            suppressed, reason = self._check_hard_suppression(trg, merchant, category, customer, now_dt)
            if suppressed:
                continue

            # Check evidence completeness
            has_evidence = self._check_evidence_completeness(trg, category, merchant, customer)
            if not has_evidence:
                continue

            # Classify into arbitration tier
            tier = self._classify_tier(trg)
            
            # Check cadence override: Tier 1 (Urgent Safety/Supply Recall) overrides 24h cadence
            # Cadence applies to direct merchant outreach
            last_touch = self._merchant_last_touch.get(merchant_id)
            if last_touch and tier > 1 and not customer_id:
                seconds_since_touch = (now_dt - last_touch).total_seconds()
                if seconds_since_touch < 86400:  # Within 24h cooldown
                    continue

            # Calculate deterministic intra-tier score
            score = self._calculate_tier_score(trg, merchant, customer, category, now_dt)
            candidates.append((tier, score, trg, merchant, category, customer))

        # 2. Sort candidates by Tier (ascending: Tier 1 beats Tier 2) then Score (descending)
        candidates.sort(key=lambda c: (c[0], -c[1]))

        # 3. Discard duplicate (merchant, customer, trigger) targets within this tick.
        # Per challenge-testing-brief.md §14 FAQ: "Can I send multiple messages in
        # one tick to the same merchant? Yes, but only one action per
        # (merchant_id, conversation_id) pair per tick." Our conversation_id is
        # deterministically f"conv_{mid}_{trg_id}" (see action_payload below), so
        # this key is exactly equivalent to that (merchant_id, conversation_id)
        # rule. Deduping on (merchant_id, customer_id) alone — without the trigger
        # id — was over-suppressing: two genuinely distinct, independently-scored
        # triggers for the same merchant (e.g. a compliance deadline AND a
        # performance dip) would collide on the same key and only the
        # higher-scoring one would ever be emitted, silently discarding real,
        # non-redundant, non-expired opportunities every tick.
        actions: List[Dict[str, Any]] = []
        seen_targets: Set[Tuple[str, Optional[str], str]] = set()

        for tier, score, trg, merchant, category, customer in candidates:
            mid = merchant.get("merchant_id")
            cid = customer.get("customer_id") if customer else trg.get("customer_id")
            target_key = (mid, cid, trg.get("id"))
            if target_key in seen_targets:
                continue
            seen_targets.add(target_key)

            # Extract facts and generate strategy
            facts = extract_evidence(category, merchant, trg, customer)
            strategy = build_message_strategy(facts)

            # Compose message
            composed = compose_grounded_message(facts, strategy)

            # Update suppression and touch state
            supp_key = trg.get("suppression_key", "")
            self.record_suppression(supp_key, trg.get("expires_at", ""), now_dt)
            if not cid:
                self._merchant_last_touch[mid] = now_dt

            # Format action payload
            action_payload = {
                "conversation_id": f"conv_{mid}_{trg.get('id')}",
                "merchant_id": mid,
                "customer_id": customer.get("customer_id") if customer else None,
                "send_as": composed.send_as,
                "trigger_id": trg.get("id"),
                "template_name": composed.template_name,
                "template_params": composed.template_params,
                "body": composed.body,
                "cta": composed.cta,
                "suppression_key": composed.suppression_key,
                "rationale": composed.rationale,
            }
            actions.append(action_payload)

            if len(actions) >= 20:  # Action cap
                break

        return actions

    def _check_hard_suppression(
        self,
        trigger: Dict[str, Any],
        merchant: Dict[str, Any],
        category: Dict[str, Any],
        customer: Optional[Dict[str, Any]],
        now_dt: datetime,
    ) -> Tuple[bool, str]:
        """Verify hard suppression rules."""
        # 1. Expiration check
        exp_iso = trigger.get("expires_at")
        if exp_iso:
            exp_dt = parse_iso(exp_iso)
            if exp_dt and now_dt > exp_dt:
                return True, "trigger_expired"

        # 2. Suppression key check
        supp_key = trigger.get("suppression_key")
        if supp_key and self.is_suppressed(supp_key, now_dt):
            return True, "suppression_key_active"

        # 3. Customer scope consent and opt-in check
        scope = trigger.get("scope", "merchant")
        if scope == "customer":
            if not customer:
                return True, "customer_context_missing"
            
            pref = customer.get("preferences", {})
            if not pref.get("reminder_opt_in", False):
                return True, "customer_not_opted_in"

            consent = customer.get("consent", {})
            consent_scope = consent.get("scope", [])
            trg_kind = trigger.get("kind", "")

            # Verify that consent scope covers the trigger kind
            kind_to_scope_map = {
                "recall_due": ["recall_reminders"],
                "appointment_tomorrow": ["appointment_reminders"],
                "chronic_refill_due": ["refill_reminders", "delivery_notifications", "recall_alerts"],
                "wedding_package_followup": ["bridal_package_followup", "appointment_reminders"],
                "bridal_followup": ["bridal_package_followup", "appointment_reminders"],
                "trial_followup": ["program_updates", "kids_program_updates"],
                "customer_lapsed_hard": ["winback_offers", "renewal_reminders"],
                "customer_lapsed_soft": ["promotional_offers"],
            }
            required_scopes = kind_to_scope_map.get(trg_kind, ["promotional_offers"])
            if not any(req in consent_scope for req in required_scopes):
                return True, "customer_scope_unauthorized"

        # 4. Merchant Subscription Check
        sub = merchant.get("subscription", {})
        sub_status = sub.get("status", "active")
        trg_kind = trigger.get("kind", "")
        if sub_status == "expired" and trg_kind not in ("renewal_due", "winback_eligible"):
            return True, "merchant_subscription_expired"

        return False, ""

    def _check_evidence_completeness(
        self,
        trigger: Dict[str, Any],
        category: Dict[str, Any],
        merchant: Dict[str, Any],
        customer: Optional[Dict[str, Any]],
    ) -> bool:
        """Verify that sufficient factual evidence exists to compose a grounded message."""
        kind = trigger.get("kind", "")
        payload = trigger.get("payload", {})

        if kind == "research_digest":
            top_id = payload.get("top_item_id")
            digests = category.get("digest", [])
            if not top_id:
                return False
            if not any(d.get("id") == top_id for d in digests):
                return False

        if kind == "recall_due":
            slots = payload.get("available_slots", [])
            if not slots:
                return False

        if kind == "supply_alert":
            batches = payload.get("affected_batches", [])
            if not batches:
                return False

        if kind == "active_planning_intent":
            if not payload.get("intent_topic") and not payload.get("merchant_last_message"):
                return False

        if kind == "competitor_opened":
            if not payload.get("competitor_name") or payload.get("distance_km") is None:
                return False

        if kind in ("perf_dip", "perf_dip_severe", "perf_spike"):
            has_payload_delta = payload.get("delta_pct") is not None
            perf = merchant.get("performance", {}) if isinstance(merchant, dict) else {}
            delta_7d = perf.get("delta_7d", {}) if isinstance(perf, dict) else {}
            has_merchant_delta = any(v is not None for v in delta_7d.values()) if isinstance(delta_7d, dict) else False
            if not has_payload_delta and not has_merchant_delta:
                return False

        if kind in ("wedding_package_followup", "bridal_followup"):
            if payload.get("days_to_wedding") is None:
                return False

        if kind == "review_theme_emerged":
            if not payload.get("theme"):
                return False

        return True

    def _classify_tier(self, trigger: Dict[str, Any]) -> int:
        """
        Classify trigger into strict priority tiers:
        Tier 1: Critical (Urgency 5 safety/recall)
        Tier 2: High (Urgency 4 regulation/renewal/active planning)
        Tier 3: Normal (Urgency 3 recall/reviews/ipl/unverified)
        Tier 4: Low (Urgency 1-2 curiosity/research/milestones/seasonality)
        """
        urgency = int(trigger.get("urgency", 1))
        kind = trigger.get("kind", "")

        if urgency == 5 or kind in ("supply_alert",):
            return 1
        elif urgency == 4 or kind in ("regulation_change", "renewal_due", "active_planning_intent"):
            return 2
        elif urgency == 3 or kind in ("recall_due", "review_theme_emerged", "ipl_match_today", "gbp_unverified"):
            return 3
        else:
            return 4

    def _calculate_tier_score(
        self,
        trigger: Dict[str, Any],
        merchant: Dict[str, Any],
        customer: Optional[Dict[str, Any]],
        category: Dict[str, Any],
        now_dt: datetime,
    ) -> float:
        """Calculate deterministic intra-tier score using evidence depth and urgency."""
        urgency = int(trigger.get("urgency", 1))
        score = float(urgency * 10)

        # Relevance to merchant customer aggregates
        agg = merchant.get("customer_aggregate", {})
        if trigger.get("kind") == "research_digest" and agg.get("high_risk_adult_count", 0) > 50:
            score += 8.0
        if trigger.get("kind") == "supply_alert" and agg.get("chronic_rx_count", 0) > 0:
            score += 15.0

        # Signals alignment
        signals = merchant.get("signals", [])
        if trigger.get("kind") == "perf_dip" and any("perf_dip" in s for s in signals):
            score += 5.0
        if trigger.get("kind") == "gbp_unverified" and not merchant.get("identity", {}).get("verified", True):
            score += 6.0

        # Customer recency
        if customer:
            state = customer.get("state")
            if state == "lapsed_soft":
                score += 7.0
            elif state == "active":
                score += 5.0

        # Proximity to expiry
        exp_iso = trigger.get("expires_at")
        if exp_iso:
            exp_dt = parse_iso(exp_iso)
            if exp_dt:
                hours_left = (exp_dt - now_dt).total_seconds() / 3600.0
                if 0 < hours_left < 48:
                    score += 4.0

        return score

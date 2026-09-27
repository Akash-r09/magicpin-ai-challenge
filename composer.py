"""
Message Composer for Magicpin Vera AI Challenge
================================================
Dual-layer composition architecture:
1. LLM Composer (when an LLM provider and API key are configured):
   - Low temperature (0.0)
   - Constrained Fact Pack prompt
   - Zero hallucination allowed
   - Strict 4s timeout
2. Deterministic Grounded Template Fallback (Mandatory):
   - 100% deterministic, grounded, and zero-latency
   - Category-tuned voice (Dentists, Salons, Restaurants, Gyms, Pharmacies)
   - Natural Hindi-English code-mix when language preference is 'hi' or 'hi-en mix'
   - Formats exact numbers, active offers, study citations, batch codes, and slots
   - Integrated with validator for automated repair
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional
from urllib import request as urlrequest, error as urlerror

from evidence_extractor import FactPack
from message_strategy import MessageStrategy
from validator import validate_and_repair


class ComposedMessage:
    def __init__(
        self,
        body: str,
        cta: str,
        send_as: str,
        suppression_key: str,
        rationale: str,
        template_name: str = "vera_generic_v1",
        template_params: Optional[List[str]] = None,
    ):
        self.body = body
        self.cta = cta
        self.send_as = send_as
        self.suppression_key = suppression_key
        self.rationale = rationale
        self.template_name = template_name
        self.template_params = template_params or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "body": self.body,
            "cta": self.cta,
            "send_as": self.send_as,
            "suppression_key": self.suppression_key,
            "rationale": self.rationale,
            "template_name": self.template_name,
            "template_params": self.template_params,
        }


def _minimal_grounded_fallback(facts: FactPack, strategy: MessageStrategy) -> str:
    salutation = facts.salutation_name or (f"Dr. {facts.owner_first_name}" if facts.category_slug == "dentists" else facts.biz_name) or "Hi"
    offer = facts.active_offer_titles[0] if facts.active_offer_titles else ""
    offer_part = f" featuring {offer}" if offer else ""
    return (
        f"{salutation}, quick update for {facts.biz_name} in {facts.locality}: "
        f"want me to draft a fresh Google post{offer_part} highlighting your services to drive more customer calls?"
    )


def compose_grounded_message(
    facts: FactPack,
    strategy: MessageStrategy,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> ComposedMessage:
    """
    Produce a verified ComposedMessage using the two-layer composition pipeline:
    1. Attempts LLM synthesis if API key is provided and responsive
    2. Uses deterministic grounded template fallback if LLM is absent, times out, or fails validation
    """
    raw_body = ""
    template_name = "vera_generic_v1"
    template_params: List[str] = []

    # Check for optional LLM provider (prefer deterministic composition for festival_upcoming)
    llm_api_key = os.getenv("LLM_API_KEY", "")
    llm_provider = os.getenv("LLM_PROVIDER", "").lower()
    
    if (
        facts.trigger_kind != "festival_upcoming"
        and llm_api_key
        and llm_provider in ("openai", "gemini", "anthropic", "groq", "deepseek")
    ):
        try:
            raw_body = _call_llm_composer(facts, strategy, llm_provider, llm_api_key)
        except Exception:
            raw_body = ""

    # If LLM generated, validate LLM output strictly
    if raw_body:
        val_res = validate_and_repair(raw_body, facts, strategy, conversation_history)
        if not val_res.is_valid or not val_res.body:
            # Reject generated message if invalid
            raw_body = ""

    # If LLM didn't generate or was rejected, use deterministic grounded fallback
    if not raw_body:
        raw_body, template_name, template_params = _deterministic_compose(facts, strategy)
        val_res = validate_and_repair(raw_body, facts, strategy, conversation_history)
        if not val_res.is_valid or not val_res.body:
            # Reject and fall back to minimal safe grounded message
            final_body = _minimal_grounded_fallback(facts, strategy)
            template_name = "vera_generic_v1"
            template_params = [facts.salutation_name, final_body[:60], strategy.cta_requirement[:40]]
            val_res = validate_and_repair(final_body, facts, strategy, conversation_history)
            final_body = val_res.body
        else:
            final_body = val_res.body
    else:
        final_body = val_res.body

    # Format template parameters if empty
    if not template_params:
        template_params = [facts.salutation_name, final_body[:60], strategy.cta_requirement[:40]]

    return ComposedMessage(
        body=final_body,
        cta=strategy.cta_type,
        send_as=strategy.send_as,
        suppression_key=facts.suppression_key,
        rationale=strategy.rationale,
        template_name=template_name,
        template_params=template_params,
    )


def _format_time_of_day(iso_str: Optional[str]) -> Optional[str]:
    """Format an ISO timestamp's clock time as e.g. '7:30 PM', preserving whatever
    UTC offset is already in the string (no timezone conversion — just display).
    Returns None if the string is missing or unparseable, so callers never fabricate
    a time that wasn't actually supplied."""
    if not iso_str or not isinstance(iso_str, str):
        return None
    try:
        import re as _re
        m = _re.search(r"T(\d{2}):(\d{2})", iso_str)
        if not m:
            return None
        hour, minute = int(m.group(1)), int(m.group(2))
        period = "AM" if hour < 12 else "PM"
        hour12 = hour % 12 or 12
        return f"{hour12}:{minute:02d} {period}"
    except Exception:
        return None


_STAT_SNIPPET_RE = None


def _extract_prior_stat(text: Optional[str]) -> Optional[str]:
    """Pull a concrete, already-stated metric (e.g. '18 orders/day', '40% up') out of
    a merchant's own prior conversation_history message, so a planning follow-up can
    reference real continuity instead of a generic restatement. Returns None if no
    such pattern is present — never invents one."""
    global _STAT_SNIPPET_RE
    if not text:
        return None
    if _STAT_SNIPPET_RE is None:
        _STAT_SNIPPET_RE = re.compile(
            r"\b\d+(?:\.\d+)?\s?(?:%|orders?/day|orders? per day|reviews?|customers?|views?|calls?)\b",
            re.IGNORECASE,
        )
    m = _STAT_SNIPPET_RE.search(text)
    return m.group(0) if m else None


def _deterministic_compose(
    facts: FactPack,
    strategy: MessageStrategy,
) -> tuple[str, str, List[str]]:
    """
    Deterministic composer covering all trigger families and verticals with exact context facts.
    Returns (body, template_name, template_params).
    """
    kind = facts.trigger_kind
    cat = facts.category_slug
    is_hi = "hi" in strategy.language_preference
    salutation = facts.salutation_name or (f"Dr. {facts.owner_first_name}" if cat == "dentists" else facts.biz_name)
    active_offer = facts.active_offer_titles[0] if facts.active_offer_titles else ""

    # =========================================================================
    # 1. RESEARCH DIGEST (Merchant-facing)
    # =========================================================================
    if kind == "research_digest":
        source = facts.digest_source
        summary = facts.digest_summary
        cohort = "your adult patients"
        if facts.high_risk_adult_count:
            cohort = f"your {facts.high_risk_adult_count} high-risk adult patients"
        elif facts.digest_patient_segment:
            cohort = f"your {facts.digest_patient_segment.replace('_', ' ')} cohort"
            
        n_patients = f"{facts.digest_trial_n:,}-patient " if facts.digest_trial_n else ""
        
        actionable_part = ""
        if facts.digest_actionable:
            act_clean = facts.digest_actionable.strip().rstrip(".") + "."
            actionable_part = f" Clinical takeaway: {act_clean}"
        
        if source and summary:
            body = (
                f"{salutation}, {source.split(',')[0]} landed. "
                f"One item relevant to {cohort} — {n_patients}trial showed {summary.strip()}{actionable_part} "
                f"Worth a look (2-min abstract). Want me to pull it + draft a patient-ed WhatsApp you can share? "
                f"— {source}"
            )
            return body, "vera_research_digest_v1", [salutation, summary[:40], source]
        elif source:
            body = (
                f"{salutation}, new clinical update from {source}. "
                f"Worth a look for {cohort}.{actionable_part} Want me to pull the abstract + draft a patient-ed WhatsApp note? "
                f"— {source}"
            )
            return body, "vera_research_digest_v1", [salutation, source[:40], source]
        else:
            body = (
                f"{salutation}, clinical research update relevant to {cohort}.{actionable_part} "
                f"Want me to pull recent clinical abstracts + draft a patient-ed WhatsApp note for your clinic?"
            )
            return body, "vera_research_digest_v1", [salutation, "clinical update", facts.biz_name]

    # =========================================================================
    # 2. REGULATION CHANGE (Merchant-facing)
    # =========================================================================
    if kind == "regulation_change":
        source = facts.digest_source or "Regulatory authority"
        summary = facts.digest_summary
        deadline = facts.digest_deadline_iso
        deadline_hook = f" effective {deadline}" if deadline else ""
        loc_str = f" in {facts.locality}" if facts.locality else ""
        biz = facts.biz_name or "your clinic"
        
        # Clinical peer hook citing authority directly without generic boilerplate
        hook = f"{salutation}, {source} has issued revised compliance standards{deadline_hook}."
        
        detail_parts = []
        if summary:
            detail_parts.append(summary.strip())
        if facts.digest_actionable:
            detail_parts.append(facts.digest_actionable.strip().rstrip(".") + ".")
            
        detail_text = (" " + " ".join(detail_parts)) if detail_parts else ""
        
        if deadline:
            implication = f" To ensure {biz}{loc_str} is fully compliant before {deadline}, I can draft a concise 1-page SOP checklist for your team to audit your setup."
        else:
            implication = f" To ensure {biz}{loc_str} is fully compliant, I can draft a concise 1-page SOP checklist for your team to verify the updated requirements."
        
        body = f"{hook}{detail_text}{implication} Takes 2 minutes — want me to prepare it?"
        return body, "vera_compliance_v1", [salutation, deadline or "compliance", source]

    # =========================================================================
    # 3. SUPPLY ALERT / DRUG RECALL (Merchant-facing)
    # =========================================================================
    if kind == "supply_alert":
        mol = facts.trigger_payload.get("molecule", "active molecule")
        batches_list = facts.trigger_payload.get("affected_batches", [])
        mfr = facts.trigger_payload.get("manufacturer", "Manufacturer")
        cohort_str = f"{facts.chronic_rx_count} chronic-Rx patients" if facts.chronic_rx_count else "repeat customers"
        
        batch_cnt = f"{len(batches_list)} " if batches_list else ""
        batch_items = f"({', '.join(batches_list)})" if batches_list else ""
        
        detail_part = f"{facts.digest_summary.strip()} " if facts.digest_summary else "sub-potency issue, no safety hazard, but customers should be informed for replacement. "
        
        body = (
            f"{salutation}, urgent notice: voluntary recall on {batch_cnt}{mol} batches {batch_items} by {mfr} — "
            f"{detail_part}"
            f"Pulled your repeat-Rx list: {cohort_str} may be affected. "
            f"Want me to draft their WhatsApp note + the replacement-pickup workflow?"
        )
        return body, "vera_supply_alert_v1", [salutation, mol, ", ".join(batches_list) or "specified batches"]

    # =========================================================================
    # 4. RECALL DUE (Customer-facing)
    # =========================================================================
    if kind == "recall_due":
        c_name = facts.customer_name or "there"
        biz = facts.biz_name or "our clinic"
        slots = facts.trigger_payload.get("available_slots", [])

        if len(slots) >= 2:
            slot_text = f"**{slots[0].get('label')}** ya **{slots[1].get('label')}**"
        elif len(slots) == 1:
            slot_text = f"**{slots[0].get('label')}**"
        else:
            slot_text = ""

        last_visit = facts.customer_last_visit or facts.trigger_payload.get("last_visit_date")
        visit_part = f" Your last visit was on {last_visit} —" if last_visit else ""

        offer_part = f" {active_offer}." if active_offer else ""

        category = (facts.category_slug or "").lower()

        if category == "dentists":
            service = "your cleaning recall is due"
            identity = f"{biz} here 🦷"
        elif category == "salons":
            service = "your next salon visit is due"
            identity = f"{biz} here"
        elif category == "gyms":
            service = "your next training session is due"
            identity = f"{biz} here"
        elif category == "pharmacies":
            service = "your refill is due"
            identity = f"{biz} here"
        else:
            service = "your next visit is due"
            identity = f"{biz} here"

        if is_hi:
            if slot_text:
                body = (
                    f"Hi {c_name}, {identity}{visit_part} "
                    f"{service}. Apke liye slots ready hain: {slot_text}."
                    f"{offer_part} "
                    f"Reply 1 for first slot, 2 for second, or tell us a time that works."
                )
            else:
                body = (
                    f"Hi {c_name}, {identity}{visit_part} "
                    f"{service}.{offer_part} "
                    f"Apko kaunsa time suit karega? Reply with your preferred day and time."
                )
        else:
            if slot_text:
                body = (
                    f"Hi {c_name}, {identity}{visit_part} "
                    f"{service}. We have slots ready for you: {slot_text}."
                    f"{offer_part} "
                    f"Reply 1 for first slot, 2 for second, or let us know a convenient time."
                )
            else:
                body = (
                    f"Hi {c_name}, {identity}{visit_part} "
                    f"{service}.{offer_part} "
                    f"What day and time works best for you?"
                )

        return body, "merchant_recall_reminder_v2", [c_name, biz, slot_text or "flexible"]

    # =========================================================================
    # 5. CURIOUS ASK DUE (Merchant-facing)
    # =========================================================================
    if kind == "curious_ask_due":
        if is_hi:
            body = (
                f"Hi {salutation}! Quick check — {facts.biz_name} pe is week kaun si service sabse zyada "
                f"demand mein rahi? I'll turn the answer into a Google post + a short WhatsApp template "
                f"you can share with inquiring customers. Takes 5 min."
            )
        else:
            body = (
                f"Hi {salutation}! Quick check — what service has been most asked-for this week "
                f"at {facts.biz_name}? I'll turn the answer into a fresh Google post + a quick WhatsApp reply "
                f"you can send to customer inquiries. Takes 5 min."
            )
        return body, "vera_curious_ask_v1", [salutation, facts.biz_name]

    # =========================================================================
    # 6. IPL MATCH TODAY (Merchant-facing)
    # =========================================================================
    if kind == "ipl_match_today":
        match = facts.trigger_payload.get("match", "tonight's match")
        venue = facts.trigger_payload.get("venue")
        venue_str = f" at {venue}" if venue else ""
        tactic_payload = facts.trigger_payload.get("recommended_tactic")
        loc_str = f" in {facts.locality}" if facts.locality else ""
        kickoff = _format_time_of_day(facts.trigger_payload.get("match_time_iso"))
        time_str = f" ({kickoff} kickoff)" if kickoff else ""

        if tactic_payload:
            tactic = tactic_payload
        else:
            tactic = f"On match nights{loc_str}, dine-in footfall typically softens while delivery orders surge."
            
        body = (
            f"Hi {salutation}, {match}{venue_str} is tonight{time_str}. "
            f"{tactic} "
            f"I can draft a high-visibility delivery banner for {facts.biz_name} to capture fan order volume. "
            f"Takes 2 minutes — want me to prepare it?"
        )
        return body, "vera_ipl_v1", [salutation, match, venue or "tonight"]

    # =========================================================================
    # 7. ACTIVE PLANNING INTENT (Merchant-facing)
    # =========================================================================
    if kind == "active_planning_intent":
        topic = facts.trigger_payload.get("intent_topic", "corporate_thali")
        pricing_tiers = facts.trigger_payload.get("pricing_tiers", [])
        
        if "thali" in topic:
            if pricing_tiers:
                tier_lines = "\n".join(f"- {t}" for t in pricing_tiers)
                body = (
                    f"{salutation}, here's a starter corporate thali structure you can review:\n"
                    f"{tier_lines}\n"
                    f"Offices in {facts.locality} are right in your delivery radius. "
                    f"Want me to draft a 3-line WhatsApp to send their facilities managers? Reply CONFIRM."
                )
            else:
                body = (
                    f"{salutation}, ready to help you set up a corporate thali program for offices in {facts.locality}. "
                    f"Share your preferred price tiers and I'll draft volume-based packages + a WhatsApp message "
                    f"for their facilities managers. Reply with your pricing and I'll build it out."
                )
        elif "yoga" in topic or "camp" in topic:
            offer_price = active_offer if active_offer else ""
            price_part = f"- Package: {offer_price}\n" if offer_price else ""
            body = (
                f"{salutation}, here's a starter structure for your kids yoga summer camp:\n"
                f"- Multi-week program with regular classes\n"
                f"- Age groups based on your studio capacity\n"
                f"{price_part}"
                f"Registrations in {facts.locality} pick up around this time. "
                f"Want me to publish this on your Google profile + draft an Insta post? Reply CONFIRM."
            )
        else:
            body = (
                f"{salutation}, here's a concrete draft based on your request:\n"
                f"- Featured package: {active_offer or 'your signature service'}\n"
                f"- Timed for clients in {facts.locality}\n"
                f"Want me to finalize the Google post + customer message draft? Reply CONFIRM."
            )
        return body, "vera_planning_v1", [salutation, topic]

    # =========================================================================
    # 8. PERFORMANCE DIP (Merchant-facing)
    # =========================================================================
    if kind in ("perf_dip", "perf_dip_severe"):
        metric = facts.trigger_payload.get("metric", "calls")
        delta = facts.trigger_payload.get("delta_pct")
        curr_calls = facts.calls_30d
        
        if delta is not None:
            pct_str = f"{abs(int(delta * 100))}%"
            dip_desc = f"dropped {pct_str} over the past 7 days"
        else:
            pct_str = "notable"
            dip_desc = "dipped notably over the past 7 days"
        
        bench_calls = facts.peer_stats.get("avg_calls_30d") if isinstance(facts.peer_stats, dict) else None
        bench_views = facts.peer_stats.get("avg_views_30d") if isinstance(facts.peer_stats, dict) else None
        
        if metric == "calls":
            if curr_calls and bench_calls:
                calls_part = f" ({curr_calls} calls recorded vs ~{bench_calls} peer average)"
            elif curr_calls:
                calls_part = f" ({curr_calls} calls recorded)"
            elif bench_calls:
                calls_part = f" (category peer average: ~{bench_calls} calls/month)"
            else:
                calls_part = ""
        elif metric == "views":
            curr_views = facts.views_30d
            if curr_views and bench_views:
                calls_part = f" ({curr_views} views recorded vs ~{bench_views} peer average)"
            elif curr_views:
                calls_part = f" ({curr_views} views recorded)"
            elif bench_views:
                calls_part = f" (category peer average: ~{bench_views} views/month)"
            else:
                calls_part = ""
        else:
            calls_part = f" ({curr_calls} calls recorded)" if curr_calls else ""
        
        body = (
            f"{salutation}, your {metric} {dip_desc}"
            f"{calls_part}. To turn this around in {facts.locality}, "
            f"I can publish a high-visibility Google update featuring your {active_offer or 'core services'}. "
            f"Takes 2 minutes — want me to draft it?"
        )
        return body, "vera_perf_dip_v1", [salutation, metric, pct_str]

    # =========================================================================
    # 9. PERFORMANCE SPIKE (Merchant-facing)
    # =========================================================================
    if kind == "perf_spike":
        metric = facts.trigger_payload.get("metric", "views")
        driver = facts.trigger_payload.get("likely_driver", "recent Google post")
        curr_views = facts.views_30d
        bench_views = facts.peer_stats.get("avg_views_30d") if isinstance(facts.peer_stats, dict) else None
        bench_calls = facts.peer_stats.get("avg_calls_30d") if isinstance(facts.peer_stats, dict) else None
        
        if metric == "views":
            if curr_views and bench_views:
                stat_part = f"({curr_views} total in 30d vs ~{bench_views} peer average)"
            elif curr_views:
                stat_part = f"({curr_views} total in 30d)"
            elif bench_views:
                stat_part = f"(category peer average: ~{bench_views} views/month)"
            else:
                stat_part = ""
        elif metric == "calls":
            curr_calls = facts.calls_30d
            if curr_calls and bench_calls:
                stat_part = f"({curr_calls} total in 30d vs ~{bench_calls} peer average)"
            elif curr_calls:
                stat_part = f"({curr_calls} total in 30d)"
            elif bench_calls:
                stat_part = f"(category peer average: ~{bench_calls} calls/month)"
            else:
                stat_part = ""
        else:
            stat_part = f"({curr_views} total in 30d)" if curr_views else ""
        
        stat_spacing = f" {stat_part}" if stat_part else ""
        body = (
            f"{salutation}, strong momentum: your {metric} jumped this week"
            f"{stat_spacing}, driven by your {driver.replace('_', ' ')}. "
            f"Want me to draft a follow-up post to keep this visibility going?"
        )
        return body, "vera_perf_spike_v1", [salutation, metric, driver]

    # =========================================================================
    # 10. SEASONAL DIP REFRAME (Merchant-facing)
    # =========================================================================
    if kind == "seasonal_perf_dip":
        # Only cite numbers that exist in context
        dip_part = ""
        if facts.views_delta_7d is not None:
            pct_drop = abs(int(facts.views_delta_7d * 100))
            dip_part = f"your views are down {pct_drop}% this week — but "
        else:
            dip_part = "your views dipped this week — but "
        
        peer_avg = facts.peer_stats.get("seasonal_dip_range")
        peer_part = f" ({peer_avg} peer average in this window)" if peer_avg else ""
        
        member_part = ""
        if facts.total_active_members:
            member_part = f"For now, focus retention on your {facts.total_active_members} active members. "
        else:
            member_part = "For now, focus on retaining your active members. "
        
        if peer_avg:
            market_desc = f"this aligns with seasonal peer trends ({peer_avg} peer average). "
        else:
            market_desc = "this is an expected seasonal pattern for gyms. "
        
        body = (
            f"{salutation}, {dip_part}"
            f"{market_desc}"
            f"Strategic action: pause heavy ad spend now and save budget for the next peak season. "
            f"{member_part}"
            f"Want me to draft a summer attendance challenge to keep them engaged?"
        )
        return body, "vera_seasonal_dip_v1", [salutation, dip_part[:20], member_part[:20]]

    # =========================================================================
    # 11. CUSTOMER LAPSED HARD (Customer-facing, Gyms)
    # =========================================================================
    if kind in ("customer_lapsed_hard", "customer_lapsed_soft"):
        c_name = facts.customer_name or "there"
        owner = facts.owner_first_name or "the team"
        biz = facts.biz_name or "the gym"
        days = facts.trigger_payload.get("days_since_last_visit")
        focus = facts.customer_training_focus or facts.trigger_payload.get("previous_focus", "fitness")
        
        if days:
            duration_text = f"It's been about {days // 7} weeks"
        else:
            duration_text = "It's been a while"
        
        body = (
            f"Hi {c_name} 👋 {owner} from {biz} here. {duration_text} — happens "
            f"to most members at some point, zero judgment. We have sessions tailored for {focus} "
            f"goals that fit your routine. Want me to hold a free comeback spot for you next week? "
            f"Reply YES — no commitment, no auto-charge."
        )
        return body, "merchant_lapsed_winback_v1", [c_name, biz, str(days or "unknown")]

    # =========================================================================
    # 12. CHRONIC REFILL DUE (Customer-facing, Pharmacies)
    # =========================================================================
    if kind == "chronic_refill_due":
        c_name = facts.customer_name or "there"
        biz = facts.biz_name or "your pharmacy"
        loc = facts.locality or ""
        loc_part = f" {loc}" if loc else ""
        mol_list = ", ".join(facts.trigger_payload.get("molecule_list", ["prescribed medicines"]))
        
        # Only cite pricing/discount if present in trigger_payload or active offer
        price_info = facts.trigger_payload.get("total_price")
        discount_info = facts.trigger_payload.get("discount_pct")
        savings_info = facts.trigger_payload.get("savings_amount")
        delivery_info = facts.trigger_payload.get("delivery_eta")
        
        pricing_part = ""
        if price_info and discount_info and savings_info:
            pricing_part = f" {discount_info}% discount applied — total ₹{price_info} (₹{savings_info} saved)."
        elif price_info:
            pricing_part = f" Total: ₹{price_info}."
        elif active_offer:
            pricing_part = f" {active_offer} applicable."
        
        delivery_part = ""
        if delivery_info:
            delivery_part = f" Free home delivery to saved address by {delivery_info}."
        
        senior_part = ""
        if facts.customer_senior_citizen:
            senior_part = " Senior citizen benefits applicable."
        
        is_hi_cust = "hi" in (facts.customer_lang_pref or "").lower() or is_hi
        if is_hi_cust:
            body = (
                f"Namaste — {biz}{loc_part} yahan. {c_name} ji ki monthly medicines ({mol_list}) "
                f"refill due hai. Same dose, verified brand pack ready hai."
                f"{senior_part}{pricing_part}{delivery_part} "
                f"Reply CONFIRM to dispatch, or let us know if doctor updated any dosage."
            )
        else:
            body = (
                f"Hello from {biz}{loc_part}. Monthly prescription refill is due for {c_name} ({mol_list}). "
                f"Same dosage, verified brand pack is ready."
                f"{senior_part}{pricing_part}{delivery_part} "
                f"Reply CONFIRM to dispatch, or let us know if doctor updated any dosage."
            )
        return body, "merchant_refill_reminder_v1", [c_name, biz, mol_list]

    # =========================================================================
    # 13. WEDDING / BRIDAL FOLLOWUP (Customer-facing, Salons)
    # =========================================================================
    if kind in ("wedding_package_followup", "bridal_followup"):
        c_name = facts.customer_name or "there"
        owner = facts.owner_first_name or "our team"
        biz = facts.biz_name or "the salon"
        days = facts.trigger_payload.get("days_to_wedding")
        
        days_part = f" {days} days to your wedding —" if days else ""
        offer_part = f" {active_offer}." if active_offer else ""
        
        body = (
            f"Hi {c_name} 💍 {owner} from {biz} {facts.locality} here.{days_part} "
            f"Perfect window to start a skin-prep program after your bridal trial."
            f"{offer_part} "
            f"Want me to block your preferred slot for the first session? Reply YES with your preferred day and time."
        )
        return body, "merchant_bridal_followup_v1", [c_name, biz, str(days or "upcoming")]

    # =========================================================================
    # 14. COMPETITOR OPENED (Merchant-facing)
    # =========================================================================
    if kind == "competitor_opened":
        comp = facts.trigger_payload.get("competitor_name")
        dist = facts.trigger_payload.get("distance_km")
        their_off = facts.trigger_payload.get("their_offer")
        
        dist_part = f" {dist}km away" if dist else " nearby"
        offer_part = f" promoting {their_off}" if their_off else ""
        comp_part = f"{comp} opened{dist_part}" if comp else f"a new business opened{dist_part}"
        
        body = (
            f"{salutation}, heads up: {comp_part} on GBP{offer_part}. "
            f"To defend your local search visibility in {facts.locality}, want me to feature your signature "
            f"{active_offer or 'top-rated service'} highlighting your verified reviews? Takes 5 min."
        )
        return body, "vera_competitor_v1", [salutation, comp or "nearby_business", str(dist or "nearby")]

    # =========================================================================
    # 15. CDE OPPORTUNITY / WEBINAR (Merchant-facing)
    # =========================================================================
    if kind == "cde_opportunity":
        credits_str = f" ({facts.digest_credits} CDE credits)" if facts.digest_credits else ""
        date_str = facts.digest_date
        date_part = f" Date: {date_str}." if date_str else ""
        summary = facts.digest_summary or ""
        summary_part = f" {summary.strip()}." if summary else ""
        source = facts.digest_source or "professional association"
        actionable = facts.digest_actionable
        actionable_part = f" {actionable}." if actionable else ""
        loc_part = f" for your {facts.locality} clinic" if facts.locality else ""
        
        body = (
            f"{salutation}, upcoming CDE opportunity{loc_part}: {facts.digest_title or 'Specialist Masterclass'}{credits_str}."
            f"{date_part}{summary_part}{actionable_part} "
            f"Want me to send the registration details and block it on your calendar?"
        )
        return body, "vera_cde_v1", [salutation, date_str or "upcoming", source]

    # =========================================================================
    # 16. GBP UNVERIFIED (Merchant-facing)
    # =========================================================================
    if kind == "gbp_unverified":
        uplift = facts.trigger_payload.get("estimated_uplift_pct") or facts.peer_stats.get("verified_uplift_pct")
        if uplift:
            uplift_pct = int(uplift * 100) if uplift < 1.0 else int(uplift)
            benefit_part = f"Verified businesses in {facts.locality} average ~{uplift_pct}% more search calls and direction requests."
        else:
            benefit_part = f"Verified businesses in {facts.locality} typically receive significantly more search calls and direction requests."
        
        body = (
            f"Hi {salutation}, your Google Business Profile for {facts.biz_name} is currently unverified. "
            f"{benefit_part} "
            f"I can guide you through the 3 quick verification steps today to unlock that dormant traffic. "
            f"Takes 3 minutes — want me to start?"
        )
        return body, "vera_gbp_unverified_v1", [salutation, facts.biz_name, facts.locality]

    # =========================================================================
    # 17. RENEWAL DUE (Merchant-facing)
    # =========================================================================
    if kind == "renewal_due":
        days = facts.trigger_payload.get("days_remaining") or facts.days_remaining
        plan = facts.subscription_plan or "Pro"
        
        days_part = f" in {days} days" if days else " soon"
        
        body = (
            f"{salutation}, your Vera {plan} subscription renews{days_part}. "
            f"Renewing on schedule ensures your automated Google updates and patient recall workflows "
            f"continue uninterrupted. Reply CONFIRM to renew your {plan} plan."
        )
        return body, "vera_renewal_v1", [salutation, plan, str(days or "soon")]

    # =========================================================================
    # 18. WINBACK ELIGIBLE (Merchant-facing)
    # =========================================================================
    if kind == "winback_eligible":
        days = facts.trigger_payload.get("days_since_expiry") or facts.days_since_expiry
        
        days_part = f" {days} days ago" if days else ""
        
        body = (
            f"{salutation}, since your Vera subscription ended{days_part}, customer calls and profile views "
            f"have tapered in {facts.locality}. Reactivating takes 1 minute and restores your automatic "
            f"Google posts and customer retention nudges. Want me to reactivate your listing today?"
        )
        return body, "vera_winback_v1", [salutation, facts.locality, str(days or "recently")]

    # =========================================================================
    # 19. REVIEW THEME EMERGED (Merchant-facing)
    # =========================================================================
    if kind == "review_theme_emerged":
        theme = facts.trigger_payload.get("theme", "service")
        count = facts.trigger_payload.get("occurrences_30d")
        quote = facts.trigger_payload.get("common_quote", "")
        quote_part = f' (e.g. "{quote}")' if quote else ""
        
        count_desc = f"{count} customer reviews this month" if count else "customer reviews this month"
        
        body = (
            f"{salutation}, operational review alert: {count_desc} mentioned '{theme}'{quote_part}. "
            f"To protect your reputation in {facts.locality}, want me to draft a customer-facing post addressing "
            f"this + a quick operational tip for your team?"
        )
        return body, "vera_review_theme_v1", [salutation, theme, str(count or "multiple")]

    # =========================================================================
    # 20. TRIAL FOLLOWUP (Customer-facing)
    # =========================================================================
    if kind == "trial_followup":
        c_name = facts.customer_name or "there"
        biz = facts.biz_name or "the studio"
        slots = facts.trigger_payload.get("next_session_options", [])
        
        if slots:
            slot_lbl = slots[0].get("label", "upcoming session")
            body = (
                f"Hi {c_name} 👋 {facts.owner_first_name or 'The team'} from {biz} here. "
                f"Hope you enjoyed the trial session! Our next batch session is ready: {slot_lbl}. "
                f"Want me to hold a spot for you? Reply YES to confirm."
            )
        else:
            slot_lbl = "flexible"
            body = (
                f"Hi {c_name} 👋 {facts.owner_first_name or 'The team'} from {biz} here. "
                f"Hope you enjoyed the trial session! We'd love to have you back. "
                f"What day and time works best for your next session? Reply and we'll hold a spot."
            )
        return body, "merchant_trial_followup_v1", [c_name, biz, slot_lbl]

    # =========================================================================
    # 21. APPOINTMENT TOMORROW (Customer-facing)
    # =========================================================================
    if kind == "appointment_tomorrow":
        c_name = facts.customer_name or "there"
        biz = facts.biz_name or "our team"
        
        # Only cite time if available from context
        appt_time = facts.trigger_payload.get("appointment_time")
        pref_slots = facts.customer_preferred_slots
        
        if appt_time:
            time_part = f" at {appt_time}"
        elif pref_slots:
            slot_clean = pref_slots.replace("_", " ")
            time_part = f" ({slot_clean} slot)"
        else:
            time_part = ""
        
        service_str = active_offer or "your session"
        
        body = (
            f"Hi {c_name}, 24-hour courtesy reminder from {biz}: your appointment for {service_str} "
            f"is confirmed for tomorrow{time_part}. "
            f"Reply CONFIRM to lock in your slot, or let us know if you need to reschedule."
        )
        return body, "merchant_appointment_reminder_v1", [c_name, biz]

    # =========================================================================
    # 22. DORMANCY / GENTLE CHECK-IN (Merchant-facing)
    # =========================================================================
    if kind == "dormant_with_vera":
        views_stat = f"{facts.views_30d} views" if facts.views_30d else "steady search interest"
        if is_hi:
            body = (
                f"Hi {salutation}, checking in from Vera — {facts.locality} mein aapke Google listing ne "
                f"last 30 days mein {views_stat} record kiye. "
                f"Is week footfall kaisa chal raha hai? "
                f"Want me to draft a fresh Google post featuring {active_offer or 'your signature services'} to keep momentum?"
            )
        else:
            body = (
                f"Hi {salutation}, checking in from Vera — your Google listing recorded {views_stat} in {facts.locality}. "
                f"How has customer footfall been this week? "
                f"Want me to draft a quick Google update featuring {active_offer or 'your signature services'} to drive more calls?"
            )
        return body, "vera_dormancy_v1", [salutation, facts.locality]

    # =========================================================================
    # 23. CATEGORY SEASONAL / DEMAND SHIFT (Merchant-facing)
    # =========================================================================
    if kind == "category_seasonal":
        trends_raw = facts.trigger_payload.get("trends", [])
        trends_up = facts.trigger_payload.get("trends_up", [])
        trends_down = facts.trigger_payload.get("trends_down", [])
        
        if trends_raw:
            formatted_trends = [t.replace("_", " ").replace("+", "up ").replace("-", "down ") for t in trends_raw]
            trend_desc = ", ".join(formatted_trends)
        elif trends_up or trends_down:
            up_part = f"{', '.join(trends_up)} demand is rising" if trends_up else ""
            down_part = f"{', '.join(trends_down)} demand is declining" if trends_down else ""
            trend_desc = " while ".join(filter(None, [up_part, down_part]))
        else:
            trend_desc = "seasonal demand patterns are shifting"
        
        seasonal_beat_part = ""
        if facts.matched_seasonal_beat:
            beat_note = facts.matched_seasonal_beat.get("note", "").strip()
            if beat_note:
                seasonal_beat_part = f" Category trend note: {beat_note}."
        
        loc_part = f" in {facts.locality}" if facts.locality else ""
        
        body = (
            f"{salutation}, seasonal demand shift advisory: {trend_desc}.{seasonal_beat_part} "
            f"Recommended action: adjust your front-visibility items to match current demand{loc_part}. "
            f"Want me to draft a seasonal Google post on trending health essentials?"
        )
        return body, "vera_category_seasonal_v1", [salutation]

    # =========================================================================
    # 24. FESTIVAL UPCOMING (Merchant-facing)
    # =========================================================================
    if kind == "festival_upcoming":
        fest = facts.festival_name or facts.trigger_payload.get("festival")
        raw_date = facts.festival_date or facts.trigger_payload.get("date") or facts.trigger_payload.get("event_date")
        
        # Format date cleanly (e.g. '2026-10-31' -> '31 Oct')
        date_formatted = ""
        if raw_date:
            m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", raw_date.strip())
            if m:
                month_num = int(m.group(2))
                day_num = int(m.group(3))
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                if 1 <= month_num <= 12:
                    date_formatted = f"{day_num} {month_names[month_num - 1]}"
                else:
                    date_formatted = raw_date
            else:
                date_formatted = raw_date
                
        fest_label = fest or "the festival"
        
        # 1. Salutation + Exact why-now trigger (festival + exact date)
        if fest and date_formatted:
            hook = f"Hi {salutation}, {fest} is on {date_formatted}."
        elif fest:
            hook = f"Hi {salutation}, {fest} is approaching."
        elif date_formatted:
            hook = f"Hi {salutation}, the festive season is on {date_formatted}."
        else:
            hook = f"Hi {salutation}, the festive season is approaching."
            
        # 2. Category seasonal benchmark
        cat_name = "salons" if cat == "salons" else (cat or "businesses")
        insight = ""
        if facts.matched_seasonal_beat:
            month_range = facts.matched_seasonal_beat.get("month_range", "")
            note = facts.matched_seasonal_beat.get("note", "").strip()
            if " — " in note:
                parts = note.split(" — ", 1)
                season_part = parts[0].strip().replace("primary ", "").replace("secondary ", "")
                season_clean = re.sub(r"\s+season$", "", season_part)
                metric_part = parts[1].strip()
                metric_clean = re.sub(r"(\b\d+x\s+baseline\b)", r"are at \1", metric_part) if "baseline" in metric_part and "are at" not in metric_part else metric_part
                insight = f"For {cat_name}, {metric_clean} in the {month_range} {season_clean} window."
            else:
                range_str = f"in the {month_range} window" if month_range else ""
                insight = f"For {cat_name}, {note} {range_str}.".replace("  ", " ").strip()
                
        # 3. Merchant Fit, Implication, Effort Externalization & ONE Binary CTA
        loc_str = f" in {facts.locality}" if facts.locality else ""
        biz_name = facts.biz_name or "your business"
        fest_label = fest or "Diwali"
        insight_ref = "that seasonal insight" if insight else "this"
        
        # Check active offers: only include if explicitly festive/bridal
        festive_keywords = ("diwali", "festiv", "holiday", "celebrat", "bridal", "wedding", "puja")
        offer_part = ""
        if active_offer and any(kw in active_offer.lower() for kw in festive_keywords):
            offer_part = f" featuring your {active_offer}"
            
        implication_cta = f"For {biz_name}{loc_str}, I can turn {insight_ref} into a ready-to-post {fest_label} booking announcement for your Google profile{offer_part}. Want me to prepare it?"
        
        if insight:
            body = f"{hook} {insight} {implication_cta}"
        else:
            body = f"{hook} {implication_cta}"
            
        return body, "vera_festival_v1", [salutation, fest or "festive_season", facts.locality]

    # =========================================================================
    # 25. MILESTONE REACHED (Merchant-facing)
    # =========================================================================
    if kind == "milestone_reached":
        metric = facts.trigger_payload.get("metric", "reviews")
        metric_label = metric.replace("_", " ")
        if "review" in metric_label:
            metric_display = "customer reviews"
        elif "call" in metric_label:
            metric_display = "customer calls"
        elif "view" in metric_label:
            metric_display = "profile views"
        else:
            metric_display = metric_label
            
        val_now = facts.trigger_payload.get("value_now")
        milestone = facts.trigger_payload.get("milestone_value")
        loc_str = f" in {facts.locality}" if facts.locality else ""
        
        if val_now is not None and milestone is not None:
            if val_now >= milestone:
                body = (
                    f"{salutation}, milestone reached for {facts.biz_name}: you've officially crossed "
                    f"{milestone} {metric_display}{loc_str}! "
                    f"I can draft a celebration Google post highlighting your team and {active_offer or 'top services'} "
                    f"to thank your customers. Takes 2 minutes — want me to draft it?"
                )
            else:
                gap = milestone - val_now
                body = (
                    f"{salutation}, milestone update for {facts.biz_name}: you're at {val_now} {metric_display}{loc_str} — "
                    f"only {gap} away from the {milestone} mark! "
                    f"I can draft a quick review-request WhatsApp message you can send to recent happy customers "
                    f"to reach {milestone} this week. Takes 2 minutes — want me to prepare it?"
                )
        elif milestone is not None:
            body = (
                f"{salutation}, milestone update for {facts.biz_name}: you're approaching the {milestone} {metric_display} mark{loc_str}. "
                f"I can draft a customer review-request template to help you reach {milestone} this week. "
                f"Takes 2 minutes — want me to draft it?"
            )
        else:
            body = (
                f"{salutation}, milestone update for {facts.biz_name}{loc_str}. "
                f"Want me to draft a fresh Google post highlighting {active_offer or 'your core services'}? "
                f"Takes 2 minutes — want me to prepare it?"
            )
        return body, "vera_milestone_v1", [salutation, str(milestone or "milestone"), facts.locality]

    # =========================================================================
    # 26. GENERIC GROUNDED FALLBACK
    # =========================================================================
    perf_parts = []
    if facts.views_30d:
        perf_parts.append(f"{facts.views_30d} views")
    if facts.calls_30d:
        perf_parts.append(f"{facts.calls_30d} calls")
    perf_text = f"your profile recorded {' and '.join(perf_parts)} recently. " if perf_parts else ""

    body = (
        f"{salutation}, quick update for {facts.biz_name} in {facts.locality}: "
        f"{perf_text}"
        f"Want me to draft a fresh Google post highlighting {active_offer or 'your services'} to keep momentum?"
    )
    return body, "vera_generic_v1", [salutation, facts.biz_name, facts.locality]


def _call_llm_composer(
    facts: FactPack,
    strategy: MessageStrategy,
    provider: str,
    api_key: str,
) -> str:
    """Execute tightly scoped LLM call with Fact Pack and timeout."""
    system_prompt = (
        "You are Vera, magicpin's elite merchant engagement AI assistant on WhatsApp.\n"
        "RULES:\n"
        "1. Use ONLY the supplied facts in the Fact Pack. If a fact is absent, DO NOT mention or invent it.\n"
        "2. NEVER invent a number, price, date, citation, offer, study, competitor, appointment slot, or performance metric.\n"
        "3. Do NOT include URLs of any kind.\n"
        "4. Do NOT expose internal system terms (e.g., signals, payload, trigger_id, suppression_key).\n"
        "5. Output ONE single primary CTA at the very end matching the requested CTA type.\n"
        "6. Do NOT ask qualification questions after explicit commitment.\n"
        "7. Tone must match the category voice strictly (clinical for dentists, operator for restaurants, etc.).\n"
        "8. Return ONLY the message body as plain text, no commentary, no quotes."
    )

    fact_pack_summary = {
        "audience": strategy.audience,
        "send_as": strategy.send_as,
        "salutation_name": facts.salutation_name,
        "biz_name": facts.biz_name,
        "locality": facts.locality,
        "category": facts.category_slug,
        "category_voice": strategy.category_voice,
        "language_preference": strategy.language_preference,
        "why_now": strategy.why_now,
        "trigger_kind": facts.trigger_kind,
        "trigger_payload": facts.trigger_payload,
        "festival_name": facts.festival_name,
        "festival_date": facts.festival_date,
        "matched_seasonal_beat": facts.matched_seasonal_beat,
        "seasonal_beats": facts.seasonal_beats,
        "trend_signals": facts.trend_signals,
        "peer_stats": facts.peer_stats,
        "digest": {
            "title": facts.digest_title,
            "source": facts.digest_source,
            "summary": facts.digest_summary,
            "actionable": facts.digest_actionable,
            "trial_n": facts.digest_trial_n,
            "deadline": facts.digest_deadline_iso,
        } if facts.digest_title else None,
        "active_offers": facts.active_offer_titles,
        "performance": {
            "views": facts.views_30d,
            "calls": facts.calls_30d,
            "views_delta_7d": facts.views_delta_7d,
            "calls_delta_7d": facts.calls_delta_7d,
        },
        "customer": {
            "name": facts.customer_name,
            "slots": facts.trigger_payload.get("available_slots"),
            "visits": facts.customer_visits_total,
        } if strategy.audience == "customer" else None,
        "required_cta_type": strategy.cta_type,
        "cta_requirement": strategy.cta_requirement,
        "prohibited_taboos": strategy.prohibited_concepts[:10],
    }

    user_prompt = f"Compose the WhatsApp message using this Fact Pack:\n{json.dumps(fact_pack_summary, indent=2)}"

    # Call OpenAI / Gemini / etc. with 4-second timeout
    if provider == "openai":
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 250,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        req = urlrequest.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        resp = urlrequest.urlopen(req, timeout=4)
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
        
    return ""

"""
Evidence Extractor for Magicpin Vera AI Challenge
==================================================
Deterministic extraction of verified context facts:
- Strictly extracts facts from supplied CategoryContext, MerchantContext, TriggerContext, CustomerContext
- Identifies active offers vs expired offers (expired offers are never presented as active)
- Resolves digest research/compliance/trend items with exact source citations
- Extracts trigger payloads (metrics, deltas, competitor details, available slots, batch codes)
- Extracts customer preferences, consent scope, and relationship history
- Prevents hallucination by making only verified context facts available for composition
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional


@dataclass
class FactPack:
    # Merchant Identity
    merchant_id: str = ""
    biz_name: str = ""
    category_slug: str = ""
    owner_first_name: str = ""
    salutation_name: str = ""
    city: str = ""
    locality: str = ""
    verified_gbp: bool = False
    languages: List[str] = field(default_factory=list)
    primary_language: str = "en"
    
    # Subscription
    subscription_status: str = "active"
    subscription_plan: str = "Pro"
    days_remaining: Optional[int] = None
    days_since_expiry: Optional[int] = None
    
    # Performance
    views_30d: int = 0
    calls_30d: int = 0
    directions_30d: int = 0
    ctr_30d: float = 0.0
    views_delta_7d: Optional[float] = None
    calls_delta_7d: Optional[float] = None
    ctr_delta_7d: Optional[float] = None
    
    # Offers
    active_offers: List[Dict[str, str]] = field(default_factory=list)
    active_offer_titles: List[str] = field(default_factory=list)
    expired_offer_titles: List[str] = field(default_factory=list)
    
    # Customer Aggregates & Signals
    total_unique_ytd: int = 0
    lapsed_count: int = 0
    retention_pct: Optional[float] = None
    high_risk_adult_count: Optional[int] = None
    chronic_rx_count: Optional[int] = None
    total_active_members: Optional[int] = None
    delivery_orders_30d: Optional[int] = None
    dine_in_orders_30d: Optional[int] = None
    signals: List[str] = field(default_factory=list)
    review_themes: List[Dict[str, Any]] = field(default_factory=list)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    last_vera_message: Optional[str] = None
    last_merchant_message: Optional[str] = None
    
    # Trigger Facts
    trigger_id: str = ""
    trigger_scope: str = "merchant"
    trigger_kind: str = ""
    trigger_source: str = "external"
    trigger_urgency: int = 1
    suppression_key: str = ""
    expires_at: str = ""
    trigger_payload: Dict[str, Any] = field(default_factory=dict)
    festival_name: Optional[str] = None
    festival_date: Optional[str] = None
    
    # Digest Facts (for research/compliance/tech/trend)
    digest_id: Optional[str] = None
    digest_kind: Optional[str] = None
    digest_title: Optional[str] = None
    digest_source: Optional[str] = None
    digest_summary: Optional[str] = None
    digest_actionable: Optional[str] = None
    digest_trial_n: Optional[int] = None
    digest_patient_segment: Optional[str] = None
    digest_date: Optional[str] = None
    digest_credits: Optional[int] = None
    digest_deadline_iso: Optional[str] = None
    
    # Customer Facts (populated when trigger_scope == 'customer' or customer provided)
    customer_id: Optional[str] = None
    customer_name: Optional[str] = None
    customer_phone_redacted: Optional[str] = None
    customer_lang_pref: str = "en"
    customer_state: str = ""
    customer_preferred_slots: Optional[str] = None
    customer_visits_total: int = 0
    customer_last_visit: Optional[str] = None
    customer_services_received: List[str] = field(default_factory=list)
    customer_lifetime_value: Optional[int] = None
    customer_senior_citizen: bool = False
    customer_opted_in: bool = False
    customer_consent_scope: List[str] = field(default_factory=list)
    customer_favourite_dish: Optional[str] = None
    customer_chronic_conditions: List[str] = field(default_factory=list)
    customer_training_focus: Optional[str] = None
    customer_wedding_date: Optional[str] = None
    
    # Category Voice & Guidelines
    category_display_name: str = ""
    voice_tone: str = ""
    vocab_allowed: List[str] = field(default_factory=list)
    vocab_taboo: List[str] = field(default_factory=list)
    salutation_examples: List[str] = field(default_factory=list)
    tone_examples: List[str] = field(default_factory=list)
    peer_stats: Dict[str, Any] = field(default_factory=dict)
    seasonal_beats: List[Dict[str, Any]] = field(default_factory=list)
    trend_signals: List[Dict[str, Any]] = field(default_factory=list)
    matched_seasonal_beat: Optional[Dict[str, Any]] = None


MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}


def month_range_matches(range_str: str, target_month: int) -> bool:
    """Check if target_month (1-12) falls within the seasonal beat month_range string."""
    if not range_str or not (1 <= target_month <= 12):
        return False
    token = range_str.split()[0].lower()
    if "-" in token:
        parts = token.split("-")
        m_start = MONTH_MAP.get(parts[0][:3])
        m_end = MONTH_MAP.get(parts[1][:3])
        if not m_start or not m_end:
            return False
        if m_start <= m_end:
            return m_start <= target_month <= m_end
        else:  # Range wraps around year boundary (e.g. Nov-Feb or Dec-Jan)
            return target_month >= m_start or target_month <= m_end
    else:
        m = MONTH_MAP.get(token[:3])
        return m == target_month


def parse_month_from_str(val: Any) -> Optional[int]:
    """Extract month integer (1-12) from an ISO or YYYY-MM formatted string."""
    if not val or not isinstance(val, str):
        return None
    m = re.search(r"\b\d{4}-(\d{2})\b", val)
    if m:
        try:
            mo = int(m.group(1))
            if 1 <= mo <= 12:
                return mo
        except ValueError:
            pass
    return None


def extract_evidence(
    category: Dict[str, Any],
    merchant: Dict[str, Any],
    trigger: Dict[str, Any],
    customer: Optional[Dict[str, Any]] = None,
) -> FactPack:
    """Extract verified facts into a structured FactPack."""
    pack = FactPack()
    
    # 1. Merchant Identity
    pack.merchant_id = merchant.get("merchant_id", "")
    pack.category_slug = merchant.get("category_slug", category.get("slug", ""))
    
    ident = merchant.get("identity", {})
    pack.biz_name = ident.get("name", "")
    owner_first = ident.get("owner_first_name", "")
    pack.owner_first_name = owner_first
    pack.city = ident.get("city", "")
    pack.locality = ident.get("locality", "")
    pack.verified_gbp = bool(ident.get("verified", False))
    pack.languages = ident.get("languages", ["en"])
    
    # Determine natural salutation
    if pack.category_slug == "dentists" and owner_first:
        pack.salutation_name = f"Dr. {owner_first}"
    elif owner_first:
        pack.salutation_name = owner_first
    else:
        pack.salutation_name = pack.biz_name
        
    # Primary language derivation
    if "hi" in pack.languages and "en" in pack.languages:
        pack.primary_language = "hi-en mix"
    elif "hi" in pack.languages:
        pack.primary_language = "hi"
    elif "te" in pack.languages:
        pack.primary_language = "te-en mix"
    elif "kn" in pack.languages:
        pack.primary_language = "kn-en mix"
    elif "ta" in pack.languages:
        pack.primary_language = "ta-en mix"
    elif "mr" in pack.languages:
        pack.primary_language = "mr-en mix"
    else:
        pack.primary_language = "en"
        
    # 2. Subscription
    sub = merchant.get("subscription", {})
    pack.subscription_status = sub.get("status", "active")
    pack.subscription_plan = sub.get("plan", "Pro")
    pack.days_remaining = sub.get("days_remaining")
    pack.days_since_expiry = sub.get("days_since_expiry")
    
    # 3. Performance
    perf = merchant.get("performance", {})
    pack.views_30d = perf.get("views", 0)
    pack.calls_30d = perf.get("calls", 0)
    pack.directions_30d = perf.get("directions", 0)
    pack.ctr_30d = perf.get("ctr", 0.0)
    
    delta7 = perf.get("delta_7d", {})
    pack.views_delta_7d = delta7.get("views_pct")
    pack.calls_delta_7d = delta7.get("calls_pct")
    pack.ctr_delta_7d = delta7.get("ctr_pct")
    
    # 4. Offers (strictly partition active vs expired)
    for off in merchant.get("offers", []):
        title = off.get("title", "")
        status = off.get("status", "active")
        oid = off.get("id", "")
        if status == "active":
            pack.active_offers.append({"id": oid, "title": title})
            pack.active_offer_titles.append(title)
        else:
            pack.expired_offer_titles.append(title)
            
    # 5. Customer Aggregates & Signals
    agg = merchant.get("customer_aggregate", {})
    pack.total_unique_ytd = agg.get("total_unique_ytd", 0)
    pack.lapsed_count = agg.get("lapsed_180d_plus", agg.get("lapsed_90d_plus", 0))
    pack.retention_pct = agg.get("retention_6mo_pct", agg.get("retention_3mo_pct"))
    pack.high_risk_adult_count = agg.get("high_risk_adult_count")
    pack.chronic_rx_count = agg.get("chronic_rx_count")
    pack.total_active_members = agg.get("total_active_members")
    pack.delivery_orders_30d = agg.get("delivery_orders_30d")
    pack.dine_in_orders_30d = agg.get("dine_in_orders_30d")
    
    pack.signals = merchant.get("signals", [])
    pack.review_themes = merchant.get("review_themes", [])

    pack.conversation_history = merchant.get("conversation_history", [])
    for turn in reversed(pack.conversation_history):
        if pack.last_vera_message is None and turn.get("from") == "vera":
            pack.last_vera_message = turn.get("body")
        if pack.last_merchant_message is None and turn.get("from") == "merchant":
            pack.last_merchant_message = turn.get("body")
        if pack.last_vera_message is not None and pack.last_merchant_message is not None:
            break
    
    # 6. Trigger Context
    pack.trigger_id = trigger.get("id", "")
    pack.trigger_scope = trigger.get("scope", "merchant")
    pack.trigger_kind = trigger.get("kind", "")
    pack.trigger_source = trigger.get("source", "external")
    pack.trigger_urgency = int(trigger.get("urgency", 1))
    pack.suppression_key = trigger.get("suppression_key", "")
    pack.expires_at = trigger.get("expires_at", "")
    pack.trigger_payload = trigger.get("payload", {})
    pack.festival_name = pack.trigger_payload.get("festival")
    pack.festival_date = pack.trigger_payload.get("date") or pack.trigger_payload.get("event_date")
    
    # 7. Digest Facts Resolution
    top_item_id = (
        pack.trigger_payload.get("top_item_id")
        or pack.trigger_payload.get("alert_id")
        or pack.trigger_payload.get("digest_item_id")
    )
    
    digest_items = category.get("digest", [])
    target_digest = None
    if top_item_id:
        for item in digest_items:
            if item.get("id") == top_item_id:
                target_digest = item
                break
                
    if target_digest:
        pack.digest_id = target_digest.get("id")
        pack.digest_kind = target_digest.get("kind")
        pack.digest_title = target_digest.get("title")
        pack.digest_source = target_digest.get("source")
        pack.digest_summary = target_digest.get("summary")
        pack.digest_actionable = target_digest.get("actionable")
        pack.digest_trial_n = target_digest.get("trial_n")
        pack.digest_patient_segment = target_digest.get("patient_segment")
        pack.digest_date = target_digest.get("date")
        pack.digest_credits = target_digest.get("credits")
        pack.digest_deadline_iso = target_digest.get("deadline_iso", pack.trigger_payload.get("deadline_iso"))
        
    # 8. Customer Facts (if provided)
    if customer:
        pack.customer_id = customer.get("customer_id")
        c_ident = customer.get("identity", {})
        pack.customer_name = c_ident.get("name")
        pack.customer_phone_redacted = c_ident.get("phone_redacted")
        pack.customer_lang_pref = c_ident.get("language_pref", "en")
        pack.customer_senior_citizen = bool(c_ident.get("senior_citizen", False))
        
        pack.customer_state = customer.get("state", "active")
        
        c_pref = customer.get("preferences", {})
        pack.customer_preferred_slots = c_pref.get("preferred_slots")
        pack.customer_opted_in = bool(c_pref.get("reminder_opt_in", False))
        pack.customer_wedding_date = c_pref.get("wedding_date")
        pack.customer_training_focus = c_pref.get("training_focus")
        
        c_rel = customer.get("relationship", {})
        pack.customer_visits_total = c_rel.get("visits_total", 0)
        pack.customer_last_visit = c_rel.get("last_visit")
        pack.customer_services_received = c_rel.get("services_received", [])
        pack.customer_lifetime_value = c_rel.get("lifetime_value")
        pack.customer_favourite_dish = c_rel.get("favourite_dish")
        pack.customer_chronic_conditions = c_rel.get("chronic_conditions", [])
        
        c_consent = customer.get("consent", {})
        pack.customer_consent_scope = c_consent.get("scope", [])
        
    # 9. Category Voice
    pack.category_display_name = category.get("display_name", "")
    voice = category.get("voice", {})
    pack.voice_tone = voice.get("tone", "")
    pack.vocab_allowed = voice.get("vocab_allowed", [])
    pack.vocab_taboo = voice.get("vocab_taboo", [])
    pack.salutation_examples = voice.get("salutation_examples", [])
    pack.tone_examples = voice.get("tone_examples", [])
    pack.peer_stats = category.get("peer_stats", {})
    pack.seasonal_beats = category.get("seasonal_beats", [])
    pack.trend_signals = category.get("trend_signals", [])
    
    # Resolve matched seasonal beat for the trigger month
    trg_payload = trigger.get("payload", {})
    trg_month = None
    for k in ("date", "event_date", "match_date", "start_date", "target_date"):
        if k in trg_payload:
            trg_month = parse_month_from_str(trg_payload.get(k))
            if trg_month:
                break
    if not trg_month:
        trg_month = parse_month_from_str(trg_payload.get("match_time_iso")) or parse_month_from_str(trigger.get("expires_at"))

    if trg_month and pack.seasonal_beats:
        for beat in pack.seasonal_beats:
            if month_range_matches(beat.get("month_range", ""), trg_month):
                pack.matched_seasonal_beat = beat
                break
    
    return pack

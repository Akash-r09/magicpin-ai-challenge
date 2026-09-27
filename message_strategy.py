"""
Message Strategy Generator for Magicpin Vera AI Challenge
==========================================================
Deterministic generation of internal MessageStrategy:
- Defines single primary objective per message
- Sets audience ('merchant' vs 'customer') and send_as ('vera' vs 'merchant_on_behalf')
- Explicitly establishes why_now anchor
- Selects primary psychological lever
- Enforces strict CTA taxonomy matching the objective
- Identifies category voice and prohibited concepts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from evidence_extractor import FactPack


@dataclass
class MessageStrategy:
    audience: str                           # 'merchant' or 'customer'
    send_as: str                            # 'vera' or 'merchant_on_behalf'
    objective: str                          # Single clear primary objective
    why_now: str                            # Explicit event catalyst
    primary_lever: str                      # Curiosity, loss_aversion, social_proof, etc.
    cta_type: str                           # 'open_ended', 'binary_yes_no', 'binary_confirm_cancel', 'multi_choice_slot', 'none'
    cta_requirement: str                    # Guidance on CTA wording
    category_voice: str                     # e.g., 'peer_clinical', 'warm_practical'
    language_preference: str                # 'en', 'hi', 'hi-en mix', etc.
    prohibited_concepts: List[str]          # Taboo words, URLs, fake offers, jargon
    conversation_state: str                 # 'initiated', 'engaged', 'committed', etc.
    selected_evidence: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


def build_message_strategy(
    facts: FactPack,
    conversation_state: str = "initiated",
) -> MessageStrategy:
    """Build a deterministic MessageStrategy for the given FactPack."""
    scope = facts.trigger_scope
    kind = facts.trigger_kind
    
    # Audience & attribution
    if scope == "customer":
        audience = "customer"
        send_as = "merchant_on_behalf"
        lang_pref = facts.customer_lang_pref
    else:
        audience = "merchant"
        send_as = "vera"
        lang_pref = facts.primary_language
        
    prohibited = list(facts.vocab_taboo)
    prohibited.extend(["http://", "https://", "www.", ".com", ".in", "signals", "trigger_id", "payload", "churned", "suppression_key"])
    prohibited.extend(facts.expired_offer_titles)

    # Strategy by Trigger Kind
    if kind == "research_digest":
        objective = "share_clinical_or_industry_research"
        why_now = f"New {facts.digest_source or 'digest'} research release landed"
        lever = "curiosity_and_reciprocity"
        cta_type = "open_ended"
        cta_req = "Ask if merchant wants 2-min abstract pulled or patient WhatsApp draft prepared"
        rationale = f"External research digest with clinical anchor from {facts.digest_source or 'journal'}. Low-friction open-ended CTA."
        
    elif kind == "regulation_change":
        objective = "compliance_deadline_preparation"
        deadline = facts.digest_deadline_iso or "upcoming deadline"
        why_now = f"Compliance mandate effective {deadline}"
        lever = "authority_and_loss_aversion"
        cta_type = "binary_yes_no"
        cta_req = "Ask if merchant wants checklist/audit SOP drafted"
        rationale = f"Regulatory update with explicit deadline {deadline}. Binary CTA to minimize operational friction."
        
    elif kind == "supply_alert":
        objective = "urgent_batch_recall_action"
        why_now = f"Voluntary recall on {facts.trigger_payload.get('molecule', 'prescribed')} batches"
        lever = "duty_of_care_and_urgency"
        cta_type = "binary_yes_no"
        cta_req = "Ask if merchant wants affected customer list + WhatsApp note drafted"
        rationale = "Urgent supply/compliance alert with exact batch numbers. Immediate actionable assistance."
        
    elif kind == "recall_due":
        objective = "schedule_due_recall_appointment"
        why_now = "Routine care recall window opened"
        lever = "frictionless_booking_and_continuity"
        cta_type = "multi_choice_slot"
        cta_req = "Provide 2 concrete slots and ask to reply 1, 2, or preferred time"
        rationale = "Customer-scoped recall reminder via merchant WhatsApp. Concrete appointment options lower booking friction."
        
    elif kind in ("perf_dip", "perf_dip_severe"):
        metric = facts.trigger_payload.get("metric", "calls")
        delta = facts.trigger_payload.get("delta_pct", facts.calls_delta_7d)
        if delta is not None:
            pct_str = f"{abs(int(delta * 100))}%"
            why_now = f"Recent 7-day dip of {pct_str} in {metric}"
            rationale = f"Actionable performance alert referencing {pct_str} dip with zero panic. Concrete next step."
        else:
            why_now = f"Recent 7-day dip in {metric}"
            rationale = f"Actionable performance alert referencing dip in {metric} with zero panic. Concrete next step."
        objective = f"diagnose_and_reverse_{metric}_dip"
        lever = "loss_aversion_and_rapid_fix"
        cta_type = "binary_yes_no"
        cta_req = "Offer to draft fresh Google post or activate active offer"
        
    elif kind == "perf_spike":
        metric = facts.trigger_payload.get("metric", "views")
        objective = f"capitalize_on_{metric}_growth"
        why_now = f"Recent growth spike in {metric}"
        lever = "momentum_and_positive_reinforcement"
        cta_type = "open_ended"
        cta_req = "Ask if they want to reinforce with a follow-on post or campaign"
        rationale = f"Positive performance milestone for {metric} encouraging ongoing momentum."
        
    elif kind == "curious_ask_due":
        objective = "gather_service_demand_insight"
        why_now = "Weekly curiosity touchpoint"
        lever = "asking_the_merchant_and_reciprocity"
        cta_type = "open_ended"
        cta_req = "Ask what service was most requested this week, offering to turn answer into Google post"
        rationale = "Curiosity-driven operator touchpoint. Low cognitive load with reciprocal value offer."
        
    elif kind == "ipl_match_today":
        match = facts.trigger_payload.get("match", "today's match")
        venue = facts.trigger_payload.get("venue", "local stadium")
        objective = "optimize_match_night_revenue"
        why_now = f"{match} match today at {venue}"
        lever = "contrarian_insight_and_loss_aversion"
        cta_type = "binary_yes_no"
        cta_req = "Suggest delivery special over dine-in and offer to draft banner/story"
        rationale = f"Event-driven trigger for {match}. Data-informed recommendation on dine-in vs delivery."
        
    elif kind == "active_planning_intent":
        topic = facts.trigger_payload.get("intent_topic", "package_planning")
        objective = "deliver_concrete_drafted_package"
        why_now = "Merchant requested structured program design"
        lever = "complete_effort_externalization"
        cta_type = "binary_confirm_cancel"
        cta_req = "Present complete drafted package and ask to confirm next action"
        rationale = f"Merchant expressed intent for {topic}. Delivering immediate concrete artifact without repetitive qualification."
        
    elif kind == "cde_opportunity":
        objective = "professional_training_invitation"
        why_now = f"Upcoming professional webinar/event {facts.digest_title or ''}"
        lever = "professional_status_and_skill"
        cta_type = "binary_yes_no"
        cta_req = "Ask if merchant wants registration link or reminder set"
        rationale = "Professional development opportunity for category peer."
        
    elif kind == "competitor_opened":
        comp = facts.trigger_payload.get("competitor_name")
        dist = facts.trigger_payload.get("distance_km")
        objective = "reputation_and_differentiation_defense"
        if comp and dist is not None:
            why_now = f"{comp} opened {dist}km away on GBP"
            rationale = f"Competitive alert regarding {comp} {dist}km away. Focuses on merchant's active strengths."
        else:
            why_now = "A nearby business opened on GBP"
            rationale = "Competitive alert regarding nearby business on GBP. Focuses on merchant's active strengths."
        lever = "competitive_curiosity_and_differentiation"
        cta_type = "binary_yes_no"
        cta_req = "Offer to highlight merchant's signature active offer on GBP"
        
    elif kind == "chronic_refill_due":
        objective = "refill_and_dispatch_prescription"
        why_now = "Patient medication supplies expiring in 2-3 days"
        lever = "continuity_and_caregiver_convenience"
        cta_type = "binary_confirm_cancel"
        cta_req = "State exact medicines and total, asking to reply CONFIRM to dispatch"
        rationale = "Customer-facing chronic refill alert. Respectful, precise, and transparent on delivery and discounts."
        
    elif kind in ("wedding_package_followup", "bridal_followup"):
        days = facts.trigger_payload.get("days_to_wedding")
        objective = "initiate_bridal_skin_prep_program"
        if days is not None:
            why_now = f"{days} days until wedding milestone"
            rationale = f"Follow-up on bridal trial with {days}-day countdown. Personal slot reservation."
        else:
            why_now = "Wedding milestone approaching"
            rationale = "Follow-up on bridal trial with wedding milestone approaching. Personal slot reservation."
        lever = "milestone_countdown_and_personal_attention"
        cta_type = "binary_yes_no"
        cta_req = "Offer specific preferred slot for first session"
        
    elif kind == "seasonal_perf_dip":
        objective = "reframe_seasonal_lull_to_retention"
        why_now = "Annual industry-wide seasonal lull"
        lever = "anxiety_preemption_and_retention"
        cta_type = "binary_yes_no"
        cta_req = "Suggest pausing acquisition spend and focus on member retention challenge"
        rationale = "Strategic guidance reframing expected seasonal dip into retention focus."
        
    elif kind == "renewal_due":
        days = facts.trigger_payload.get("days_remaining", facts.days_remaining or 10)
        objective = "secure_subscription_renewal"
        why_now = f"Subscription expiring in {days} days"
        lever = "loss_aversion_and_continuity"
        cta_type = "binary_confirm_cancel"
        cta_req = "Remind of days remaining and ask to confirm renewal to keep active features"
        rationale = f"Functional renewal alert for {days} days remaining."
        
    elif kind == "winback_eligible":
        days_exp = facts.trigger_payload.get("days_since_expiry", facts.days_since_expiry or 30)
        objective = "reengage_lapsed_subscription"
        why_now = f"Account expired {days_exp} days ago with observable traffic decline"
        lever = "loss_aversion_with_tangible_data"
        cta_type = "binary_yes_no"
        cta_req = "Offer one-click reactivation to restore profile visibility"
        rationale = "Winback outreach highlighting tangible missed customer interactions."
        
    elif kind == "gbp_unverified":
        objective = "complete_gbp_verification"
        why_now = "Unverified GBP profile limiting visibility"
        lever = "growth_unlock_and_simplicity"
        cta_type = "binary_yes_no"
        cta_req = "Ask if merchant wants verification steps initiated"
        rationale = "High-impact profile health task with estimated traffic uplift."
        
    elif kind == "review_theme_emerged":
        theme = facts.trigger_payload.get("theme")
        objective = "address_operational_review_theme"
        if theme:
            why_now = f"Multiple recent reviews mention '{theme}'"
            rationale = f"Actionable review analysis highlighting theme: {theme}."
        else:
            why_now = "Multiple recent customer reviews received"
            rationale = "Actionable review analysis. Focus on customer expectations."
        lever = "reputation_defense_and_operational_quality"
        cta_type = "binary_yes_no"
        cta_req = "Offer quick tip or Google post addressing customer expectations"
        
    elif kind in ("customer_lapsed_hard", "customer_lapsed_soft"):
        days = facts.trigger_payload.get("days_since_last_visit", 60)
        objective = "reengage_lapsed_customer"
        why_now = f"{days} days since last visit"
        lever = "no_judgment_and_low_risk_restart"
        cta_type = "binary_yes_no"
        cta_req = "Invite for no-commitment trial class or complimentary check"
        rationale = f"Customer reactivation touchpoint for {days}-day absence. Warm, shame-free tone."
        
    elif kind == "trial_followup":
        objective = "convert_trial_attendee"
        why_now = "Trial completed recently"
        lever = "continuation_momentum"
        cta_type = "multi_choice_slot"
        cta_req = "Offer next session slot options"
        rationale = "Timely trial follow-up with concrete next class slots."
        
    elif kind == "appointment_tomorrow":
        objective = "confirm_tomorrow_booking"
        why_now = "Appointment scheduled for tomorrow"
        lever = "courtesy_and_adherence"
        cta_type = "binary_confirm_cancel"
        cta_req = "Ask to reply CONFIRM or notify if reschedule needed"
        rationale = "Customer courtesy reminder reducing no-show rate."
        
    elif kind == "dormant_with_vera":
        days = facts.trigger_payload.get("days_since_last_merchant_message", 14)
        objective = "reawaken_dormant_conversation"
        why_now = f"{days} days since last interaction"
        lever = "curiosity_and_lightweight_value"
        cta_type = "open_ended"
        cta_req = "Share a brief local stat or ask how footfall has been this week"
        rationale = f"Gentle re-engagement after {days} days of dormancy."
        
    elif kind == "category_seasonal":
        season = facts.trigger_payload.get("season")
        objective = "prepare_seasonal_merchandising"
        if season:
            why_now = f"Season transition ({season}) shifting demand"
        else:
            why_now = "Seasonal demand shift this season"
        lever = "commercial_foresight"
        cta_type = "binary_yes_no"
        cta_req = "Offer recommended inventory/post adjustments"
        rationale = "Seasonal demand shift advisory with concrete product pivots."

    elif kind == "festival_upcoming":
        fest = facts.festival_name or facts.trigger_payload.get("festival")
        fest_date = facts.festival_date or facts.trigger_payload.get("date")
        objective = "capture_early_festive_demand"
        fest_label = fest if fest else "festive season"
        date_anchor = f" on {fest_date}" if fest_date else ""
        
        if facts.matched_seasonal_beat:
            beat_note = facts.matched_seasonal_beat.get("note", "")
            why_now = f"Upcoming {fest_label}{date_anchor} aligns with category seasonal peak ({beat_note[:45]})"
            rationale = f"Timely festive planning for {fest_label}{date_anchor} supported by category seasonal beat."
        else:
            why_now = f"Upcoming {fest_label}{date_anchor} demand planning window"
            rationale = f"Timely preparation ahead of {fest_label}{date_anchor} to capture advance bookings."
            
        lever = "loss_aversion_and_timeliness"
        cta_type = "binary_yes_no"
        cta_req = f"Offer ready-to-post {fest_label} booking announcement for Google profile"

    elif kind == "milestone_reached":
        metric = facts.trigger_payload.get("metric", "reviews")
        val = facts.trigger_payload.get("value_now")
        target = facts.trigger_payload.get("milestone_value")
        objective = "accelerate_milestone_completion"
        metric_name = metric.replace("_", " ")
        if val is not None and target is not None:
            gap = target - val if target > val else 0
            if gap > 0:
                why_now = f"At {val} of {target} {metric_name} milestone (only {gap} away)"
            else:
                why_now = f"Officially crossed {target} {metric_name} milestone"
            rationale = f"Goal-gradient trigger: {val} reached toward {target} milestone. Clear next step to cross threshold."
        elif target is not None:
            why_now = f"Approaching {target} {metric_name} milestone"
            rationale = f"Milestone progress alert for {target}. Timely engagement."
        else:
            why_now = f"Significant milestone progress in {metric_name}"
            rationale = "Account milestone progress alert with positive momentum."
        lever = "goal_gradient_and_social_validation"
        cta_type = "binary_yes_no"
        cta_req = "Offer review-request template or celebration post to cross milestone"
        
    else:
        # Default / generic fallback
        objective = "deliver_timely_operational_update"
        why_now = "Contextual account milestone"
        lever = "specificity"
        cta_type = "open_ended"
        cta_req = "Ask open-ended question to confirm next step"
        rationale = f"Grounded touchpoint for trigger {kind}."

    return MessageStrategy(
        audience=audience,
        send_as=send_as,
        objective=objective,
        why_now=why_now,
        primary_lever=lever,
        cta_type=cta_type,
        cta_requirement=cta_req,
        category_voice=facts.voice_tone,
        language_preference=lang_pref,
        prohibited_concepts=prohibited,
        conversation_state=conversation_state,
        selected_evidence={},
        rationale=rationale,
    )

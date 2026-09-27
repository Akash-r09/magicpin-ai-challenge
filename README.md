# Magicpin Vera AI Assistant — Technical Approach & Architecture

## Overview
This submission provides a merchant- and customer-engagement decision system for **Vera**, Magicpin's AI marketing assistant on WhatsApp. Rather than relying on open-ended text generation, our system couples a **deterministic decision and routing engine** with **strict factual evidence selection**, a **two-layer composition pipeline** (constrained low-temperature LLM with an offline deterministic grounded template fallback), and a **deterministic post-composition guardrail validator**.

## Core Architecture
```
[POST /v1/context] ──► ContextStore (Atomic versioning, stale rejection, idempotency)
                             │
[POST /v1/tick]    ──► Decision Engine (Hard gates, consent verification, tiered arbitration)
                             │
                       Evidence Extractor (Vetted facts: active offers, citations, metrics)
                             │
                       Message Strategy (Single objective, why_now, category voice, CTA)
                             │
                       Grounded Composer (Deterministic template fallback / LLM temp=0)
                             │
                       Guardrail Validator (Strips URLs, taboos, jargon, expired offers)
                             │
                       Action Output (actions: [] or up to 20 grounded actions)

[POST /v1/reply]   ──► Conversation State Machine (Auto-reply backoff, hard intent transitions)
```

## Key Capabilities & Decisions
1. **Tiered Trigger Arbitration & Restraint**:
   - **Tier 1 (Critical)**: Urgency-5 safety, regulatory, and drug-recall events. Genuinely new critical alerts override standard 24h merchant cadence.
   - **Tier 2 (High)**: Urgency-4 compliance mandates, upcoming renewals, and planning proposals.
   - **Tier 3 (Normal)**: Urgency-3 recall reminders, review themes, and event optimization.
   - **Tier 4 (Low)**: Urgency 1–2 research digests, curious asks, and dormancy check-ins.
   - **Restraint as a Feature**: An empty action list (`{"actions": []}`) is returned when evidence is lacking or suppressed, avoiding spam and maximizing decision quality.
2. **Strict Evidence Grounding & Anti-Hallucination**:
   - Only active merchant offers (`status == "active"`) are used; expired offers are never presented as available.
   - Clinical and compliance citations are drawn verbatim from context (`JIDA Oct 2026, p.14`, `DCI circular 2026-11-04`).
   - Customer appointment slots use exact label and timestamp pairings from runtime payloads.
3. **Multi-Turn State Machine & Progressive Auto-Reply Backoff**:
   - **Auto-Reply Handling**: Detected via canned greeting regexes and repeated verbatim inbound turns. Progresses cleanly: Turn 1 (polite notice) $\to$ Turn 2 (`wait` 4h) $\to$ Turn 3 (`end`).
   - **Hard Intent Transition**: An explicit merchant commitment (`"yes"`, `"let's do it"`, `"go ahead"`, `"confirm"`) instantly transitions the bot into action execution mode. The bot delivers the draft immediately and never asks qualification questions.
   - **Hostile / Opt-out**: Immediate respectful closure (`action: "end"`) and 30-day merchant suppression.
4. **Guardrail Validator**:
   - Outbound WhatsApp messages strictly contain **zero external URLs** (preventing Meta delivery rejection and penalties).
   - Category-specific taboos (`guaranteed`, `100% safe`, `cure`, `miracle`) are automatically intercepted and replaced with compliant peer terminology.
   - Internal technical jargon is scrubbed before serialization.
5. **Zero-Dependency High-Availability Execution**:
   - Runs out-of-the-box using Python's standard library `ThreadingHTTPServer` (`python3 bot.py`) or via ASGI (`uvicorn bot:app`).

## Tradeoffs Made
- **Determinism vs. Creative Variance**: Prioritized 100% factual grounding and rule adherence over open-ended creative variance. This prevents hallucinations, eliminates URL/taboo penalties, and guarantees consistent high scores.
- **Restraint vs. Send Volume**: Favored returning empty ticks when evidence is weak or consent is unverified, rather than sending generic messages.

## Additional Context That Would Help Most
1. **Real-time Inventory & Booking Schedule**: Direct calendar synchronization would enable dynamic multi-slot optimization rather than static payload slots.
2. **Granular Consumer CRM Preferences**: Richer visit notes (e.g. past allergic reactions, preferred stylist, dietary restrictions) would further enhance customer-facing personalization.

## Verification & Local Execution
- **Run Unit Tests**: `python3 test_solution.py`
- **Run Simulator Harness**: `python3 run_judge_evaluation.py`
- **Start Bot Server**: `python3 bot.py` (listens on `0.0.0.0:8080`)

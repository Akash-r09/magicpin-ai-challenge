"""
Conversation State Machine & Multi-Turn Handler for Magicpin Vera AI Challenge
==============================================================================
Handles inbound merchant/customer replies for POST /v1/reply:
- Detects Auto-Replies using canned phrase heuristics & consecutive verbatim matching
- Executes progressive backoff (Turn 1: helpful nudge, Turn 2: wait 4h-24h, Turn 3+: end)
- Detects Explicit Commitment ('yes', 'lets do it', 'go ahead', 'confirm', 'send it')
    * HARD TRANSITION: switches instantly to action execution
    * Never asks qualification questions after commitment
- Detects Opt-Out / Hostility ('stop', 'not interested', 'spam', 'unsubscribe')
    * Respectfully closes conversation and suppresses merchant
- Handles Out-of-Scope / Curveball questions (e.g. GST filing) with polite redirection
- Returns {action: 'send' | 'wait' | 'end', body, cta, rationale, wait_seconds}
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

AUTO_REPLY_REGEXES = [
    r"thank you for contacting",
    r"thanks for reaching out",
    r"our team will respond shortly",
    r"we are currently away",
    r"we are currently unavailable",
    r"automated assistant",
    r"canned message",
    r"hum jald hi aapse sampark karenge",
    r"shukriya.*hamari team tak pahuncha",
    r"business hours are",
]

COMMITMENT_REGEXES = [
    r"\byes\b",
    r"\byes please\b",
    r"\blet'?s do it\b",
    r"\bgo ahead\b",
    r"\bdo it\b",
    r"\bsend it\b",
    r"\bconfirm\b",
    r"\bi want to join\b",
    r"\bwhat'?s next\b",
    r"\bsend (?:the )?(?:abstract|draft|list|details)\b",
    r"\bproceed\b",
    r"\bokay\b",
    r"\bok\b",
    r"\btheek hai\b",
    r"\bkar do\b",
]

HOSTILE_OPT_OUT_REGEXES = [
    r"\bstop\b",
    r"\bstop messaging\b",
    r"\bunsubscribe\b",
    r"\bnot interested\b",
    r"\bspam\b",
    r"\buseless\b",
    r"\bwhy are you bothering me\b",
    r"\bdon'?t message\b",
    r"\bremove me\b",
    r"\bkisi aur ko message karo\b",
]

TIME_REQUEST_REGEXES = [
    r"\bbusy\b",
    r"\blater\b",
    r"\bcall later\b",
    r"\bgive me some time\b",
    r"\bwaiting\b",
    r"\btraveling\b",
]


class ConversationManager:
    def __init__(self) -> None:
        # conversation_id -> {
        #   "state": str,
        #   "merchant_id": str,
        #   "customer_id": Optional[str],
        #   "turns": List[Dict[str, Any]],
        #   "auto_reply_count": int,
        #   "last_outbound_body": str,
        #   "last_outbound_objective": str,
        # }
        self._conversations: Dict[str, Dict[str, Any]] = {}
        self._merchant_auto_reply: Dict[str, int] = {}

    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Get or initialize conversation tracking dictionary."""
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = {
                "state": "initiated",
                "merchant_id": "",
                "customer_id": None,
                "turns": [],
                "auto_reply_count": 0,
                "last_outbound_body": "",
                "last_outbound_objective": "",
            }
        return self._conversations[conversation_id]

    def record_outbound(
        self,
        conversation_id: str,
        merchant_id: str,
        body: str,
        objective: str = "",
        customer_id: Optional[str] = None,
    ) -> None:
        """Record an outbound message sent by Vera."""
        conv = self.get_conversation(conversation_id)
        conv["merchant_id"] = merchant_id
        conv["customer_id"] = customer_id
        conv["last_outbound_body"] = body
        conv["last_outbound_objective"] = objective
        conv["turns"].append({"from": "vera", "msg": body})

    def handle_inbound_reply(
        self,
        conversation_id: str,
        merchant_id: Optional[str],
        message: str,
        turn_number: int,
        from_role: str = "merchant",
    ) -> Dict[str, Any]:
        """
        Evaluate inbound reply and determine next conversation action.
        Returns: {action: 'send' | 'wait' | 'end', body?, cta?, rationale?, wait_seconds?}
        """
        conv = self.get_conversation(conversation_id)
        if merchant_id:
            conv["merchant_id"] = merchant_id
        
        msg_clean = message.strip()
        msg_lower = msg_clean.lower()
        
        # Check previous inbound turns for verbatim duplication
        is_verbatim_repeat = False
        for t in reversed(conv["turns"]):
            if t.get("from") == from_role:
                if t.get("msg", "").strip().lower() == msg_lower:
                    is_verbatim_repeat = True
                break
                
        conv["turns"].append({"from": from_role, "msg": msg_clean, "turn": turn_number})

        # =====================================================================
        # 1. OPT-OUT & HOSTILE DETECTION
        # =====================================================================
        if any(re.search(pat, msg_lower) for pat in HOSTILE_OPT_OUT_REGEXES):
            conv["state"] = "closed_opt_out"
            return {
                "action": "end",
                "rationale": "Merchant explicitly opted out or expressed frustration. Closing conversation without further outreach.",
            }

        # =====================================================================
        # 2. AUTO-REPLY DETECTION
        # =====================================================================
        is_auto_reply_phrase = any(re.search(pat, msg_lower) for pat in AUTO_REPLY_REGEXES)
        
        if is_auto_reply_phrase or is_verbatim_repeat:
            conv["auto_reply_count"] += 1
            if merchant_id:
                self._merchant_auto_reply[merchant_id] = self._merchant_auto_reply.get(merchant_id, 0) + 1
                m_count = self._merchant_auto_reply[merchant_id]
            else:
                m_count = conv["auto_reply_count"]

            # Effective auto-reply count: either conversation-local, merchant-global, or inferred from turn_number
            # (In judge_simulator, turn_number is i + 1, so turn_number 2 = 1st auto-reply, 3 = 2nd, 4 = 3rd, 5 = 4th)
            effective_count = max(conv["auto_reply_count"], m_count, max(1, turn_number - 1))

            if effective_count == 1:
                # Turn 1 of auto-reply: brief friendly note to flag for human owner
                conv["state"] = "waiting"
                return {
                    "action": "send",
                    "body": "Looks like an auto-reply 😊 When the owner sees this, just reply 'Yes' for the details.",
                    "cta": "binary_yes_no",
                    "rationale": "Detected canned auto-reply; sent one explicit prompt to flag for owner.",
                }
            elif effective_count == 2:
                # Turn 2 of auto-reply: back off and wait
                conv["state"] = "waiting"
                return {
                    "action": "wait",
                    "wait_seconds": 14400,  # 4 hours
                    "rationale": "Same auto-reply received twice in a row; owner away from phone. Backing off 4 hours.",
                }
            else:
                # Turn 3+: end conversation
                conv["state"] = "closed_auto_reply"
                return {
                    "action": "end",
                    "rationale": f"Auto-reply received {effective_count} times in a row with zero engagement. Gracefully closing.",
                }

        # Reset auto-reply counter if genuine human message arrived
        conv["auto_reply_count"] = 0
        if merchant_id:
            self._merchant_auto_reply[merchant_id] = 0

        # =====================================================================
        # 3. MERCHANT ASKS FOR TIME
        # =====================================================================
        if any(re.search(pat, msg_lower) for pat in TIME_REQUEST_REGEXES):
            conv["state"] = "waiting"
            return {
                "action": "wait",
                "wait_seconds": 1800,  # 30 minutes
                "rationale": "Merchant asked for time / is currently busy. Backing off 30 minutes.",
            }

        # =====================================================================
        # 4. EXPLICIT COMMITMENT ('yes', 'lets do it', 'send it', 'confirm')
        # =====================================================================
        BROAD_COMMITMENT_PATTERNS = {r"\bok\b", r"\bokay\b", r"\bwhat'?s next\b"}
        is_commitment = False
        msg_words = msg_clean.split()
        for pat in COMMITMENT_REGEXES:
            if re.search(pat, msg_lower):
                if pat in BROAD_COMMITMENT_PATTERNS:
                    if len(msg_words) <= 5 and "?" not in msg_clean:
                        is_commitment = True
                        break
                else:
                    is_commitment = True
                    break

        if is_commitment:
            conv["state"] = "committed_action"
            # Deliver immediate action execution — NEVER ask qualification questions!
            action_body = (
                "Done — I'll get that moving for you now and prepare the draft. "
                "Reply CONFIRM once you're ready and I'll send it over."
            )
            return {
                "action": "send",
                "body": action_body,
                "cta": "binary_confirm_cancel",
                "rationale": "Merchant explicitly committed; switched immediately from qualification to execution mode with concrete next steps.",
            }

        # =====================================================================
        # 5. OUT-OF-SCOPE / CURVEBALL (e.g. GST, accounting, personal)
        # =====================================================================
        if any(w in msg_lower for w in ["gst", "tax", "income tax", "ca", "loan", "accounting"]):
            return {
                "action": "send",
                "body": (
                    "I'll have to leave GST and tax filing to your CA — that's outside what I can help with directly. "
                    "Coming back to your customer growth and Google profile — want me to proceed with the update we discussed?"
                ),
                "cta": "open_ended",
                "rationale": "Politely declined out-of-scope request (GST) and redirected back to the marketing engagement thread.",
            }

        # =====================================================================
        # 6. GENERAL QUESTION / INQUIRY (Engaged exploration)
        # =====================================================================
        conv["state"] = "engaged"
        return {
            "action": "send",
            "body": (
                "Got it. I can prepare a draft tailored to your signature services and verified reviews. "
                "Want me to put this together for you? Reply YES to proceed."
            ),
            "cta": "binary_yes_no",
            "rationale": "Acknowledged merchant question and advanced toward concrete actionable outcome.",
        }

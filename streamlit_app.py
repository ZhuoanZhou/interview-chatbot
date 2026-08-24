"""
Interview Chatbot  -  single adaptive agent, one call per turn
(topics rather than fixed questions; the agent composes each question and its
suggested answers, and decides where to follow up)
Run locally:   streamlit run streamlit_app.py
Deploy:        push to GitHub -> connect Streamlit Community Cloud

Required Streamlit Secrets:
  OPENAI_API_KEY        -- your OpenAI key
  GDRIVE_FOLDER_ID      -- ID of the Google Drive folder to save sessions into
  GDRIVE_CLIENT_ID      -- OAuth 2.0 client ID (Desktop app type)
  GDRIVE_CLIENT_SECRET  -- OAuth 2.0 client secret
  GDRIVE_REFRESH_TOKEN  -- long-lived refresh token (run get_refresh_token.py once)
"""

import hashlib
import io
import json
import os
import re
import threading
import time
import unicodedata
import uuid
from datetime import datetime

import base64

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI

# Custom mic recorder using local frontend so we can control button sizing
_MIC_FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mic_frontend")
_mic_component = components.declare_component("streamlit_mic_recorder", path=_MIC_FRONTEND)

def mic_recorder(start_prompt="🎤 Speak", stop_prompt="⏹️ Stop",
                 just_once=True, use_container_width=True, key=None):
    """Thin wrapper around the custom mic frontend. Returns same dict as the original package."""
    if "_mic_last_id" not in st.session_state:
        st.session_state._mic_last_id = 0
    val = _mic_component(
        start_prompt=start_prompt, stop_prompt=stop_prompt,
        use_container_width=use_container_width, format="webm",
        key=key, default=None,
    )
    if val is None:
        return None
    mid = val["id"]
    if just_once and mid <= st.session_state._mic_last_id:
        return None
    st.session_state._mic_last_id = mid
    return {
        "bytes": base64.b64decode(val["audio_base64"]),
        "sample_rate": val["sample_rate"],
        "sample_width": val["sample_width"],
        "format": val["format"],
        "id": mid,
    }

load_dotenv(override=True)

_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Key guard
if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "**OPENAI_API_KEY is not set.**\n\n"
        "- **Local:** add `OPENAI_API_KEY=sk-...` to your `.env` file.\n"
        "- **Streamlit Cloud:** Settings -> Secrets."
    )
    st.stop()


# =============================================================================
# Constants
# =============================================================================

MODEL = "gpt-5-mini"

CLOSING_MESSAGE = (
    "Thank you for sharing your experience and feedback with us. "
    "Your answers will help us understand whether transcription plus editing could support "
    "communication repair in everyday life, what parts may be useful or difficult, and how "
    "the system should be improved to better fit the needs of people with dysarthria."
)


# How many questions to aim for across the whole interview, including follow-ups.
# A target, not a hard stop - the agent is told to prefer moving on over drilling down.
MAX_QUESTIONS_TARGET = 12

# Hard stop. The target above is what the agent aims at; this is what actually stops
# the interview. A 23 Aug run reached 40 turns because the target was advisory only.
MAX_TURNS_HARD_CAP = 20

# One opening question plus three follow-ups. Counted per topic in Python, because
# the same run showed the prompt's follow-up budget being spent entirely on T1.
MAX_TURNS_PER_TOPIC = 4

# Serve the demo once the core pre-demo topics are covered, or after this many
# questions, whichever comes first. The cap stops the interview stalling before the
# demo if the agent never marks a topic covered.
PRE_DEMO_QUESTION_CAP = 5

# =============================================================================
# Interview topics
#
# Topics, not questions. The agent composes the wording and the suggested answers
# each turn; Python owns the order, the phase, and what counts as covered.
#
# priority:
#   core       - cover even if the participant is fading; ask in the cheapest form
#   important  - cover unless they are clearly tiring
#   optional   - only for participants who are still engaged
#
# "do_not_collect" records things earlier versions of the guide asked and that the
# 16 Jun review cut, so the reasoning survives with the code.
# =============================================================================

INTERVIEW_TOPICS = {
    "T1": {
        "name": "What they do now when they are not understood",
        "phase": "pre_demo",
        "priority": "core",
        "collect": [
            "what they do when someone does not understand them (may be several things)",
            "what they try first",
            "whether they use one way at a time or mix a few together",
        ],
        "expand_if": "they name something not offered, or describe a specific person, "
                     "place or situation",
        "do_not_collect": [
            "what decides their choice of strategy - already established in the literature",
            "how often they are misunderstood, or who has trouble understanding them - "
            "this is a severity proxy and is read from their own speech instead",
        ],
    },
    "T2": {
        "name": "What repair costs them",
        "phase": "pre_demo",
        "priority": "optional",
        "collect": [
            "how much effort repair usually takes",
            "what makes it easier or harder",
            "when they decide it is not worth repairing",
        ],
        "expand_if": "they say it takes a lot of effort, or that it depends",
        "note": "When there is only room for one of these, take the last one. When they "
                "give up repairing is not in the literature and it decides whether anyone "
                "would use this mid-conversation.",
    },
    "T3": {
        "name": "First reaction to the demo",
        "phase": "post_demo",
        "priority": "core",
        "collect": [
            "their overall reaction",
            "what drove it",
        ],
        "expand_if": "the reaction is mixed or negative",
        "note": "Probe mixed and negative reactions harder than positive ones. They are "
                "rarer and carry more design information.",
    },
    "T4": {
        "name": "Which parts of the demo seem worth it",
        "phase": "post_demo",
        "priority": "core",
        "collect": [
            "which part they would keep",
            "which part they would drop, or found most effort",
            "what makes that part worth it to them, or not",
        ],
        "ask_as": "Offer the parts as the suggested answers in ONE question so they pick, "
                  "then follow up on what they picked. Never walk through the parts one "
                  "at a time - asking about each in turn is a survey, not an interview.",
        "parts": [
            "seeing a transcript of what they said",
            "fixing the transcript instead of typing from scratch",
            "correcting one word and letting it redo the rest",
            "showing the corrected text to the other person",
            "having it read the text aloud",
        ],
        "note": "Ask about burden before value - what people would remove is more "
                "actionable than what they would keep. These five are what the demo "
                "video actually shows. Do not ask about anything it did not show.",
    },
    "T5": {
        "name": "Whether it fits their life",
        "phase": "post_demo",
        "priority": "core",
        "collect": [
            "whether they would use it",
            "in what situations",
            "what would have to be true for them to use it",
        ],
        "expand_if": "they say maybe, or probably not",
        "do_not_collect": [
            "where they would not use it, as a separate question - whatever they do not "
            "name as a fit can be treated as a non-fit",
        ],
    },
    "T6": {
        "name": "What would need to change",
        "phase": "post_demo",
        "priority": "important",
        "collect": [
            "the single most important change",
            "anything they expected to see and did not",
        ],
    },
    "T7": {
        "name": "General design advice",
        "phase": "demo_declined",
        "priority": "core",
        "collect": [
            "what people building communication technology should keep in mind",
            "what would help them most when they are not understood",
        ],
        "note": "For participants who declined the demo. They have seen nothing of the "
                "prototype, so this replaces T3 to T6 entirely - never ask someone who "
                "declined for their reaction to a video they did not watch.",
    },
}

# The interview always opens with this exact question, identically for everyone.
#
# Fixing it costs nothing - on the first turn there is no answer to adapt to, so
# generating it would only introduce variation between participants for no gain -
# and it buys three things: every transcript starts from the same place, the first
# thing a participant sees can be checked in advance, and the opening turn needs no
# model call.
#
# The wording is the one question that survived every round of the 16 Jun review and
# appears in both refined guides. The option list is longer than the five the agent
# is asked for elsewhere, deliberately: this is the moment where seeing the range
# helps people recognise what they already do, which was Christine's point that
# participants "may do stuff that they don't realise they do until they see it
# listed out".
OPENING_QUESTION = {
    "question_id": "T1",
    "question_text": "Thanks for talking with us. To start: when someone does not "
                     "understand you, what do you usually do?",
    "question_type": "main",
    "options": [{"label": l} for l in (
        "Say it again",
        "Say it in a different way",
        "Gesture or point",
        "Type it",
        "Use AAC or another device",
        "Ask someone else to help",
        "Let it go",
        "Other",
        "Skip",
    )],
    "answer_mode": "multiple_choice",
    "input_mode": "free",
}


# =============================================================================
# LLM prompts (short, focused)
# =============================================================================

_TURN_AGENT_SYSTEM = """\
Role:
You are a warm, patient research interviewer talking with a person who has dysarthria. Their speech is sometimes hard for others to understand. You are running a short formative interview about how they repair communication today, and what they think of an early prototype that transcribes their speech and lets them correct the text.

Core objectives:
- Cover the assigned topics and collect the variables listed under each.
- Adapt to how much this participant wants to give, and where their interest is.
- Minimise burden. Never ask for something you already have.
- Sound like a person having a conversation, not a form.

This is not a test of the participant. There are no right answers. Never evaluate their communication or suggest they are answering badly.

You are given each turn:
- PHASE - pre_demo, post_demo, or demo_declined. If the participant declined the demo they have seen nothing of the prototype, so ask only the topics for that phase and never for a reaction to the video.
- TOPICS - every topic in this interview, with its phase, its priority, the things to collect under it, when to expand it, and anything you must not ask about. Ask only about topics whose phase matches PHASE. The rest are listed so you can pace yourself against what is still ahead, and so you can recognise when an answer has already covered something you will not reach until later.
- COVERAGE - which topics are already covered. Do not re-open a covered topic.
- TRANSCRIPT - the conversation so far. Suggestions the participant tapped are kept separate from what they typed, so you can tell a deliberate sentence from a tap.
- SIGNALS - how this participant has been answering: typed words per answer against their own median, and typing speed. Use this to judge engagement. Never read it as an absolute measure of anything.
- QUESTIONS_ASKED - how many questions so far, against the target.

The interview opens with one fixed question, which has already been asked before you are first called: "when someone does not understand you, what do you usually do?". Start from the participant's answer to it. Do not repeat it or reintroduce yourself.

Reading the participant:

Judge two things before deciding what to ask next.

engagement - how invested this particular answer is, compared with how THIS participant has been answering so far. It is not a length measure. Brevity is not disengagement; some people answer completely in four words. Look instead for:
- they added something you did not ask for
- they named a specific person, place or situation rather than a category
- they used evaluative language ("annoying", "I hate it when", "that would help")
- they asked you a question
- they returned to an earlier topic on their own
Low engagement looks like: answers that stopped addressing the question, repeated skipping, or a clear drop from their own earlier pattern.

information_value - how much a follow-up would add beyond what they already told you, beyond what a later topic will cover, and beyond what is already well established about communication repair.

Then decide:
- engagement normal or high, information_value high -> ask the follow-up.
- engagement low, information_value high -> ask it, but phrased so it can be answered with one tap.
- information_value low -> move on, whatever their engagement.

Never ask a follow-up whose answer is already contained in something they said.

Conversation rules:
- Ask exactly one question per turn. One question means one thing. Do not join two questions with "and" or a comma.
- Keep each message to one or two short sentences.
- Acknowledge what they said before asking the next thing, briefly and specifically. Refer to their own words.
- When the participant taps a suggested answer, that tells you which option they chose, not their words. Do not quote it back as something they said, and do not ask what made them say it — they picked from your list. Ask about the thing behind their choice instead.
- Do not ask them to restate anything they have already told you, including whether something is easier, harder, better, worse, common or rare.
- If they answer several topics at once, mark all of them covered and do not re-ask.
- If they mention something interesting that belongs to a later topic, let them finish it now rather than making them repeat it later.
- A "why" question is fine when it can be answered in a short phrase or a tap. Do not ask for stories or extended explanations, and do not pressure for detail.
- Never mention topic codes, variable names, or anything about how you are structured.

Budget:
- At most 3 follow-ups on the topic the participant is most engaged with, and at most 1 on each other topic.
- Aim to finish in about 12 questions in total.
- Pace yourself against the topics still ahead in TOPICS. Do not spend the interview on the first thing that interests you and arrive at the later topics with nothing left. Save room for a good opportunity rather than taking the first one.
- It is acceptable to leave things uncollected. Anything missing is recorded for the researcher. Prefer moving on over drilling down.
- One or two variables per topic is usually enough.

Topic priority:
- core topics must be covered even if the participant is fading. Ask them in their cheapest form.
- important topics are covered unless the participant is clearly tiring.
- optional topics are only for participants who are still engaged.
Cover topics in the order given unless the participant has opened one early themselves.

Suggested answers:
- Whenever your reply asks a question, give exactly five short suggested answers, plus nothing else. They are optional taps, not a questionnaire.
- Make them meaningfully different from each other. Where it fits, cover positive, negative, neutral and uncertain.
- Write them in the participant's own register, short enough to read at a glance.
- Never include a generic filler option such as "Other" or "Something else" - the interface adds those itself.
- When your reply does not ask a question, return an empty list.

Participant wellbeing:
Treat it as a flag when the participant says the interview is tiring or too long, asks to stop, asks more than once how much is left, expresses distress about their communication, or says something that needs a person rather than a chatbot.
When flagged:
- Acknowledge it warmly and specifically.
- Offer to skip ahead, take a break, or stop - as a real option, not a formality.
- Do not add follow-ups on that topic.
- Put a short note in wellbeing_flag.
- This session is not monitored in real time. Do not give advice or reassurance beyond acknowledging what they said.

Closing:
When the assigned topics are covered, ask one final open question, naming in a few words what you talked about together. For example: "We've talked about what you do when people don't understand you, and what you thought of the demo. Before we finish, is there anything I didn't ask about?"
- On that turn is_complete must be false.
- If they raise something new, treat it as a real topic, ask one follow-up, then return to the closing question.
- Only set is_complete true after they answer the closing question with nothing new.
- is_complete must never be true on a turn where you ask a question.

Respond with valid JSON and these keys only:
{
  "reply": "what the participant sees",
  "suggested_answers": ["exactly five short options when reply asks a question, otherwise empty"],
  "topic": "the topic id this turn belongs to, or empty",
  "topics_done": ["ids of topics you are finished with"],
  "engagement": "high | normal | low",
  "information_value": "high | medium | low",
  "wellbeing_flag": "short note, or empty",
  "is_complete": true or false,
  "researcher_summary": "3-4 sentences when complete, otherwise empty"
}

The participant sees only "reply" and "suggested_answers".
"""

_SUMMARIZER_SYSTEM = """\
Summarize what we have learned from this interview participant so far in 2-4 plain sentences: who they communicate with, their strategies, difficulties, and reactions to the technology demo. Also note their answering style (clicks suggestions, types short answers, or types full sentences). You are given the previous summary and the latest question and answer. Return JSON only: {"summary": "..."}
"""


# =============================================================================
# OpenAI helpers
# =============================================================================

def _strip_controls(obj):
    """Recursively strip Unicode control characters (category Cc) from all strings,
    keeping only tab and newline as legitimate whitespace."""
    if isinstance(obj, str):
        return ''.join(
            ch for ch in obj
            if unicodedata.category(ch) != 'Cc' or ch in '\t\n'
        )
    if isinstance(obj, dict):
        return {k: _strip_controls(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_controls(v) for v in obj]
    return obj


def _call_llm_json(system_prompt, user_prompt, label="agent"):
    """Call the LLM and return a parsed JSON dict. Appends raw log to session state."""
    raw_text = None
    try:
        resp = _openai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw_text = resp.choices[0].message.content
        result = _strip_controls(json.loads(raw_text))
    except Exception as first_err:
        if "agent_logs" in st.session_state:
            st.session_state.agent_logs.append({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "label": label + "_first_attempt_error",
                "error": f"{type(first_err).__name__}: {first_err}",
                "raw_response": raw_text,
            })
        resp = _openai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_text = resp.choices[0].message.content or ""
        m = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if m:
            result = _strip_controls(json.loads(m.group()))
        else:
            if "agent_logs" in st.session_state:
                st.session_state.agent_logs.append({
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "label": label + "_error",
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "raw_response": raw_text,
                    "error": "No JSON found in response",
                })
            raise ValueError(f"LLM did not return valid JSON. Raw: {raw_text[:300]}")

    if "agent_logs" in st.session_state:
        st.session_state.agent_logs.append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "label": label,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw_text,
            "parsed_response": result,
        })

    return result


# =============================================================================
# Flow engine
#
# One agent call per turn. Python owns three things only: the demo interlude,
# what counts as covered, and the behavioural signals the agent reasons over.
# Everything the participant reads is composed by the agent.
# =============================================================================

def _typed_residual(user_msg):
    """The part of an answer the participant actually typed, with tapped
    suggestions removed."""
    free = user_msg.get("free_text", user_msg.get("content", "")) or ""
    residual = free
    for phrase in user_msg.get("selected_suggestions", []):
        residual = residual.replace(phrase, "")
    return re.sub(r"[;,.\s]+", " ", residual).strip()


def _parse_ts(value):
    try:
        return datetime.strptime((value or "").rstrip("Z"), "%Y-%m-%dT%H:%M:%S.%f")
    except Exception:
        return None


def _turn_signals(chat):
    """Per-answer behavioural signals, from timestamps already on the messages.

    Typing speed counts typed characters only. Tapping a suggestion inserts text in
    one second and would otherwise register as superhuman typing.
    """
    rows = []
    prev_assistant = None
    for m in chat:
        role = m.get("role")
        if role == "assistant":
            prev_assistant = m
            continue
        if role != "user":
            continue
        typed = _typed_residual(m)
        secs = None
        if prev_assistant:
            a = _parse_ts(prev_assistant.get("timestamp"))
            b = _parse_ts(m.get("timestamp"))
            if a and b:
                secs = max(0.0, (b - a).total_seconds())
        wpm = None
        if typed and secs and secs > 2:
            wpm = round((len(typed) / 5) / (secs / 60), 1)
        rows.append({
            "typed_words": len(typed.split()) if typed else 0,
            "tapped": len(m.get("selected_suggestions") or []),
            "seconds_to_answer": round(secs, 1) if secs is not None else None,
            "typing_wpm": wpm,
        })
    return rows


def _signal_summary(chat):
    """What the agent needs to judge engagement: this participant against themselves."""
    rows = _turn_signals(chat)
    if not rows:
        return {"note": "no answers yet"}
    typed = [r["typed_words"] for r in rows]
    wpms = [r["typing_wpm"] for r in rows if r["typing_wpm"]]
    median = sorted(typed)[len(typed) // 2]
    return {
        "answers_so_far": len(rows),
        "median_typed_words": median,
        "latest_typed_words": typed[-1],
        "answers_with_any_typing": sum(1 for t in typed if t > 0),
        "median_typing_wpm": (sorted(wpms)[len(wpms) // 2] if wpms else None),
        "how_to_read_this": (
            "This participant is mostly tapping suggestions rather than typing."
            if median == 0 else
            "Compare the latest answer with this participant's own median, never with "
            "an absolute length. A short answer can be a complete one."
        ),
    }


def _covered_topics(chat):
    """Topic ids the agent has reported as finished, accumulated over the session.

    "Done" means the agent would not ask about the topic again -- not merely that an
    answer touched it. The prompt draws that distinction explicitly, because reading
    "this answer was about T1" as "T1 is finished" is what sent the 23 Aug session
    to the demo after two questions.
    """
    out = set()
    for m in chat:
        if m.get("role") == "assistant":
            out.update(m.get("topics_done") or [])
    # A topic that has used its whole turn budget counts as finished whether or not the
    # agent said so. Without this the per-topic limit is advisory, and advisory limits
    # did not hold: T1 took four turns of enumerating strategies in the 23 Aug run.
    out |= {t for t, n in _topic_turns(chat).items() if n >= MAX_TURNS_PER_TOPIC}
    return out


def _topic_turns(chat):
    """How many interviewer turns each topic has used."""
    counts = {}
    for m in chat:
        if m.get("role") == "assistant":
            qid = m.get("question_id", "")
            if qid in INTERVIEW_TOPICS:
                counts[qid] = counts.get(qid, 0) + 1
    return counts


def _coverage_report(chat):
    covered = _covered_topics(chat)
    turns = _topic_turns(chat)
    return {tid: {"status": "done" if tid in covered else "not yet",
                  "turns_used": turns.get(tid, 0),
                  "turns_left": max(0, MAX_TURNS_PER_TOPIC - turns.get(tid, 0))}
            for tid in INTERVIEW_TOPICS}


def _questions_asked(chat):
    return sum(1 for m in chat if m.get("role") == "assistant"
               and m.get("question_type") in ("main", "transition"))


# ---- Demo interlude ---------------------------------------------------------
# Still on Python rails. A seam to revisit once the rest is settled.

DEMO_CONSENT = {
    "question_id": "DemoConsent",
    "question_text": "Next, we would like to show a short demo video of an early idea. "
                     "Is now an okay time to watch it?",
    "question_type": "transition",
    # "I have a question first" was removed: there was no handling for it, and the
    # turn has no text box, so a participant had no way to actually ask anything.
    "options": [{"label": "Yes"}, {"label": "Skip the demo"}],
    "answer_mode": "multiple_choice",
    "input_mode": "single_choice",
}

DEMO_SHOW = {
    "question_id": "DemoShow",
    "question_text": "Great - please watch the short demo now. "
                     "After that, we will ask a few questions.",
    "question_type": "transition",
    "options": [{"label": "Done"}, {"label": "Skip"}],
    "answer_mode": "multiple_choice",
    "input_mode": "single_choice",
}


def _demo_settled(chat):
    """True once the demo has been played or explicitly declined."""
    if any(m.get("role") == "video" for m in chat):
        return True
    for i, m in enumerate(chat):
        if m.get("role") == "assistant" and m.get("question_id") == "DemoConsent":
            if i + 1 < len(chat) and chat[i + 1].get("role") == "user":
                if "skip" in (chat[i + 1].get("content", "") or "").lower():
                    return True
    return False


def _demo_watched(chat):
    """True only if the video actually played. Declining also settles the demo."""
    return any(m.get("role") == "video" for m in chat)


def _phase(chat):
    """pre_demo, post_demo, or demo_declined.

    The third value matters: a participant who declines has seen nothing of the
    prototype, so T3-T6 are unanswerable for them and T7 replaces the lot. An earlier
    version collapsed declined into post_demo and asked ten questions about a video
    that was never watched.
    """
    if not _demo_settled(chat):
        return "pre_demo"
    return "post_demo" if _demo_watched(chat) else "demo_declined"


def _pre_demo_ready(chat, extra=()):
    """Whether it is time to offer the demo.

    `extra` carries topic ids the current response has just marked done. Those are
    not in `chat` yet, so without it this check would always be one turn behind.
    """
    core = {t for t, e in INTERVIEW_TOPICS.items()
            if e["phase"] == "pre_demo" and e["priority"] == "core"}
    if core and core <= (_covered_topics(chat) | set(extra)):
        return True
    return _questions_asked(chat) >= PRE_DEMO_QUESTION_CAP


def _demo_step(chat, last_q, last_user):
    """Take the turn if the demo needs handling, else (False, None) to let the agent run.

    Consent is granted only by an explicit "Yes". Since the consent turn is
    single_choice, the answer is always one of its two labels; anything else means
    something went wrong, and falling through leaves the demo unsettled so consent is
    offered again rather than assumed.
    """
    if _demo_settled(chat):
        return False, None
    qid = last_q.get("question_id") if last_q else None
    if qid == "DemoConsent" and last_user is not None:
        ans = (last_user.get("content") or "").strip().lower()
        if ans.startswith("yes"):
            st.session_state.demo_status = "shown"
            return True, dict(DEMO_SHOW)
        if "skip" in ans:
            st.session_state.demo_status = "skipped"
            return False, None
        return False, None
    if qid == "DemoShow" and last_user is not None:
        return False, None
    return False, None


# ---- The agent turn ---------------------------------------------------------

def _topics_for_prompt():
    """Every topic, including ones for the other phase.

    The agent needs the whole list even though it may only ask about the current
    phase: it cannot pace its follow-up budget without knowing what is still ahead,
    and it cannot notice that an answer has already covered a later topic if it does
    not know that topic exists. The prompt gates asking on the phase field instead.
    """
    keys = ("name", "phase", "priority", "collect", "parts", "expand_if",
            "do_not_collect", "note")
    return {tid: {k: e[k] for k in keys if e.get(k)}
            for tid, e in INTERVIEW_TOPICS.items()}


def _transcript_for_prompt(chat):
    """Full conversation, with tapped suggestions kept separate from typed text so the
    agent can tell a deliberate sentence from a tap."""
    out = []
    for m in chat:
        role = m.get("role")
        if role == "assistant":
            out.append({"interviewer": m.get("content", ""),
                        "topic": m.get("question_id", "")})
        elif role == "user":
            out.append({"participant_tapped": m.get("selected_suggestions") or [],
                        "participant_typed": _typed_residual(m)})
        elif role == "video":
            out.append({"event": "demo video shown"})
    return out


def _build_payload(chat, phase):
    return (
        f"PHASE:\n{phase}\n\n"
        f"TOPICS:\n{json.dumps(_topics_for_prompt(), ensure_ascii=False, indent=2)}\n\n"
        f"COVERAGE:\n{json.dumps(_coverage_report(chat), ensure_ascii=False, indent=2)}\n\n"
        f"TRANSCRIPT:\n{json.dumps(_transcript_for_prompt(chat), ensure_ascii=False, indent=2)}\n\n"
        f"SIGNALS:\n{json.dumps(_signal_summary(chat), ensure_ascii=False, indent=2)}\n\n"
        f"QUESTIONS_ASKED:\n{_questions_asked(chat)} so far. Aim for about "
        f"{MAX_QUESTIONS_TARGET}. Hard maximum {MAX_TURNS_HARD_CAP}, after which the "
        f"interview closes automatically."
    )


FORCED_CLOSING = {
    "question_id": "Closing",
    "question_text": "Before we finish, is there anything important I did not ask about?",
    "question_type": "main",
    "options": [{"label": l} for l in
                ("No, that's everything", "Yes, there is something", "Other", "Skip")],
    "answer_mode": "multiple_choice",
    "input_mode": "free",
}


def _closing_asked(chat):
    return any(m.get("role") == "assistant" and m.get("question_id") == "Closing"
               for m in chat)


_RETRY_RESULT = {
    "question_id": "",
    "question_text": "Sorry - something went wrong on my end. Could you send that again?",
    "question_type": "main",
    "options": [],
    "answer_mode": "multiple_choice",
    "input_mode": "free",
}


def run_agent_turn():
    """Decide and return the next interviewer turn: (show_video, result).

    result=None with interview_ended set means the interview is over.
    All state is derived from the chat history, so resumed sessions work.
    """
    chat = st.session_state.chat

    # ---- First turn: the fixed opener, no model call ----
    if not any(m.get("role") == "user" for m in chat):
        return False, dict(OPENING_QUESTION)

    last_q = next((m for m in reversed(chat) if m.get("role") == "assistant"), None)
    last_user = next((m for m in reversed(chat) if m.get("role") == "user"), None)

    show_video, demo_result = _demo_step(chat, last_q, last_user)
    if demo_result is not None:
        return show_video, demo_result

    phase = _phase(chat)
    user_prompt = _build_payload(chat, phase)

    try:
        result = _call_llm_json(_TURN_AGENT_SYSTEM, user_prompt, label="turn_agent")
    except Exception as e:
        if "agent_logs" in st.session_state:
            st.session_state.agent_logs.append({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "label": "turn_agent_error",
                "error": f"{type(e).__name__}: {e}",
                "user_prompt": user_prompt,
            })
        return False, dict(_RETRY_RESULT)

    reply = (result.get("reply") or "").strip()

    if result.get("is_complete"):
        st.session_state.interview_ended = True
        st.session_state.researcher_summary = (result.get("researcher_summary") or "").strip()
        if reply:
            st.session_state.final_message = reply
        return False, None

    # The prompt tells the agent not to produce "Other" and "Skip"; the interface adds
    # them. Kept as a failsafe so they are never missing.
    opts = [o.strip() for o in (result.get("suggested_answers") or [])
            if isinstance(o, str) and o.strip()]
    if opts:
        for extra in ("Other", "Skip"):
            if extra not in opts:
                opts.append(extra)

    done = [t for t in (result.get("topics_done") or []) if t in INTERVIEW_TOPICS]

    # ---- Hard turn cap ----
    # Checked before anything else so it always wins: better to close than to start a
    # demo or a new topic at turn 20. The agent is told the cap in the payload and is
    # expected to close itself; this is what happens when it does not.
    asked = _questions_asked(chat)
    if asked >= MAX_TURNS_HARD_CAP:
        st.session_state.interview_ended = True
        st.session_state.researcher_summary = (result.get("researcher_summary") or "").strip()
        st.session_state.final_message = reply or CLOSING_MESSAGE
        return False, None
    if asked >= MAX_TURNS_HARD_CAP - 1 and not _closing_asked(chat):
        return False, dict(FORCED_CLOSING)

    # Checked here, after the agent has seen the participant's answer -- not in
    # _demo_step, which runs before the call. Triggering it there meant the demo
    # interrupted on the turn *after* a question, so the answer to that question was
    # never processed. The condition itself is unchanged.
    if phase == "pre_demo" and _pre_demo_ready(chat, done):
        consent = dict(DEMO_CONSENT)
        # Carry the topics this turn finished onto the consent message. Without this
        # they are dropped, and COVERAGE would keep reporting them unfinished for the
        # rest of the interview.
        consent["topics_done"] = done
        return False, consent

    return False, {
        "question_id": (result.get("topic") or "").strip(),
        "question_text": reply or "Could you tell me a little more?",
        "question_type": "main",
        "options": [{"label": o} for o in opts],
        "answer_mode": "multiple_choice",
        "input_mode": "free",
        "topics_done": done,
        "engagement": (result.get("engagement") or "").strip(),
        "information_value": (result.get("information_value") or "").strip(),
        "wellbeing_flag": (result.get("wellbeing_flag") or "").strip(),
    }


# =============================================================================
# Google Drive helpers
# =============================================================================

def _get_drive_config():
    try:
        return {
            "folder_id": st.secrets.get("GDRIVE_FOLDER_ID", ""),
            "client_id": st.secrets.get("GDRIVE_CLIENT_ID", ""),
            "client_secret": st.secrets.get("GDRIVE_CLIENT_SECRET", ""),
            "refresh_token": st.secrets.get("GDRIVE_REFRESH_TOKEN", ""),
        }
    except Exception:
        return {"folder_id": "", "client_id": "", "client_secret": "", "refresh_token": ""}


def _make_service(config):
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    creds = Credentials(
        token=None,
        refresh_token=config["refresh_token"],
        client_id=config["client_id"],
        client_secret=config["client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    creds.refresh(Request())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _get_or_create_folder(name, parent_id, svc):
    q = (
        f"name='{name}' and '{parent_id}' in parents "
        "and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    results = svc.files().list(q=q, fields="files(id)").execute().get("files", [])
    if results:
        return results[0]["id"]
    return svc.files().create(
        body={"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]},
        fields="id",
    ).execute()["id"]


def _upsert_bytes(name, data, folder_id, svc):
    from googleapiclient.http import MediaIoBaseUpload
    q = f"name='{name}' and '{folder_id}' in parents and trashed=false"
    existing = svc.files().list(q=q, fields="files(id)").execute().get("files", [])
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype="application/octet-stream")
    if existing:
        svc.files().update(fileId=existing[0]["id"], media_body=media).execute()
    else:
        svc.files().create(
            body={"name": name, "parents": [folder_id]},
            media_body=media,
        ).execute()


def _download_bytes(file_id, svc):
    from googleapiclient.http import MediaIoBaseDownload
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, svc.files().get_media(fileId=file_id))
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()


def _update_participants_log(user_id, root_folder_id, svc):
    try:
        q = f"name='participants_log.json' and '{root_folder_id}' in parents and trashed=false"
        existing = svc.files().list(q=q, fields="files(id)").execute().get("files", [])
        if existing:
            data = json.loads(_download_bytes(existing[0]["id"], svc).decode("utf-8"))
        else:
            data = {}
        if user_id not in data:
            data[user_id] = {
                "first_seen": datetime.utcnow().isoformat() + "Z",
                "last_seen": datetime.utcnow().isoformat() + "Z",
                "status": "in_progress",
                "turns": 0,
            }
        else:
            data[user_id]["last_seen"] = datetime.utcnow().isoformat() + "Z"
            data[user_id]["turns"] = data[user_id].get("turns", 0) + 1
        _upsert_bytes(
            "participants_log.json",
            json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
            root_folder_id, svc,
        )
    except Exception:
        pass


def _do_save(user_id, chat, agent_logs, config):
    if not config.get("folder_id") or not config.get("refresh_token"):
        missing = [k for k in ("folder_id", "refresh_token") if not config.get(k)]
        raise RuntimeError(f"Drive not configured  -  missing secrets: {', '.join(missing)}")
    svc = _make_service(config)
    root = config["folder_id"]
    pfolder = _get_or_create_folder(f"participant_{user_id}", root, svc)
    _upsert_bytes(
        "chat_history.json",
        json.dumps(chat, ensure_ascii=False, indent=2).encode("utf-8"),
        pfolder, svc,
    )
    if agent_logs:
        _upsert_bytes(
            "agent_logs.json",
            json.dumps(agent_logs, ensure_ascii=False, indent=2).encode("utf-8"),
            pfolder, svc,
        )
    _update_participants_log(user_id, root, svc)
    return True, "Saved."


@st.cache_resource
def _get_drive_errors():
    """Persistent error list.

    Module-level variables are wiped on every rerun because Streamlit re-executes the
    whole script; a cached resource is created once per server process.
    """
    return []


_drive_errors = _get_drive_errors()

def save_async(user_id, chat, agent_logs, config):
    def _run():
        try:
            _do_save(user_id, chat, agent_logs, config)
        except Exception as e:
            _drive_errors.append(str(e))
    threading.Thread(target=_run, daemon=True).start()


def save_sync(user_id, chat, agent_logs, config):
    try:
        return _do_save(user_id, chat, agent_logs, config)
    except Exception as e:
        return False, str(e)


def _save_audio_async(user_id, question_id, audio_bytes, transcript, config):
    """Save a .webm recording and update audio_log.json in Drive (async)."""
    def _run():
        try:
            if not config.get("folder_id") or not config.get("refresh_token"):
                return
            svc = _make_service(config)
            root = config["folder_id"]
            pfolder = _get_or_create_folder(f"participant_{user_id}", root, svc)
            afolder = _get_or_create_folder("audio", pfolder, svc)

            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            filename = f"{question_id}_{ts}.webm"

            # Upload the audio file
            from googleapiclient.http import MediaIoBaseUpload
            media = MediaIoBaseUpload(
                io.BytesIO(audio_bytes), mimetype="audio/webm"
            )
            svc.files().create(
                body={"name": filename, "parents": [afolder]},
                media_body=media,
            ).execute()

            # Update audio_log.json
            q = f"name='audio_log.json' and '{pfolder}' in parents and trashed=false"
            existing = svc.files().list(q=q, fields="files(id)").execute().get("files", [])
            log = json.loads(_download_bytes(existing[0]["id"], svc).decode("utf-8")) if existing else []
            log.append({
                "timestamp": ts,
                "question_id": question_id,
                "filename": filename,
                "transcript": transcript,
            })
            _upsert_bytes(
                "audio_log.json",
                json.dumps(log, ensure_ascii=False, indent=2).encode("utf-8"),
                pfolder, svc,
            )
        except Exception as e:
            _drive_errors.append(f"Audio save error: {e}")
    threading.Thread(target=_run, daemon=True).start()


def restore_from_drive(participant_id, config):
    try:
        if not config.get("folder_id") or not config.get("refresh_token"):
            return [], False
        svc = _make_service(config)
        root = config["folder_id"]
        q = (
            f"name='participant_{participant_id}' and '{root}' in parents "
            "and mimeType='application/vnd.google-apps.folder' and trashed=false"
        )
        folders = svc.files().list(q=q, fields="files(id)").execute().get("files", [])
        if not folders:
            return [], False
        pfolder = folders[0]["id"]
        files = {
            f["name"]: f["id"]
            for f in svc.files().list(
                q=f"'{pfolder}' in parents and trashed=false",
                fields="files(id, name)",
            ).execute().get("files", [])
        }
        chat = []
        if "chat_history.json" in files:
            chat = json.loads(_download_bytes(files["chat_history.json"], svc).decode("utf-8"))
        return chat, bool(chat)
    except Exception:
        return [], False



# =============================================================================
# Demo video
# =============================================================================

@st.cache_data(show_spinner=False)
def _load_demo_video_bytes():
    try:
        from googleapiclient.http import MediaIoBaseDownload
        config = _get_drive_config()
        service = _make_service(config)
        file_id = "1FCfzZslMnuyQAPhcZoiACrx0sWaYskxV"
        request = service.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buf.getvalue()
    except Exception:
        return None


# =============================================================================
# Whisper transcription
# =============================================================================

def _transcribe(audio_bytes):
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=("recording.wav", io.BytesIO(audio_bytes), "audio/wav"),
        )
        return result.text.strip()
    except Exception:
        return ""


# =============================================================================
# Page setup & state init
# =============================================================================

st.set_page_config(page_title="Interview", page_icon="mic", layout="wide")
st.title("Interview")

if "phase" not in st.session_state:
    st.session_state.update(
        phase="id_entry",
        user_id=None,
        chat=[],
        waiting=False,
        drive_config=None,
        session_saved=False,
        last_audio_hash=None,
        user_draft="",
        demo_status="not_shown",
        interview_ended=False,
        form_generation=0,
        agent_logs=[],
        typed_lengths=[],
    )


st.markdown("""
<style>
html, body, [class*="css"], .stMarkdown, .stChatMessage { font-size: 20px !important; }
div[data-testid="stChatMessage"] p { font-size: 1.05rem !important; line-height: 1.7 !important; }
div[data-testid="stTextArea"] textarea {
    min-height: 80px !important; font-size: 1.1rem !important;
    line-height: 1.7 !important; border-radius: 14px !important;
    padding: 14px 18px !important; resize: none !important;
}
div[data-testid="stButton"] button[kind="primary"] {
    font-size: 1rem !important;
    border-radius: 8px !important; width: 100% !important;
}
/* Keep Speak + text area on the same line; pin mic column to fixed width */
div[data-testid="stHorizontalBlock"]:has(iframe),
div[data-testid="stColumns"]:has(iframe) {
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: stretch !important;
}
div[data-testid="stHorizontalBlock"]:has(iframe) > div[data-testid="stColumn"]:first-child,
div[data-testid="stColumns"]:has(iframe) > div[data-testid="stColumn"]:first-child {
    flex: 0 0 110px !important;
    min-width: 110px !important;
    max-width: 110px !important;
}
[data-testid="stColumn"] iframe {
    height: 100px !important; min-height: 100px !important;
    width: 100% !important;
}
/* Suggested Phrases toggle button */
div[data-testid="stButton"] button[kind="secondary"] {
    height: 100px !important;
    width: 100% !important;
    font-size: 1rem !important;
    border-radius: 8px !important;
    background-color: #f0f4ff !important;
    color: #1a237e !important;
}
/* Option grid cards */
div[data-testid="stColumn"] div[data-testid="stButton"] button[kind="secondary"],
div[data-testid="stColumn"] div[data-testid="stButton"] button[kind="primary"] {
    min-height: 90px !important; height: auto !important;
    white-space: normal !important; word-break: break-word !important;
    border-radius: 12px !important; font-size: 1rem !important;
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Phase: participant ID entry
# =============================================================================

if st.session_state.phase == "id_entry":

    st.markdown("### Welcome")
    st.info(
        "After clicking **Start**, you will be given a **Participant ID**.  \n"
        "Please **write it down** -- you will need it to continue the interview "
        "later if you close the browser or need a break."
    )

    tab_new, tab_return = st.tabs(["New participant", "Returning participant"])

    with tab_new:
        st.markdown("Click the button to begin a new interview session.")
        if st.button("Start interview ->", type="primary", key="btn_new"):
            user_id = "P-" + uuid.uuid4().hex[:6].upper()
            cfg = _get_drive_config()
            st.session_state.update(user_id=user_id, drive_config=cfg, phase="intro")
            st.rerun()

    with tab_return:
        st.markdown("Enter the Participant ID you received when you started.")
        pid_input = st.text_input("Participant ID (e.g. P-ABC123):", key="pid_input")
        if st.button("Resume interview ->", key="btn_return"):
            pid = pid_input.strip().upper()
            if not pid:
                st.warning("Please enter your Participant ID.")
            else:
                cfg = _get_drive_config()
                with st.spinner(f"Looking up session for {pid}..."):
                    chat, found = restore_from_drive(pid, cfg)
                if found:
                    video_shown = any(m.get("role") == "video" for m in chat)
                    # If the session ended mid-turn (last message is from user,
                    # agent never responded), resume in waiting state so the
                    # agent fires immediately  -  this also handles the case where
                    # the participant answered "yes" to the demo consent question
                    # but the video was never shown.
                    last_role = chat[-1].get("role") if chat else None
                    resume_waiting = last_role == "user"
                    st.session_state.update(
                        user_id=pid, drive_config=cfg, chat=chat,
                        demo_status="shown" if video_shown else "not_shown",
                        waiting=resume_waiting, phase="active",
                    )
                    st.rerun()
                else:
                    st.error(
                        f"No session found for **{pid}**.  \n"
                        "Please double-check your ID and try again.  \n"
                        "If you have not started before, use the **New participant** tab."
                    )

    st.stop()


# =============================================================================
# Phase: intro
# =============================================================================

if st.session_state.phase == "intro":

    INTRO_TEXT = (
        "Thank you for meeting with us.\n\n"
        "We are interested in your everyday experiences communicating with other people, "
        "especially times when someone has trouble understanding you.\n"
        "Later, we will show you a short demo of an early technology idea and ask what you think about it.\n\n"
        "This is not a test of you. We are learning from your experience.\n"
        "There are no right or wrong answers. Short answers are fine. You can skip any question.\n\n"
        "It usually takes about 30 minutes.\n\n"
        "You can answer by speaking, typing, choosing suggested answers, or using a mix of these.\n"
        "If helpful, you can press the suggestions button to see possible answers."
    )

    st.markdown(INTRO_TEXT)
    st.markdown("")

    if st.button("Continue to interview ->", type="primary", key="btn_intro_continue"):
        st.session_state.chat = []
        st.session_state.waiting = True
        st.session_state.phase = "active"
        st.rerun()

    st.stop()


# =============================================================================
# Phase: active interview
# =============================================================================

user_id = st.session_state.user_id
cfg = st.session_state.drive_config

if "_pending_draft" in st.session_state:
    st.session_state.user_draft = st.session_state.pop("_pending_draft")

with st.sidebar:
    st.markdown("### Your Participant ID")
    st.code(user_id, language=None)
    st.caption(
        "Keep this ID safe. If you need to leave and continue later, "
        "use the **Returning participant** tab on the start screen and enter this ID."
    )
    if _drive_errors:
        st.error(f"⚠️ Drive save error: {_drive_errors[-1]}")


# Render chat history
_first_assistant_seen = False
for msg in st.session_state.chat:
    if msg.get("role") == "video":
        st.markdown("#### Demo Video")
        st.markdown("<p style='font-size:18px; color:black;'>Please watch the short demo video below before answering the next question.</p>", unsafe_allow_html=True)
        _video_bytes = _load_demo_video_bytes()
        if _video_bytes:
            _, vid_col, _ = st.columns([1, 5, 1])
            with vid_col:
                st.video(_video_bytes, format="video/mp4")
        else:
            st.info("Video unavailable -- please ask the researcher to share the demo link.")
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.write(msg["content"])
            if not _first_assistant_seen:
                st.caption(
                    'You can respond in whatever way works best for you: '
                    'You can type your answer, click Speak to record it, '
                    'or choose one or more example answers using the button below. '
                    'You can also combine these options. '
                    'When using Speak, press Stop when you are finished, '
                    'and your words will appear in the text box.'
                )
                _first_assistant_seen = True
    elif msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])

# State machine
if st.session_state.waiting:
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                show_video, result = run_agent_turn()
            except Exception as e:
                st.session_state.waiting = False
                st.error(f"Something went wrong: {e}")
                st.stop()

    if show_video:
        st.session_state.chat.append({"role": "video"})
    if result:
        st.session_state.chat.append({
            "role": "assistant",
            "content": result["question_text"],
            "question_id": result.get("question_id", ""),
            "question_type": result.get("question_type", ""),
            "support_reason": result.get("support_reason", ""),
            "answer_mode": result.get("answer_mode", "multiple_choice"),
            "input_mode": result.get("input_mode", "free"),
            "options": result.get("options", []),
            # The agent's own judgements. Persisted so coverage survives a resumed
            # session and so the researcher can audit why each follow-up was asked.
            "topics_done": result.get("topics_done", []),
            "engagement": result.get("engagement", ""),
            "information_value": result.get("information_value", ""),
            "wellbeing_flag": result.get("wellbeing_flag", ""),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

    st.session_state.waiting = False
    save_async(user_id, st.session_state.chat, st.session_state.agent_logs, cfg)
    st.rerun()

elif st.session_state.get("interview_ended"):
    with st.chat_message("assistant"):
        # The agent writes its own sign-off; CLOSING_MESSAGE is the fallback when the
        # interview ended some other way (a stop request, or an error).
        st.write(st.session_state.get("final_message") or CLOSING_MESSAGE)
    st.success("The interview has ended. Thank you for your time!")
    if not st.session_state.session_saved:
        with st.spinner("Saving your session to Google Drive..."):
            ok, save_msg = save_sync(user_id, st.session_state.chat, st.session_state.agent_logs, cfg)
        st.session_state.session_saved = True
        if ok:
            st.info(f"Session saved. Your Participant ID was **`{user_id}`**.")
        else:
            st.caption(f"(Note: auto-save encountered an issue: {save_msg})")

else:
    current_q_msg = None
    for msg in reversed(st.session_state.chat):
        if msg.get("role") == "assistant":
            current_q_msg = msg
            break

    gen = st.session_state.form_generation
    draft_key = f"user_draft_{gen}"

    # Apply any pending pre-fill (from audio transcription) before widgets render
    if "_prefill" in st.session_state:
        _new_text = st.session_state.pop("_prefill")
        _existing = st.session_state.get(draft_key, "")
        if _existing:
            _existing = _existing.rstrip()
            if not _existing.endswith(";"):
                _existing += ";"
            _existing += " "
        st.session_state[draft_key] = _existing + _new_text

    # ── Interactive options (metadata only  -  rendered below input row) ──────────
    answer_mode = current_q_msg.get("answer_mode", "multiple_choice") if current_q_msg else "multiple_choice"
    options = current_q_msg.get("options", []) if current_q_msg else []
    q_key = current_q_msg.get("question_id", "q") if current_q_msg else "q"
    input_mode = current_q_msg.get("input_mode", "free") if current_q_msg else "free"

    # ── Single-choice turns (demo consent / demo show) ───────────────────────
    # Exactly one option, no text box and no mic, and the click submits straight away.
    # Consent has to be an unambiguous act, so there is no free text that could be
    # misread as agreement and no half-filled draft to leave behind.
    if input_mode == "single_choice" and options:
        st.markdown("**Please choose one:**")
        n_cols = min(3, len(options))
        choice_cols = st.columns(n_cols)
        for i, opt in enumerate(options):
            with choice_cols[i % n_cols]:
                if st.button(opt["label"], key=f"single_{gen}_{q_key}_{i}",
                             type="primary", use_container_width=True):
                    st.session_state.form_generation += 1
                    st.session_state.chat.append({
                        "role": "user",
                        "content": opt["label"],
                        "selected_suggestions": [opt["label"]],
                        "free_text": opt["label"],
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    })
                    st.session_state.waiting = True
                    st.rerun()
        st.stop()

    # ── Speak | Text area | Send ────────────────────────────────────────────
    mic_col, text_col, send_col = st.columns([1, 8, 2])

    with mic_col:
        audio = mic_recorder(
            start_prompt="🎤  Speak",
            stop_prompt="⏹️  Stop",
            just_once=True,
            use_container_width=True,
            key="mic",
        )

    with text_col:
        typed = st.text_area(
            "response",
            key=draft_key,
            height=100,
            placeholder="Type your response here, or click 🎤 Speak to record...",
            label_visibility="collapsed",
        )

    with send_col:
        send_clicked = st.button("Send →", type="primary", use_container_width=True, key=f"send_btn_{gen}")

    # Enter key sends (Shift+Enter = newline)
    components.html("""
    <script>
    (function() {
        function attach() {
            var ta = window.parent.document.querySelector('textarea[aria-label="response"]');
            if (!ta || ta._enterBound) return;
            ta._enterBound = true;
            ta.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                    // Ctrl+Enter = insert newline (block Streamlit's "apply")
                    e.preventDefault();
                    e.stopPropagation();
                    var start = ta.selectionStart, end = ta.selectionEnd, val = ta.value;
                    var setter = Object.getOwnPropertyDescriptor(
                        window.parent.HTMLTextAreaElement.prototype, 'value').set;
                    setter.call(ta, val.slice(0, start) + String.fromCharCode(10) + val.slice(end));
                    ta.selectionStart = ta.selectionEnd = start + 1;
                    ta.dispatchEvent(new Event('input', {bubbles: true}));
                } else if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    ta.blur();
                    setTimeout(function() {
                        var btns = window.parent.document.querySelectorAll('button');
                        for (var i = 0; i < btns.length; i++) {
                            if (btns[i].innerText.trim().startsWith('Send')) {
                                btns[i].click(); break;
                            }
                        }
                    }, 150);
                }
            });
        }
        function fixHint() {
            var hints = window.parent.document.querySelectorAll('[data-testid="InputInstructions"]');
            for (var i = 0; i < hints.length; i++) {
                if (hints[i].textContent.indexOf('Ctrl') !== -1) {
                    hints[i].textContent = 'Press Enter to send';
                }
            }
        }
        attach();
        fixHint();
        new MutationObserver(function() { attach(); fixHint(); })
            .observe(window.parent.document.body, {childList:true, subtree:true, characterData:true});
    })();
    </script>
    """, height=0)

    # ── Suggested Phrases ────────────────────────────────────────────────────
    if options:
        show_key = f"show_opts_{gen}_{q_key}"
        if show_key not in st.session_state:
            st.session_state[show_key] = False

        if not st.session_state[show_key]:
            if st.button("Suggested Phrases", key=f"show_opts_btn_{gen}_{q_key}"):
                st.session_state[show_key] = True
                st.rerun()
        else:
            if answer_mode in ("multiple_choice", "ranking"):
                grid_cols = st.columns(4)
                _pick_key = f"phrase_picks_{gen}_{q_key}"
                st.session_state.setdefault(_pick_key, set())
                for i, opt in enumerate(options):
                    with grid_cols[i % 4]:
                        if st.button(opt["label"], key=f"mbtn_{gen}_{q_key}_{i}",
                                     type="secondary",
                                     use_container_width=True):
                            _phrase = opt["label"]
                            st.session_state[_pick_key].add(_phrase)
                            st.session_state._prefill = _phrase
                            st.rerun()

            elif answer_mode == "yes_no_plus_optional_text":
                st.markdown("**Choose one (you can add details below):**")
                n_cols = min(3, len(options))
                cols = st.columns(n_cols)
                for i, opt in enumerate(options):
                    with cols[i % n_cols]:
                        if st.button(opt["label"], key=f"ynopt_{gen}_{q_key}_{i}",
                                     use_container_width=True):
                            st.session_state[draft_key] = opt["label"]

    if send_clicked:
        typed_text = (typed or st.session_state.get(draft_key) or "").strip()

        selected = []
        if answer_mode in ("multiple_choice", "ranking"):
            _pick_key = f"phrase_picks_{gen}_{q_key}"
            _picks = st.session_state.get(_pick_key, set())
            _draft = st.session_state.get(draft_key, "")
            # Sort by position in the draft so picks keep click order -- they are
            # shown back to the participant as follow-up options.
            selected = sorted((p for p in _picks if p in _draft), key=_draft.index)

        answer = typed_text or None

        if answer:
            st.session_state.form_generation += 1
            st.session_state.chat.append({
                "role": "user",
                "content": answer,
                "selected_suggestions": selected,
                "free_text": typed_text,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })
            st.session_state.waiting = True
            st.rerun()
        else:
            st.warning("Please type a response or choose an option before sending.")

    elif audio:
        audio_bytes = audio["bytes"]
        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        if audio_hash != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = audio_hash
            with st.spinner("Transcribing..."):
                transcript = _transcribe(audio_bytes)
            if transcript:
                st.session_state._prefill = transcript
                _save_audio_async(
                    user_id,
                    q_key,
                    audio_bytes,
                    transcript,
                    cfg,
                )
                st.rerun()

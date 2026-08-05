"""
Interview Chatbot  -  Python-driven flow with targeted LLM calls
(predefined questions; LLM consulted only for typed input + background summarizer)
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


# =============================================================================
# Interview guide (Python-driven flow -- questions and options are predefined)
# =============================================================================

MAX_FOLLOWUPS_TOTAL = 5
MAX_FOLLOWUPS_PER_QUESTION = 1

INTERVIEW_GUIDE = {
    "A1": {
        "question": "Who do you communicate with most often?",
        "purpose": "Understand the participant's everyday communication context.",
        "options": ["Family", "Friends", "Caregivers or support workers",
                    "Doctors or health workers", "People at work or school",
                    "Store or service workers", "Other", "Skip"],
        "followup": "Who is easiest to communicate with?",
        "followup_options": ["Family", "Friends", "Caregivers or support workers",
                             "Doctors or health workers", "People who know me well",
                             "No one is easy", "Other", "Skip"],
    },
    "A2": {
        "question": "When someone does not understand you, what do you usually do?",
        "purpose": "Learn the participant's own communication strategies without assuming that repeating speech is the main strategy.",
        "options": ["Say it again", "Say it differently", "Gesture or point",
                    "Type or write", "Use AAC, sign, or another device",
                    "Ask someone else to help", "Let it go", "Other", "Skip"],
        "followup": "Do you usually use one way, or more than one?",
        "followup_options": ["One way", "More than one", "It depends",
                             "I am not sure", "Other", "Skip"],
    },
    "A3": {
        "question": "What is usually hardest when someone does not understand you?",
        "purpose": "Identify burdens that a new technology should reduce, not add to.",
        "options": ["Repeating myself", "Saying it another way", "Typing or using a device",
                    "Feeling rushed", "The other person gets impatient",
                    "Losing what I wanted to say", "Nothing is especially hard", "Other", "Skip"],
        "followup": "Which one is hardest?",
        "followup_options": ["Repeating myself", "Saying it another way",
                             "Typing or using a device", "Feeling rushed",
                             "Other person gets impatient", "Losing my thought", "Other", "Skip"],
    },
    "A4": {
        "question": "What can other people do that helps you be understood?",
        "purpose": "Understand listener-side and environment-side supports.",
        "options": ["Be patient", "Wait longer", "Ask yes/no questions", "Guess from context",
                    "Watch my gestures", "Read what I type or show", "Move to a quieter place",
                    "Nothing helps much", "Other", "Skip"],
        "followup": "What is most helpful?",
        "followup_options": ["Patience", "Waiting", "Yes/no questions", "Guessing from context",
                             "Watching gestures", "Reading what I type or show", "Other", "Skip"],
    },
    "DemoConsent": {
        "question": "Next, we would like to show a short demo video of an early idea. Is now an okay time to watch it?",
        "purpose": "Ask permission before showing the demo.",
        "options": ["Yes", "Skip the demo", "I'm not sure"],
        "followup": None,
        "followup_options": [],
        "type": "transition",
    },
    "DemoShow": {
        "question": "Great - please watch the short demo now. After that, we will ask a few questions.",
        "purpose": "Show the demo video.",
        "options": ["Done", "Skip", "I need help"],
        "followup": None,
        "followup_options": [],
        "type": "transition",
    },
    "B1": {
        "question": "What is your first reaction to the demo?",
        "purpose": "Capture initial reaction without assuming the idea is good.",
        "options": ["I like it", "I partly like it", "I do not like it",
                    "Interesting, but I am not sure", "Seems too much work",
                    "Not useful for me", "Other", "Skip"],
        "followup": None,
        "followup_options": [],
    },
    "B2-useful": {
        "question": "What seems useful in the demo video?",
        "purpose": "Understand possible perceived benefits without forcing a positive reaction.",
        "options": ["Transcript", "Word choices", "Less repeating", "Helps the other person",
                    "Gives me control", "Could save time", "Nothing seems useful",
                    "I am not sure", "Other", "Skip"],
        "followup": "Which part seems most useful?",
        "followup_options": ["Transcript", "Word choices", "Less repeating",
                             "Helps the other person", "Control", "Saving time",
                             "None", "Other", "Skip"],
    },
    "B2-concern": {
        "question": "What seems not useful or concerning in the demo video?",
        "purpose": "Understand concerns, disliked parts, and possible barriers without assuming the participant dislikes the idea.",
        "options": ["Too slow", "Too much effort", "Transcript may be wrong",
                    "Hard to choose options", "Typing is hard", "Other person may not wait",
                    "I have better ways now", "Nothing concerns me", "I am not sure",
                    "Other", "Skip"],
        "followup": "Which concern matters most?",
        "followup_options": ["Too slow", "Too much effort", "Wrong transcript", "Hard to choose",
                             "Typing is hard", "Other person may not wait", "None",
                             "Other", "Skip"],
    },
    "B3": {
        "question": "If the system guessed wrong, what would be easiest?",
        "purpose": "Identify low-effort correction options without assuming that speaking again, typing, or detailed editing is easy.",
        "options": ["Pick the right word", "Pick from a few choices", "Tap the wrong word",
                    "Type a short fix", "Use a saved phrase", "Gesture or point",
                    "Let the other person help", "Do not fix it", "Other", "Skip"],
        "followup": "Which would take the least effort?",
        "followup_options": ["Pick the right word", "Pick from choices", "Tap the wrong word",
                             "Type a short fix", "Use a saved phrase", "Gesture or point",
                             "Other person helps", "Skip"],
    },
    "B4": {
        "question": "What should the people making this remember?",
        "purpose": "Elicit participant-centered design implications.",
        "options": ["Keep it low effort", "Do not assume typing is easy",
                    "Do not assume speaking again works", "Support gesture, AAC, or sign",
                    "Make it work in real conversations", "Let the other person help",
                    "Give me control", "Other", "Skip"],
        "followup": "What is most important?",
        "followup_options": ["Low effort", "Typing is not easy", "Speaking again may not work",
                             "Support gesture, AAC, or sign", "Real conversations",
                             "Other person can help", "Control", "Skip"],
    },
    "B4-general": {
        "question": "What should people making communication technology remember?",
        "purpose": "Elicit participant-centered design implications (demo skipped).",
        "options": ["Keep it low effort", "Do not assume typing is easy",
                    "Do not assume speaking again works", "Support gesture, AAC, or sign",
                    "Make it work in real conversations", "Let the other person help",
                    "Give me control", "Other", "Skip"],
        "followup": None,
        "followup_options": [],
    },
    "C1": {
        "question": "Is there anything important we did not ask?",
        "purpose": "Allow participant-led concerns or insights not anticipated by the guide.",
        "options": ["Yes", "No", "I'm not sure", "Other", "Skip"],
        "followup": "What else should we know?",
        "followup_options": ["Something about communication", "Something about the technology",
                             "Something about access or effort", "Something about privacy",
                             "Something else", "Skip"],
    },
}

# Question order. DemoShow is inserted by the flow only when the demo is accepted.
SEQUENCE_DEFAULT = ["A1", "A2", "A3", "A4", "DemoConsent", "DemoShow",
                    "B1", "B2-useful", "B2-concern", "B3", "B4", "C1"]
SEQUENCE_DEMO_SKIPPED = ["A1", "A2", "A3", "A4", "DemoConsent", "B4-general", "C1"]


# =============================================================================
# LLM prompts (short, focused)
# =============================================================================

_TURN_AGENT_SYSTEM = """\
You are helping run an interview with a person with dysarthria about everyday communication and a technology demo. Participants may type slowly, use shorthand, or make typos. Be respectful and never pressure them.

You are given: the current question, its research purpose, the participant's answer, a candidate follow-up question, a summary of the interview so far, and the participant's answering style.

Return JSON only:
{"understood": true, "acknowledgment": "one short natural sentence acknowledging their answer", "ask_followup": false, "followup_question": "", "followup_reason": "one line"}

Rules:
- Set ask_followup true only if the answer raises something design-relevant that a short follow-up could usefully deepen, and the interview summary does not already cover it. Never follow up just because an answer is short.
- You may rephrase the candidate follow-up or write a better one, but keep it answerable in one word or short phrase. Do not ask for stories or "why?" questions.
- If the answer is impossible to interpret, set understood false and put a gentle check in followup_question following this pattern: "It sounds like you mean [brief interpretation]. Is that right?"
- The acknowledgment must not mention internal question IDs.
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
    except Exception:
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
# Background summarizer (runs in a thread; results read on the next turn)
# =============================================================================

_summary_store = {}  # {user_id: {"summary": str, "pending_logs": [..]}}


def _update_summary_async(user_id, question_text, answer_text, prev_summary):
    def _run():
        try:
            user_prompt = (
                f"PREVIOUS_SUMMARY:\n{prev_summary or '(none yet)'}\n\n"
                f"LATEST_QUESTION:\n{question_text}\n\n"
                f"LATEST_ANSWER:\n{answer_text}"
            )
            resp = _openai_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": _SUMMARIZER_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            result = _strip_controls(json.loads(raw))
            entry = _summary_store.setdefault(user_id, {"summary": "", "pending_logs": []})
            entry["summary"] = result.get("summary", prev_summary or "")
            entry["pending_logs"].append({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "label": "summarizer",
                "system_prompt": _SUMMARIZER_SYSTEM,
                "user_prompt": user_prompt,
                "raw_response": raw,
                "parsed_response": result,
            })
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()


def _get_summary(user_id):
    return _summary_store.get(user_id, {}).get("summary", "")


def _drain_summary_logs(user_id):
    entry = _summary_store.get(user_id)
    if entry and entry.get("pending_logs") and "agent_logs" in st.session_state:
        st.session_state.agent_logs.extend(entry["pending_logs"])
        entry["pending_logs"] = []


# =============================================================================
# Flow engine (Python drives the interview; LLM consulted only for typed input)
# =============================================================================

_SKIP_WORDS = {"skip", "skip it", "next", "pass", "i don't know", "i dont know",
               "idk", "dont know", "don't know", "no", "none", "nothing"}


def _make_question_result(qid, ack="", is_followup=False, followup_text=None):
    """Build the result dict the UI expects for a guide question or its follow-up."""
    entry = INTERVIEW_GUIDE[qid]
    if is_followup:
        text = followup_text or entry["followup"]
        options = entry["followup_options"]
        q_type = "follow_up"
        q_id = qid + "_followup"
    else:
        text = entry["question"]
        options = entry["options"]
        q_type = entry.get("type", "main")
        q_id = qid
    content_text = (ack.strip() + " " + text).strip() if ack else text
    return {
        "question_id": q_id,
        "question_text": content_text,
        "question_type": q_type,
        "options": [{"label": o} for o in options],
        "answer_mode": "multiple_choice",
    }


def _clarification_result(qid, clarification_text):
    return {
        "question_id": qid + "_clarification",
        "question_text": clarification_text,
        "question_type": "clarification",
        "options": [{"label": "Yes"}, {"label": "No, I meant something else"}],
        "answer_mode": "multiple_choice",
    }


def _base_qid(question_id):
    """Strip _followup/_clarification suffixes."""
    for suffix in ("_followup", "_clarification"):
        if question_id.endswith(suffix):
            return question_id[: -len(suffix)]
    return question_id


def _get_sequence():
    """Return the active question order, accounting for a skipped demo."""
    chat = st.session_state.chat
    for i, m in enumerate(chat):
        if m.get("role") == "assistant" and m.get("question_id") == "DemoConsent":
            if i + 1 < len(chat) and chat[i + 1].get("role") == "user":
                ans = chat[i + 1].get("content", "").lower()
                if "skip" in ans or "not sure" in ans:
                    return SEQUENCE_DEMO_SKIPPED
    return SEQUENCE_DEFAULT


def _count_followups(chat):
    return sum(1 for m in chat
               if m.get("role") == "assistant" and m.get("question_type") == "follow_up")


def _answer_style():
    """Describe how the participant answers, from typed-residual lengths."""
    lengths = st.session_state.get("typed_lengths", [])
    if not lengths or max(lengths) == 0:
        return "So far the participant only clicks suggested answers."
    avg = sum(lengths) / len(lengths)
    if avg < 30:
        return "The participant types short answers (a few words)."
    return "The participant is comfortable typing full sentences."


def _typed_residual(user_msg):
    """Return the part of the answer the participant actually typed
    (free text minus clicked suggested phrases and separators)."""
    free = user_msg.get("free_text", user_msg.get("content", "")) or ""
    residual = free
    for phrase in user_msg.get("selected_suggestions", []):
        residual = residual.replace(phrase, "")
    residual = re.sub(r"[;,.\s]+", " ", residual).strip()
    return residual


def _is_skip_answer(user_msg):
    txt = (user_msg.get("content", "") or "").strip().lower().rstrip(".!")
    return txt in _SKIP_WORDS or txt == ""


def run_agent_turn():
    """Decide and return the next interviewer turn: (show_video, result).

    result=None with interview_ended set means the interview is over.
    Derives all flow state from the chat history, so resumed sessions work.
    """
    chat = st.session_state.chat
    user_id = st.session_state.get("user_id", "")
    _drain_summary_logs(user_id)

    # ---- First turn: serve A1, no LLM ----
    if not any(m.get("role") == "user" for m in chat):
        return False, _make_question_result("A1")

    # ---- Identify the question just answered ----
    last_q = None
    for m in reversed(chat):
        if m.get("role") == "assistant":
            last_q = m
            break
    last_user = None
    for m in reversed(chat):
        if m.get("role") == "user":
            last_user = m
            break
    q_id = last_q.get("question_id", "") if last_q else ""
    q_type = last_q.get("question_type", "") if last_q else ""
    base_id = _base_qid(q_id)
    sequence = _get_sequence()

    # ---- Track answering style ----
    residual = _typed_residual(last_user)
    st.session_state.setdefault("typed_lengths", []).append(len(residual))

    # ---- Kick off background summary update ----
    _update_summary_async(
        user_id,
        last_q.get("content", "") if last_q else "",
        last_user.get("content", "") if last_user else "",
        _get_summary(user_id),
    )

    def _next_main(ack="Thanks."):
        """Serve the next main question after base_id (deterministic)."""
        seq = _get_sequence()
        try:
            idx = seq.index(base_id)
        except ValueError:
            idx = -1
        if idx + 1 >= len(seq):
            st.session_state.interview_ended = True
            return False, None
        next_id = seq[idx + 1]
        show_video = False
        if next_id == "DemoShow":
            show_video = True
            st.session_state.demo_status = "shown"
        return show_video, _make_question_result(next_id, ack=ack)

    # ---- Demo consent: deterministic branch ----
    if base_id == "DemoConsent":
        ans = (last_user.get("content", "") or "").lower()
        if "skip" in ans or "not sure" in ans:
            st.session_state.demo_status = "skipped"
            return False, _make_question_result("B4-general", ack="No problem.")
        st.session_state.demo_status = "shown"
        return True, _make_question_result("DemoShow")

    # ---- Demo show acknowledgement: proceed to B1 ----
    if base_id == "DemoShow":
        return False, _make_question_result("B1", ack="Thanks.")

    # ---- Answer to a clarification ----
    if q_type == "clarification":
        ans = (last_user.get("content", "") or "").lower()
        served_count = sum(1 for m in chat
                           if m.get("role") == "assistant" and m.get("question_id") == base_id)
        if "no" in ans and served_count < 2:
            return False, _make_question_result(
                base_id, ack="Sorry about that. Let's try again -")
        return _next_main()

    # ---- Answer to a follow-up: always advance ----
    if q_type == "follow_up":
        return _next_main()

    # ---- End of interview after C1 (deterministic "Yes" follow-up) ----
    if base_id == "C1":
        ans = (last_user.get("content", "") or "").strip().lower()
        if ans.startswith("yes") and INTERVIEW_GUIDE["C1"]["followup"]:
            return False, _make_question_result("C1", ack="Sure.", is_followup=True)
        st.session_state.interview_ended = True
        return False, None

    # ---- Skip or clicks-only: advance, no LLM ----
    if _is_skip_answer(last_user):
        return _next_main(ack="No problem.")
    if not residual:
        return _next_main(ack="Thanks.")

    # ---- Typed input: consult the turn agent ----
    entry = INTERVIEW_GUIDE.get(base_id, {})
    followups_used = _count_followups(chat)
    quota_left = (followups_used < MAX_FOLLOWUPS_TOTAL
                  and entry.get("followup") is not None)
    already_clarified = any(
        m.get("role") == "assistant" and m.get("question_id") == base_id + "_clarification"
        for m in chat
    )

    user_prompt = (
        f"CURRENT_QUESTION:\n{entry.get('question', last_q.get('content', ''))}\n\n"
        f"RESEARCH_PURPOSE:\n{entry.get('purpose', '')}\n\n"
        f"PARTICIPANT_ANSWER:\n{last_user.get('content', '')}\n\n"
        f"CANDIDATE_FOLLOWUP:\n{entry.get('followup') or '(none - do not ask a follow-up)'}\n\n"
        f"INTERVIEW_SUMMARY_SO_FAR:\n{_get_summary(user_id) or '(interview just started)'}\n\n"
        f"ANSWER_STYLE:\n{_answer_style()}\n\n"
        f"FOLLOWUP_BUDGET:\n"
        f"{'A follow-up is allowed for this question.' if quota_left else 'Follow-ups are NOT allowed for this question - set ask_followup false.'}"
    )

    try:
        result = _call_llm_json(_TURN_AGENT_SYSTEM, user_prompt, label="turn_agent")
    except Exception:
        return _next_main()

    ack = (result.get("acknowledgment") or "Thanks.").strip()

    if not result.get("understood", True) and not already_clarified:
        clar = (result.get("followup_question") or "").strip()
        if clar:
            return False, _clarification_result(base_id, clar)

    if quota_left and result.get("ask_followup") and result.get("followup_question"):
        return False, _make_question_result(
            base_id, ack=ack, is_followup=True,
            followup_text=result["followup_question"].strip(),
        )

    return _next_main(ack=ack)


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


_drive_errors = []

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
        "There are about 10 questions in total.\n\n"
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
            "participant_message_type": result.get("participant_message_type", ""),
            "answer_mode": result.get("answer_mode", "multiple_choice"),
            "options": result.get("options", []),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

    st.session_state.waiting = False
    save_async(user_id, st.session_state.chat, st.session_state.agent_logs, cfg)
    st.rerun()

elif st.session_state.get("interview_ended"):
    with st.chat_message("assistant"):
        st.write(CLOSING_MESSAGE)
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
            selected = [p for p in _picks if p in _draft]

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

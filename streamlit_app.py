"""
Interview Chatbot — Two-agent pipeline (Decision Maker + Question Generator)
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
import uuid
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder

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

MODEL = "gpt-5-nano"

INTERVIEW_GUIDE = """\
## 1. Study Purpose

This interview explores how people with dysarthria respond to an idea for using speech transcription plus editing to support communication repair.

The system concept is that a person speaks, the system generates a transcript, and the person can edit or correct the transcript when needed. The system also allows the user to correct one word, then re-transcribe the partial transcript after the corrected word, which can potentially correct the remaining mistakes in the transcript after where the user corrected. 

The goal is to understand:

* How participants currently handle communication breakdowns.
* What makes communication repair difficult, tiring, or not worth the effort.
* Whether transcription plus editing could be useful in real-life communication.
* Which parts of the system seem useful, difficult, unrealistic, or unnecessary.
* In what situations participants might or might not use this kind of system.
* What design changes may make the system more usable, accessible, and practical.

Participants will watch a short demo video of the prototype, but they will not be asked to try the prototype directly in this chatbot interview.

This interview is not a test of the participant or their abilities. There are no right or wrong answers.
"""

QUESTIONS = [
    {
        "id": "A1",
        "main_question": "When someone does not understand you, what do you usually do?",
        "probes": [
            "Does your strategy depend on the person, situation, or importance of the message?",
            "Are there times when you decide not to keep trying?",
        ],
    },
    {
        "id": "A2",
        "main_question": "What makes communication repair difficult or tiring for you?",
        "probes": [
            "Is the hard part speech effort, typing effort, time, frustration, stress, or something else?",
            "Are there situations where repair feels too slow or too much work?",
            "What currently helps reduce the effort?",
        ],
    },
    {
        "id": "B1",
        "main_question": "What is your first reaction to this idea after seeing the demo? (Positive, neutral, or negative)",
        "probes": [
            "Could you imagine yourself using something like this?",
        ],
    },
    {
        "id": "B2",
        "main_question": "Which parts of the system seem useful?",
        "probes": [
            "Seeing a transcript of what you said.",
            "Editing the transcript.",
            "Correcting one word and asking the system to re-transcribe the rest.",
            "Starting from a transcript instead of typing everything from scratch.",
            "Showing the corrected text to another person.",
            "Having the corrected text spoken aloud.",
            "Is there any feature missing from the system?",
        ],
    },
    {
        "id": "B3",
        "main_question": "In what situations would you want to use something like this?",
        "probes": [
            "Would it fit better with strangers, familiar people, medical appointments, ordering food, work, school, or other situations?",
            "Would it be more useful for short conversations, longer conversations, or important messages?",
            "Are there situations where this system would feel too slow, awkward, tiring, or unnecessary?",
        ],
    },
    {
        "id": "B4",
        "main_question": "What concerns would you have about using this with another person in a real conversation?",
        "probes": [
            "Would the other person wait while you edit?",
            "Would using the system feel natural or awkward?",
            "Would privacy, attention to the screen, or social pressure be a concern?",
            "Would the other person's reaction affect whether you use it?",
        ],
    },
    {
        "id": "C1",
        "main_question": "When would the transcript be good enough to share with another person?",
        "probes": [
            "Does it need to be almost perfect, or is the main meaning enough?",
            "What kinds of mistakes would matter most?",
            "Are there mistakes you would be willing to leave unchanged?",
        ],
    },
    {
        "id": "C2",
        "main_question": "Overall, would something like this be useful for you?",
        "probes": [
            "Would it be better than repeating, typing from scratch, or what you currently use?",
            "Would it only be useful in certain situations or with certain people?",
            "Would the effort be worth it?",
        ],
    },
    {
        "id": "C3",
        "main_question": "What would need to change to make this system more useful for you?",
        "probes": [
            "Better transcription accuracy?",
            "Less typing?",
            "Easier editing?",
            "Word suggestions?",
            "Highlighting important mistakes?",
            "Easier repeat or re-record option?",
            "Support for shorthand, abbreviations, or first-letter input?",
            "A better way to show or speak the message to another person?",
        ],
    },
]

B1_INDEX = 2  # demo video shown when current_question_index first reaches this value

OPENING_QUESTION = {
    "question_id": "A1",
    "question_text": "When someone doesn't understand you, what do you usually do?",
    "answer_mode": "multiple_choice",
    "options": [
        {"label": "Repeat"},
        {"label": "Rephrase"},
        {"label": "Write down / type"},
        {"label": "Ask for clarification"},
        {"label": "Gesture / point"},
        {"label": "Use AAC"},
        {"label": "Other / type your answer"},
        {"label": "Skip"},
    ],
    "participant_instruction": "You can choose one option or type your own answer.",
}

CLOSING_MESSAGE = (
    "Thank you for sharing your experience and feedback with us. "
    "Your answers will help us understand whether transcription plus editing could support "
    "communication repair in everyday life, what parts may be useful or difficult, and how "
    "the system should be improved to better fit the needs of people with dysarthria."
)


# =============================================================================
# Agent system prompts + user-message templates
# =============================================================================

_QG_SYSTEM = """\
You are the Accessible Interview Question Composer.
You write short, simple, participant-facing interview prompts for people who may have dysarthric speech and may also have difficulty typing.
Your goal is to make each question easy to answer in a few words while still collecting useful qualitative data.
You receive a decision from the Interview State Manager. Follow it exactly. Do not change the interview direction.

Participant context: The participants have dysarthric speech — a condition that makes their speech difficult for others to understand. In this interview, they are the SPEAKER whose speech is being misunderstood. When generating options for questions like "what do you do when someone doesn't understand you?", options must reflect what a speaker with dysarthria would do (e.g., repeat, rephrase, type, write it down, use AAC, gesture). Always think from the participant's perspective as the speaker.

How to use the Decision Maker output:
- MOVE_NEXT: ask the `current_main_question` shown in the prompt. It has already been selected for you — do not skip ahead to the one after it.
- FOLLOW_UP: do NOT repeat or rephrase the main question. Instead, write a new question focused on the `target_information_gap` from the decision. The question must be clearly different from what was already asked. Look at the chat history to see what the participant already said, and build on it.
  - Options for a FOLLOW_UP must be fresh -- they must fit the follow-up topic, not recycle options already shown or already answered.
  - It is acceptable to briefly acknowledge what the participant said (e.g. "You mentioned using several strategies.") before the follow-up question.
- CLARIFY: ask a short clarifying question about the unclear part of the participant's last answer.
- REDUCE_BURDEN: shorten the question and offer fewer options.

Question design rules:
1. Ask only one question.
2. Avoid broad prompts like "Can you tell me more?"
3. Prefer narrowed questions that can be answered with one word, a short phrase, or one sentence.
4. Provide answer options.
5. Include "Skip" as an option.
6. Do not suggest that one answer is better than another.
7. Do not ask for names, exact age, address, phone number, email, or other personally identifying information.
8. Do not ask the participant to design a solution unless the interview guide explicitly asks for design preferences.
9. If mentioning technical terms such as "communication repair," "AAC," or "strategy", explain them in simple language.
10. Do not combine multiple questions into one.

Option design rules:
- Options should be short labels, not long sentences.
- Options should cover common possibilities without forcing the participant.
- Options should not imply judgment.
- Options should include "Other / type your answer."
- Options may include "I'm not sure" if appropriate.
- For sensitive or difficult topics, include "Prefer not to answer."
"""

_QG_USER_TEMPLATE = """\
# Interview guide
{interview_guide}

# Chat history:
{chat_history}

# Current main question:
{current_main_question}

# Optional probes for the current question:
{current_probes}

# Decision from Interview State Manager:
{decision}

Generate the next participant-facing prompt.
Return JSON in this format:
{{
  "question_id": "A1",
  "question_text": "...",
  "answer_mode": "multiple_choice | yes_no_plus_optional_text | ranking",
  "options": [
    {{"label": "..."}},
    ...
  ],
  "participant_instruction": "You can choose one option or type your own answer.",
  "why_this_question": "...",
  "target_information_gap": "..."
}}"""

_DM_SYSTEM = """\
You are the Interview State Manager for an accessibility-aware semi-structured interview.
The interview is with a participant who may have dysarthric speech and may also have difficulty typing. The interview should reduce response burden while still collecting useful qualitative data.
Your job is NOT to write the final participant-facing question. Your job is to decide the next interview action.

Core goals:
1. Maintain semi-structured interview coverage.
2. Avoid repeating questions already asked.
3. Decide whether a follow-up is necessary.
4. Prefer short, narrowed, answerable prompts over broad open-ended questions.
5. Respect participant effort, fatigue, and accessibility needs.
6. Preserve qualitative validity by avoiding leading questions.
7. Allow participants to answer by selecting options or typing their own answer.
8. Move forward when the current topic is sufficiently covered.

## Process
1. Determine Subtopic Nature
   - STAR-appropriate: event, project, or experience.
   - Descriptive: background, motivation, reasoning, or conceptual understanding.

2. Evaluate Completeness
   - STAR: Situation, Task, Action, Result all present -> covered.
   - Descriptive: main question explained with sufficient clarity -> covered.
   - If notes are already comprehensive, mark as covered and move on.

3. Aggregation
   - Synthesize covered subtopics into a concise final summary.

Decision rules:
1. Selected options create branches. Explore one branch at a time.
2. Keep an active branch. Ask follow-up about that branch before moving on.
3. Do not jump to a new topic too early.
4. Prefer branch-specific, narrow questions.
5. One small gap at a time. `target_information_gap` must describe exactly one thing to find out -- a single, atomic question. Never combine "find out X" with "get an example of X" or "also find out Y" in the same gap. If you need an example after a factual question, that becomes a separate follow-up turn once the factual question is answered.
6. Move on when the current branch has enough detail.
7. Accept multiple selections on a narrowing question. If the previous turn asked the participant to identify a single most-used or most-important item, and the participant responded by selecting multiple options, treat that as a valid answer and MOVE_NEXT. Do not ask the same narrowing question again.
8. When generating a FOLLOW_UP, first look at the optional probes for the current question in the interview guide. If one is relevant to what the participant said, use it as the basis for target_information_gap. Do not invent ranking or narrowing questions like "which one do you use most?" or "which is most important?" — these are not research questions and are not in the interview guide.
9. Decision options:
   - If current_subtopic_status is sufficiently_covered, you must choose MOVE_NEXT. Do not choose FOLLOW_UP on a sufficiently covered topic.
   - MOVE_NEXT: enough information for current subtopic.
   - FOLLOW_UP: one important detail missing.
   - CLARIFY: participant answer is unclear.
   - REDUCE_BURDEN: participant seems tired or frustrated.
   - END_INTERVIEW: all topics covered or participant wants to stop.
   - If participant refuses, skips, or says they do not know, accept and move on.
"""

_DM_USER_TEMPLATE = """\
# Interview guide:
{interview_guide}

# Current main question:
{current_main_question}

# Optional probes for the current question:
{current_probes}

Chat history:
{chat_history}

Available response modes:
- participant can select one or more options
- participant can type their own answer
- participant can skip
- participant can ask for clarification

Decide the next action.
Return JSON in this format:
{{
  "current_subtopic_status": "not_started | partially_covered | sufficiently_covered | skipped",
  "subtopic_type": "event_based | descriptive",
  "decision": "FOLLOW_UP | MOVE_NEXT | CLARIFY | REDUCE_BURDEN | END_INTERVIEW",
  "active_branch": {{
    "branch_label": "...",
    "branch_context": "...",
    "branch_status": "needs_story"
  }},
  "pending_branches": [
    {{
      "branch_label": "...",
      "branch_status": "not_explored"
    }}
  ],
  "target_information_gap": "...",
  "reason_for_decision": "..."
}}"""


# =============================================================================
# OpenAI helpers
# =============================================================================

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
        result = json.loads(raw_text)
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
            result = json.loads(m.group())
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


def _format_chat_for_prompt(chat):
    lines = []
    for msg in chat:
        if msg.get("role") == "video":
            lines.append("[Demo video was shown to the participant here]")
        elif msg["role"] == "assistant":
            lines.append(f"Interviewer: {msg['content']}")
            if msg.get("options"):
                opts = " | ".join(o["label"] for o in msg["options"])
                lines.append(f"[Options shown: {opts}]")
        elif msg["role"] == "user":
            lines.append(f"Participant: {msg['content']}")
    return "\n".join(lines) if lines else "(No conversation yet)"


# =============================================================================
# Agent turn
# =============================================================================

def _get_probes_str(q_idx):
    if q_idx < len(QUESTIONS):
        probes = QUESTIONS[q_idx].get("probes", [])
        return "\n".join(f"- {p}" for p in probes) if probes else "(none)"
    return "(none)"


def run_agent_turn(skip_dm=False):
    q_idx = st.session_state.current_question_index
    chat_str = _format_chat_for_prompt(st.session_state.chat)
    current_q = QUESTIONS[q_idx]["main_question"] if q_idx < len(QUESTIONS) else ""
    current_probes = _get_probes_str(q_idx)

    decision = None
    if not skip_dm:
        decision = _run_decision_maker(chat_str, current_q, current_probes)
        action = decision.get("decision", "FOLLOW_UP")
        if action == "MOVE_NEXT":
            q_idx += 1
            st.session_state.current_question_index = q_idx
            if q_idx >= len(QUESTIONS):
                st.session_state.interview_ended = True
                return False, None
            current_q = QUESTIONS[q_idx]["main_question"]
            current_probes = _get_probes_str(q_idx)
        elif action == "END_INTERVIEW":
            st.session_state.interview_ended = True
            return False, None

    show_video = q_idx == B1_INDEX and not st.session_state.get("video_shown", False)
    if show_video:
        st.session_state.video_shown = True

    if q_idx >= len(QUESTIONS):
        st.session_state.interview_ended = True
        return show_video, None

    result = _run_question_generator(chat_str, current_q, decision, current_probes)
    result["question_id"] = QUESTIONS[q_idx]["id"]
    return show_video, result


def _run_decision_maker(chat_str, current_main_question, current_probes):
    user_prompt = _DM_USER_TEMPLATE.format(
        interview_guide=INTERVIEW_GUIDE,
        chat_history=chat_str,
        current_main_question=current_main_question,
        current_probes=current_probes,
    )
    return _call_llm_json(_DM_SYSTEM, user_prompt, label="decision_maker")


def _run_question_generator(chat_str, current_main_question, decision, current_probes):
    decision_str = (
        json.dumps(decision, indent=2) if decision
        else "None -- this is the opening question. Generate the first question for this topic."
    )
    user_prompt = _QG_USER_TEMPLATE.format(
        interview_guide=INTERVIEW_GUIDE,
        chat_history=chat_str,
        current_main_question=current_main_question,
        current_probes=current_probes,
        decision=decision_str,
    )
    return _call_llm_json(_QG_SYSTEM, user_prompt, label="question_generator")


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
        return False, "Drive not configured."
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


def save_async(user_id, chat, agent_logs, config):
    threading.Thread(target=lambda: _do_save(user_id, chat, agent_logs, config), daemon=True).start()


def save_sync(user_id, chat, agent_logs, config):
    try:
        return _do_save(user_id, chat, agent_logs, config)
    except Exception as e:
        return False, str(e)


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


def _infer_question_index(chat):
    id_to_idx = {q["id"]: i for i, q in enumerate(QUESTIONS)}
    for msg in reversed(chat):
        if msg.get("role") == "assistant" and msg.get("question_id"):
            idx = id_to_idx.get(msg["question_id"])
            if idx is not None:
                return idx
    return 0


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
        current_question_index=0,
        video_shown=False,
        interview_ended=False,
        form_generation=0,
        agent_logs=[],
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
[data-testid="stColumn"] div[data-testid="stButton"] button[kind="primary"] {
    height: 100px !important;
}
[data-testid="stColumn"] iframe {
    height: 100px !important; min-height: 100px !important;
    width: 100px !important; min-width: 100px !important;
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
                    q_idx = _infer_question_index(chat)
                    video_shown = any(m.get("role") == "video" for m in chat)
                    st.session_state.update(
                        user_id=pid, drive_config=cfg, chat=chat,
                        current_question_index=q_idx, video_shown=video_shown,
                        waiting=False, phase="active",
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
        "Thank you for attending this interview today.\n\n"
        "We are studying an idea for helping people when others have trouble understanding their speech. "
        "The idea is to use speech transcription as a starting point, and allow the text to be edited "
        "if needed to help repair meaning.\n\n"
        "Later in the interview, we will show you a short demo of the idea and ask what you think about it.\n\n"
        "This is not a test of you. We are testing the idea and learning from your experience.\n\n"
        "You can answer by selecting choices and typing extra comments if you want. "
        "You can skip any question, take a break, or stop at any time."
    )

    st.markdown(INTRO_TEXT)
    st.markdown("")

    if st.button("Continue to interview ->", type="primary", key="btn_intro_continue"):
        new_chat = [{
            "role": "assistant",
            "content": OPENING_QUESTION["question_text"],
            "question_id": OPENING_QUESTION["question_id"],
            "answer_mode": OPENING_QUESTION["answer_mode"],
            "options": OPENING_QUESTION["options"],
        }]
        st.session_state.chat = new_chat
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


# Render chat history
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
            if msg.get("options"):
                opts_text = "   ".join(f"`{o['label']}`" for o in msg["options"])
                st.caption(f"Options: {opts_text}")
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
            "answer_mode": result.get("answer_mode", "multiple_choice"),
            "options": result.get("options", []),
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
        st.session_state[draft_key] = st.session_state.pop("_prefill")

    # ── Interactive options ───────────────────────────────────────────────────
    answer_mode = current_q_msg.get("answer_mode", "multiple_choice") if current_q_msg else "multiple_choice"
    options = current_q_msg.get("options", []) if current_q_msg else []
    q_key = current_q_msg.get("question_id", "q") if current_q_msg else "q"

    if options:
        if answer_mode in ("multiple_choice", "ranking"):
            # Check boxes — collected at Send time
            st.markdown("**Choose all that apply:**")
            for i, opt in enumerate(options):
                st.checkbox(opt["label"], key=f"mopt_{gen}_{q_key}_{i}")

        elif answer_mode == "yes_no_plus_optional_text":
            # Click pre-fills text area; participant can add details before sending
            st.markdown("**Choose one (you can add details below):**")
            n_cols = min(3, len(options))
            cols = st.columns(n_cols)
            for i, opt in enumerate(options):
                with cols[i % n_cols]:
                    if st.button(opt["label"], key=f"ynopt_{gen}_{q_key}_{i}",
                                 use_container_width=True):
                        st.session_state[draft_key] = opt["label"]

    # ── Speak | Text area | Send ──────────────────────────────────────────────
    mic_col, text_col, send_col = st.columns([2, 7, 2])

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
        send_clicked = st.button("Send →", type="primary", use_container_width=True)

    # Enter key sends (Shift+Enter = newline)
    components.html("""
    <script>
    (function() {
        function attach() {
            var ta = window.parent.document.querySelector('textarea[aria-label="response"]');
            if (!ta || ta._enterBound) return;
            ta._enterBound = true;
            ta.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    var btns = window.parent.document.querySelectorAll('button');
                    for (var i = 0; i < btns.length; i++) {
                        if (btns[i].innerText.trim().startsWith('Send')) {
                            btns[i].click(); break;
                        }
                    }
                }
            });
        }
        attach();
        new MutationObserver(attach).observe(window.parent.document.body, {childList:true, subtree:true});

        // Inject CSS into mic-recorder iframe to make its button fill the full 100px height
        function resizeMicBtn() {
            var iframes = window.parent.document.querySelectorAll('[data-testid="stColumn"] iframe');
            iframes.forEach(function(iframe) {
                if (iframe._micStyled) return;
                try {
                    var d = iframe.contentDocument || iframe.contentWindow.document;
                    if (!d || !d.head) return;
                    var style = d.createElement('style');
                    style.textContent =
                        'html, body { margin: 0 !important; padding: 0 !important; ' +
                        '  height: 100px !important; overflow: hidden !important; }' +
                        'button { width: 100% !important; height: 100px !important; ' +
                        '  box-sizing: border-box !important; border-radius: 8px !important; ' +
                        '  font-size: 1.1rem !important; cursor: pointer !important; }';
                    d.head.appendChild(style);
                    iframe._micStyled = true;
                } catch(e) {}
            });
        }
        setInterval(resizeMicBtn, 300);
    })();
    </script>
    """, height=0)

    if send_clicked:
        typed_text = (typed or st.session_state.get(draft_key) or "").strip()

        selected = []
        if answer_mode in ("multiple_choice", "ranking"):
            selected = [
                opt["label"]
                for i, opt in enumerate(options)
                if st.session_state.get(f"mopt_{gen}_{q_key}_{i}")
            ]

        parts = []
        if selected:
            parts.append("; ".join(selected))
        if typed_text:
            parts.append(typed_text)
        answer = ". ".join(parts) if parts else None

        if answer:
            st.session_state.form_generation += 1
            st.session_state.chat.append({"role": "user", "content": answer})
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
                st.rerun()
            else:
                st.warning("Could not transcribe. Please try again or type your response.")

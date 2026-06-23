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

MODEL = "gpt-5-nano"

CLOSING_MESSAGE = (
    "Thank you for sharing your experience and feedback with us. "
    "Your answers will help us understand whether transcription plus editing could support "
    "communication repair in everyday life, what parts may be useful or difficult, and how "
    "the system should be improved to better fit the needs of people with dysarthria."
)


# =============================================================================
# Single-agent system prompt + user-message builder
# =============================================================================

_AGENT_SYSTEM = """\
You are an accessibility-aware semi-structured interview chatbot.
You are interviewing people with dysarthria about everyday communication and about an early technology idea that may help when other people have trouble understanding speech.
The participant may have difficulty speaking, typing, using automatic speech recognition, or sustaining effort. Some participants may type slowly. Some participants may choose ready-made suggestions to reduce effort. Keep the interview respectful, brief, flexible, and low-burden.
The goal is to understand the participant's lived communication experience, not to make them produce long answers.
The interview guide and the interview history are provided below.
Core behavior
Ask one question at a time.
Use plain language.
Keep each message short.
Ask interview questions, not survey questions.
Avoid broad questions that sound like they require a long answer. Prefer questions that can be answered with one word, a short phrase, selected suggestions, or skip.
Do not ask the participant to describe a specific past event unless they volunteer one.
Avoid questions like:
"Can you think of a time when..."
"Tell me about a situation where..."
"What happened?"
"Can you walk me through..."
These can create too much burden.
Let the participant answer in their own words first.
The participant may answer by:
speaking,
typing,
selecting one or more suggestions,
combining suggestions with typed text,
giving a short answer,
giving a partial answer,
saying "I don't know,"
or skipping.
Treat all of these as valid forms of participation.
Do not ask the participant to explain more just because the answer is short.
Do not ask the participant to rephrase unless the meaning is unclear and the clarification is important.
The interface includes a textbox, a microphone button, and a "suggestions" button.
Suggestions are an optional accessibility support. They are not the default interview mode.
Only show suggestions when:
the participant clicks or asks for suggestions,
the participant seems unsure or asks for examples,
the participant gives no answer and may need support,
or the interface explicitly asks you to generate suggestions.
When suggestions are shown, the participant may select one, select several, type their own answer, combine selected answers with typed text, or skip.
Suggestions should be easy to choose from, but they should not feel like the expected answers.
Usually provide 4–6 suggestions, plus "Other" and "Skip."
Use "None of these" when appropriate.
Avoid vague references such as "it," "that," or "these moments." Be clear about what you mean.
Avoid using the word "repair" with participants unless you explain it. Prefer:
"when someone does not understand you"
"help them understand"
"make the message clearer"
"correct the transcript"
"decide whether to keep trying"
Do not assume that repeating speech is the main strategy. People may use speech, gesture, pointing, writing, typing, AAC, ASL/sign, saved messages, partner help, context, or may decide to move on.
Do not assume typing is easy.
Do not assume speech recognition works well.
Do not assume the participant wants to keep trying until the other person understands.
Use follow-up questions sparingly. A follow-up is okay when:
the answer is unclear and clarification is important,
the participant says something especially important or surprising,
the answer helps explain why a technology would or would not fit,
or the participant seems comfortable giving more detail.
Do not ask a follow-up only because the answer is short.
Ask at most one follow-up after a main question unless the participant clearly wants to say more.
If the participant skips, says "I don't know," seems tired, gives minimal answers, or appears frustrated, accept the answer and move on.
If the participant has already answered a later topic, do not ask the same thing again. Mark that topic as covered and move to the next useful topic.
Target length: 6–8 main questions, with 0–3 total follow-ups.
If participant burden appears high, use the short version, reduce follow-ups, and prioritize the most important questions.
Managing participant burden
Watch for signs that the participant may want a lower-burden interview, such as:
very short answers,
repeated skips,
"I don't know,"
frustration,
long pauses,
difficulty typing,
difficulty using speech recognition,
or comments about being tired.
When burden seems high:
ask fewer follow-ups,
use simpler wording,
move through the interview more quickly,
offer the suggestions button as an option if appropriate,
and consider switching to the short version.
Do not say the participant is doing badly.
Do not pressure the participant to give longer answers.
Opening message
If this is the start of the interview, use this opening:
Thank you for meeting with us.
We are interested in your everyday experiences communicating with other people, especially times when someone has trouble understanding you.
Later, we will show you a short demo of an early technology idea and ask what you think about it.
This is not a test of you. We are learning from your experience.
There are no right or wrong answers. Short answers are fine. You can skip any question.
You can answer by speaking, typing, choosing suggested answers, or using a mix of these.
If helpful, you can press the suggestions button to see possible answers.
Do you have any questions before we begin?
If the opening has already been shown and the participant has no questions, proceed to A1.
Interview guide
Section A. Everyday communication
Design principle for Section A:
Questions should not ask the participant to tell a story, recall a specific episode, imagine a situation, or explain a sequence of events. Each question should be answerable with a short word, phrase, selected suggestion, or skip. Free text is always allowed, but not required.
A1. People they communicate with
Main question:
Who do you communicate with most often?
Suggestions if requested:
Family
Friends
Caregivers or support workers
Doctors or health workers
People at work or school
Store or service workers
Other
Skip
Possible follow-up:
Who is usually easiest to communicate with?
Research purpose:
Understand the participant's everyday communication context without asking for a story.
A2. Current ways of helping someone understand
Main question:
When someone does not understand you, what do you usually do?
Suggestions if requested:
Say it again
Say it differently
Gesture or point
Type or write
Use AAC, sign, or another device
Ask someone else to help
Let it go
Other
Skip
Possible follow-up:
Do you usually use one of these, or more than one?
Research purpose:
Learn the participant's own communication strategies without assuming that repeating speech is the main strategy.
A3. What is hardest
Main question:
What is usually hardest when someone does not understand you?
Suggestions if requested:
Saying it again
Saying it another way
Typing or writing
Using a device
Feeling rushed
The other person gets impatient
Losing what I wanted to say
Nothing is especially hard
Other
Skip
Possible follow-up:
Which one is hardest?
Research purpose:
Identify burdens that a new technology should reduce, not add to.
A4. What the participant does that helps
Main question:
What usually works best for helping someone understand you?
Suggestions if requested:
Saying it again
Saying it differently
Using fewer words
Gesturing or pointing
Typing or writing
Using AAC, sign, or saved messages
Asking someone else to help
Letting it go
Nothing works well
Other
Skip
Possible follow-up:
Which way takes the least effort?
Research purpose:
Understand which participant-side strategies are most effective or least burdensome.
A5. What other people can do that helps
Main question:
What can other people do that helps you be understood?
Suggestions if requested:
Be patient
Wait longer
Ask yes/no questions
Guess from context
Watch my gestures
Read what I type or show
Ask someone who knows me
Move to a quieter place
Nothing helps much
Other
Skip
Possible follow-up:
What is most helpful for other people to do?
Research purpose:
Understand listener-side and environment-side supports, separate from what the participant does.
A6. When it is harder
Main question:
When is it harder for people to understand you?
Suggestions if requested:
With strangers
In noisy places
When people are rushed
When I am tired
When the message is important
On the phone or video call
In groups
It is about the same
Other
Skip
Possible follow-up:
Which situation is hardest?
Research purpose:
Understand variation by listener, setting, fatigue, urgency, and communication channel without asking a yes/no question or requiring a story.
A7. Desired support before demo
Main question:
What would you want help with, if anything?
Suggestions if requested:
Helping others understand my speech
Reducing how much I repeat
Making typing easier
Giving me word choices
Saving common messages
Helping the other person wait
Helping in noisy places
I do not want technology help
I am not sure
Other
Skip
Possible follow-up:
Which kind of help would matter most?
Research purpose:
Elicit participant-centered needs before showing the prototype, without requiring the participant to invent a technology idea.
Section B. Reaction to demo
Before Section B, ask:
Next, we would like to show a short demo video of an early idea. Is now an okay time to watch it?
Suggestions if requested:
Yes
I need a break
Skip the demo
I'm not sure
If the participant is ready, show the short demo video.
After the demo, say:
That was an early idea, not a finished system. We want to learn what seems useful, not useful, realistic, unrealistic, or too much work for you.
B1. First reaction after demo
Main question:
After seeing the demo, what is your first reaction?
Suggestions if requested:
I like it
I partly like it
I do not like it
Interesting, but I am not sure
Seems too much work
Not useful for me
Other
Skip
Possible follow-up:
What is the main reason?
Research purpose:
Capture initial reaction without assuming the idea is good.
Branching instruction:
If the participant likes it or partly likes it, ask B2-like next.
If the participant does not like it, says it is not useful, or says it is too much work, ask B2-dislike next.
If the participant is unsure, mixed, or skips, ask B2-mixed next.
B2-like. What they like
Main question:
What seems useful?
Suggestions if requested:
Transcript
Word choices
Less repeating
Helps the other person
Gives me control
Could save time
I am not sure
Other
Skip
Possible follow-up:
Which part seems most useful?
Research purpose:
If the participant reacts positively, understand the perceived benefit before asking about concerns.
B2-dislike. What they do not like
Main question:
What seems not useful?
Suggestions if requested:
Too slow
Too much effort
Transcript may be wrong
Hard to choose options
Typing is hard
Other person may not wait
I have better ways now
I am not sure
Other
Skip
Possible follow-up:
Which problem matters most?
Research purpose:
If the participant reacts negatively, understand the main objection before asking about possible benefits.
B2-mixed. Useful or not useful
Main question:
What seems useful or not useful?
Suggestions if requested:
Some parts seem useful
Some parts seem too much work
Depends where I use it
Depends who I talk to
I worry the transcript will be wrong
I am not sure
Other
Skip
Possible follow-up:
Which part matters most?
Research purpose:
Allow a mixed reaction without forcing either positive or negative framing.
B3. Where it might help
Main question:
Where might this help?
Suggestions if requested:
Doctor or appointment
Store or restaurant
With strangers
Work or school
Phone or video call
At home
Nowhere
Other
Skip
Possible follow-up:
Where would it help most?
Research purpose:
Identify possible use contexts with a low-burden question.
B4. Where it might not help
Main question:
Where would this not help?
Suggestions if requested:
Fast conversation
Noisy place
Public place
Private conversation
With people who know me well
When I am tired
Anywhere
I am not sure
Other
Skip
Possible follow-up:
Where would it be hardest to use?
Research purpose:
Identify boundaries of use without combining helpful and not-helpful situations in one confusing question.
B5. If the transcript is wrong
Main question:
If the transcript is wrong, could it still help?
Suggestions if requested:
Yes, if key words are right
Yes, if the main idea is right
Yes, if it helps the other person guess
No, mistakes would confuse people
No, I would not trust it
Depends
I am not sure
Other
Skip
Possible follow-up:
What kind of mistake would be worst?
Research purpose:
Explore whether imperfect speech recognition can still support understanding.
B6. Easiest way to correct or clarify
Main question:
If the system guessed wrong, what would be easiest?
Suggestions if requested:
Pick the right word
Pick from a few choices
Tap the wrong word
Type a short fix
Use a saved phrase
Gesture or point
Let the other person help
Do not fix it
Other
Skip
Possible follow-up:
Which would take the least effort?
Research purpose:
Identify low-effort correction options without assuming that speaking again, typing, or detailed editing is easy.
B7. Concerns after discussing possible benefits
Main question:
What would worry you about using this?
Suggestions if requested:
Too slow
Too much effort
Transcript mistakes
Hard to use while talking
Other person may not wait
Privacy
Feeling awkward
No worries
Other
Skip
Possible follow-up:
Which worry matters most?
Research purpose:
Surface major concerns after giving space for benefits, especially for participants who initially liked the idea.
Branching instruction:
If the participant already gave strong concerns in B2-dislike, do not repeat this question unless there is a new concern to ask about. Move to B8.
B8. What designers should understand
Main question:
What should designers remember?
Suggestions if requested:
Keep it low effort
Do not assume typing is easy
Do not assume speaking again works
Support gesture, AAC, or sign
Make it work in real conversations
Let the other person help
Give me control
Other
Skip
Possible follow-up:
What is most important?
Research purpose:
Elicit participant-centered design implications without asking for generic feature improvements.
Closing
C1. Anything missing
Main question:
Is there anything important we did not ask?
Suggestions if requested:
Yes
No
I'm not sure
Other
Skip
Possible follow-up:
What else should we know?
Research purpose:
Allow participant-led concerns or insights not anticipated by the guide.
Short version if participant fatigue or burden is high
Use only these questions:
Who do you communicate with most often?
When someone does not understand you, what do you usually do?
What is usually hardest when someone does not understand you?
What can other people do that helps you be understood?
After seeing the demo, what is your first reaction?
Based on that reaction, ask the most relevant B2 question: useful, not useful, or mixed.
If the system guessed wrong, what would be easiest?
What should designers remember?
Is there anything important we did not ask?
In the short version, ask few or no follow-ups.
Runtime inputs
The system should provide the chatbot with these inputs each turn.
INTERVIEW_HISTORY:
A compact record of the interview so far. This should include the question IDs already asked, the participant's selected suggestions, and any typed or spoken free text.
DEMO_STATUS:
One of:
"not_shown"
"ready_to_show"
"shown"
"skipped"
PARTICIPANT_BURDEN_NOTES:
Any observed signs of burden, fatigue, frustration, slow typing, repeated skipping, or preference for suggestions.
Task
Generate the next interview question or follow-up according to the guide and the interview history.
Use the participant's previous answers to avoid repetition.
Prefer moving forward over asking for more detail when the participant gives a short answer.
Choose the next question based on the participant's prior answer when the guide gives branching instructions.
Return only JSON.
Output format
Use this format:
{
  "question_id": "...",
  "message_to_participant": "...",
  "suggestions_if_requested": [
    {"label": "..."}
  ],
  "question_type": "main | follow_up | transition | closing"
}
Do not include internal reasoning in the JSON.
The participant should see only message_to_participant.
The suggestions in suggestions_if_requested are for the suggestions button. Do not show them automatically unless the participant clicks the suggestions button or the interface requests them.
The participant may always type, speak, select one suggestion, select multiple suggestions, combine selected suggestions with typed text, or skip. The interface should allow these options by default.
"""


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

def _build_interview_history(chat):
    """Build the compact INTERVIEW_HISTORY list the single agent expects."""
    history = []
    i = 0
    msgs = [m for m in chat if m.get("role") in ("assistant", "user", "video")]
    while i < len(msgs):
        msg = msgs[i]
        if msg.get("role") == "assistant":
            entry = {
                "question_id": msg.get("question_id", ""),
                "message_to_participant": msg.get("content", ""),
                "participant_response": None,
            }
            # Look for the immediately following user message
            if i + 1 < len(msgs) and msgs[i + 1].get("role") == "user":
                i += 1
                user_msg = msgs[i]
                raw = user_msg.get("content", "")
                # Split selected suggestions (prefixed with "✓ ") from free text
                selected = [
                    p.strip().lstrip("✓").strip()
                    for p in raw.split("\n")
                    if p.strip().startswith("✓")
                ]
                free = " ".join(
                    p.strip() for p in raw.split("\n")
                    if not p.strip().startswith("✓")
                ).strip()
                entry["participant_response"] = {
                    "selected_suggestions": selected,
                    "free_text": free,
                }
            history.append(entry)
        i += 1
    return history


def run_agent_turn():
    chat = st.session_state.chat
    demo_status = st.session_state.get("demo_status", "not_shown")

    # Count skips / short answers for burden notes
    skip_count = sum(
        1 for m in chat
        if m.get("role") == "user" and m.get("content", "").lower().strip() in ("skip", "i don't know", "")
    )
    burden_notes = f"{skip_count} skip(s) or empty answers so far." if skip_count else "No signs of high burden observed."

    history = _build_interview_history(chat)
    user_prompt = (
        f"INTERVIEW_HISTORY:\n{json.dumps(history, indent=2)}\n\n"
        f"DEMO_STATUS:\n{demo_status}\n\n"
        f"PARTICIPANT_BURDEN_NOTES:\n{burden_notes}"
    )

    result = _call_llm_json(_AGENT_SYSTEM, user_prompt, label="agent")

    # Detect end-of-interview
    q_type = result.get("question_type", "")
    q_id = result.get("question_id", "")
    if q_type == "closing" and not result.get("message_to_participant", "").strip():
        st.session_state.interview_ended = True
        return False, None

    # Trigger demo video when agent first moves into Section B
    show_video = (
        demo_status == "not_shown"
        and q_id.upper().startswith("B")
    )
    if show_video:
        st.session_state.demo_status = "shown"

    # Normalise output to the fields the UI expects
    result["question_text"] = result.get("message_to_participant", "")
    result["options"] = result.get("suggestions_if_requested", [])
    result["answer_mode"] = "multiple_choice"

    return show_video, result


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
    width: 100% !important;
}
/* Option grid cards */
div[data-testid="stColumn"] div[data-testid="stButton"] button[kind="secondary"] {
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
                    st.session_state.update(
                        user_id=pid, drive_config=cfg, chat=chat,
                        demo_status="shown" if video_shown else "not_shown",
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

    # ── Interactive options (metadata only — rendered below input row) ──────────
    answer_mode = current_q_msg.get("answer_mode", "multiple_choice") if current_q_msg else "multiple_choice"
    options = current_q_msg.get("options", []) if current_q_msg else []
    q_key = current_q_msg.get("question_id", "q") if current_q_msg else "q"

    # ── Speak | Text area | Send ──────────────────────────────────────────────
    mic_col, text_col, send_col = st.columns([1, 9, 2])

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
    })();
    </script>
    """, height=0)

    # ── Suggested answers (hidden until toggled) ──────────────────────────────
    if options:
        show_key = f"show_opts_{gen}_{q_key}"
        if show_key not in st.session_state:
            st.session_state[show_key] = False

        if not st.session_state[show_key]:
            if st.button("💡 Show suggested answers", key=f"show_opts_btn_{gen}_{q_key}"):
                st.session_state[show_key] = True
                st.rerun()
        else:
            if answer_mode in ("multiple_choice", "ranking"):
                grid_cols = st.columns(4)
                _opt_toggled = False
                for i, opt in enumerate(options):
                    sel_key = f"mopt_{gen}_{q_key}_{i}"
                    if sel_key not in st.session_state:
                        st.session_state[sel_key] = False
                    is_sel = st.session_state[sel_key]
                    label = f"✓  {opt['label']}" if is_sel else opt["label"]
                    with grid_cols[i % 4]:
                        if st.button(label, key=f"mbtn_{gen}_{q_key}_{i}",
                                     type="primary" if is_sel else "secondary",
                                     use_container_width=True):
                            st.session_state[sel_key] = not is_sel
                            _opt_toggled = True
                if _opt_toggled:
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

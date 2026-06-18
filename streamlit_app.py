"""
SparkMe -- Streamlit web chatbot
Run locally:   streamlit run streamlit_app.py
Deploy:        push to GitHub -> connect Streamlit Community Cloud

Required Streamlit Secrets:
  OPENAI_API_KEY        -- your OpenAI key
  GDRIVE_FOLDER_ID      -- ID of the Google Drive folder to save sessions into
  GDRIVE_CLIENT_ID      -- OAuth 2.0 client ID (Desktop app type)
  GDRIVE_CLIENT_SECRET  -- OAuth 2.0 client secret
  GDRIVE_REFRESH_TOKEN  -- long-lived refresh token (run get_refresh_token.py once)
"""

import asyncio
import hashlib
import io
import json
import os
import threading
import time
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder

# Load .env for local dev; Streamlit Cloud injects secrets as env vars
load_dotenv(override=True)

# SparkMe reads many env vars at import time.
# Set defaults so the app works without adding every variable to Streamlit Secrets.
os.environ.setdefault("LOGS_DIR", "logs")
os.environ.setdefault("DATA_DIR", "data")
os.environ.setdefault("USER_AGENT_PROFILES_DIR", "data/sample_user_profiles")
os.environ.setdefault("INTERVIEW_PLAN_PATH", "data/configs/topics.json")
os.environ.setdefault("MODEL_NAME", "gpt-5-nano")
os.environ.setdefault("AGENDA_MANAGER_MODEL_NAME", "gpt-5-nano")
os.environ.setdefault("EXPLORATION_PLANNER_MODEL_NAME", "gpt-5-nano")
os.environ.setdefault("EMBEDDING_BACKEND", "openai")
os.environ.setdefault("MAX_EVENTS_LEN", "10")
os.environ.setdefault("MAX_CONSIDERATION_ITERATIONS", "4")
os.environ.setdefault("USE_BASELINE_PROMPT", "false")
os.environ.setdefault("EVAL_MODE", "false")
os.environ.setdefault("COMPLETION_METRIC", "minimum_threshold")
os.environ.setdefault("SESSION_TIMEOUT_MINUTES", "10")
os.environ.setdefault("MEMORY_THRESHOLD_FOR_UPDATE", "10")
os.environ.setdefault("EXPLORATION_PLANNER_GAMMA", "0")

# --- Guard: OpenAI key must be present ------------------------------------
if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "**OPENAI_API_KEY is not set.**\n\n"
        "- **Local:** add `OPENAI_API_KEY=sk-...` to your `.env` file.\n"
        "- **Streamlit Cloud:** Settings → Secrets."
    )
    st.stop()

# --- SparkMe imports -------------------------------------------------------
from src.interview_session.interview_session import InterviewSession
from src.interview_session.session_models import Message, MessageType


# ==========================================================================
# Google Drive helpers
# ==========================================================================

def _get_drive_config():
    """Capture Drive credentials from Streamlit secrets (call in main thread only)."""
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
    """Build a Drive service authenticated as the real user via OAuth refresh token."""
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
    """Return the Drive folder ID for name under parent_id, creating it if needed."""
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


def _upload_dir_tree(local_dir, drive_folder_id, svc, _folder_cache=None):
    """Recursively mirror local_dir into drive_folder_id, creating subfolders as needed."""
    if _folder_cache is None:
        _folder_cache = {}
    if not os.path.isdir(local_dir):
        return
    for entry in os.scandir(local_dir):
        if entry.is_file():
            try:
                with open(entry.path, "rb") as f:
                    _upsert_bytes(entry.name, f.read(), drive_folder_id, svc)
            except Exception:
                pass  # skip unreadable files silently
        elif entry.is_dir():
            key = (entry.name, drive_folder_id)
            if key not in _folder_cache:
                _folder_cache[key] = _get_or_create_folder(entry.name, drive_folder_id, svc)
            _upload_dir_tree(entry.path, _folder_cache[key], svc, _folder_cache)


def _upsert_bytes(name, data, folder_id, svc):
    """Upload bytes as name into folder_id, overwriting any existing file."""
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
    """Download a Drive file and return its bytes."""
    from googleapiclient.http import MediaIoBaseDownload
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, svc.files().get_media(fileId=file_id))
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()


def _latest_agenda_path(user_id):
    """Return (session_num, local_path) for the highest-numbered session_agenda.json, or (None, None)."""
    base = os.path.join("logs", user_id, "execution_logs")
    if not os.path.exists(base):
        return None, None
    dirs = [d for d in os.listdir(base) if d.startswith("session_") and os.path.isdir(os.path.join(base, d))]
    if not dirs:
        return None, None
    dirs.sort(key=lambda d: int(d.split("_")[1]), reverse=True)
    for d in dirs:
        path = os.path.join(base, d, "session_agenda.json")
        if os.path.exists(path):
            return int(d.split("_")[1]), path
    return None, None


def _do_save(user_id, chat, config):
    """Core Drive save: chat history + latest session agenda. Returns (ok, message)."""
    if not config.get("folder_id") or not config.get("refresh_token"):
        return False, "Drive not configured."
    svc = _make_service(config)
    root = config["folder_id"]

    # Per-participant subfolder
    pfolder = _get_or_create_folder(f"participant_{user_id}", root, svc)

    # Chat history
    _upsert_bytes(
        "chat_history.json",
        json.dumps(chat, ensure_ascii=False, indent=2).encode("utf-8"),
        pfolder, svc,
    )

    # Latest session agenda (enables resume)
    session_num, agenda_path = _latest_agenda_path(user_id)
    if agenda_path:
        with open(agenda_path, "rb") as f:
            _upsert_bytes(f"session_agenda_s{session_num}.json", f.read(), pfolder, svc)

    # Upload full agent logs tree (raw agent responses, token stats, etc.)
    logs_user_dir = os.path.join("logs", user_id)
    if os.path.isdir(logs_user_dir):
        logs_drive = _get_or_create_folder("logs", pfolder, svc)
        _upload_dir_tree(logs_user_dir, logs_drive, svc)

    # Upload data files (memory bank, question bank) for resume continuity
    data_user_dir = os.path.join("data", user_id)
    if os.path.isdir(data_user_dir):
        data_drive = _get_or_create_folder("data", pfolder, svc)
        _upload_dir_tree(data_user_dir, data_drive, svc)

    # Update the researcher's participant log in the root folder
    _update_participants_log(user_id, root, svc)

    return True, "Saved."


def _update_participants_log(user_id, root_folder_id, svc):
    """Maintain participants_log.json in the root Drive folder."""
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
        pass  # log errors silently; don't disrupt the interview


def save_async(user_id, chat, config):
    """Fire-and-forget background save (called after each interviewer turn)."""
    threading.Thread(
        target=lambda: _do_save(user_id, chat, config),
        daemon=True,
    ).start()


def save_sync(user_id, chat, config):
    """Blocking save (called at session end). Returns (ok, message)."""
    try:
        return _do_save(user_id, chat, config)
    except Exception as e:
        return False, str(e)


def restore_from_drive(participant_id, config):
    """
    Download a returning participant's data from Drive.
    Returns (chat_history: list, found: bool).
    Side-effect: writes session_agenda file to the local filesystem so SparkMe
    picks up the existing session state automatically on InterviewSession init.
    """
    try:
        if not config.get("folder_id") or not config.get("refresh_token"):
            return [], False
        svc = _make_service(config)
        root = config["folder_id"]

        # Find participant subfolder
        q = (
            f"name='participant_{participant_id}' and '{root}' in parents "
            "and mimeType='application/vnd.google-apps.folder' and trashed=false"
        )
        folders = svc.files().list(q=q, fields="files(id)").execute().get("files", [])
        if not folders:
            return [], False
        pfolder = folders[0]["id"]

        # List files in the folder
        files = {
            f["name"]: f["id"]
            for f in svc.files().list(
                q=f"'{pfolder}' in parents and trashed=false",
                fields="files(id, name)",
            ).execute().get("files", [])
        }

        # Download chat history for UI display
        chat = []
        if "chat_history.json" in files:
            chat = json.loads(_download_bytes(files["chat_history.json"], svc).decode("utf-8"))

        # Restore latest session_agenda so SparkMe resumes the existing session
        agenda_files = [(n, fid) for n, fid in files.items() if n.startswith("session_agenda_s")]
        if agenda_files:
            # Sort by session number (session_agenda_sN.json)
            agenda_files.sort(key=lambda x: int(x[0].replace("session_agenda_s", "").replace(".json", "")))
            latest_name, latest_id = agenda_files[-1]
            session_num = int(latest_name.replace("session_agenda_s", "").replace(".json", ""))
            local_path = os.path.join("logs", participant_id, "execution_logs", f"session_{session_num}", "session_agenda.json")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(_download_bytes(latest_id, svc))

        # Restore data files (memory bank, question bank) so topic memory carries over
        q_data = (
            f"name='data' and '{pfolder}' in parents "
            "and mimeType='application/vnd.google-apps.folder' and trashed=false"
        )
        data_folders = svc.files().list(q=q_data, fields="files(id)").execute().get("files", [])
        if data_folders:
            data_drive_id = data_folders[0]["id"]
            local_data_dir = os.path.join("data", participant_id)
            os.makedirs(local_data_dir, exist_ok=True)
            data_files_list = svc.files().list(
                q=f"'{data_drive_id}' in parents and trashed=false",
                fields="files(id, name)",
            ).execute().get("files", [])
            for df in data_files_list:
                local_file = os.path.join(local_data_dir, df["name"])
                with open(local_file, "wb") as f:
                    f.write(_download_bytes(df["id"], svc))

        return chat, True

    except Exception:
        return [], False


# ==========================================================================
# Whisper transcription
# ==========================================================================

def _transcribe(audio_bytes):
    """Send audio bytes to OpenAI Whisper API and return the transcript text."""
    try:
        import openai
        client = openai.OpenAI()
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=("recording.wav", io.BytesIO(audio_bytes), "audio/wav"),
        )
        return result.text.strip()
    except Exception as e:
        return ""


# ==========================================================================
# SparkMe session management
# ==========================================================================

def _run_bg(session, loop):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(session.run())


def _create_session(user_id, previous_chat=None):
    loop = asyncio.new_event_loop()
    session = InterviewSession(
        interaction_mode="api",
        user_config={"user_id": user_id},
        interview_config={
            "interview_description": "Understanding the impact of AI in the workforce",
            "interview_plan_path": os.getenv("INTERVIEW_PLAN_PATH", "data/configs/topics.json"),
            "interview_evaluation": os.getenv("COMPLETION_METRIC", "minimum_threshold"),
        },
    )

    # For returning participants: the Interviewer decides whether to give a fresh
    # introduction or a natural continuation by checking two things:
    #   1. Its own event stream (always empty at session start — can't inject)
    #   2. session_agenda.last_meeting_summary (non-empty → continuation prompt)
    #
    # SparkMe never writes session_agenda.json during a session (only snapshots),
    # so restore_from_drive has nothing to download.  We therefore inject a
    # summary directly into the live session_agenda object here, before run()
    # fires.  The Interviewer will then use "introduction_continue_session".
    if previous_chat:
        if not session.session_agenda.last_meeting_summary:
            turns = []
            for msg in previous_chat:
                prefix = "Interviewer" if msg["role"] == "assistant" else "Participant"
                body = msg["content"]
                if len(body) > 300:
                    body = body[:300] + "..."
                turns.append(f" - {prefix}: {body}")
            summary = "Previous session transcript:\n" + "\n".join(turns[:30])

            # Append covered subtopics from the restored session agenda so the
            # AgendaManager knows which subtopics are already done and won't
            # re-evaluate or re-ask about them in the new session.
            _, agenda_path = _latest_agenda_path(user_id)
            if agenda_path and os.path.exists(agenda_path):
                try:
                    with open(agenda_path) as f:
                        agenda = json.load(f)
                    covered_lines = []
                    topic_dict = (
                        agenda.get("interview_topic_manager", {})
                              .get("core_topic_dict", {})
                    )
                    for topic in topic_dict.values():
                        for sub_id, sub in topic.get("required_subtopics", {}).items():
                            if sub.get("is_covered") and sub.get("final_summary"):
                                covered_lines.append(
                                    f" - {sub_id} ({sub['description']}): {sub['final_summary']}"
                                )
                    if covered_lines:
                        summary += (
                            "\n\nSubtopics already fully covered in previous sessions"
                            " (do NOT revisit these):\n"
                            + "\n".join(covered_lines)
                        )
                except Exception:
                    pass  # If parsing fails, fall back to transcript-only summary

            session.session_agenda.last_meeting_summary = summary

    threading.Thread(target=_run_bg, args=(session, loop), daemon=True).start()
    return session, loop


# ==========================================================================
# Page setup
# ==========================================================================

st.set_page_config(page_title="SparkMe Interview", page_icon="mic", layout="centered")
st.title("SparkMe Interview")

if "phase" not in st.session_state:
    st.session_state.update(
        phase="id_entry",   # "id_entry" | "intro" | "active"
        user_id=None,
        session=None,
        loop=None,
        chat=[],
        waiting=False,
        drive_config=None,
        session_saved=False,
        last_audio_hash=None,   # dedup guard for audio recorder
        user_draft="",          # text currently in the input box
    )


# ==========================================================================
# Phase: participant ID entry
# ==========================================================================

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
        if st.button("Start interview →", type="primary", key="btn_new"):
            user_id = "P-" + uuid.uuid4().hex[:6].upper()
            cfg = _get_drive_config()
            st.session_state.update(
                user_id=user_id,
                drive_config=cfg,
                phase="intro",
            )
            st.rerun()

    with tab_return:
        st.markdown("Enter the Participant ID you received when you started.")
        pid_input = st.text_input("Participant ID (e.g. P-ABC123):", key="pid_input")
        if st.button("Resume interview →", key="btn_return"):
            pid = pid_input.strip().upper()
            if not pid:
                st.warning("Please enter your Participant ID.")
            else:
                cfg = _get_drive_config()
                with st.spinner(f"Looking up session for {pid}..."):
                    chat, found = restore_from_drive(pid, cfg)
                if found:
                    with st.spinner("Resuming your session..."):
                        session, loop = _create_session(pid, previous_chat=chat)
                    # If the chatbot was the last to speak, SparkMe will still
                    # generate a continuation message internally (we can't stop it),
                    # but we should silently discard it — the chatbot already asked
                    # something, so just show the history and let the participant reply.
                    # If the participant was last to speak, we DO want SparkMe's response.
                    last_role = chat[-1]["role"] if chat else "user"
                    skip_first = last_role == "assistant"
                    st.session_state.update(
                        user_id=pid,
                        drive_config=cfg,
                        chat=chat,
                        session=session,
                        loop=loop,
                        waiting=not skip_first,   # don't show spinner if we'll discard the msg
                        skip_first_bot_msg=skip_first,
                        phase="active",
                    )
                    st.rerun()
                else:
                    st.error(
                        f"No session found for **{pid}**.  \n"
                        "Please double-check your ID and try again.  \n"
                        "If you have not started before, use the **New participant** tab."
                    )

    st.stop()


# ==========================================================================
# Phase: intro (new participants only — shown before session starts)
# ==========================================================================

if st.session_state.phase == "intro":

    INTRO_TEXT = """\
Thank you for attending this semi-structured interview today.

We are studying an idea for helping people when others have trouble understanding their speech. \
The idea is to use speech transcription as a starting point, and then allow the text to be edited \
if needed to help repair meaning.

In this session, we will show you a short demo of the idea. After that, we will ask what you think about it.

This is not a test of you. We are testing the idea and learning from your experience.

You can answer by selecting choices and typing extra comments if you want. \
You can skip any question, take a break, or stop at any time."""

    st.markdown(INTRO_TEXT)

    st.markdown("#### Demo Video")
    st.markdown(
        '<iframe src="https://drive.google.com/file/d/1FCfzZslMnuyQAPhcZoiACrx0sWaYskxV/preview" '
        'width="100%" height="450" allow="autoplay" style="border:none;"></iframe>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    if st.button("Continue to interview →", type="primary", key="btn_intro_continue"):
        uid = st.session_state.user_id
        with st.spinner("Starting your session..."):
            session, loop = _create_session(uid)
        FIRST_QUESTION = "What is your first reaction to this idea?"
        st.session_state.update(
            session=session,
            loop=loop,
            chat=[{"role": "assistant", "content": FIRST_QUESTION}],
            waiting=False,
            skip_first_bot_msg=True,
            phase="active",
        )
        st.rerun()

    st.stop()


# ==========================================================================
# Phase: active interview
# ==========================================================================

user_id = st.session_state.user_id
session  = st.session_state.session
cfg      = st.session_state.drive_config

# Apply any pending draft (transcript or clear) BEFORE the text_area renders.
# We can't set a widget key after the widget renders, so we use a bridge variable.
if "_pending_draft" in st.session_state:
    st.session_state.user_draft = st.session_state.pop("_pending_draft")

# Sidebar: show participant ID as a persistent reminder
with st.sidebar:
    st.markdown("### Your Participant ID")
    st.code(user_id, language=None)
    st.caption(
        "Keep this ID safe. If you need to leave and continue later, "
        "use the **Returning participant** tab on the start screen and enter this ID."
    )

# Poll for new interviewer messages
new_msgs = session.user.get_and_clear_messages()
if new_msgs:
    if st.session_state.get("skip_first_bot_msg"):
        # Chatbot was last to speak: discard SparkMe's automatic continuation
        # message so the participant just sees their history and the input box.
        st.session_state.skip_first_bot_msg = False
        new_msgs = new_msgs[1:]
    for m in new_msgs:
        st.session_state.chat.append({"role": "assistant", "content": m["content"]})
    # Only stop waiting if we actually received real messages (not just discarded one).
    # If new_msgs is empty after discarding, leave waiting as-is so the
    # polling loop keeps running if the user has already sent a message.
    if new_msgs:
        st.session_state.waiting = False
    if new_msgs:
        # Persist session_agenda.json so the next session can restore agenda state.
        try:
            session.session_agenda.save()
        except Exception:
            pass
        # Non-blocking save after every interviewer turn
        save_async(user_id, st.session_state.chat, cfg)

# Render chat history
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- State machine ---

if st.session_state.waiting:
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            time.sleep(0.8)
    st.rerun()

elif not session.session_in_progress:
    st.success("The interview has ended. Thank you for your time!")

    if not st.session_state.session_saved:
        with st.spinner("Saving your session to Google Drive..."):
            ok, msg = save_sync(user_id, st.session_state.chat, cfg)
        st.session_state.session_saved = True
        if ok:
            st.info(f"Session saved. Your Participant ID was **`{user_id}`**.")
        else:
            st.caption(f"(Note: auto-save encountered an issue: {msg})")

else:
    # ------------------------------------------------------------------ #
    # Input area: full-width textbox, then Send + Mic buttons on one row  #
    # ------------------------------------------------------------------ #
    st.markdown("""
    <style>
    /* Larger base font throughout the app */
    html, body, [class*="css"], .stMarkdown, .stChatMessage {
        font-size: 18px !important;
    }
    /* Chat messages */
    div[data-testid="stChatMessage"] p {
        font-size: 1.05rem !important;
        line-height: 1.7 !important;
    }
    /* Text area */
    div[data-testid="stTextArea"] textarea {
        min-height: 130px !important;
        font-size: 1.1rem !important;
        line-height: 1.7 !important;
        border-radius: 14px !important;
        padding: 14px 18px !important;
        resize: none !important;
    }
    /* Send button */
    div[data-testid="stButton"] button[kind="primaryFormSubmit"],
    div[data-testid="stButton"] button[kind="primary"] {
        font-size: 1rem !important;
        height: 60px !important;
        padding: 0 1.2rem !important;
        border-radius: 8px !important;
        width: 100% !important;
    }
    /* Mic recorder iframe — taller so the button is larger */
    [data-testid="stColumn"]:first-child iframe {
        height: 60px !important;
        min-height: 60px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Full-width text area
    st.text_area(
        "response",
        key="user_draft",
        height=140,
        placeholder="Type your response here, or click 🎤 Speak to record...",
        label_visibility="collapsed",
    )

    # Mic (left-aligned) and Send (right-aligned) below the text area
    mic_col, spacer_col, send_col = st.columns([3, 5, 2])

    with mic_col:
        audio = mic_recorder(
            start_prompt="🎤  Speak",
            stop_prompt="⏹️  Stop",
            just_once=True,
            use_container_width=True,
            key="mic",
        )

    with send_col:
        send_clicked = st.button("Send →", type="primary", use_container_width=True)

    # Handle new recording
    if audio:
        audio_bytes = audio["bytes"]
        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        if audio_hash != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = audio_hash
            with st.spinner("Transcribing..."):
                transcript = _transcribe(audio_bytes)
            if transcript:
                st.session_state._pending_draft = transcript
                st.rerun()
            else:
                st.warning("Could not transcribe. Please try again or type your response.")

    if send_clicked:
        prompt = (st.session_state.get("user_draft") or "").strip()
        if prompt:
            st.session_state._pending_draft = ""   # clear box on next run
            st.session_state.chat.append({"role": "user", "content": prompt})

            async def _submit(text=prompt):
                session.user.add_user_message(text)

            asyncio.run_coroutine_threadsafe(
                _submit(), st.session_state.loop
            ).result(timeout=10)

            st.session_state.waiting = True
            st.rerun()

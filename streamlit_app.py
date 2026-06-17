"""
SparkMe -- Streamlit web chatbot
Run locally:   streamlit run streamlit_app.py
Deploy:        push to GitHub -> connect Streamlit Community Cloud

Required Streamlit Secrets:
  OPENAI_API_KEY           -- your OpenAI key
  GDRIVE_FOLDER_ID         -- ID of the Google Drive folder to save sessions into
  [google_service_account] -- service account credentials (see deployment guide)
"""

import asyncio
import os
import threading
import time
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# Load .env for local dev; Streamlit Cloud injects secrets as env vars
load_dotenv(override=True)

# SparkMe reads many env vars at import time. Set defaults so the app works
# on Streamlit Cloud without requiring every variable in Streamlit Secrets.
os.environ.setdefault("LOGS_DIR", "logs")
os.environ.setdefault("DATA_DIR", "data")
os.environ.setdefault("USER_AGENT_PROFILES_DIR", "data/sample_user_profiles")
os.environ.setdefault("INTERVIEW_PLAN_PATH", "data/configs/topics.json")
os.environ.setdefault("USER_PORTRAIT_PATH", "data/configs/user_portrait.json")
os.environ.setdefault("MODEL_NAME", "gpt-4.1-mini")
os.environ.setdefault("AGENDA_MANAGER_MODEL_NAME", "gpt-4.1-mini")
os.environ.setdefault("EXPLORATION_PLANNER_MODEL_NAME", "gpt-4.1-mini")
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
        "- **Streamlit Cloud:** Settings -> Secrets."
    )
    st.stop()

# --- SparkMe imports -------------------------------------------------------
from src.interview_session.interview_session import InterviewSession


# --- Google Drive helpers --------------------------------------------------

def _drive_service():
    """Build a Google Drive API client from Streamlit secrets."""
    from googleapiclient.discovery import build
    from google.oauth2.service_account import Credentials

    creds = Credentials.from_service_account_info(
        dict(st.secrets["google_service_account"]),
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _upload_directory(local_dir, parent_folder_id, service):
    """Upload every file inside local_dir (recursively) to a Drive folder.
    Returns the number of files uploaded."""
    from googleapiclient.http import MediaFileUpload

    count = 0
    if not os.path.exists(local_dir):
        return count

    for root, _dirs, files in os.walk(local_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            # Flatten the relative path into the filename so it is readable in Drive
            rel = os.path.relpath(filepath, local_dir).replace(os.sep, "__")
            service.files().create(
                body={"name": rel, "parents": [parent_folder_id]},
                media_body=MediaFileUpload(filepath, resumable=False),
            ).execute()
            count += 1

    return count


def upload_session_to_drive(user_id):
    """Upload all logs and data for this session to Google Drive.

    Creates a subfolder named session_{user_id}_{timestamp} inside the
    folder pointed to by GDRIVE_FOLDER_ID.

    Files uploaded:
      logs/{user_id}/  -- chat transcript, execution log, token usage
      data/{user_id}/  -- memory bank, question bank (AI internal notes)

    Returns (success: bool, message: str).
    """
    folder_id = st.secrets.get("GDRIVE_FOLDER_ID", "")
    if not folder_id:
        return False, "GDRIVE_FOLDER_ID not found in Streamlit Secrets."

    try:
        service = _drive_service()

        # Create a per-session subfolder so sessions never overwrite each other
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subfolder = service.files().create(
            body={
                "name": f"session_{user_id}_{timestamp}",
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [folder_id],
            },
            fields="id",
        ).execute()
        subfolder_id = subfolder["id"]

        n = _upload_directory(f"logs/{user_id}", subfolder_id, service)
        n += _upload_directory(f"data/{user_id}", subfolder_id, service)

        return True, f"Session saved -- {n} files uploaded to Google Drive."

    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# --- Background async session runner --------------------------------------

def _run_bg(session, loop):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(session.run())


def _create_session():
    user_id = f"web_{uuid.uuid4().hex[:8]}"
    loop = asyncio.new_event_loop()
    session = InterviewSession(
        interaction_mode="api",
        user_config={"user_id": user_id},
        interview_config={
            "interview_description": "Understanding the impact of AI in the workforce",
            "interview_plan_path": os.getenv("INTERVIEW_PLAN_PATH", "data/configs/topics.json"),
            "interview_evaluation": os.getenv("COMPLETION_METRIC", "minimum_threshold"),
            "initial_user_portrait_path": os.getenv(
                "USER_PORTRAIT_PATH", "data/configs/user_portrait.json"
            ),
        },
    )
    threading.Thread(target=_run_bg, args=(session, loop), daemon=True).start()
    return session, loop, user_id


# --- Page config ----------------------------------------------------------
st.set_page_config(
    page_title="SparkMe Interview",
    page_icon="mic",
    layout="centered",
)
st.title("SparkMe Interview")


# --- Per-user session state -----------------------------------------------
# Each browser tab gets its own Streamlit session, so participants are isolated.

if "session" not in st.session_state:
    with st.spinner("Starting interview session..."):
        session, loop, user_id = _create_session()

    st.session_state.session  = session
    st.session_state.loop     = loop
    st.session_state.user_id  = user_id
    st.session_state.chat     = []      # list of {"role": "assistant"|"user", "content": str}
    st.session_state.waiting  = True    # waiting for the interviewer's first message
    st.session_state.uploaded = False   # whether the Drive upload has already run


# --- Poll for new interviewer messages ------------------------------------
session = st.session_state.session
new_msgs = session.user.get_and_clear_messages()
if new_msgs:
    for m in new_msgs:
        st.session_state.chat.append({"role": "assistant", "content": m["content"]})
    st.session_state.waiting = False


# --- Render chat history --------------------------------------------------
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# --- State machine --------------------------------------------------------

if st.session_state.waiting:
    # Interviewer is generating -- show typing indicator and auto-refresh
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            time.sleep(0.8)
    st.rerun()

elif not session.session_in_progress:
    # Session ended -- thank participant and upload data to Drive
    st.success("The interview has ended. Thank you for participating!")

    if not st.session_state.uploaded:
        with st.spinner("Saving your session to Google Drive..."):
            ok, message = upload_session_to_drive(st.session_state.user_id)
        st.session_state.uploaded = True

        if ok:
            st.info(f"Saved: {message}")
        else:
            # Show a quiet note rather than alarming the participant
            st.caption(f"(Auto-save note: {message})")

else:
    # Active session -- show text input
    prompt = st.chat_input("Type your response...")
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})

        # Inject the message into the session's event loop.
        # Required because add_message_to_chat_history() calls asyncio.create_task() internally.
        async def _submit():
            session.user.add_user_message(prompt)

        asyncio.run_coroutine_threadsafe(
            _submit(), st.session_state.loop
        ).result(timeout=10)

        st.session_state.waiting = True
        st.rerun()

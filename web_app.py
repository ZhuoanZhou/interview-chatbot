"""
SparkMe Web Chatbot
===================
A simple Flask web interface for the SparkMe interview system.

Usage (from the SparkMe repo root):
    python web_app.py

Then open http://localhost:5000 in your browser.

Prerequisites:
    - .env file with OPENAI_API_KEY (copy from .env_sample and fill in)
    - pip install -r requirements.txt
"""

import asyncio
import os
import threading
import uuid
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv

load_dotenv(override=True)

# These imports rely on the SparkMe src/ package being on the path.
# Run this script from the repo root so that `src/` is importable.
from src.interview_session.interview_session import InterviewSession

app = Flask(__name__)

# ── In-memory session store ─────────────────────────────────────────────────
# Maps session_id -> { session, loop, thread }
_sessions: dict = {}


# ── Background runner ────────────────────────────────────────────────────────

def _run_session(session: InterviewSession, loop: asyncio.AbstractEventLoop):
    """Run an async interview session inside a dedicated event loop thread."""
    asyncio.set_event_loop(loop)
    loop.run_until_complete(session.run())


def _start_session() -> str:
    """Create and launch a new InterviewSession; return the session_id."""
    user_id = f"web_{uuid.uuid4().hex[:8]}"
    loop = asyncio.new_event_loop()

    session = InterviewSession(
        interaction_mode="api",
        user_config={"user_id": user_id},
        interview_config={
            "interview_description": "Understanding the impact of AI in the workforce",
            "interview_plan_path": os.getenv("INTERVIEW_PLAN_PATH", "data/configs/topics.json"),
            "interview_evaluation": os.getenv("COMPLETION_METRIC", "minimum_threshold"),
            "initial_user_portrait_path": os.getenv("USER_PORTRAIT_PATH", "data/configs/user_portrait.json"),
        },
    )

    t = threading.Thread(target=_run_session, args=(session, loop), daemon=True)
    t.start()

    session_id = uuid.uuid4().hex
    _sessions[session_id] = {"session": session, "loop": loop, "thread": t}
    return session_id


# ── API routes ───────────────────────────────────────────────────────────────

@app.route("/api/start", methods=["POST"])
def api_start():
    """Start a new interview session."""
    try:
        session_id = _start_session()
        return jsonify({"session_id": session_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/poll")
def api_poll():
    """
    Poll for new messages from the interviewer.
    Returns any buffered messages and whether the session is still active.
    """
    sid = request.args.get("session_id", "")
    if sid not in _sessions:
        return jsonify({"error": "unknown session"}), 400

    data = _sessions[sid]
    session: InterviewSession = data["session"]

    # get_and_clear_messages() is thread-safe (uses a Lock internally)
    messages = session.user.get_and_clear_messages()

    return jsonify({
        "messages": messages,
        "active": session.session_in_progress,
    })


@app.route("/api/send", methods=["POST"])
def api_send():
    """
    Submit a user message to the interview session.
    The message is injected into the session's event loop so that
    asyncio.create_task() inside add_message_to_chat_history() works correctly.
    """
    body = request.get_json(force=True) or {}
    sid = body.get("session_id", "")
    text = (body.get("message") or "").strip()

    if sid not in _sessions:
        return jsonify({"error": "unknown session"}), 400
    if not text:
        return jsonify({"error": "empty message"}), 400

    data = _sessions[sid]
    session: InterviewSession = data["session"]
    loop: asyncio.AbstractEventLoop = data["loop"]

    if not session.session_in_progress:
        return jsonify({"error": "session has ended"}), 400

    # Schedule add_user_message inside the session's running event loop.
    # This is required because add_message_to_chat_history() calls
    # asyncio.create_task(), which must run inside the event loop.
    async def _submit():
        session.user.add_user_message(text)

    future = asyncio.run_coroutine_threadsafe(_submit(), loop)
    try:
        future.result(timeout=10)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "ok"})


# ── Frontend ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(CHAT_HTML)


CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SparkMe Interview</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f0f2f5;
      height: 100dvh;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    /* ── Header ── */
    header {
      width: 100%;
      background: #1a73e8;
      color: #fff;
      padding: 14px 24px;
      font-size: 17px;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 10px;
      flex-shrink: 0;
    }
    .dot {
      width: 9px; height: 9px;
      border-radius: 50%;
      background: #81c995;
      flex-shrink: 0;
    }
    .dot.off { background: #d93025; }

    /* ── Start screen ── */
    #start-screen {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 14px;
      padding: 32px 24px;
      text-align: center;
    }
    #start-screen h2 { font-size: 22px; color: #202124; }
    #start-screen p  { color: #5f6368; max-width: 460px; font-size: 15px; line-height: 1.6; }
    #start-btn {
      background: #1a73e8;
      color: #fff;
      border: none;
      border-radius: 24px;
      padding: 12px 36px;
      font-size: 15px;
      cursor: pointer;
      margin-top: 6px;
    }
    #start-btn:disabled { background: #9aa0a6; cursor: default; }
    #start-error { color: #d93025; font-size: 14px; display: none; }

    /* ── Chat UI ── */
    #chat-ui {
      display: none;
      flex-direction: column;
      width: 100%;
      max-width: 760px;
      flex: 1;
      min-height: 0;
    }

    #messages {
      flex: 1;
      overflow-y: auto;
      padding: 20px 16px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .msg-group { display: flex; flex-direction: column; gap: 3px; max-width: 72%; }
    .msg-group.them { align-self: flex-start; }
    .msg-group.me   { align-self: flex-end;   }

    .label {
      font-size: 11px;
      color: #80868b;
      padding: 0 4px;
    }
    .msg-group.me .label { text-align: right; }

    .bubble {
      padding: 11px 15px;
      border-radius: 18px;
      font-size: 15px;
      line-height: 1.55;
      word-wrap: break-word;
    }
    .them .bubble {
      background: #fff;
      color: #202124;
      border-bottom-left-radius: 4px;
      box-shadow: 0 1px 2px rgba(0,0,0,.12);
    }
    .me .bubble {
      background: #1a73e8;
      color: #fff;
      border-bottom-right-radius: 4px;
    }

    /* Typing indicator */
    .typing {
      display: flex; gap: 5px; align-items: center;
      padding: 11px 16px;
      background: #fff;
      border-radius: 18px;
      border-bottom-left-radius: 4px;
      width: fit-content;
      box-shadow: 0 1px 2px rgba(0,0,0,.12);
    }
    .typing span {
      width: 7px; height: 7px;
      background: #bdc1c6;
      border-radius: 50%;
      animation: pulse 1.4s infinite both;
    }
    .typing span:nth-child(2) { animation-delay: .2s; }
    .typing span:nth-child(3) { animation-delay: .4s; }
    @keyframes pulse {
      0%, 80%, 100% { opacity: .25; }
      40%           { opacity: 1;   }
    }

    /* ── Footer / input area ── */
    footer {
      width: 100%;
      max-width: 760px;
      padding: 10px 16px 20px;
      flex-shrink: 0;
    }
    #ended-note {
      text-align: center;
      color: #80868b;
      font-size: 13px;
      padding: 8px;
      display: none;
    }
    .input-row {
      display: flex;
      align-items: flex-end;
      gap: 8px;
      background: #fff;
      border-radius: 24px;
      padding: 8px 8px 8px 16px;
      box-shadow: 0 2px 6px rgba(0,0,0,.14);
    }
    textarea {
      flex: 1;
      border: none;
      outline: none;
      resize: none;
      font-size: 15px;
      font-family: inherit;
      line-height: 1.5;
      min-height: 24px;
      max-height: 120px;
      background: transparent;
      padding: 2px 0;
    }
    #send-btn {
      background: #1a73e8;
      color: #fff;
      border: none;
      border-radius: 20px;
      padding: 7px 18px;
      font-size: 14px;
      cursor: pointer;
      white-space: nowrap;
      flex-shrink: 0;
    }
    #send-btn:disabled { background: #9aa0a6; cursor: default; }
  </style>
</head>
<body>

<header>
  <div class="dot" id="dot"></div>
  SparkMe Interview
</header>

<!-- ── Start screen ── -->
<div id="start-screen">
  <h2>Welcome to SparkMe</h2>
  <p>This AI interviewer will guide you through a structured conversation.
     Click below when you're ready to begin.</p>
  <button id="start-btn">Start Interview</button>
  <span id="start-error"></span>
</div>

<!-- ── Chat UI ── -->
<div id="chat-ui">
  <div id="messages"></div>
  <footer>
    <div id="ended-note">The interview has ended. Thank you for participating!</div>
    <div class="input-row" id="input-row">
      <textarea id="input" placeholder="Type your response…" rows="1"></textarea>
      <button id="send-btn" disabled>Send</button>
    </div>
  </footer>
</div>

<script>
  /* ── State ── */
  let sessionId   = null;
  let polling     = false;
  let waiting     = false;   // waiting for interviewer response
  let typingEl    = null;

  /* ── DOM refs ── */
  const startScreen = document.getElementById("start-screen");
  const chatUI      = document.getElementById("chat-ui");
  const messages    = document.getElementById("messages");
  const inputEl     = document.getElementById("input");
  const sendBtn     = document.getElementById("send-btn");
  const dot         = document.getElementById("dot");
  const startBtn    = document.getElementById("start-btn");
  const startError  = document.getElementById("start-error");
  const endedNote   = document.getElementById("ended-note");
  const inputRow    = document.getElementById("input-row");

  /* ── Helpers ── */
  function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

  function scrollBottom() {
    messages.scrollTop = messages.scrollHeight;
  }

  function showTyping() {
    if (typingEl) return;
    const g = document.createElement("div");
    g.className = "msg-group them";
    typingEl = document.createElement("div");
    typingEl.className = "typing";
    typingEl.innerHTML = "<span></span><span></span><span></span>";
    g.appendChild(typingEl);
    messages.appendChild(g);
    scrollBottom();
    typingEl._group = g;
  }

  function hideTyping() {
    if (!typingEl) return;
    typingEl._group.remove();
    typingEl = null;
  }

  function appendMessage(role, content) {
    const isMe = role === "User";
    const g = document.createElement("div");
    g.className = `msg-group ${isMe ? "me" : "them"}`;

    const lbl = document.createElement("div");
    lbl.className = "label";
    lbl.textContent = isMe ? "You" : "Interviewer";

    const b = document.createElement("div");
    b.className = "bubble";
    b.textContent = content;

    g.appendChild(lbl);
    g.appendChild(b);
    messages.appendChild(g);
    scrollBottom();
  }

  function setWaiting(val) {
    waiting = val;
    sendBtn.disabled = val;
    if (val) showTyping(); else hideTyping();
  }

  function markEnded() {
    polling = false;
    dot.classList.add("off");
    endedNote.style.display = "block";
    inputRow.style.display = "none";
    hideTyping();
  }

  /* ── Session start ── */
  async function startSession() {
    startBtn.disabled = true;
    startBtn.textContent = "Starting…";
    startError.style.display = "none";

    try {
      const res  = await fetch("/api/start", { method: "POST" });
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      sessionId = data.session_id;
      startScreen.style.display = "none";
      chatUI.style.display = "flex";

      setWaiting(true);   // interviewer is composing first message
      startPolling();
    } catch (e) {
      startBtn.disabled = false;
      startBtn.textContent = "Start Interview";
      startError.textContent = "Failed to start: " + e.message;
      startError.style.display = "block";
    }
  }

  /* ── Polling loop ── */
  function startPolling() {
    if (polling) return;
    polling = true;
    (async () => {
      while (polling && sessionId) {
        try {
          const res  = await fetch(`/api/poll?session_id=${sessionId}`);
          const data = await res.json();

          if (data.messages && data.messages.length) {
            hideTyping();
            for (const m of data.messages) appendMessage(m.role, m.content);
            waiting  = false;
            sendBtn.disabled = false;
          }

          if (!data.active) { markEnded(); break; }
        } catch (e) {
          console.warn("Poll error:", e);
        }
        await sleep(1000);
      }
    })();
  }

  /* ── Send message ── */
  async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text || waiting || !sessionId) return;

    inputEl.value = "";
    inputEl.style.height = "auto";
    appendMessage("User", text);
    setWaiting(true);

    try {
      const res  = await fetch("/api/send", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ session_id: sessionId, message: text }),
      });
      const data = await res.json();
      if (data.error) {
        hideTyping();
        waiting = false;
        sendBtn.disabled = false;
        alert("Error: " + data.error);
      }
    } catch (e) {
      hideTyping();
      waiting = false;
      sendBtn.disabled = false;
      console.error("Send error:", e);
    }
  }

  /* ── Event listeners ── */
  startBtn.addEventListener("click", startSession);
  sendBtn.addEventListener("click", sendMessage);

  inputEl.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  inputEl.addEventListener("input", () => {
    inputEl.style.height = "auto";
    inputEl.style.height = inputEl.scrollHeight + "px";
  });
</script>
</body>
</html>"""


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # use_reloader=False is required — the reloader forks the process and
    # breaks the background asyncio threads.
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

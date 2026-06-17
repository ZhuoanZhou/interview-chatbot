"""
Flask Web Application for Interview Session
Supports both text and voice input/output with authentication
"""

from flask import Flask, request, jsonify, render_template, Response, redirect, url_for, flash
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import traceback
import asyncio
import threading
import os
import uuid
import argparse
import time
import logging
import secrets
import hashlib
import json
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

# Your backend imports
from src.utils.speech.text_to_speech import TextToSpeechBase, create_tts_engine
from src.utils.speech.audio_player import AudioPlayerBase, create_audio_player
from src.utils.speech.speech_to_text import create_stt_engine
from src.interview_session.interview_session import InterviewSession

load_dotenv(override=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

SESSION_TIMEOUT_SECONDS = 3600  # 1 hour
START_TIME = time.time()

class AppConfig:
    """Application configuration"""
    def __init__(self):
        self.default_user_id = "web_user"
        self.host = "0.0.0.0"
        self.port = 5000
        self.debug = False
        self.restart = False
        self.max_turns = None
        self.additional_context_path = None

config = AppConfig()

# TTS/STT engines
TTS_PROVIDER = os.getenv('TTS_PROVIDER', 'openai')
TTS_VOICE = os.getenv('TTS_VOICE', 'alloy')
tts_engine: TextToSpeechBase = create_tts_engine(provider=TTS_PROVIDER, voice=TTS_VOICE)
stt_engine = create_stt_engine()

# =============================================================================
# FLASK APP SETUP
# =============================================================================

app = Flask(__name__,
            static_folder='web/static',
            template_folder='web/templates')

app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(32))
CORS(app)

# =============================================================================
# AUTHENTICATION SETUP
# =============================================================================

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access the interview.'

USERS_FILE = os.path.join(os.getenv('DATA_DIR', 'data'), 'users.json')

class User(UserMixin):
    def __init__(self, user_id, username):
        self.id = user_id
        self.username = username

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users):
    """Save users to JSON file"""
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

@login_manager.user_loader
def load_user(user_id):
    users = load_users()
    if user_id in users:
        return User(user_id, users[user_id]['username'])
    return None

# =============================================================================
# LOGGING SETUP
# =============================================================================

if not app.debug:
    os.makedirs('logs', exist_ok=True)
    file_handler = RotatingFileHandler(
        'logs/flask_app.log',
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Interview application startup')

# =============================================================================
# ASYNC EVENT LOOP MANAGEMENT
# =============================================================================

loop = asyncio.new_event_loop()

def start_background_loop(loop):
    """Run async event loop in background thread."""
    asyncio.set_event_loop(loop)
    loop.run_forever()

threading.Thread(target=start_background_loop, args=(loop,), daemon=True).start()

def run_async_task(coro):
    """Submit coroutine to background loop."""
    return asyncio.run_coroutine_threadsafe(coro, loop)

# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

class SessionWrapper:
    def __init__(self, session_token: str, interview_session: InterviewSession,
                 user_id: str):
        self.session_token = session_token
        self.interview_session = interview_session
        self.user_id = user_id
        self.created_at = time.time()

active_sessions: Dict[str, SessionWrapper] = {}
last_messages_by_session: Dict[str, Dict[str, str]] = {}
session_audio_cache: Dict[str, Dict[str, object]] = {}

def create_interview_session(user_id: str) -> tuple[InterviewSession, str]:
    """Create interview session with authenticated user_id"""
    session_token = str(uuid.uuid4())
    
    interview_session = InterviewSession(
        interaction_mode='api',
        user_config={
            "user_id": user_id,
            "enable_voice": False,
            "restart": config.restart
        },
        interview_config={
            "enable_voice": False,
            "interview_description": os.getenv(
                'INTERVIEW_DESCRIPTION', 
                "Understanding the impact of AI in the workforce"
            ),
            "interview_plan_path": os.getenv('INTERVIEW_PLAN_PATH'),
            "interview_evaluation": os.getenv('COMPLETION_METRIC'),
            "additional_context_path": config.additional_context_path,
            "initial_user_portrait_path": os.getenv('USER_PORTRAIT_PATH'),
        },
        max_turns=config.max_turns
    )
    
    wrapper = SessionWrapper(
        session_token=session_token,
        interview_session=interview_session,
        user_id=user_id,
    )
    active_sessions[session_token] = wrapper
    
    session_loop = asyncio.new_event_loop()
    def _start_loop(l):
        asyncio.set_event_loop(l)
        l.run_forever()
    t = threading.Thread(target=_start_loop, args=(session_loop,), daemon=True)
    t.start()

    wrapper.loop = session_loop
    wrapper.loop_thread = t
    asyncio.run_coroutine_threadsafe(interview_session.run(), session_loop)
    
    return interview_session, session_token

def get_session(session_token: str) -> Optional[InterviewSession]:
    wrapper = active_sessions.get(session_token)
    return wrapper.interview_session if wrapper is not None else None

def get_session_wrapper(session_token: str) -> Optional[SessionWrapper]:
    return active_sessions.get(session_token)

# =============================================================================
# AUTHENTICATION ROUTES (NO @login_required)
# =============================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))  # Changed: redirect to index with instructions
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        users = load_users()
        
        # Find user
        user_id = None
        for uid, user_data in users.items():
            if user_data['username'] == username:
                user_id = uid
                break
        
        if user_id and users[user_id]['password'] == hash_password(password):
            user = User(user_id, username)
            login_user(user)
            app.logger.info(f"User logged in: {username} ({user_id})")
            
            next_page = request.args.get('next')
            return redirect(next_page if next_page else url_for('index'))  # Changed: go to index first
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))  # Changed: redirect to index with instructions
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not username or not password:
            flash('Username and password are required', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters', 'error')
            return render_template('register.html')
        
        users = load_users()
        
        # Check if username exists
        for user_data in users.values():
            if user_data['username'] == username:
                flash('Username already exists', 'error')
                return render_template('register.html')
        
        # Create new user
        user_id = secrets.token_urlsafe(16)
        users[user_id] = {
            'username': username,
            'password': hash_password(password),
            'created_at': time.time()
        }
        save_users(users)
        
        # Create user directories
        os.makedirs(os.path.join(os.getenv('LOGS_DIR', 'logs'), user_id), exist_ok=True)
        os.makedirs(os.path.join(os.getenv('DATA_DIR', 'data'), user_id), exist_ok=True)
        
        app.logger.info(f"New user registered: {username} ({user_id})")
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    """Logout"""
    username = current_user.username
    logout_user()
    app.logger.info(f"User logged out: {username}")
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# =============================================================================
# PAGE ROUTES - PROTECTED (REQUIRE LOGIN)
# =============================================================================

@app.route('/')
@login_required  # MUST BE LOGGED IN TO SEE INSTRUCTIONS
def index():
    """Landing page with instructions - shown after login"""
    return render_template('index.html', username=current_user.username)

@app.route('/chat')
@login_required  # MUST BE LOGGED IN
def unified_chat():
    """Unified chat interface"""
    return render_template('chat.html', username=current_user.username)

# =============================================================================
# API ENDPOINTS - PROTECTED (REQUIRE LOGIN)
# All endpoints that handle interview data need @login_required
# =============================================================================

@app.route('/api/start-session', methods=['POST'])
@login_required  # PROTECTED
def start_session():
    """Initialize a new interview session using authenticated user's ID"""
    user_id = current_user.id
    
    # DEBUG: Log every request
    call_stack = ''.join(traceback.format_stack()[-3:-1])
    app.logger.info(f"[DEBUG] start-session called by user {current_user.username}")
    print(f"[DEBUG] start-session request from {current_user.username} (user_id: {user_id})")
    
    # Check if user already has an active session
    for token, wrapper in active_sessions.items():
        if wrapper.user_id == user_id:
            # Return existing session instead of creating duplicate
            app.logger.info(f"Returning existing session {token} for user {current_user.username}")
            print(f"[Session] Reusing existing session {token} for {current_user.username}")
            return jsonify({
                'success': True,
                'session_token': token,
                'session_id': wrapper.interview_session.session_id,
                'user_id': user_id,
                'username': current_user.username,
                'message': 'Using existing session',
                'was_existing': True
            })
    
    # Create new session only if none exists
    interview_session, session_token = create_interview_session(user_id=user_id)

    app.logger.info(f"Session created: {session_token} | User: {current_user.username} ({user_id})")
    print(f"[Session] Created NEW session {session_token} for user {current_user.username}")

    return jsonify({
        'success': True,
        'session_token': session_token,
        'session_id': interview_session.session_id,
        'user_id': user_id,
        'username': current_user.username,
        'message': 'Session started successfully',
        'was_existing': False
    })

@app.route('/api/send-message', methods=['POST'])
@login_required  # PROTECTED
def send_message():
    """Send a text message to the interview session"""
    data = request.json
    session_token = data.get('session_token')
    user_message = data.get('message')

    session = get_session(session_token)
    if not session:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session'
        }), 400

    if not session.session_in_progress:
        return jsonify({
            'success': False,
            'error': 'Session has ended',
            'session_completed': True
        }), 400

    wrapper = get_session_wrapper(session_token)
    if wrapper and hasattr(wrapper, 'loop'):
        wrapper.loop.call_soon_threadsafe(wrapper.interview_session.user.add_user_message, user_message)
    else:
        session.user.add_user_message(user_message)

    bot_reply = wait_for_agent_response(session)

    return jsonify({
        'success': True,
        'message': 'Message sent successfully'
    })

@app.route('/api/send-voice', methods=['POST'])
@login_required  # PROTECTED
def send_voice():
    """Send a voice message to the interview session"""
    session_token = request.form.get('session_token')
    audio_file = request.files.get('audio')

    if not audio_file:
        return jsonify({
            'success': False,
            'error': 'No audio file provided'
        }), 400

    session = get_session(session_token)
    if not session:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session'
        }), 400

    # Check if STT is available
    if stt_engine is None:
        return jsonify({
            'success': False,
            'error': 'Voice input is not enabled on this server (PyAudio not available)'
        }), 503

    temp_audio_path = Path(f"temp_audio_{uuid.uuid4().hex}.wav")
    audio_file.save(temp_audio_path)

    try:
        transcribed_text = transcribe_audio_to_text(temp_audio_path)

        wrapper = get_session_wrapper(session_token)
        if wrapper and hasattr(wrapper, 'loop'):
            wrapper.loop.call_soon_threadsafe(wrapper.interview_session.user.add_user_message, transcribed_text)
        else:
            session.user.add_user_message(transcribed_text)

        return jsonify({
            'success': True,
            'transcribed_text': transcribed_text,
            'message': 'Voice message processed successfully'
        })
    finally:
        if temp_audio_path.exists():
            temp_audio_path.unlink()

@app.route('/api/get-messages', methods=['GET'])
@login_required  # PROTECTED
def get_messages():
    """Get new messages from the session (polling endpoint)"""
    session_token = request.args.get('session_token')

    session = get_session(session_token)
    if not session:
        print(f"[get_messages] Invalid session_token={session_token}")
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session',
            'active_sessions_count': len(active_sessions)
        }), 400

    messages = []
    if session.user:
        if hasattr(session.user, 'get_new_messages'):
            messages = session.user.get_new_messages() or []
        elif hasattr(session.user, '_message_buffer'):
            lock = getattr(session.user, '_lock', None)
            if lock:
                with lock:
                    messages = list(getattr(session.user, '_message_buffer', []))
            else:
                messages = list(getattr(session.user, '_message_buffer', []))
        elif hasattr(session.user, 'get_and_clear_messages'):
            messages = session.user.get_and_clear_messages() or []

    is_session_done = session.session_completed
    
    if not is_session_done:
        current_turns = getattr(session, 'turns', 0)
        max_turns = getattr(session, 'max_turns', float('inf'))
        if current_turns is not None and max_turns is not None and current_turns >= max_turns:
            is_session_done = True
        elif not session.session_in_progress and len(session.chat_history) > 0:
            is_session_done = True

    if is_session_done:
        end_msg_id = f"system_end_{session_token}"
        if not any(m.get('id') == end_msg_id for m in messages):
            messages.append({
                'id': end_msg_id,
                'role': 'Interviewer',
                'content': "Session has been completed! Thank you for your participation in our interview! Your responses have been recorded!",
                'timestamp': time.time()
            })

    return jsonify({
        'success': True,
        'messages': messages,
        'session_active': session.session_in_progress,
        'session_completed': is_session_done
    })

@app.route('/api/acknowledge-messages', methods=['POST'])
@login_required  # PROTECTED
def acknowledge_messages():
    """Mark messages as acknowledged by the client"""
    data = request.json
    session_token = data.get('session_token')
    message_ids = data.get('message_ids', [])

    session = get_session(session_token)
    if not session:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session'
        }), 400

    if session.user and hasattr(session.user, '_message_buffer'):
        lock = getattr(session.user, '_lock', None)
        if lock:
            with lock:
                buffer = getattr(session.user, '_message_buffer', [])
                session.user._message_buffer = [
                    m for m in buffer 
                    if m.get('id') not in message_ids
                ]
        
    return jsonify({'success': True})

@app.route('/api/get-voice-response', methods=['GET'])
@login_required  # PROTECTED
def get_voice_response():
    """Get the latest interviewer message as voice audio"""
    session_token = request.args.get('session_token')
    message_id = request.args.get('message_id')

    session = get_session(session_token)
    if not session:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session'
        }), 400

    if not message_id:
        return jsonify({
            'success': False,
            'error': 'message_id required'
        }), 400

    target_msg = None
    for m in session.chat_history:
        if hasattr(m, 'id') and m.id == message_id:
            target_msg = m
            break
    
    if not target_msg:
        return jsonify({
            'success': False,
            'error': 'Message not found'
        }), 404

    cache = session_audio_cache.setdefault(session_token, {})
    entry = cache.get(message_id)

    # Check cache entry status
    if isinstance(entry, dict):
        # New format: {'status': 'pending'|'ready'|'failed', 'data': bytes|None, 'error': str|None}
        status = entry.get('status')
        if status == 'ready' and entry.get('data'):
            return Response(entry['data'], mimetype='audio/mpeg')
        elif status == 'failed':
            return jsonify({
                'success': False,
                'error': f"TTS generation failed: {entry.get('error', 'Unknown error')}"
            }), 500
        elif status == 'pending':
            return ('', 202)  # Still generating
    elif isinstance(entry, (bytes, bytearray)):
        # Legacy format support
        return Response(entry, mimetype='audio/mpeg')
    elif entry == 'pending':
        # Legacy format support
        return ('', 202)

    # No cache entry exists - start generation
    cache[message_id] = {'status': 'pending', 'data': None, 'error': None, 'timestamp': time.time()}
    wrapper = get_session_wrapper(session_token)

    if not wrapper or not hasattr(wrapper, 'loop'):
        cache[message_id] = {
            'status': 'failed',
            'data': None,
            'error': 'Session loop not available'
        }
        return jsonify({
            'success': False,
            'error': 'Session not properly initialized'
        }), 500

    async def _synth_and_cache():
        try:
            text = target_msg.content
            app.logger.info(f"Starting TTS generation for message {message_id}")

            def _blocking_tts():
                out_path = Path(f"temp_speech_{uuid.uuid4().hex}.mp3")
                try:
                    tts_engine.text_to_speech(text=text, output_path=str(out_path))
                    data = out_path.read_bytes()
                    app.logger.info(f"TTS generated {len(data)} bytes for message {message_id}")
                    return data
                finally:
                    if out_path.exists():
                        out_path.unlink(missing_ok=True)

            audio_bytes = await wrapper.loop.run_in_executor(None, _blocking_tts)

            # Update cache with success
            cache[message_id] = {
                'status': 'ready',
                'data': audio_bytes,
                'error': None,
                'timestamp': time.time()
            }
            app.logger.info(f"TTS cached successfully for message {message_id}")

        except Exception as e:
            error_msg = str(e)
            app.logger.error(f"TTS error for message {message_id}: {error_msg}", exc_info=True)

            # Update cache with failure
            cache[message_id] = {
                'status': 'failed',
                'data': None,
                'error': error_msg,
                'timestamp': time.time()
            }

    # Schedule async TTS generation
    future = asyncio.run_coroutine_threadsafe(_synth_and_cache(), wrapper.loop)

    # Log if scheduling failed
    try:
        future.result(timeout=0.1)  # Quick check if it errored immediately
    except Exception:
        pass  # Will complete async

    return ('', 202)  # Tell client to poll again

@app.route('/api/end-session', methods=['POST'])
@login_required  # PROTECTED
def end_session():
    """End the interview session - background tasks will complete gracefully"""
    data = request.json
    session_token = data.get('session_token')

    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session'
        }), 400

    session = wrapper.interview_session

    # End the session (marks as not in progress, stops new tasks)
    session.end_session()

    app.logger.info(f"Session {session_token} ended by user, background tasks will complete")

    # Don't remove from active_sessions yet - let background tasks complete
    # They'll be cleaned up when session.run() completes or by timeout cleanup

    return jsonify({
        'success': True,
        'message': 'Session ending, background tasks will complete shortly',
        'session_id': session.session_id,
        'user_id': session.user_id
    })

@app.route('/api/session-status', methods=['GET'])
@login_required  # PROTECTED
def session_status():
    """Get current session status including background task progress"""
    session_token = request.args.get('session_token')

    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session'
        }), 400

    session = wrapper.interview_session

    # Get background task count if available
    background_tasks_count = 0
    if hasattr(session, '_background_tasks'):
        try:
            import asyncio
            # Try to get count safely
            if hasattr(session, '_background_tasks_lock'):
                # Can't acquire lock in sync context, just get len
                background_tasks_count = len(session._background_tasks)
        except:
            background_tasks_count = 0

    return jsonify({
        'success': True,
        'session_active': session.session_in_progress,
        'session_completed': session.session_completed,
        'background_tasks_remaining': background_tasks_count,
        'message_count': len(session.chat_history),
        'session_id': session.session_id,
        'user_id': session.user_id
    })

@app.route('/api/debug-session', methods=['GET'])
@login_required  # PROTECTED
def debug_session():
    """Development-only: return session internals"""
    session_token = request.args.get('session_token')
    if not session_token:
        return jsonify({'success': False, 'error': 'session_token required'}), 400

    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({'success': False, 'error': 'Invalid or expired session', 'active_sessions_count': len(active_sessions)}), 400

    session = wrapper.interview_session

    last_msgs = []
    for m in session.chat_history[-20:]:
        last_msgs.append({
            'id': getattr(m, 'id', None),
            'role': getattr(m, 'role', None),
            'content': getattr(m, 'content', None),
            'timestamp': getattr(m, 'timestamp', None).isoformat() if getattr(m, 'timestamp', None) else None,
        })

    user_buffer = []
    user = session.user
    if hasattr(user, '_message_buffer'):
        try:
            lock = getattr(user, '_lock', None)
            if lock:
                lock.acquire()
            user_buffer = list(getattr(user, '_message_buffer', []))
        finally:
            if lock:
                lock.release()

    return jsonify({
        'success': True,
        'session_id': session.session_id,
        'session_active': session.session_in_progress,
        'session_completed': session.session_completed,
        'message_count': len(session.chat_history),
        'chat_history': last_msgs,
        'user_buffer': user_buffer,
        'active_sessions_count': len(active_sessions)
    })

@app.route('/process_audio', methods=['POST'])
@login_required  # PROTECTED
def process_audio():
    """Compatibility route for older speech_chat.html template"""
    session_token = request.form.get('session_token')
    audio_file = request.files.get('audio')

    if not audio_file:
        return jsonify({'success': False, 'error': 'No audio file provided'}), 400

    user_id = current_user.id
    
    if not session_token:
        interview_session, session_token = create_interview_session(user_id=user_id)
    else:
        interview_session = get_session(session_token)
        if not interview_session:
            interview_session, session_token = create_interview_session(user_id=user_id)

    temp_audio_path = Path(f"temp_audio_{uuid.uuid4().hex}.wav")
    audio_file.save(temp_audio_path)

    try:
        transcribed_text = transcribe_audio_to_text(temp_audio_path)
        wrapper = get_session_wrapper(session_token)
        if wrapper and hasattr(wrapper, 'loop'):
            wrapper.loop.call_soon_threadsafe(
                wrapper.interview_session.user.add_user_message, 
                transcribed_text
            )
        else:
            interview_session.user.add_user_message(transcribed_text)

        bot_reply = wait_for_agent_response(interview_session, timeout=15.0)
        
        last_messages_by_session[session_token] = {
            'user_message': transcribed_text,
            'bot_reply': bot_reply or ''
        }

        if bot_reply:
            out_path = Path(f"temp_speech_{uuid.uuid4().hex}.mp3")
            generate_speech_from_text(bot_reply, out_path)
            audio_bytes = out_path.read_bytes()
            out_path.unlink(missing_ok=True)
            return Response(audio_bytes, mimetype='audio/mpeg')

        return jsonify({
            'success': True, 
            'user_message': transcribed_text, 
            'bot_reply': bot_reply
        }), 200

    finally:
        if temp_audio_path.exists():
            temp_audio_path.unlink()

@app.route('/get_last_messages', methods=['GET'])
@login_required  # PROTECTED
def get_last_messages():
    """Get last messages for session"""
    session_token = request.args.get('session_token')
    if not session_token:
        return jsonify({'success': False, 'error': 'session_token required'}), 400

    msgs = last_messages_by_session.get(session_token, {})
    return jsonify({
        'success': True,
        'user_message': msgs.get('user_message', ''),
        'bot_reply': msgs.get('bot_reply', '')
    })

# =============================================================================
# HEALTH CHECK - NOT PROTECTED (for monitoring)
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint - no login required for monitoring"""
    current_time = time.time()
    session_ages = []
    for wrapper in active_sessions.values():
        age_minutes = (current_time - wrapper.created_at) / 60
        session_ages.append(age_minutes)
    
    avg_age = sum(session_ages) / len(session_ages) if session_ages else 0
    
    return jsonify({
        'status': 'healthy',
        'active_sessions': len(active_sessions),
        'avg_session_age_minutes': round(avg_age, 2),
        'tts_provider': TTS_PROVIDER,
        'tts_voice': TTS_VOICE,
        'uptime_seconds': round(current_time - START_TIME, 2)
    })

# =============================================================================
# VOICE PROCESSING UTILITIES
# =============================================================================

def generate_speech_from_text(text: str, output_path: Path) -> Path:
    """Generate speech audio from text using TTS"""
    global tts_engine
    if tts_engine is None:
        raise NotImplementedError("TTS engine is not configured")
    os.makedirs(os.path.dirname(str(output_path)) or '.', exist_ok=True)
    result_path = tts_engine.text_to_speech(text=text, output_path=str(output_path))
    return Path(result_path)

def transcribe_audio_to_text(audio_path: Path) -> str:
    """Transcribe audio file to text using speech recognition"""
    global stt_engine
    return stt_engine.transcribe(str(audio_path))

def wait_for_agent_response(session, timeout: float = 60.0, poll_interval: float = 0.5):
    """Wait for the Interviewer/Agent to produce an output"""
    start_time = None
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            start_time = None
        else:
            start_time = loop.time()
    except Exception:
        start_time = None

    elapsed = 0.0
    import time as _time

    while elapsed < timeout:
        try:
            msgs = []
            if hasattr(session.user, 'get_and_clear_messages'):
                msgs = session.user.get_and_clear_messages() or []
            interviewer_msgs = [m for m in msgs if m.get('role') == 'Interviewer']
            if interviewer_msgs:
                return interviewer_msgs[-1].get('content')
        except Exception:
            pass
        _time.sleep(poll_interval)
        elapsed += poll_interval
    return None

# =============================================================================
# SESSION CLEANUP
# =============================================================================

def cleanup_old_sessions():
    """Remove sessions older than timeout threshold and clean stale audio cache"""
    current_time = time.time()
    to_remove = []

    for token, wrapper in list(active_sessions.items()):
        age = current_time - wrapper.created_at
        if age > SESSION_TIMEOUT_SECONDS:
            to_remove.append(token)
            session_audio_cache.pop(token, None)
            last_messages_by_session.pop(token, None)

    for token in to_remove:
        wrapper = active_sessions.pop(token, None)
        if wrapper:
            print(f"[Cleanup] Removed session {token} (age: {age/60:.1f}min, user: {wrapper.user_id})")

    if to_remove:
        print(f"[Cleanup] Removed {len(to_remove)} old sessions. Active: {len(active_sessions)}")

    # Clean up stale pending audio cache entries (older than 5 minutes)
    audio_cleaned = 0
    for session_token, cache in list(session_audio_cache.items()):
        for message_id, entry in list(cache.items()):
            if isinstance(entry, dict):
                entry_age = current_time - entry.get('timestamp', current_time)
                # Remove pending entries older than 5 minutes (likely failed)
                if entry.get('status') == 'pending' and entry_age > 300:
                    cache.pop(message_id, None)
                    audio_cleaned += 1
                # Remove failed entries older than 1 hour
                elif entry.get('status') == 'failed' and entry_age > 3600:
                    cache.pop(message_id, None)
                    audio_cleaned += 1

    if audio_cleaned > 0:
        print(f"[Cleanup] Removed {audio_cleaned} stale audio cache entries")

def start_cleanup_thread():
    """Start background thread for session cleanup"""
    def cleanup_loop():
        while True:
            time.sleep(300)  # Every 5 minutes
            try:
                cleanup_old_sessions()
            except Exception as e:
                print(f"[Cleanup] Error: {e}")
    
    t = threading.Thread(target=cleanup_loop, daemon=True, name="SessionCleanup")
    t.start()
    print("[Cleanup] Started session cleanup thread")

# =============================================================================
# MAIN
# =============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Flask Interview Session Web Application'
    )
    parser.add_argument('--user-id', type=str, help='Default user ID')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--additional_context_path', default=None)
    parser.add_argument('--restart', action='store_true', default=False)
    parser.add_argument('--max_turns', type=int, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    if args.restart and args.user_id:
        os.system(f"rm -rf {os.getenv('LOGS_DIR')}/{args.user_id}")
        os.system(f"rm -rf {os.getenv('DATA_DIR')}/{args.user_id}")
        print(f"Cleared data for user {args.user_id}")
    
    config.default_user_id = args.user_id if args.user_id else "web_guest"
    config.host = args.host
    config.port = args.port
    config.debug = args.debug
    config.restart = args.restart
    config.max_turns = args.max_turns
    config.additional_context_path = args.additional_context_path
    
    start_cleanup_thread()
    
    print("\n" + "="*70)
    print("Flask Interview Session Server - Multi-User Mode")
    print("="*70)
    print(f"🌐 Host:              {config.host}")
    print(f"🔌 Port:              {config.port}")
    print(f"🐛 Debug:             {config.debug}")
    print(f"🔐 Authentication:    Enabled")
    print(f"🧹 Session Cleanup:   Every 5 minutes (timeout: {SESSION_TIMEOUT_SECONDS/60:.0f} min)")
    print(f"🗣️  TTS Provider:      {TTS_PROVIDER} ({TTS_VOICE})")
    print("="*70)
    print(f"\n📍 Login at: http://{config.host}:{config.port}/login")
    print(f"📊 Health check: http://{config.host}:{config.port}/health")
    print("="*70 + "\n")
    
    if not config.debug:
        print("⚠️  For production, use: gunicorn -w 2 --threads 4 -b 0.0.0.0:8080 flask_app:app\n")
    
    app.run(
        host=config.host,
        port=config.port,
        debug=config.debug,
        use_reloader=False,
        threaded=True
    )
import argparse
import asyncio
import json
import logging
import os
import queue
import threading
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# Configuration and Constants
# ============================================================================

DEFAULT_SUPERVISOR_MODEL = "gpt-4.1-mini"
DEFAULT_RESPONDER_MODEL = "gpt-4.1-mini"
DEFAULT_USER_MODEL = "gpt-4.1-mini"

SUPERVISOR_DEFAULT_FREQUENCY = 3
SUPERVISOR_DEFAULT_MAX_TOKENS = 8192

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TurnLog:
    """Turn log for JSONL logging - simple turn-level data only"""
    timestamp: str
    turn_number: int
    user_response: Optional[str]
    assistant_message: Optional[str]
    # Notes captured from AI
    notes: Optional[str]
    raw_model_json: Optional[dict] = None
    # Dual-agent tracking
    supervisor_analysis: Optional[str] = None
    supervisor_used: bool = False
    supervisor_tokens: Optional[int] = None
    # Token usage
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def initialize_model_client(model: str, base_url: Optional[str] = None):
    """
    Initialize client with vLLM support for both OpenAI models.

    Args:
        model: Model name. If starts with "vllm:", uses vLLM backend.
        base_url: Base URL for vLLM server (required for vLLM models).

    Returns:
        Client instance (OpenAI)
    """
    if model.startswith("vllm:"):
        from openai import OpenAI
        if not base_url:
            raise ValueError("base_url required for vLLM models")
        # vLLM models use OpenAI-compatible API
        return OpenAI(base_url=base_url, api_key="EMPTY")
    else:
        from openai import OpenAI
        # Standard OpenAI models
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable required for OpenAI models")
        return OpenAI(api_key=openai_api_key)


def load_spec(spec_path: str) -> dict:
    """Load interview specification from JSON file"""
    with open(spec_path, 'r') as f:
        spec = json.load(f)

    # Validate structure
    if 'topics' not in spec:
        raise ValueError("Interview spec must contain 'topics' array")

    for topic in spec['topics']:
        if 'topic_name' not in topic or 'questions' not in topic:
            raise ValueError("Each topic must have 'topic_name' and 'questions' fields")

    return spec


def load_data_source(data_source_path: Optional[str]) -> str:
    """Load background context from file"""
    if not data_source_path:
        return ""

    with open(data_source_path, 'r') as f:
        return f.read()

class AsyncJSONLLogger:
    """
    JSONL logger that writes each record immediately to disk.
    Uses synchronous writes with flush + fsync for GCS FUSE compatibility.
    """
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self._lock = threading.Lock()
        # Ensure directory exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        # Touch the file to ensure it exists
        self.filepath.touch(exist_ok=True)

    def log(self, record: dict):
        """Write a record immediately to the JSONL file with fsync for GCS."""
        with self._lock:
            with self.filepath.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())  # Force OS to write to disk (important for GCS FUSE)

    def close(self):
        """No-op for compatibility - writes are already synced."""
        pass


# ============================================================================
# Notes Management
# ============================================================================

def save_notes(notes_path: str, notes_list: List[dict]):
    """
    Save chronological notes to JSON file.

    Args:
        notes_path: Path to notes JSON file
        notes_list: List of note dicts with turn_number, timestamp, notes
    """
    notes_data = {
        "interview_timestamp": datetime.now().isoformat(),
        "total_turns": len(notes_list),
        "notes": notes_list
    }

    with open(notes_path, 'w') as f:
        json.dump(notes_data, f, indent=2)


# ============================================================================
# Supervisor Trigger Logic
# ============================================================================

def should_trigger_supervisor(history: List[dict], frequency: int = 3) -> bool:
    """
    Intelligent supervisor trigger - cost optimization.

    Args:
        history: Conversation history
        frequency: Call supervisor every N user responses

    Returns:
        True if supervisor should be called this turn
    """
    user_messages = [msg for msg in history if msg.get('role') == 'user']
    num_user_messages = len(user_messages)

    # Strategy: Detect short/repetitive responses
    if len(history) >= 4:
        recent_messages = history[-4:]
        user_responses = [msg['content'] for msg in recent_messages if msg.get('role') == 'user']
        if len(user_responses) >= 2:
            if any(len(resp.strip()) < 10 for resp in user_responses[-2:]):
                return True  # Trigger if responses too short

    # Always trigger in initial phase
    if num_user_messages <= 1:
        return True

    # Frequency control: call every N exchanges by default
    if num_user_messages % frequency != 0:
        return False

    return True  # Triggered by frequency rule


# ============================================================================
# Synchronous Supervisor Call (SIMPLIFIED)
# ============================================================================

def call_supervisor_sync(
    spec: dict,
    history: List[dict],
    supervisor_client,
    supervisor_model: str,
    max_tokens: int = SUPERVISOR_DEFAULT_MAX_TOKENS,
    logger: Optional[logging.Logger] = None
) -> Tuple[str, int]:
    """
    Call supervisor model synchronously and return analysis.

    Returns:
        Tuple of (analysis_text, tokens_used)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        # Prepare supervisor prompt
        recent_history = history[-10:] if len(history) > 10 else history

        supervisor_prompt = f"""You are an AI interview supervision expert, analyzing interview quality and providing strategic guidance.

**Full Interview Guide** (all topics and subtopics):
{json.dumps(spec, indent=2)}

**Interview Type**: Semi-structured / Flexible (AI-driven progression)

**Analysis Dimensions**:
1. Interview depth and quality
2. Interviewee engagement level
3. Topic coverage completeness across ALL topics
4. Conversation flow and natural transitions
5. Follow-up opportunities

**Your Task**:
Analyze the conversation history and provide strategic guidance:
- Which topics/subtopics have been covered adequately
- Whether to probe deeper or transition to new areas
- Quality of information gathered so far
- Suggested angles, follow-ups, or transitions to pursue
- Coverage gaps that should be addressed

**Note**: The interviewer AI will decide the next question - your role is strategic guidance only.

**Conversation History**:
{json.dumps(recent_history, indent=2)}
"""

        # Call supervisor model
        response = supervisor_client.chat.completions.create(
            model=supervisor_model.removeprefix("vllm:"),
            messages=[{"role": "user", "content": supervisor_prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        analysis = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0

        logger.info(f"Supervisor analysis completed: {len(analysis)} chars, {tokens_used} tokens")

        # Print supervisor output to console
        print(f"\n{'='*60}")
        print(f"[SUPERVISOR ANALYSIS - {tokens_used} tokens]")
        print(f"{'='*60}")
        print(analysis)
        print(f"{'='*60}\n")

        return (analysis, tokens_used)

    except Exception as e:
        logger.error(f"Supervisor analysis failed: {e}")
        raise


# ============================================================================
# Main AI Response Generation (SYNCHRONOUS SUPERVISOR)
# ============================================================================

async def generate_ai_response(
    text: str,
    spec: dict,
    history: List[dict],
    responder_client,
    responder_model: str,
    supervisor_client,
    supervisor_model: str,
    supervisor_frequency: int = 3,
    supervisor_max_tokens: int = SUPERVISOR_DEFAULT_MAX_TOKENS,
    logger: Optional[logging.Logger] = None
) -> dict:
    """
    Generate AI interviewer response with synchronous supervision.

    Flow: User input → Supervisor analysis (if triggered) → Responder generates question

    This function is called ONCE PER TURN. The AI model decides:
    - What topic to explore
    - What subtopic to ask about
    - Whether to probe deeper or move forward
    - When to transition between topics

    NO sequential loop or question_idx counter - AI-driven progression.
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    # === DUAL-AGENT INTEGRATION START ===

    # 1. Check if we should trigger supervisor (intelligent frequency control)
    use_supervisor = should_trigger_supervisor(history, supervisor_frequency)

    # 2. Call supervisor SYNCHRONOUSLY if triggered
    supervisor_analysis = None
    supervisor_tokens = 0
    if use_supervisor:
        print(f"\n[Triggering supervisor analysis...]\n")
        supervisor_analysis, supervisor_tokens = call_supervisor_sync(
            spec=spec,
            history=history,
            supervisor_client=supervisor_client,
            supervisor_model=supervisor_model,
            max_tokens=supervisor_max_tokens,
            logger=logger
        )

    # 3. Generate responder question with supervisor guidance
    full_spec_json = json.dumps(spec, indent=2)

    if supervisor_analysis:
        # Prompt WITH supervisor guidance
        system_prompt = f"""You are a professional AI interviewer conducting an in-depth, conversational interview.

**Supervisor's Strategic Guidance**:
{supervisor_analysis}

**Full Interview Guide** (all topics and subtopics):
{full_spec_json}

**Your Task**:
You have full autonomy to conduct the interview naturally. Based on the conversation history and supervisor guidance:
- Decide what topic/subtopic to explore next based on conversation flow
- Determine whether to probe deeper on current topic or transition to new areas
- Ask exactly ONE question per turn
- Paraphrase subtopics into conversational questions (never use subtopic text verbatim)
- Build on prior answers to maintain natural flow
- Cover topics in the guide over the course of the interview

**Notes Capture** (REQUIRED):
Generate structured notes from the user's LAST response:
- Concise (1-2 sentences max)
- Factual: dates, names, metrics, technologies, achievements
- Third person (e.g., "Worked at X from 2020-2022...")

**Output Format** (strict JSON):
{{
  "question_to_ask": "<your next question>",
  "notes": "<structured notes from user's last response>"
}}

Output ONLY the JSON, no other text."""
    else:
        # Prompt without supervisor (basic mode)
        system_prompt = f"""You are a professional AI interviewer conducting an in-depth, conversational interview.

**Full Interview Guide** (all topics and subtopics):
{full_spec_json}

**Your Task**:
Conduct a natural interview, deciding what to ask based on conversation history:
- Ask exactly ONE question per turn
- Decide what to explore based on conversation flow and guide coverage
- Paraphrase subtopics into conversational questions
- Build natural flow and transitions

**Notes Capture** (REQUIRED):
Generate structured notes from the user's LAST response:
- Concise (1-2 sentences max)
- Factual: dates, names, metrics, technologies, achievements
- Third person (e.g., "Worked at X from 2020-2022...")

**Output Format** (strict JSON):
{{
  "question_to_ask": "<question>",
  "notes": "<notes>"
}}

Output ONLY the JSON, no other text."""

    # Call responder model
    try:
        # OpenAI-compatible API (vLLM or OpenAI)
        messages = [{"role": "system", "content": system_prompt}] + history
        if text:
            messages.append({"role": "user", "content": text})

        response = responder_client.chat.completions.create(
            model=responder_model.removeprefix("vllm:"),
            messages=messages,
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        assistant_content = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') else 0
        completion_tokens = response.usage.completion_tokens if hasattr(response, 'usage') else 0
        total_tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0

        # Parse JSON response
        response_json = json.loads(assistant_content)

        assistant_msg = response_json.get("question_to_ask", "")
        notes = response_json.get("notes", "")

        logger.info(f"Responder generated question ({total_tokens} tokens)")
        if supervisor_analysis:
            logger.info(f"Supervisor guidance used ({supervisor_tokens} tokens)")

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.error(f"Response content: {assistant_content}")
        assistant_msg = "I apologize, I encountered an error. Could you please repeat that?"
        notes = ""
        response_json = {}
        prompt_tokens = completion_tokens = total_tokens = 0

    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        assistant_msg = "I apologize, I encountered an error. Could you please try again?"
        notes = ""
        response_json = {}
        prompt_tokens = completion_tokens = total_tokens = 0

    # === DUAL-AGENT INTEGRATION END ===

    # Return response (no progression logic - AI decides next move)
    return {
        "assistant_message": assistant_msg,
        "notes": notes,
        "supervisor_analysis": supervisor_analysis,
        "supervisor_used": supervisor_analysis is not None,
        "supervisor_tokens": supervisor_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "response_json": response_json
    }

class LLMUserAgent:
    """LLM-powered user agent for automated interviews"""

    def __init__(self, user_profile: str, model: str, client, logger):
        self.user_profile = user_profile
        if model.startswith("vllm:"):
            self.model = model[len("vllm:"):]
        else:
            self.model = model
        self.client = client
        self.logger = logger
        self.conversation_history = []

    async def generate_response(self, interviewer_question: str) -> str:
        """Generate user response using LLM"""
        self.conversation_history.append({"role": "interviewer", "content": interviewer_question})

        # Build prompt
        system_prompt = f"""You are playing the role of a real person being interviewed in a semi-structured interview.

You are not trying to be maximally informative.
You are trying to be natural, human-like, and interview-realistic.

If this is the first turn, respond only that you are happy to start the interview.

# BACKGROUND INFORMATION
You have access to the following background information about yourself.
This represents your full life history, but not everything should be disclosed at once.

<profile_background>
{self.user_profile}
</profile_background>

# GENERAL INTERVIEW RULES
- Always answer the question asked.
- Never skip a question.
- Do not anticipate follow-up questions.
- Treat this as a real interview: the interviewer controls depth and direction.
- Answer only what is necessary for the current question.

# HUMAN STOPPING HEURISTIC (CRITICAL)
Humans stop talking once they have given a sufficient answer, not a complete one.

- Aim for the first reasonable stopping point.
- Assume the interviewer may interrupt or follow up.
- Do not try to close the topic yourself.

# BREVITY & DEPTH CONTROL (STRICT)
- Answer length: 1-2 sentences.
- Typical answers should reveal approximately one concrete fact or signal.
- Do not compress multiple ideas, timelines, or facts into one answer.

# VAGUE OR OPEN-ENDED QUESTIONS (CRITICAL)
If a question is vague, ambiguous, or open-ended to answer without guessing, for example if it sounds like listing some list of topics, then:
- Do NOT invent scope or details.
- Briefly acknowledge the ambiguity (e.g., "I'm not sure which aspect you mean").
- Either ask one short clarification question, OR state one reasonable assumption and answer briefly under that assumption.
- Do not do both and do not expand beyond the assumed or clarified scope.

# CONTENT GUIDELINES
- Stay tightly focused on the question’s scope.
- Do not expand across time, roles, or institutions unless asked.
- Do not repeat prior answers unless explicitly prompted.
- Avoid lists unless the interviewer asks for them.
- Avoid meta-commentary about motivation, passion, or energy.

# EMERGENCE (ALLOWED AND ENCOURAGED)
You may introduce emergent content that is not explicitly listed in your background, such as:
- Interpretations
- Personal insights
- Opinions
- Non-obvious takeaways

Constraints on emergence:
- Emergent content must be reflective, not biographical.
- Do not introduce new life events, credentials, dates, or timeline facts unless asked.
- At most one emergent insight per answer.
- Emergence should add depth, not breadth.

Preferred pattern:
- One profile-grounded anchor
- Optional one emergent insight
- Stop

# STYLE
- Natural, conversational, confident.
- Professional but unscripted.
- Sound like a strong candidate who knows when to stop talking.

# STOPPING RULE (ABSOLUTE)
- End your response immediately after your main point.
- Do not summarize.
- Do not add closing remarks such as “happy to elaborate” or “let me know if you’d like more.”

Respond directly as the interviewee.
Do not include tags, reasoning, or preamble.
"""

        history_str = ""
        if len(self.conversation_history) > 1:
            recent = self.conversation_history[-10:]
            history_str = "\n\nRecent conversation:\n"
            for msg in recent:
                role = "Interviewer" if msg["role"] == "interviewer" else "You"
                history_str += f"{role}: {msg['content']}\n"

        user_prompt = f"""{history_str}

Current question: {interviewer_question}

Your response:"""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.7,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        response_text = response.choices[0].message.content.strip()
        self.conversation_history.append({"role": "user", "content": response_text})
        return response_text

    def save_history(self, path: str):
        """Save conversation history to JSON"""
        with open(path, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)


# ============================================================================
# Main Interview Loop (AI-driven like mimitalk.py)
# ============================================================================

async def run_interview(
    spec: dict,
    responder_client,
    responder_model: str,
    supervisor_client,
    supervisor_model: str,
    log_path: str,
    input_mode: str = "user",
    user_agent: Optional[LLMUserAgent] = None,
    max_turns: int = 80,
    supervisor_frequency: int = 3,
    supervisor_max_tokens: int = SUPERVISOR_DEFAULT_MAX_TOKENS,
    logger: Optional[logging.Logger] = None
):
    """
    Main interview loop - simple wrapper around AI-driven function.
    NO nested loops over topics/subtopics - AI handles progression.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Initialize
    interview_uuid = str(uuid.uuid4())
    history = []
    turn_count = 0
    notes_list = []

    # Initialize logging
    async_logger = AsyncJSONLLogger(Path(log_path))

    logger.info(f"Starting interview (UUID: {interview_uuid})")
    logger.info(f"Responder: {responder_model}, Supervisor: {supervisor_model}")
    logger.info(f"Supervisor frequency: every {supervisor_frequency} turns")

    try:
        # Initial greeting (optional - call with empty text)
        initial_response = await generate_ai_response(
            text="",
            spec=spec,
            history=history,
            responder_client=responder_client,
            responder_model=responder_model,
            supervisor_client=supervisor_client,
            supervisor_model=supervisor_model,
            supervisor_frequency=supervisor_frequency,
            supervisor_max_tokens=supervisor_max_tokens,
            logger=logger
        )

        print(f"\nInterviewer: {initial_response['assistant_message']}\n")
        history.append({"role": "assistant", "content": initial_response['assistant_message']})

        # Main interview loop
        while turn_count < max_turns:
            # Get user input
            if input_mode == "llm" and user_agent:
                user_text = await user_agent.generate_response(initial_response['assistant_message'] if turn_count == 0 else assistant_msg)
                print(f"You: {user_text}\n")
            else:
                user_text = input("You: ").strip()
                if not user_text or user_text.lower() in ["exit", "quit", "q"]:
                    print("Interview ended by user.")
                    break

            history.append({"role": "user", "content": user_text})

            # Generate next question (AI-driven, no loop)
            response = await generate_ai_response(
                text=user_text,
                spec=spec,  # ALWAYS pass full spec
                history=history,
                responder_client=responder_client,
                responder_model=responder_model,
                supervisor_client=supervisor_client,
                supervisor_model=supervisor_model,
                supervisor_frequency=supervisor_frequency,
                supervisor_max_tokens=supervisor_max_tokens,
                logger=logger
            )

            # Extract and display assistant response
            assistant_msg = response['assistant_message']
            print(f"Interviewer: {assistant_msg}\n")
            history.append({"role": "assistant", "content": assistant_msg})

            # Log turn with notes and supervisor info
            turn_log = TurnLog(
                timestamp=datetime.now().isoformat(),
                turn_number=turn_count,
                user_response=user_text,
                assistant_message=assistant_msg,
                notes=response["notes"],
                raw_model_json=response["response_json"],
                supervisor_analysis=response.get("supervisor_analysis"),
                supervisor_used=response.get("supervisor_used", False),
                supervisor_tokens=response.get("supervisor_tokens"),
                prompt_tokens=response.get("prompt_tokens"),
                completion_tokens=response.get("completion_tokens"),
                total_tokens=response.get("total_tokens")
            )
            async_logger.log(asdict(turn_log))

            # Save notes (if present)
            if response["notes"]:
                notes_list.append({
                    "turn_number": turn_count,
                    "timestamp": datetime.now().isoformat(),
                    "notes": response["notes"]
                })

            turn_count += 1

        # Save final notes
        if notes_list:
            notes_path = log_path.replace(".jsonl", "_notes.json")
            save_notes(notes_path, notes_list)
            logger.info(f"Saved {len(notes_list)} notes to {notes_path}")

        # Save user agent history if used
        if input_mode == "llm" and user_agent:
            user_history_path = log_path.replace(".jsonl", "_user_history.json")
            user_agent.save_history(user_history_path)
            logger.info(f"Saved user agent history to {user_history_path}")

    finally:
        async_logger.close()
        logger.info("Interview completed")


# ============================================================================
# Command-Line Interface
# ============================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Mimitalk Adapted: Dual-Agent AI Interviewer with vLLM Support"
    )

    # Required arguments
    ap.add_argument("--spec", required=True, help="Path to interview spec JSON file")

    # Interview settings
    ap.add_argument("--log", default="interview_log.jsonl", help="JSONL log file path")
    ap.add_argument("--input-mode", choices=["user", "llm"], default="user", help="Input mode: user (manual) or llm (automated)")
    ap.add_argument("--max-turns", type=int, default=80, help="Max conversation turns")
    ap.add_argument("--verbose-logging", action="store_true", help="Print detailed logging messages")

    # Dual-agent model arguments
    ap.add_argument("--responder-model", default=DEFAULT_RESPONDER_MODEL, help="Responder model (use vllm:model-name for vLLM)")
    ap.add_argument("--responder-base-url", default=None, help="vLLM server URL for responder")
    ap.add_argument("--supervisor-model", default=DEFAULT_SUPERVISOR_MODEL, help="Supervisor model (use vllm:model-name for vLLM)")
    ap.add_argument("--supervisor-base-url", default=None, help="vLLM server URL for supervisor")

    # LLM user agent arguments
    ap.add_argument("--user-profile", default=None, help="Path to user profile text file (required if --input-mode llm)")
    ap.add_argument("--user-model", default=DEFAULT_USER_MODEL, help="Model for LLM user agent")
    ap.add_argument("--user-base-url", default=None, help="vLLM server URL for user agent")

    # Supervisor optimization
    ap.add_argument("--supervisor-frequency", type=int, default=SUPERVISOR_DEFAULT_FREQUENCY, help="Call supervisor every N user responses")
    ap.add_argument("--supervisor-max-tokens", type=int, default=SUPERVISOR_DEFAULT_MAX_TOKENS, help="Max tokens for supervisor analysis")

    return ap.parse_args()


async def main():
    """Main entry point"""
    args = parse_args()

    # Setup logging
    logger = setup_logging(args.verbose_logging)

    # Load spec and data source
    spec = load_spec(args.spec)

    # Initialize model clients
    responder_client = initialize_model_client(args.responder_model, args.responder_base_url)
    supervisor_client = initialize_model_client(args.supervisor_model, args.supervisor_base_url)

    # Initialize LLM user agent if needed
    user_agent = None
    if args.input_mode == "llm":
        if not args.user_profile:
            raise ValueError("--user-profile required when --input-mode llm")
        user_profile = load_data_source(args.user_profile)
        user_client = initialize_model_client(args.user_model, args.user_base_url)
        user_agent = LLMUserAgent(user_profile, args.user_model, user_client, logger)

    # Run interview
    await run_interview(
        spec=spec,
        responder_client=responder_client,
        responder_model=args.responder_model,
        supervisor_client=supervisor_client,
        supervisor_model=args.supervisor_model,
        log_path=args.log,
        input_mode=args.input_mode,
        user_agent=user_agent,
        max_turns=args.max_turns,
        supervisor_frequency=args.supervisor_frequency,
        supervisor_max_tokens=args.supervisor_max_tokens,
        logger=logger
    )


if __name__ == "__main__":
    asyncio.run(main())

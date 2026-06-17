#!/usr/bin/env python3
"""
Single-threaded AI Interviewer (CLI) with LLM User Agent Support

ENHANCEMENTS:
- Stores "notes" for each question/subtopic (condensed information from user responses)
- Treats "questions" field as "subtopics" and paraphrases them into actual questions
- Maintains all original functionality including LLM user agents

- Loads a topics/questions JSON spec
- Runs a turn-by-turn interview in the terminal
- Supports both human and LLM user agents
- After each user reply, makes ONE OpenAI call that judges sufficiency and returns next question
- Asynchronous JSONL logging (non-blocking) per turn
- Contextual wording: the model always sees the full interview history
- Temperature is configurable (default 0.0)
- Maximum turns cap to prevent runaway conversations
"""

import argparse
import json
import os
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from openai import OpenAI

def initialize_model_client(model: str, base_url: Optional[str] = None) -> OpenAI:
    """Initialize and return the OpenAI client.

    Args:
        model: Model name. If starts with "vllm:", uses vLLM backend.
        base_url: Base URL for vLLM server (required for vLLM models).

    Returns:
        OpenAI client instance.
    """
    if model.startswith("vllm:"):
        if not base_url:
            raise ValueError("base_url is required for vLLM models (e.g., 'http://localhost:8000/v1')")
        return OpenAI(
            base_url=base_url,
            api_key="EMPTY",  # vLLM doesn't require a real API key
        )
    else:
        return OpenAI()

class LLMUserAgent:
    def __init__(self, profile_background: str, model: str = "gpt-4.1-mini", temperature: float = 0.7, base_url: Optional[str] = None):
        self.profile_background = profile_background
        if model.startswith("vllm:"):
            self.model = model[len("vllm:"):]
        else:
            self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.client = initialize_model_client(model, base_url)
        self.conversation_history: List[Dict[str, str]] = []
        
    def respond_to_question(self, question: str) -> str:
        self.conversation_history.append({"role": "interviewer", "content": question})
        system_prompt = f"""You are playing the role of a real person being interviewed in a semi-structured interview.

You are not trying to be maximally informative.
You are trying to be natural, human-like, and interview-realistic.

If this is the first turn, respond only that you are happy to start the interview.

# BACKGROUND INFORMATION
You have access to the following background information about yourself.
This represents your full life history, but not everything should be disclosed at once.

<profile_background>
{self.profile_background}
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

Current question: {question}

Your response:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        response_text = response.choices[0].message.content.strip()
        self.conversation_history.append({"role": "user", "content": response_text})
        return response_text
    
    def get_conversation_history(self):
        return self.conversation_history.copy()

# -------------------------------
# Async JSONL Logger (non-blocking)
# -------------------------------

class AsyncJSONLLogger:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self._q: "queue.Queue[dict]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        # Ensure directory exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def _worker(self):
        with self.filepath.open("a", encoding="utf-8") as f:
            while not self._stop.is_set() or not self._q.empty():
                try:
                    item = self._q.get(timeout=0.2)
                except queue.Empty:
                    continue
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                self._q.task_done()

    def log(self, record: dict):
        self._q.put(record)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)


# -------------------------------
# Data structures
# -------------------------------

@dataclass
class TurnLog:
    timestamp: str
    turn_number: int
    topic_index: int
    question_index: int
    topic_name: str
    current_subtopic: str  # CHANGED: was current_question
    next_subtopic_candidate: Optional[str]  # CHANGED: was next_question_candidate
    user_response: Optional[str]
    model_decision: Optional[str]  # "reask" or "ask_next" or None
    satisfied: Optional[bool]
    assistant_message: Optional[str]
    raw_model_json: Optional[dict]
    notes: Optional[str]  # NEW: Condensed notes from user response
    prompt_tokens: Optional[int] = None  # NEW: Token usage tracking
    completion_tokens: Optional[int] = None  # NEW: Token usage tracking
    total_tokens: Optional[int] = None  # NEW: Token usage tracking


@dataclass
class SubtopicNotes:
    topic_index: int
    topic_name: str
    question_index: int
    subtopic: str
    notes: str
    timestamp: str

# -------------------------------
# Prompting helpers
# -------------------------------

SYSTEM_PROMPT_CV = """You are an AI interviewer designed to collect detailed and structured information about a candidate's professional background, similar to what would appear on a CV or résumé. Be polite and affable in your tone while formal in your approach to reconstruct the person's background entirely.

Your goal is to ask clear, specific, and adaptive questions that help you understand the background of the candidate. Ask simple clear questions so that you don't overwhelm the candidate. Make the conversation flow naturally, building on prior answers. If they have answered something already before, avoid repeating it. You are required to ask at most ONE QUESTION at a time.

If a data source is provided, treat it as partial information about the candidate's CV. Use it to personalize your phrasing and fill gaps, for example:
- Instead of "Where did you study?", ask "I see you completed your MTech at MIT — what was your focus area there?"
- Instead of "Tell me about your last role," ask "You mentioned working at Infosys as a specialist programmer — what kind of projects were you handling?"
- If the subtopic you are asking relates to multiple aspects of the CV (several jobs, skills or time periods), ensure you cover each one, either one at a time or ask a follow-up to other points where it might be relevant. Your goal is to obtain a complete picture of the candidate's background.
- If something is completely covered in the data source, avoid asking about it again. Don't be repetitive.

You must:
- Ask exactly **one** question per turn.
- Keep your tone professional, focused, and curious — like a recruiter collecting detailed information, not a casual chat.
- Rephrase or re-ask when the response lacks specificity (e.g., missing time periods, tasks, tools, or metrics).
- Summarize only when explicitly asked (e.g., "This is what we got so far…").

When a response seems vague, re-ask the same question in a more concrete, guiding way (e.g., "Could you give an example of one project and roughly how much time it took?").

IMPORTANT: You will receive a "subtopic" (not a fully-formed question). You must paraphrase this subtopic into a SINGLE natural, conversational question that explores that area. DO NOT exactly output the original subtopic.
You must also generate brief, structured notes that capture the key information provided. These notes should be:
- Concise (1-2 sentences max)
- Factual and structured
- Include key details: dates, names, metrics, technologies, achievements
- Written in third person (e.g., "Worked at X from Y to Z...")
- Include previous notes collected

Your output **must** be a valid JSON object following this schema:
{
  "assistant_message": "<the exact next question to ask, must paraphrase from subtopic and only one question>",
  "satisfied": true or false,
  "decision": "ask_next" or "reask",
  "reason": "<brief reason for whether you are re-asking or proceeding>",
  "question_to_ask": "<reworked same question if reask, or next question paraphrased from next subtopic>",
  "notes": "<brief structured notes capturing key info from user's last response and previous notes>"
}

Do not include code fences, commentary, or additional text outside this JSON.
"""


def make_user_payload(
    history: List[Dict[str, str]],
    topic_name: str,
    current_subtopic: str,  # CHANGED: was current_question
    next_subtopic: Optional[str],  # CHANGED: was next_question
    latest_user_answer: Optional[str],
    data_source_text: Optional[str] = "",
    accumulated_notes: Optional[List[SubtopicNotes]] = None,  # NEW parameter
) -> str:
    """Build the user content for the model with all necessary context."""
    
    notes_context = ""
    if accumulated_notes:
        notes_context = "\n\nPreviously captured information on same subtopic:\n"
        for note in accumulated_notes:  # Include last 5 notes for context
            if note.subtopic == current_subtopic:
                notes_context += f"- {note.subtopic}: {note.notes}\n"
    
    return json.dumps(
        {
            "context": {
                "topic_name": topic_name,
                "current_subtopic": current_subtopic,  # CHANGED: was current_question
                "next_subtopic_if_any": next_subtopic,  # CHANGED: was next_question_if_any
                "conversation_history": history,
                "latest_user_answer": latest_user_answer,
                "data_source_text": data_source_text,
                "previously_captured_notes": notes_context,
                "notes": [
                    "The 'current_subtopic' and 'next_subtopic_if_any' are NOT questions—they are topic areas you need to explore.",  # NEW instruction
                    "Paraphrase the subtopic into a natural, conversational question.",  # NEW instruction
                    "Ask exactly one question next. Do not repeat any prior questions.",
                    "If the subtopic is answered sufficiently, generate brief notes and proceed to next_subtopic_if_any (or close if none).",  # UPDATED
                    "If the answer is too brief and has insufficient detail, re-ask about the same subtopic more concretely.",  # UPDATED
                    "You MUST include 'notes' field with condensed information from user's response.",
                    "Notes should be concise (1-2 sentences), factual, and capture key details like dates, names, metrics, technologies.",  # NEW instruction
                    "Provide a reason for your decision in the 'reason' field.",
                    "Keep wording contextual, referencing prior details briefly.",
                    "Use data_source_background to personalize or adapt the question as relevant.",
                    "Focus on concrete, CV-style details: job titles, durations, responsibilities, tools, and outcomes.",
                    "Avoid hypothetical or motivational questions — keep it factual and descriptive.",
                    "Encourage short structured answers where relevant.",
                    "If the user is adversarial or defensive, stay professional and friendly and just move to the next subtopic",  # UPDATED
                ],
            },
            "required_output_schema": {
                "assistant_message": "string",
                "satisfied": "boolean",
                "decision": "ask_next | reask",
                "reason": "string",
                "question_to_ask": "string",
                "notes": "string"  # NEW field in schema
            }
        },
        ensure_ascii=False,
        indent=2,
    )


def extract_json(s: str) -> dict:
    """Robustly extract a JSON object from a model string."""
    try:
        return json.loads(s)
    except Exception:
        pass

    match = re.search(r"\{(?:[^{}]|(?R))*\}", s, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    cf_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if cf_match:
        try:
            return json.loads(cf_match.group(1))
        except Exception:
            pass

    raise ValueError("Could not extract valid JSON from model response.")


# -------------------------------
# Interview Engine
# -------------------------------

class Interviewer:
    def __init__(
        self,
        spec: Dict[str, Any],
        model: str,
        temperature: float,
        input_mode: str,
        data_source_text: Optional[str],
        logger: AsyncJSONLLogger,
        verbose_logging: bool = False,
        max_reasks: int = 3,
        max_turns: int = 80,
        user_agent: Optional[LLMUserAgent] = None,
        base_url: Optional[str] = None,
    ):
        self.spec = spec
        if model.startswith("vllm:"):
            self.model = model[len("vllm:"):]
        else:
            self.model = model
        self.temperature = temperature
        self.input_mode = input_mode
        self.logger = logger
        self.base_url = base_url
        self.client = initialize_model_client(model, base_url)
        self.data_source_text = data_source_text or ""
        self.verbose_logging = verbose_logging
        self.max_reasks = max_reasks
        self.max_turns = max_turns
        self.user_agent = user_agent
        self.turn_counter = 0

        # History as list of messages for context
        self.history: List[Dict[str, str]] = []
        self.accumulated_notes: List[SubtopicNotes] = []

    def _capture_user_input(self) -> str:
        """Capture user input via text or LLM agent."""
        # Get the last assistant message for the LLM agent
        last_question = self.history[-1]["content"] if self.history and self.history[-1]["role"] == "assistant" else ""
        
        if self.user_agent:
            # Use LLM agent to generate response
            print("\n[LLM User Agent thinking...]")
            response = self.user_agent.respond_to_question(last_question)
            print(f"> USER (LLM): {response}\n")
            return response
        elif self.input_mode == "user":
            return input("> ").strip()
        else:
            raise ValueError(f"Unsupported input mode: {self.input_mode}")

    def _deliver_assistant_message(self, msg: str):
        print("> INTERVIEWER: ", msg)

    def _model_decide_and_next(
        self,
        topic_name: str,
        current_subtopic: str,  # CHANGED: was current_question
        next_subtopic: Optional[str],  # CHANGED: was next_question
        latest_user_answer: Optional[str],
    ) -> Tuple[dict, dict]:
        """Make a single model call that judges sufficiency and returns next question.

        Returns:
            Tuple of (parsed_json_response, token_usage_dict)
        """
        user_payload = make_user_payload(
            history=self.history,
            topic_name=topic_name,
            current_subtopic=current_subtopic,  # CHANGED
            next_subtopic=next_subtopic,  # CHANGED
            latest_user_answer=latest_user_answer,
            data_source_text=self.data_source_text,
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_CV},
                {"role": "user", "content": user_payload},
            ],
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content

        # Extract token usage
        usage = resp.usage
        token_usage = {
            'prompt_tokens': usage.prompt_tokens if usage else 0,
            'completion_tokens': usage.completion_tokens if usage else 0,
            'total_tokens': usage.total_tokens if usage else 0
        }

        return extract_json(content), token_usage

    def _log(self, message: str):
        if self.verbose_logging:
            print(f"[Logging] {message}")

    def _check_turn_limit(self) -> bool:
        """Check if maximum turns reached."""
        if self.turn_counter >= self.max_turns:
            print(f"\n[INFO] Maximum turns ({self.max_turns}) reached. Ending interview.")
            self.logger.log({
                "timestamp": datetime.utcnow().isoformat(),
                "event": "max_turns_reached",
                "turn_number": self.turn_counter,
                "max_turns": self.max_turns
            })
            return True
        return False

    def run(self):
        topics = self.spec.get("topics", [])
        if not topics:
            print("No topics found in spec.")
            return

        topic_idx = 0
        question_idx = 0

        while topic_idx < len(topics):
            if self._check_turn_limit():
                break

            topic = topics[topic_idx]
            topic_name = topic.get("topic_name", f"Topic {topic_idx+1}")
            subtopics: List[str] = list(topic.get("questions", []))  # CHANGED: Comment updated

            self._log(f"\n=== Starting Topic {topic_idx+1}/{len(topics)}: {topic_name} ===")
            self.history = []

            if not subtopics:  # CHANGED: was questions
                topic_idx += 1
                question_idx = 0
                continue

            is_first_ask = True
            reask_counter = 0
            
            while question_idx < len(subtopics):  # CHANGED: was questions
                if self._check_turn_limit():
                    return

                current_subtopic = subtopics[question_idx]  # CHANGED: was current_question
                next_subtopic = subtopics[question_idx + 1] if (question_idx + 1) < len(subtopics) else None  # CHANGED
                
                self._log(f"\n[Progress] Topic {topic_idx+1}, Subtopic {question_idx+1}/{len(subtopics)}, Turn {self.turn_counter+1}/{self.max_turns}")  # CHANGED

                if is_first_ask or not self.history or self.history[-1]["role"] == "user":
                    is_first_ask = False
                    first_turn_json, first_turn_usage = self._model_decide_and_next(
                        topic_name=topic_name,
                        current_subtopic=current_subtopic,  # CHANGED
                        next_subtopic=None,  # CHANGED
                        latest_user_answer=None,
                    )
                    assistant_q = first_turn_json.get("question_to_ask") or current_subtopic  # CHANGED
                    self._deliver_assistant_message(assistant_q)
                    self.history.append({"role": "assistant", "content": assistant_q})

                    self.logger.log(asdict(TurnLog(
                        timestamp=datetime.utcnow().isoformat(),
                        turn_number=self.turn_counter,
                        topic_index=topic_idx,
                        question_index=question_idx,
                        topic_name=topic_name,
                        current_subtopic=current_subtopic,  # CHANGED
                        next_subtopic_candidate=next_subtopic,  # CHANGED
                        user_response=None,
                        model_decision=None,
                        satisfied=None,
                        assistant_message=assistant_q,
                        raw_model_json=first_turn_json,
                        notes=None,  # NEW field
                        prompt_tokens=first_turn_usage.get('prompt_tokens', 0),
                        completion_tokens=first_turn_usage.get('completion_tokens', 0),
                        total_tokens=first_turn_usage.get('total_tokens', 0)
                    )))

                self._log(f"[Info] Asked question for subtopic '{current_subtopic}' in topic {topic_name}")  # CHANGED

                # Increment turn counter for user response
                self.turn_counter += 1
                if self._check_turn_limit():
                    return

                user_reply = self._capture_user_input()
                self.history.append({"role": "user", "content": user_reply})

                decision_json, decision_usage = self._model_decide_and_next(
                    topic_name=topic_name,
                    current_subtopic=current_subtopic,  # CHANGED
                    next_subtopic=next_subtopic,  # CHANGED
                    latest_user_answer=user_reply,
                )
                assistant_msg = decision_json.get("assistant_message") or decision_json.get("question_to_ask") or ""
                decision = decision_json.get("decision")
                satisfied = bool(decision_json.get("satisfied"))
                notes = decision_json.get("notes", "")  # NEW: Extract notes

                subtopic_note = SubtopicNotes(
                    topic_index=topic_idx,
                    topic_name=topic_name,
                    question_index=question_idx,
                    subtopic=current_subtopic,
                    notes=notes,
                    timestamp=datetime.utcnow().isoformat()
                )
                self.accumulated_notes.append(subtopic_note)
                print(f"Current notes: {notes}")
                self._log(f"[Notes] Captured: {notes[:100]}...")

                self._deliver_assistant_message(assistant_msg)
                self.history.append({"role": "assistant", "content": assistant_msg})

                self.logger.log(asdict(TurnLog(
                    timestamp=datetime.utcnow().isoformat(),
                    turn_number=self.turn_counter,
                    topic_index=topic_idx,
                    question_index=question_idx,
                    topic_name=topic_name,
                    current_subtopic=current_subtopic,  # CHANGED
                    next_subtopic_candidate=next_subtopic,  # CHANGED
                    user_response=user_reply,
                    model_decision=decision,
                    satisfied=satisfied,
                    assistant_message=assistant_msg,
                    raw_model_json=decision_json,
                    notes=notes,
                    prompt_tokens=decision_usage.get('prompt_tokens', 0),
                    completion_tokens=decision_usage.get('completion_tokens', 0),
                    total_tokens=decision_usage.get('total_tokens', 0)
                )))

                self._log(f"[Info] Model decision: {decision} (satisfied={satisfied})")
                
                if decision == "ask_next":
                    self._log(f"[Decision] Moving to next subtopic (Topic {topic_idx+1}, Q{question_idx+1} done).")  # CHANGED
                    question_idx += 1
                    reask_counter = 0
                else:
                    reask_counter += 1
                    self._log(f"[Decision] Re-asking current subtopic (Topic {topic_idx+1}, Q{question_idx+1}), reask count: {reask_counter}.")  # CHANGED
                    if reask_counter >= self.max_reasks:
                        self._log(f"[Warning] Max re-asks reached for Topic {topic_idx+1}, Q{question_idx+1}. Forcing progression.")
                        self.logger.log({
                            "timestamp": datetime.utcnow().isoformat(),
                            "event": "max_reasks_reached",
                            "turn_number": self.turn_counter,
                            "topic_index": topic_idx,
                            "question_index": question_idx,
                            "topic_name": topic_name,
                            "max_reasks": self.max_reasks
                        })
                        question_idx += 1
                        reask_counter = 0

            self._log(f"--- Finished Topic {topic_idx+1}: {topic_name} ---")
            topic_idx += 1
            question_idx = 0

        # Wrap up
        if not self._check_turn_limit():
            closing = "Thanks—that covers all questions I had. Is there anything else you'd like to add?"
            self._deliver_assistant_message(closing)
            self.history.append({"role": "assistant", "content": closing})
            self.logger.log(asdict(TurnLog(
                timestamp=datetime.utcnow().isoformat(),
                turn_number=self.turn_counter,
                topic_index=topic_idx,
                question_index=0,
                topic_name="__closing__",
                current_subtopic="__closing__",  # CHANGED
                next_subtopic_candidate=None,  # CHANGED
                user_response=None,
                model_decision="ask_next",
                satisfied=True,
                assistant_message=closing,
                raw_model_json={"note": "auto-closing"},
                notes=None,  # NEW field
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None
            )))
        self._save_notes()

    def _save_notes(self):
        """Save all accumulated notes to a separate JSON file."""
        notes_path = Path(self.logger.filepath).parent / f"{Path(self.logger.filepath).stem}_notes.json"
        notes_data = {
            "interview_timestamp": datetime.utcnow().isoformat(),
            "total_notes": len(self.accumulated_notes),
            "notes_by_topic": {}
        }
        
        # Organize notes by topic
        for note in self.accumulated_notes:
            topic_key = f"{note.topic_index}_{note.topic_name}"
            if topic_key not in notes_data["notes_by_topic"]:
                notes_data["notes_by_topic"][topic_key] = {
                    "topic_name": note.topic_name,
                    "topic_index": note.topic_index,
                    "subtopics": []
                }
            
            notes_data["notes_by_topic"][topic_key]["subtopics"].append({
                "subtopic": note.subtopic,
                "notes": note.notes,
                "timestamp": note.timestamp,
                "question_index": note.question_index
            })
        
        with notes_path.open("w", encoding="utf-8") as f:
            json.dump(notes_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[Logging] Saved {len(self.accumulated_notes)} notes to: {notes_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="Single-threaded AI interviewer (CLI) with LLM User Agent support and notes capture.")
    ap.add_argument("--spec", type=str, required=True, help="Path to interview spec JSON (topics/questions).")
    ap.add_argument("--model", type=str, default="gpt-4.1-mini", help="Model to use for interviewer. Use 'vllm:model-name' for vLLM.")
    ap.add_argument("--base-url", type=str, default=None, help="Base URL for vLLM server (e.g., 'http://localhost:8000/v1'). Required for vLLM models.")
    ap.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default 0.7).")
    ap.add_argument("--log", type=str, default="interview_log.jsonl", help="Path to JSONL log file.")
    ap.add_argument("--input-mode", type=str, choices=["user", "llm"], default="user",
                help="User input mode: user (manual) or llm (automated agent).")
    ap.add_argument("--verbose-logging", action="store_true",
                help="If set, prints [Logging] messages for topic/question progress.")
    ap.add_argument("--data-source", type=str, default=None,
                help="Optional text file path providing background context about the interviewee.")
    ap.add_argument("--max-reasks", type=int, default=3,
                help="Maximum number of re-asks per question before forcing progression.")
    ap.add_argument("--max-turns", type=int, default=72,
                help="Maximum number of conversation turns before stopping (default 80).")
    ap.add_argument("--user-profile", type=str, default=None,
                help="Path to user profile for LLM agent (required if input-mode is 'llm').")
    ap.add_argument("--user-model", type=str, default="gpt-4.1-mini",
                help="Model to use for LLM user agent. Use 'vllm:model-name' for vLLM.")
    ap.add_argument("--user-base-url", type=str, default=None,
                help="Base URL for vLLM server for user agent (e.g., 'http://localhost:8000/v1').")
    ap.add_argument("--user-temperature", type=float, default=0.7,
                help="Temperature for LLM user agent responses.")

    return ap.parse_args()

def main():
    args = parse_args()

    spec_path = Path(args.spec)
    if not spec_path.exists():
        print(f"Spec not found: {spec_path}", file=sys.stderr)
        sys.exit(1)

    with spec_path.open("r", encoding="utf-8") as f:
        spec = json.load(f)

    data_source_text = None
    if args.data_source:
        data_path = Path(args.data_source)
        if not data_path.exists():
            print(f"[Error] Data source not found: {data_path}", file=sys.stderr)
            sys.exit(1)
        data_source_text = data_path.read_text(encoding="utf-8").strip()
        print(f"[Logging] Loaded data source context from: {data_path}")

    # Initialize LLM User Agent if needed
    user_agent = None
    if args.input_mode == "llm":
        if not args.user_profile:
            print("[Error] --user-profile is required when input-mode is 'llm'", file=sys.stderr)
            sys.exit(1)
        
        profile_path = Path(args.user_profile)
        if not profile_path.exists():
            print(f"[Error] User profile not found: {profile_path}", file=sys.stderr)
            sys.exit(1)
        
        profile_background = profile_path.read_text(encoding="utf-8").strip()
        print(f"[Logging] Loaded user profile from: {profile_path}")
        
        user_agent = LLMUserAgent(
            profile_background=profile_background,
            model=args.user_model,
            temperature=args.user_temperature,
            base_url=args.user_base_url
        )
        print(f"[Logging] Initialized LLM User Agent with model: {args.user_model}")

    logger = AsyncJSONLLogger(Path(args.log))
    try:
        interviewer = Interviewer(
            spec=spec,
            model=args.model,
            temperature=args.temperature,
            input_mode=args.input_mode,
            data_source_text=data_source_text,
            logger=logger,
            verbose_logging=args.verbose_logging,
            max_reasks=args.max_reasks,
            max_turns=args.max_turns,
            user_agent=user_agent,
            base_url=args.base_url,
        )
        interviewer.run()
        
        # Save user agent conversation history if available
        if user_agent:
            history_path = Path(args.log).parent / f"{Path(args.log).stem}_user_history.json"
            with history_path.open("w", encoding="utf-8") as f:
                json.dump(user_agent.get_conversation_history(), f, indent=2, ensure_ascii=False)
            print(f"\n[Logging] Saved user agent conversation history to: {history_path}")
            
    finally:
        logger.close()

if __name__ == "__main__":
    main()
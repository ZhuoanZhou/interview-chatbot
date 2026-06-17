import argparse
import json
import logging
import time
import os
import re
import ast
import hashlib
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# Constants
OPENAI_RETRIES = 3
MAX_TOKENS = 8192
OPENAI_CLIENT = None

# Global model client (will be initialized based on config)
MODEL_CLIENT = None
MODEL_CONFIG = None
TOKENIZER = None

LEADING_JUDGE_PROMPT = """You are Qwen3, an AI assistant that reasons, thinks, and answers strictly in English.

# Instruction

You will be given the full transcript of a semi-structured interview and one interviewer question to evaluate.
Evaluate the interviewer’s question-asking corresponding to the given question using a Cleanness rubric. In particular, assess whether the question (and any immediate follow-up phrasing) is grounded in the interviewee’s prior responses or introduces interviewer-generated content.

A leading question is defined as an interviewer utterance that introduces new content, embeds presuppositions, or evaluates the interviewee’s response.
Use the interviewee’s response only to assess whether the question compromises the authorship of the answer, not to judge the quality or correctness of the response itself.

# Evaluation Rubric

1: Strongly Leading - The utterance clearly introduces content, presupposition, or evaluation that suggests a specific answer, and the interviewee’s response raises reasonable doubt about whether the answer reflects their independent experience.
2: Mildly Leading - The utterance introduces some new content, presupposition, or evaluative framing that suggests a direction of response, but the interviewee’s reply does not indicate compromised authorship or compliance.
3: Contextually Clean - The utterance introduces the interview topic or follows the interviewee’s logic while minimizing superfluous content. It remains within the interviewee’s framing and does not introduce presuppositions or evaluations.
4: Clean Repeat - The question reformulates, repeats, or logically extends interviewee-generated content without introducing new content, presupposition, or evaluation.
5: Classically Clean - The utterance uses only interviewee-generated content or universal constructs and conforms to a classically clean question form, introducing no interviewer-generated structure beyond the question itself.

# Response Format
{'type': 'object', 'properties': {'explanation': {'type': 'string', 'description': 'A brief reasoning behind the final contingency responsiveness score.'}, 'score': {'type': 'string', 'description': "The verdict label from the rubric: one of '1', '2', '3', '4', or '5'.", 'enum': ['1', '2', '3', '4', '5']}}, 'required': ['explanation', 'score']}
"""

def compute_message_hash(msg: List[Dict]) -> str:
    """
    Compute a deterministic hash of a message for deduplication.
    Hashes the user content (ground_truth + all_notes).

    Args:
        msg: Message list with system and user prompts

    Returns:
        MD5 hash of the user content
    """
    # Extract user content (second element in message list)
    user_content = msg[1]['content']
    # Create deterministic hash
    return hashlib.md5(user_content.encode('utf-8')).hexdigest()

class EvaluationConfig:
    """Configuration for evaluation run"""
    def __init__(self, mode: str, base_path: str, max_workers: int = 16):
        self.mode = mode
        self.base_path = base_path
        self.max_workers = max_workers

        # Validate mode
        if mode not in ['sparkme', 'storysage', 'llmroleplay', 'freeform']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'sparkme', 'storysage', 'llmroleplay', or 'freeform'")


def initialize_model(config_path: Optional[str] = None):
    """Initialize model client based on configuration"""
    global MODEL_CLIENT, MODEL_CONFIG, TOKENIZER

    if config_path is None:
        # Default to OpenAI
        MODEL_CLIENT = OPENAI_CLIENT
        MODEL_CONFIG = {
            "provider_name": "openai",
            "model_name": "gpt-4.1-nano",
            "generation_args": {
                "temperature": 0,
                "max_tokens": MAX_TOKENS
            }
        }
        logging.info("Using default OpenAI client")
        return

    # Load config from file
    with open(config_path, 'r') as f:
        MODEL_CONFIG = json.load(f)

    provider = MODEL_CONFIG.get("provider_name", "openai")

    if provider == "openai":
        MODEL_CLIENT = OPENAI_CLIENT
        logging.info(f"Using OpenAI client with model {MODEL_CONFIG.get('model_name', 'gpt-4.1-nano')}")
    elif provider == "local":
        # Initialize vLLM client
        try:
            from vllm import LLM

            model_name = MODEL_CONFIG["model_name"]
            model_args = MODEL_CONFIG.get("model_args", {})

            logging.info(f"Initializing vLLM with model {model_name}")
            MODEL_CLIENT = LLM(model=model_name, **model_args)

            # Initialize tokenizer for chat template
            logging.info(f"Loading tokenizer for {model_name}")
            TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            logging.info("vLLM client and tokenizer initialized successfully")
        except ImportError:
            logging.error("vLLM not installed. Please install with: pip install vllm")
            raise
        except Exception as e:
            logging.error(f"Error initializing vLLM: {e}")
            raise
    else:
        raise ValueError(f"Unknown provider: {provider}")


def safe_parse_json(text: str):
    """Parse JSON from various formats including code blocks"""
    text = text.strip()
    if not text:
        return None

    # Try to find ```json ... ``` fenced block first
    codeblock_match = re.search(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if codeblock_match:
        candidate = codeblock_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Try parsing entire text as JSON directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try Python literal eval (last resort)
    try:
        parsed_dict = ast.literal_eval(text)
        if isinstance(parsed_dict, dict):
            return parsed_dict
    except Exception:
        pass

    return None

def _strip_log_prefix(line: str) -> str:
    """Strip optional timestamp/log prefix from a line.

    Handles both formats:
        2025-12-25 19:00:21,252 - INFO - Interviewer: ...
        Interviewer: ...
    """
    match = re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - \w+ - ', line)
    if match:
        return line[match.end():]
    return line


def parse_interviewer_questions(chat_log: str) -> List[str]:
    """Parse all interviewer questions from the chat log.

    Handles lines with or without a timestamp/log prefix.
    Skips empty lines.
    """
    questions = []
    lines = chat_log.split('\n')
    current_speaker = None
    current_text = []

    for line in lines:
        line = _strip_log_prefix(line)
        if line.startswith('Interviewer:'):
            # Save previous interviewer block if any
            if current_speaker == 'Interviewer' and current_text:
                questions.append('\n'.join(current_text))
            current_speaker = 'Interviewer'
            current_text = [line[len('Interviewer:'):].strip()]
        elif line.startswith('User:'):
            # Save previous interviewer block if any
            if current_speaker == 'Interviewer' and current_text:
                questions.append('\n'.join(current_text))
                current_text = []
            current_speaker = 'User'
        else:
            # Continuation line for current speaker
            if current_speaker == 'Interviewer':
                current_text.append(line)

    # Don't forget the last block
    if current_speaker == 'Interviewer' and current_text:
        questions.append('\n'.join(current_text))

    return questions


def prepare_session_data(user_id: str, config: EvaluationConfig):
    """Prepare session data and paths without evaluation.

    Returns a list of (message, question_text) tuples, one per interviewer question.
    """
    # Construct paths based on mode
    if config.mode == 'sparkme' or config.mode == 'storysage':
        chat_path = f"{config.base_path}/{user_id}/execution_logs/session_1/chat_history.log"
    elif config.mode == 'llmroleplay' or config.mode == 'freeform':
        chat_path = f"{config.base_path}/{user_id}/chat_history.log"
    else:
        raise ValueError(f"Invalid mode: {config.mode}")

    # Skip if session file doesn't exist
    if not os.path.exists(chat_path):
        return None

    with open(chat_path, 'r') as f:
        chat_log = f.read()

    # Parse each interviewer question from the chat log
    questions = parse_interviewer_questions(chat_log)
    if not questions:
        return None

    # Create one evaluation message per question
    messages = []
    for question in questions:
        msg = [
            {"role": "system", "content": LEADING_JUDGE_PROMPT},
            {"role": "user", "content": f"# Interview Transcript\n\n<interview_transcript>{chat_log}</interview_transcript>\n\n# Question to Focus On\n\n<question>{question}</question>\n\n# Your Output\n"},
        ]
        messages.append(msg)

    return messages

def main():
    parser = argparse.ArgumentParser(description='Evaluate interview sessions')
    parser.add_argument('--mode', type=str, required=True, choices=['sparkme', 'storysage', 'llmroleplay', 'freeform'],
                        help='Evaluation mode: interviewer, storysage, llmroleplay, or freeform')
    parser.add_argument('--base-path', type=str, required=True,
                        help='Base path to logs directory')
    parser.add_argument('--output-path', type=str,
                        default='output.json',
                        help='Output path')
    parser.add_argument('--model-config', type=str, default=None,
                        help='Path to model config JSON file (optional, defaults to OpenAI)')
    parser.add_argument('--max-workers', type=int, default=16,
                        help='Maximum number of parallel workers')
    parser.add_argument('--surgery', action='store_true',
                        help='Re-evaluate only error entries from an existing output file')

    args = parser.parse_args()

    # Initialize model
    initialize_model(args.model_config)

    # Create config
    config = EvaluationConfig(
        mode=args.mode,
        base_path=args.base_path,
        max_workers=args.max_workers
    )

    provider = MODEL_CONFIG.get("provider_name", "openai")

    if provider == "local":
        from vllm import SamplingParams

        # vLLM: Batch ALL evaluations at once
        logging.info("Using vLLM - collecting all sessions for batch inference")

        # Step 1: Discover user directories and prepare all session data
        user_dirs = sorted([
            d for d in os.listdir(config.base_path)
            if os.path.isdir(os.path.join(config.base_path, d))
        ])
        logging.info(f"Found {len(user_dirs)} user directories")

        all_messages = []
        all_user_ids = []  # one entry per message, tracking which user it belongs to
        for user_id in tqdm(user_dirs, desc="Preparing sessions"):
            session_data = prepare_session_data(user_id, config)
            if session_data is None:
                continue
            for msg in session_data:
                all_messages.append(msg)
                all_user_ids.append(user_id)

        # Surgery mode: load existing results, keep successes, only re-eval errors
        existing_results = None
        if args.surgery:
            if not os.path.exists(args.output_path):
                logging.error(f"--surgery requires existing output file: {args.output_path}")
                return
            with open(args.output_path, 'r') as f:
                existing_results = json.load(f)

            # Build set of (user_id, question) pairs that already succeeded
            success_keys = set()
            for r in existing_results:
                if not r.get("error", False):
                    success_keys.add((r["user_id"], r["question"]))

            # Filter to only messages that need re-evaluation
            filtered_messages = []
            filtered_user_ids = []
            for msg, uid in zip(all_messages, all_user_ids):
                user_content = msg[1]["content"]
                q_match = re.search(r"<question>(.*?)</question>", user_content, re.DOTALL)
                q_text = q_match.group(1).strip() if q_match else ""
                if (uid, q_text) not in success_keys:
                    filtered_messages.append(msg)
                    filtered_user_ids.append(uid)

            logging.info(f"Surgery mode: {len(all_messages)} total, {len(success_keys)} already succeeded, {len(filtered_messages)} to re-evaluate")
            all_messages = filtered_messages
            all_user_ids = filtered_user_ids

        if not all_messages:
            logging.info("No messages to evaluate.")
            return

        # Step 2: Apply chat templates and run batch inference
        prompts = []
        for msg in tqdm(all_messages, desc="Applying chat templates"):
            prompt = TOKENIZER.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        gen_args = MODEL_CONFIG.get("generation_args", {})
        sampling_params = SamplingParams(**gen_args)

        logging.info("Running batch inference...")
        outputs = MODEL_CLIENT.generate(prompts, sampling_params)

        # Step 3: Parse results
        logging.info("Parsing results...")

        new_results = []

        for idx, output in tqdm(enumerate(outputs), total=len(outputs), desc="Processing outputs"):
            generated_text = output.outputs[0].text.strip()

            if "</think>" in generated_text:
                generated_text = generated_text.split("</think>")[-1]

            parsed_response = safe_parse_json(generated_text)

            # Extract the question text from the original message
            user_content = all_messages[idx][1]["content"]
            question_match = re.search(r"<question>(.*?)</question>", user_content, re.DOTALL)
            question_text = question_match.group(1).strip() if question_match else ""

            if parsed_response:
                result = {
                    "user_id": all_user_ids[idx],
                    "question": question_text,
                    "score": parsed_response["score"],
                    "explanation": parsed_response["explanation"],
                    "error": False,
                }
            else:
                result = {
                    "user_id": all_user_ids[idx],
                    "question": question_text,
                    "score": 1,
                    "explanation": generated_text,
                    "error": True,
                }

            new_results.append(result)

        # Step 4: Merge results if surgery mode
        if args.surgery and existing_results is not None:
            # Build lookup from new results
            new_results_map = {}
            for r in new_results:
                new_results_map[(r["user_id"], r["question"])] = r

            # Replace error entries in existing results with new results
            merged = []
            for r in existing_results:
                key = (r["user_id"], r["question"])
                if r.get("error", False) and key in new_results_map:
                    merged.append(new_results_map[key])
                else:
                    merged.append(r)
            all_results = merged

            error_count = sum(1 for r in all_results if r.get("error", False))
            logging.info(f"Surgery complete: {error_count} errors remaining")
        else:
            all_results = new_results

        # Step 5: Write results
        logging.info(f"Writing results...")
        with open(args.output_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        logging.info("vLLM batch evaluation complete!")
    else:
        raise NotImplementedError("Not implemented yet other than vLLM implementation")


if __name__ == "__main__":
    main()

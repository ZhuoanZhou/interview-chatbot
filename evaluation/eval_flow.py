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

COHERENCE_JUDGE_PROMPT = """You are Qwen3, an AI assistant that reasons, thinks, and answers strictly in English.

# Instruction
You will be given the full transcript of a semi-structured interview.

Evaluate the interviewer’s question-asking behavior across the entire interview in terms of local coherence between consecutive questions, i.e. whether each interviewer question is logically connected to the immediately preceding conversational context.
Do NOT score individual turns. The score should reflect a global, holistic judgment over the full interview.

# Evaluation Rubric

1: Consecutive questions are frequently unrelated, disruptive, or non-sequitur, breaking sequential coherence throughout the interview.
2: Consecutive questions are often weakly connected or abruptly shift focus, frequently disrupting local coherence.
3: Logical connections are inconsistent; some questions follow naturally while others introduce noticeable but tolerable shifts.
4: Most consecutive questions are logically connected, with minor or infrequent lapses that do not disrupt overall coherence.
5: Consecutive questions are consistently well-threaded and clearly motivated by the immediately prior context, with no disruptive jumps.

# Response Format
{'type': 'object', 'properties': {'explanation': {'type': 'string', 'description': 'A brief reasoning behind the final coherence score.'}, 'score': {'type': 'string', 'description': "The verdict label from the rubric: one of '1', '2', '3', '4', or '5'.", 'enum': ['1', '2', '3', '4', '5']}}, 'required': ['explanation', 'score']}
"""

TRANSITION_JUDGE_PROMPT = """You are Qwen3, an AI assistant that reasons, thinks, and answers strictly in English.

# Instruction
You will be given the full transcript of a semi-structured interview.

Evaluate the interviewer’s question-asking behavior across the entire interview in terms of transition quality across topics, i.e. how smoothly the interviewer transitions between topics, subtopics, or sections.
Do NOT score individual turns. The score should reflect a global, holistic judgment over the full interview.

# Evaluation Rubric

1: Topic shifts are consistently abrupt and confusing, severely disrupting the interview's structure.
2: Topic transitions are frequently abrupt, un-signposted, or disruptive to conversational flow.
3: Transitions are mixed; some are smooth while others are abrupt or weakly motivated.
4: Most transitions are smooth and intelligible, though some lack explicit signposting.
5: Topic transitions are consistently smooth, well-signposted, or clearly motivated, preserving conversational flow and intelligibility.

# Response Format
{'type': 'object', 'properties': {'explanation': {'type': 'string', 'description': 'A brief reasoning behind the final transition quality score.'}, 'score': {'type': 'string', 'description': "The verdict label from the rubric: one of '1', '2', '3', '4', or '5'.", 'enum': ['1', '2', '3', '4', '5']}}, 'required': ['explanation', 'score']}
"""

CONTINGENCY_JUDGE_PROMPT = """You are Qwen3, an AI assistant that reasons, thinks, and answers strictly in English.

# Instruction
You will be given the full transcript of a semi-structured interview.

Evaluate the interviewer’s question-asking behavior across the entire interview in terms of contingency responsiveness of follow-up questions, i.e. whether follow-up questions are grounded in the User's prior responses.
Do NOT score individual turns. The score should reflect a global, holistic judgment over the full interview.

# Evaluation Rubric

1: Follow-ups consistently fail to engage with prior responses and feel largely non-contingent (or no follow-ups at all). 
2: Follow-ups frequently ignore or misinterpret prior responses, introducing unwarranted assumptions.
3: Follow-ups sometimes reflect prior responses, but often feel generic or only loosely grounded.
4: Follow-ups usually reflect prior responses, with occasional partial mismatches or missed cues.
5: Follow-ups consistently demonstrate clear uptake, accurately building on what the interviewee has said without introducing unwarranted assumptions.

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
    def __init__(self, mode: str, base_path: str, sample_users_path: str,
                 ground_truth_path: str, max_workers: int = 16):
        self.mode = mode
        self.base_path = base_path
        self.sample_users_path = sample_users_path
        self.ground_truth_path = ground_truth_path
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

def prepare_session_data(user_id: str, config: EvaluationConfig):
    """Prepare session data and paths without evaluation"""
    # Construct paths based on mode
    if config.mode == 'sparkme' or config.mode == 'storysage':
        chat_path = f"{config.base_path}/{user_id}/execution_logs/session_1/chat_history.log"
    elif config.mode == 'llmroleplay' or config.mode == 'freeform':
        chat_path = f"{config.base_path}/{user_id}/interview_log.jsonl"
    else:
        raise ValueError(f"Invalid mode: {config.mode}")

    # Skip if session file doesn't exist
    if not os.path.exists(chat_path):
        return None

    # Load session data
    if config.mode == 'llmroleplay':
        chat_log = []
        with open(chat_path, 'r') as f:
            for obj in f.readlines():
                loaded_obj = json.loads(obj)
                
                if loaded_obj.get("user_message", ""):
                    chat_log.append(f"User: {loaded_obj.get('user_message')}")
                if loaded_obj.get("assistant_message", ""):
                    chat_log.append(f"Interviewer: {loaded_obj.get('assistant_message')}")

        chat_log = "\n".join(chat_log)
    elif config.mode == 'freeform':
        chat_log = []
        turn = 0
        with open(chat_path, 'r') as f:
            for obj in f.readlines():
                loaded_obj = json.loads(obj)
                
                if loaded_obj.get("user_message", ""):
                    chat_log.append(f"User: {loaded_obj.get('user_message')}")
                    turn += 1
                if loaded_obj.get("assistant_message", ""):
                    chat_log.append(f"Interviewer: {loaded_obj.get('assistant_message')}")

                if turn == 10:
                    break

        chat_log = "\n".join(chat_log)  
    else:
        # For interviewer and storysage modes, load JSON
        with open(chat_path, 'r') as f:
            chat_log = f.read()

    # Create evaluation messages
    messages = [
        [{"role": "system", "content": COHERENCE_JUDGE_PROMPT},
           {"role": "user", "content": f"# Interview Transcript\n\n<interview_transcript>{chat_log}</interview_transcript>\n\n# Your Output\n"}],
        [{"role": "system", "content": TRANSITION_JUDGE_PROMPT},
           {"role": "user", "content": f"# Interview Transcript\n\n<interview_transcript>{chat_log}</interview_transcript>\n\n# Your Output\n"}],
        [{"role": "system", "content": CONTINGENCY_JUDGE_PROMPT},
           {"role": "user", "content": f"# Interview Transcript\n\n<interview_transcript>{chat_log}</interview_transcript>\n\n# Your Output\n"}],
    ]

    return messages

def main():
    parser = argparse.ArgumentParser(description='Evaluate interview sessions')
    parser.add_argument('--mode', type=str, required=True, choices=['sparkme', 'storysage', 'llmroleplay', 'freeform'],
                        help='Evaluation mode: interviewer, storysage, llmroleplay, or freeform')
    parser.add_argument('--base-path', type=str, required=True,
                        help='Base path to logs directory')
    parser.add_argument('--sample-users-path', type=str,
                        default='analysis/sample_users_50.json',
                        help='Path to sample users JSON')
    parser.add_argument('--ground-truth-path', type=str,
                        default='data/sample_user_profiles',
                        help='Path to ground truth data directory')
    parser.add_argument('--output-path', type=str,
                        default='output.json',
                        help='Output path')
    parser.add_argument('--model-config', type=str, default=None,
                        help='Path to model config JSON file (optional, defaults to OpenAI)')
    parser.add_argument('--max-workers', type=int, default=16,
                        help='Maximum number of parallel workers')
    parser.add_argument('--num-users', type=int, default=200,
                        help='Number of users to process')

    args = parser.parse_args()

    # Initialize model
    initialize_model(args.model_config)

    # Create config
    config = EvaluationConfig(
        mode=args.mode,
        base_path=args.base_path,
        sample_users_path=args.sample_users_path,
        ground_truth_path=args.ground_truth_path,
        max_workers=args.max_workers
    )

    # Load sample users
    with open(config.sample_users_path, 'r') as f:
        sample_users = json.load(f)

    provider = MODEL_CONFIG.get("provider_name", "openai")

    if provider == "local":
        # vLLM: Batch ALL evaluations at once
        logging.info("Using vLLM - collecting all sessions for batch inference")

        # Step 1: Prepare all session data
        all_messages = []
        all_user_ids = []
        for person_profile in tqdm(sample_users[:args.num_users], desc="Preparing sessions"):
            user_id = person_profile["User ID"]
            all_messages.extend(prepare_session_data(user_id, config))
            all_user_ids.append(user_id)

        # Step 3: Deduplicate messages before batch inference
        from vllm import SamplingParams

        # Apply chat template only to unique messages
        prompts = []
        for msg in tqdm(all_messages, desc="Applying chat templates"):
            prompt = TOKENIZER.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        gen_args = MODEL_CONFIG.get("generation_args", {})
        sampling_params = SamplingParams(**gen_args)

        logging.info("Running batch inference...")
        outputs = MODEL_CLIENT.generate(prompts, sampling_params)

        # Step 4: Parse all results and group by session
        logging.info("Parsing results and mapping back to all duplicates...")

        # Create result list for ALL original messages (including duplicates)
        all_results = []

        # Process unique outputs and map to all duplicate indices
        for unique_idx, output in tqdm(enumerate(outputs), total=len(outputs), desc="Processing outputs"):
            generated_text = output.outputs[0].text.strip()

            if "</think>" in generated_text:
                generated_text = generated_text.split("</think>")[-1]

            parsed_response = safe_parse_json(generated_text)
            
            metric = None
            if unique_idx % 3 == 0:
                metric = "coherence"
            elif unique_idx % 3 == 1:
                metric = "transition"
            elif unique_idx % 3 == 2:
                metric = "contingency"

            if parsed_response:
                result = {
                    "user_id": all_user_ids[unique_idx // 3],
                    "metric": metric,
                    "score": parsed_response["score"],
                    "explanation": parsed_response["explanation"],
                    "error": False,
                }
            else:
                result = {
                    "user_id": all_user_ids[unique_idx // 3],
                    "metric": metric,
                    "score": 1,
                    "explanation": generated_text,
                    "error": True,
                }
            
            all_results.append(result)

        # Step 5: Write all results to files
        logging.info(f"Writing results...")
        with open(args.output_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        logging.info("vLLM batch evaluation complete!")
    else:
        raise NotImplementedError("Not implemented yet other than vLLM implementation")


if __name__ == "__main__":
    main()

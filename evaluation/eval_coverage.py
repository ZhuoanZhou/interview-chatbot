import argparse
import json
import logging
import time
import os
import re
import ast
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

JUDGE_PROMPT = """# Instruction
Your task is to evaluate recall accuracy in interview notes. Check whether the **ground truth facts** appear **explicitly** in the interview notes.

Rules:
1. Facts must be stated explicitly (no inference).
2. Components of a fact may be spread across the notes.
3. Extra information does not affect the score.

# Evaluation Rubric

- **5 (Perfect):** All ground truth facts are explicitly found in the interview note.
- **4 (Minor Omission):** One minor fact or sub-bullet is missing in the interview note.
- **3 (Partial):** About half of the facts are found in the interview notes.
- **2 (Vague Overlap):** General topic mentioned, specifics missing in the interview notes.
- **1 (No Recall):** Ground truth facts are absent in the interview notes.

# Output Format (JSON)
{{
    "score": 1-5
}}
"""

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


def request_openai_completion(msg: List[Dict]) -> str:
    """Request completion from OpenAI with retries"""
    global MODEL_CLIENT, MODEL_CONFIG

    model_name = MODEL_CONFIG.get("model_name", "gpt-4.1-nano")
    gen_args = MODEL_CONFIG.get("generation_args", {})

    for attempt in range(OPENAI_RETRIES):
        try:
            response = MODEL_CLIENT.chat.completions.create(
                model=model_name,
                messages=msg,
                **gen_args
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.warning(f"Error calling OpenAI API (attempt {attempt + 1}/{OPENAI_RETRIES}): {e}")
            time.sleep(61)

    logging.error(f"Could not resolve error after {OPENAI_RETRIES} attempts")
    return None


def build_ground_truth_index(ground_truth: List[Dict]) -> Dict[str, Dict]:
    """Build an index of ground truth subtopics by ID"""
    gt_index = {}

    for topic_idx in range(len(ground_truth)):
        for subtopic in ground_truth[topic_idx].get('subtopics', []):
            subtopic_id = subtopic['subtopic_id']
            subtopic_description = subtopic['subtopic_description']
            subtopic_notes = "\n -".join(subtopic['notes'])

            gt_index[subtopic_id] = {
                'description': subtopic_description,
                'notes': subtopic_notes,
                'formatted': f"#### Subtopic Description: {subtopic_description}\n\n#### Notes\n{subtopic_notes}"
            }

    return gt_index


def extract_all_notes_interviewer(session_agenda: Dict) -> str:
    """Extract ALL notes from interviewer session agenda format"""
    all_notes_parts = []

    for core_topic_id, core_topic in session_agenda["interview_topic_manager"]["core_topic_dict"].items():
        topic_desc = core_topic.get("description", "")

        # Process required subtopics
        for subto_id, subto in core_topic.get('required_subtopics', {}).items():
            subto_desc = subto.get("description", "")
            notes = list(set(subto.get("notes", [])))
            final_summary = subto.get("final_summary", "")
            final_note = ""
                
            if final_summary:
                final_note += f"**Topic: {topic_desc}**\n**Subtopic: {subto_desc}**\nSummary: {final_summary}"
            elif notes:
                notes_str = "\n -".join(notes)
                final_note += f"**Topic: {topic_desc}**\n**Subtopic: {subto_desc}**\n{notes_str}"
                
            if len(final_note) > 1:
                all_notes_parts.append(final_note)

        # Process emergent subtopics
        for subto_id, subto in core_topic.get('emergent_subtopics', {}).items():
            subto_desc = subto.get("description", "")
            notes = list(set(subto.get("notes", [])))
            final_summary = subto.get("final_summary", "")
            final_note = ""

            if final_summary:
                final_note += f"**Topic: {topic_desc}**\n**Subtopic: {subto_desc}**\nSummary: {final_summary}"
            elif notes:
                notes_str = "\n -".join(notes)
                final_note += f"**Topic: {topic_desc}**\n**Subtopic: {subto_desc}**\n{notes_str}"
                
            if len(final_note) > 1:
                all_notes_parts.append(final_note)

    return "\n\n".join(all_notes_parts)


def extract_notes_for_subtopic_interviewer(
    session_agenda: Dict,
    target_subtopic_id: str
) -> str:
    """Extract notes ONLY for a given subtopic ID (interviewer format)."""
    notes_parts = []

    for core_topic_id, core_topic in session_agenda["interview_topic_manager"]["core_topic_dict"].items():
        if str(core_topic_id) != target_subtopic_id.split(".")[0]:
            continue

        topic_desc = core_topic.get("description", "")
        
        # Required subtopics
        for subto_id, subto in core_topic.get("required_subtopics", {}).items():
            if subto_id != target_subtopic_id:
                continue
            
            subto_desc = subto.get("description", "")
            notes = list(set(subto.get("notes", [])))
            final_summary = subto.get("final_summary", "")

            if final_summary:
                notes_parts.append(
                    f"**Topic: {topic_desc}**\n"
                    f"**Subtopic: {subto_desc}**\n"
                    f"Summary: {final_summary}"
                )
            elif notes:
                notes_str = "\n - ".join(notes)
                notes_parts.append(
                    f"**Topic: {topic_desc}**\n"
                    f"**Subtopic: {subto_desc}**\n"
                    f"- {notes_str}"
                )

        # Emergent subtopics
        for subto_id, subto in core_topic.get("emergent_subtopics", {}).items():
            subto_desc = subto.get("description", "")
            notes = list(set(subto.get("notes", [])))
            final_summary = subto.get("final_summary", "")

            if final_summary:
                notes_parts.append(
                    f"**Topic: {topic_desc}**\n"
                    f"**Subtopic: {subto_desc}**\n"
                    f"Summary: {final_summary}"
                )
            elif notes:
                notes_str = "\n - ".join(notes)
                notes_parts.append(
                    f"**Topic: {topic_desc}**\n"
                    f"**Subtopic: {subto_desc}**\n"
                    f"- {notes_str}"
                )

    return "\n\n".join(notes_parts)


def extract_all_notes_storysage(memory_bank: dict, max_turn: int) -> str:
    """Extract ALL notes from storysage format"""
    all_notes_parts = []

    for mem in memory_bank.get("memories", []):
        turn = mem.get("metadata", {}).get("turn", -1)
        if turn > max_turn:
            continue

        notes = mem.get("text", "").strip()

        if notes:
            all_notes_parts.append(f"- {notes}")

    return "\n".join(all_notes_parts)


def get_covered_subtopics_interviewer(session_agenda: Dict, gt_index: Dict) -> List[str]:
    """Get list of subtopic IDs that have any notes in interviewer format"""
    covered_subtopics = []

    for core_topic_id, core_topic in session_agenda["interview_topic_manager"]["core_topic_dict"].items():
        for subto_id, subto in core_topic.get('required_subtopics', {}).items():
            if subto_id in gt_index and (len(subto.get("notes", [])) > 0 or len(subto.get("final_summary", "")) > 0):
                covered_subtopics.append(subto_id)

    return covered_subtopics


def get_covered_subtopics_storysage(topics_filled: List[Dict], gt_index: Dict) -> List[str]:
    """Get list of subtopic IDs that have any notes in storysage format"""
    covered_subtopics = []

    for topic in topics_filled:
        for subtopic in topic.get("subtopics", []):
            subto_id = subtopic.get('subtopic_id')
            if subto_id in gt_index and len(subtopic.get("notes", [])) > 0:
                covered_subtopics.append(subto_id)

    return covered_subtopics


def extract_all_notes_llmroleplay(history: List[Dict], max_turn: int) -> str:
    """Extract ALL notes from llmroleplay baseline format (interview_log.jsonl)

    Notes: Only keeps the LATEST version of notes for each subtopic,
    since notes get aggregated and updated over multiple turns.
    """
    # Dictionary to store latest notes for each subtopic
    subtopic_notes = {}

    for i, item in enumerate(history):
        if i > max_turn:
            break

        notes = item.get("notes", None)
        if notes is not None and len(notes.strip()) > 0:
            topic_idx = item.get("topic_index", 0)
            question_idx = item.get("question_index", 0)
            subtopic_id = f"{topic_idx + 1}.{question_idx + 1}"
            # Update with latest notes (overwrites previous)
            subtopic_notes[subtopic_id] = notes

    # Format all collected notes
    all_notes_parts = [f"**Subtopic {sid}**\n{notes}" for sid, notes in subtopic_notes.items()]
    return "\n\n".join(all_notes_parts)


def get_covered_subtopics_llmroleplay(history: List[Dict], max_turn: int, gt_index: Dict) -> List[str]:
    """Get list of subtopic IDs that have any notes in llmroleplay format

    Notes: Only considers subtopics that have notes in the final aggregated state,
    since notes get updated over multiple turns.
    """
    # Dictionary to track latest notes for each subtopic
    subtopic_has_notes = {}

    for i, item in enumerate(history):
        if i > max_turn:
            break

        notes = item.get("notes", None)
        topic_idx = item.get("topic_index", 0)
        question_idx = item.get("question_index", 0)
        subtopic_id = f"{topic_idx + 1}.{question_idx + 1}"

        # Update latest state for this subtopic
        if notes is not None and len(notes.strip()) > 0:
            subtopic_has_notes[subtopic_id] = True
        elif subtopic_id in subtopic_has_notes:
            # If notes were cleared/emptied, update that too
            subtopic_has_notes[subtopic_id] = False

    # Return only subtopics that have notes in final state and exist in ground truth
    covered_subtopics = [sid for sid, has_notes in subtopic_has_notes.items()
                         if has_notes and sid in gt_index]

    return covered_subtopics

def extract_all_notes_freeform(history: List[Dict], max_turn: int) -> str:
    """Extract ALL notes from freeform baseline format (interview_log.jsonl)

    Notes: Only keeps the LATEST version of notes for each subtopic,
    since notes get aggregated and updated over multiple turns.
    """
    # Dictionary to store latest notes for each subtopic
    subtopic_notes = []

    for i, item in enumerate(history):
        if i > max_turn:
            break

        notes = item.get("notes", None)
        if notes is not None and len(notes.strip()) > 0:
            subtopic_notes.append(notes)

    # Format all collected notes
    return "\n- ".join(subtopic_notes)

def create_evaluation_messages(ground_truth: List[Dict], session_data: Dict, mode: str, snapshot_idx: int = None) -> Tuple[List[List[Dict]], List[str]]:
    """
    Create evaluation messages comparing ALL notes against each ground truth subtopic

    Args:
        ground_truth: Ground truth data
        session_data: Session agenda (interviewer), memory bank (storysage), or history list (llmroleplay/freeform)
        mode: 'sparkme', 'storysage', 'llmroleplay', or 'freeform'
        snapshot_idx: For llmroleplay/freeform mode, the turn number to evaluate up to

    Returns:
        Tuple of (list of messages, list of subtopic IDs)
    """
    gt_index = build_ground_truth_index(ground_truth)

    # Extract ALL notes from the session
    if mode == 'sparkme':
        all_notes = extract_all_notes_interviewer(session_data)
        covered_subtopics = list(gt_index.keys())
    elif mode == 'storysage':
        all_notes = extract_all_notes_storysage(session_data, snapshot_idx)
        covered_subtopics = list(gt_index.keys())
    elif mode == 'llmroleplay':
        all_notes = extract_all_notes_llmroleplay(session_data, snapshot_idx)
        covered_subtopics = list(gt_index.keys())
    elif mode == "freeform":
        all_notes = extract_all_notes_freeform(session_data, snapshot_idx)
        covered_subtopics = list(gt_index.keys())
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Create evaluation messages for each covered subtopic
    messages = []
    subtopic_ids = []

    for subtopic_id in covered_subtopics:
        if subtopic_id not in gt_index:
            continue

        gt_info = gt_index[subtopic_id]
        
        msg = [{
            "role": "system",
            "content": JUDGE_PROMPT
        },{
            "role": "user",
            "content": "# Input\n\n ### Ground Truth Facts\n\n{ground_truth}\n\n### Interview Notes\n\n{all_notes}\n\n### Your Output\n".format(
                ground_truth=gt_info['formatted'],
                all_notes=all_notes
            )
        }]

        messages.append(msg)
        subtopic_ids.append(subtopic_id)

    return messages, subtopic_ids


def evaluate_subtopic_openai(msg: List[Dict], subtopic_id: str) -> Dict:
    """Evaluate a single subtopic with OpenAI"""
    response = request_openai_completion(msg)

    if response is None:
        return {
            'subtopic_id': subtopic_id,
            'score': None,
            'rationale': 'API request failed',
            'error': True
        }

    parsed_response = safe_parse_json(response)

    if parsed_response is None:
        return {
            'subtopic_id': subtopic_id,
            'score': None,
            'rationale': 'Failed to parse response',
            'error': True
        }

    return {
        'subtopic_id': subtopic_id,
        'score': parsed_response.get('score'),
        'rationale': parsed_response.get('rationale', ''),
        'error': False
    }


def prepare_session_data(user_id: str, snapshot_idx: int, ground_truth: List[Dict], config: EvaluationConfig, surgery: bool = False):
    """Prepare session data and paths without evaluation"""
    # Construct paths based on mode
    if config.mode == 'sparkme':
        session_path = f"{config.base_path}/{user_id}/execution_logs/session_0/session_agenda_snap_{snapshot_idx}.json"
        eval_dir = f"{config.base_path}/{user_id}/evaluations_all_to_subtopics"
    elif config.mode == 'storysage':
        session_path = f"{config.base_path}/{user_id}/memory_bank_content.json"
        eval_dir = f"{config.base_path}/{user_id}/evaluations_all_to_subtopics"
    elif config.mode == 'llmroleplay':
        session_path = f"{config.base_path}/{user_id}/interview_log.jsonl"
        eval_dir = f"{config.base_path}/{user_id}/evaluations_all_to_subtopics"
    elif config.mode == 'freeform':
        session_path = f"{config.base_path}/{user_id}/interview_log.jsonl"
        eval_dir = f"{config.base_path}/{user_id}/evaluations_all_to_subtopics"  
    else:
        raise ValueError(f"Invalid mode: {config.mode}")

    save_eval_path = f"{eval_dir}/snap_eval_{snapshot_idx}.json"

    # Skip if session file doesn't exist
    if not os.path.exists(session_path):
        return None

    # Skip if already evaluated (and not empty)
    if not surgery and os.path.exists(save_eval_path):
        with open(save_eval_path, 'r') as f:
            existing_results = json.load(f)
            if len(existing_results) > 0:
                return None

    # Load session data
    if config.mode == 'llmroleplay' or config.mode == 'freeform':
        # For llmroleplay mode, load the JSONL file as a list of history items
        history = []
        with open(session_path) as f:
            for line in f.readlines():
                history.append(json.loads(line))

        # Check if snapshot_idx is beyond the history length
        if snapshot_idx >= len(history) + 4:
            return None

        session_data = history        
    else:
        # For interviewer and storysage modes, load JSON
        with open(session_path) as f:
            session_data = json.load(f)

    # Create evaluation messages
    messages, subtopic_ids = create_evaluation_messages(ground_truth, session_data, config.mode, snapshot_idx)

    if len(messages) == 0:
        return None

    return {
        'user_id': user_id,
        'snapshot_idx': snapshot_idx,
        'messages': messages,
        'subtopic_ids': subtopic_ids,
        'save_path': save_eval_path,
        'eval_dir': eval_dir
    }


def process_session_openai(user_id: str, snapshot_idx: int, ground_truth: List[Dict], config: EvaluationConfig, surgery: bool = False):
    """Process a single session snapshot with OpenAI (for threading)"""
    session_info = prepare_session_data(user_id, snapshot_idx, ground_truth, config, surgery=surgery)

    if session_info is None:
        return

    # Evaluate each subtopic individually
    results = []
    for msg, subtopic_id in zip(session_info['messages'], session_info['subtopic_ids']):
        result = evaluate_subtopic_openai(msg, subtopic_id)
        results.append(result)

    # Save results
    os.makedirs(session_info['eval_dir'], exist_ok=True)
    with open(session_info['save_path'], 'w') as f:
        json.dump(results, f, indent=2)


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
    parser.add_argument('--model-config', type=str, default=None,
                        help='Path to model config JSON file (optional, defaults to OpenAI)')
    parser.add_argument('--max-workers', type=int, default=16,
                        help='Maximum number of parallel workers')
    parser.add_argument('--num-users', type=int, default=30,
                        help='Number of users to process')
    parser.add_argument('--snapshot-start', type=int, default=1,
                        help='Starting snapshot index')
    parser.add_argument('--snapshot-end', type=int, default=80,
                        help='Ending snapshot index')
    parser.add_argument('--snapshot-step', type=int, default=1,
                        help='Snapshot step size')
    parser.add_argument('--surgery', action="store_true", help="Enable surgery")

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
        all_session_infos = []
        for person_profile in tqdm(sample_users[:args.num_users], desc="Preparing sessions"):
            user_id = person_profile["User ID"]

            # Load ground truth
            gt_path = f"{config.ground_truth_path}/{user_id}/{user_id}_topics_filled.json"
            with open(gt_path) as f:
                ground_truth = json.load(f)

            # Prepare all snapshots for this user
            for snapshot_idx in range(args.snapshot_start, args.snapshot_end, args.snapshot_step):
                session_info = prepare_session_data(user_id, snapshot_idx, ground_truth, config, surgery=args.surgery)
                if session_info is not None:
                    all_session_infos.append(session_info)

        logging.info(f"Prepared {len(all_session_infos)} sessions for evaluation")

        # Step 2: Collect ALL messages and metadata
        all_messages = []
        all_metadata = []  # Track which session each message belongs to

        for session_info in all_session_infos:
            for msg, subtopic_id in zip(session_info['messages'], session_info['subtopic_ids']):
                all_messages.append(msg)
                all_metadata.append({
                    'session_info': session_info,
                    'subtopic_id': subtopic_id
                })

        logging.info(f"Total prompts to evaluate: {len(all_messages)}")

        # Step 3: Batch inference ALL at once
        from vllm import SamplingParams

        prompts = []
        for msg in tqdm(all_messages, desc="Applying chat templates"):
            prompt = TOKENIZER.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        gen_args = MODEL_CONFIG.get("generation_args", {})
        sampling_params = SamplingParams(**gen_args)

        logging.info("Running batch inference...")
        outputs = MODEL_CLIENT.generate(prompts, sampling_params)

        # Step 4: Parse all results and group by session
        logging.info("Parsing results and grouping by session...")
        session_results = {}  # {(user_id, snapshot_idx): [results]}

        for metadata, output in tqdm(zip(all_metadata, outputs), total=len(outputs), desc="Processing outputs"):
            session_info = metadata['session_info']
            subtopic_id = metadata['subtopic_id']
            session_key = (session_info['user_id'], session_info['snapshot_idx'])

            generated_text = output.outputs[0].text.strip()
            parsed_response = safe_parse_json(generated_text)

            if parsed_response is None:
                result = {
                    'subtopic_id': subtopic_id,
                    'score': generated_text,
                    'rationale': 'Failed to parse response',
                    'error': True
                }
            else:
                try:
                    result = {
                        'subtopic_id': subtopic_id,
                        'score': parsed_response.get('score'),
                        'rationale': parsed_response.get('rationale', ''),
                        'error': False
                    }
                except:
                    result = {
                        'subtopic_id': subtopic_id,
                        'score': generated_text,
                        'rationale': 'Failed to parse response',
                        'error': True
                    }

            if session_key not in session_results:
                session_results[session_key] = {
                    'results': [],
                    'save_path': session_info['save_path'],
                    'eval_dir': session_info['eval_dir']
                }
            session_results[session_key]['results'].append(result)

        # Step 5: Write all results to files
        logging.info(f"Writing results to {len(session_results)} files...")
        for session_key, session_data in tqdm(session_results.items(), desc="Writing results"):
            os.makedirs(session_data['eval_dir'], exist_ok=True)
            with open(session_data['save_path'], 'w') as f:
                json.dump(session_data['results'], f, indent=2)

        logging.info("vLLM batch evaluation complete!")

    else:
        # OpenAI: Use threading for parallel API calls
        logging.info("Using OpenAI - processing sessions in parallel with threading")

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = []

            for person_profile in sample_users[:args.num_users]:
                user_id = person_profile["User ID"]

                # Load ground truth
                gt_path = f"{config.ground_truth_path}/{user_id}/{user_id}_topics_filled.json"
                with open(gt_path) as f:
                    ground_truth = json.load(f)

                # Submit evaluation tasks
                for snapshot_idx in range(args.snapshot_start, args.snapshot_end, args.snapshot_step):
                    futures.append(
                        executor.submit(process_session_openai, user_id, snapshot_idx, ground_truth, config, surgery=args.surgery)
                    )

            # Wait for all tasks to complete
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating sessions ({config.mode} mode)"):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error processing session: {e}")


if __name__ == "__main__":
    main()

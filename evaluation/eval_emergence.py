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

JUDGE_PROMPT = """"You are a Research Auditor identifying **Emergent Subtopics** in LLM-led interviews.

An emergent subtopic is defined as a **NEW SUBTOPIC** that should be added to the interview agenda.

Emergence is **rare**. Most interviews produce **no new subtopics**.

### Definition of Emergent Subtopic

A candidate subtopic qualifies as an emergent subtopic ONLY if it satisfies ALL of the following:

1. It clearly falls **within an existing interview topic**
2. It **does NOT belong to ANY existing subtopic** under that topic or any other topic
   - If it can reasonably be addressed (even loosely) within an existing subtopic, it is NOT emergent
3. It enables a **qualitatively new line of inquiry**, not just deeper questioning of an existing subtopic
4. It reveals at least ONE of the following:
   (a) A **new dimension, pattern, or tradeoff** not previously captured  
   (b) A **cross-cutting constraint or mental model** that reframes multiple subtopics  
   (c) A **latent strategy, failure mode, or decision criterion** that would change how the interview is conducted

Fluent elaborations, clarifications, examples, or refinements of existing subtopics are **NOT emergent**.

### Ground Truth Facts
<ground_truth_facts>
{ground_truth}
</ground_truth_facts>

### Interview Notes
<interview_notes>
{all_notes}
</interview_notes>

### Decision Rule (MANDATORY)

First, determine whether ANY candidate emergent subtopic exists.

If ALL interview content can reasonably be placed under existing subtopics
(even if they could be refined or expanded), then:
- You MUST return an empty list []
- You MUST NOT propose reframed, abstracted, or renamed versions of existing subtopics

Returning [] is considered a successful and correct outcome.

Disallowed as Emergent:
- Reframing existing behaviors as "strategies", "mental models", or "dimensions"
- Renaming known concepts (e.g., training, learning, skill maintenance)
- Abstracting routine practices into higher-level labels without new constraints

Emergent subtopics MUST require at least one NEW interview question
that could not reasonably be asked under any existing subtopic.

If no such question exists, the subtopic is NOT emergent.

### Output Format (STRICT JSON ONLY)

Return a LIST of ALL emergent subtopics in the following format:
[
    {{
        "rationale": "Why the emergent subtopic is MOST RELEVANT to the topic given, why the subtopic cannot be placed under any existing subtopic, and what qualitatively new inquiry it enables especially since emergence is rare",
        "emergent_subtopic": "Name or concise description of the emergent subtopic",
        "topic": "Parent interview topic this subtopic belongs to"
    }},
    {{
        "rationale": "Why the emergent subtopic is MOST RELEVANT to the topic given, why the subtopic cannot be placed under any existing subtopic, and what qualitatively new inquiry it enables especially since emergence is rare",
        "emergent_subtopic": "Name or concise description of the emergent subtopic",
        "topic": "Parent interview topic this subtopic belongs to"
    }},
    ...
]

If no emergent subtopic, return:

[]

### Your Response
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
            subtopic_notes = "\n- ".join(subtopic['notes'])

            topic_desc = ""
            if str(subtopic_id.split(".")[-1]) == "1":
                topic_desc = f"#### Topic: {ground_truth[topic_idx]['topic']}\n\n" 

            gt_index[subtopic_id] = {
                'description': subtopic_description,
                'notes': subtopic_notes,
                'formatted': f"{topic_desc}##### Subtopic Description: {subtopic_description}"
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
                final_note += f"#### Topic: {topic_desc}\n##### Subtopic: {subto_desc}\nSummary: {final_summary}"
            elif notes:
                notes_str = "\n -".join(notes)
                final_note += f"#### Topic: {topic_desc}\n##### Subtopic: {subto_desc}\n{notes_str}"
                
            if len(final_note) > 1:
                all_notes_parts.append(final_note)

        # Process emergent subtopics
        for subto_id, subto in core_topic.get('emergent_subtopics', {}).items():
            subto_desc = subto.get("description", "")
            notes = list(set(subto.get("notes", [])))
            final_summary = subto.get("final_summary", "")
            final_note = ""

            if final_summary:
                final_note += f"#### Topic: {topic_desc}\n##### Emergent Subtopic: {subto_desc}\nSummary: {final_summary}"
            elif notes:
                notes_str = "\n -".join(notes)
                final_note += f"#### Topic: {topic_desc}\n##### Emergent Subtopic: {subto_desc}\n{notes_str}"
                
            if len(final_note) > 1:
                all_notes_parts.append(final_note)

    return "\n\n".join(all_notes_parts)


def extract_all_notes_storysage(memory_bank: dict, snapshot_idx: int) -> str:
    """Extract notes from storysage format for a specific turn only"""
    all_notes_parts = []

    for mem in memory_bank.get("memories", []):
        turn = mem.get("metadata", {}).get("turn", -1)
        # Only get notes from this specific turn
        if turn == snapshot_idx:
            notes = mem.get("text", "").strip()
            if notes:
                all_notes_parts.append(f"- {notes}")
    
    all_notes_parts = list(set(all_notes_parts))

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


def extract_all_notes_llmroleplay(history: List[Dict], snapshot_idx: int) -> str:
    """Extract notes from llmroleplay baseline format for a specific turn only (interview_log.jsonl)

    Only returns the notes from the exact turn specified by snapshot_idx.
    """
    if snapshot_idx >= len(history):
        return ""

    item = history[snapshot_idx]
    notes = item.get("notes", None)

    if notes is not None and len(notes.strip()) > 0:
        topic_idx = item.get("topic_index", 0)
        question_idx = item.get("question_index", 0)
        subtopic_id = f"{topic_idx + 1}.{question_idx + 1}"
        return f"**Subtopic {subtopic_id}**\n{notes}"

    return ""


def get_covered_subtopics_llmroleplay(history: List[Dict], max_turn: int, gt_index: Dict) -> List[str]:
    """Get list of subtopic IDs that have any notes in llmroleplay format

    Notes: Only considers subtopics that have notes in the final aggregated state,
    since notes get updated over multiple turns.
    """
    # Dictionary to track latest notes for each subtopic
    subtopic_has_notes = {}

    for item in history:
        if item["turn_number"] > max_turn:
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

    for item in history:
        if item["turn_number"] > max_turn:
            break

        notes = item.get("notes", None)
        if notes is not None and len(notes.strip()) > 0:
            subtopic_notes.append(notes)

    # Format all collected notes
    return "\n- ".join(subtopic_notes)

def create_evaluation_messages(ground_truth: List[Dict], session_data: Dict, mode: str, snapshot_idx: int = None) -> Tuple[List[List[Dict]], List[str]]:
    """
    Create evaluation messages comparing ALL notes against ALL ground truth facts

    Args:
        ground_truth: Ground truth data
        session_data: Session agenda (interviewer), memory bank (storysage), or history list (llmroleplay/freeform)
        mode: 'sparkme', 'storysage', 'llmroleplay', or 'freeform'
        snapshot_idx: For llmroleplay/freeform mode, the turn number to evaluate up to

    Returns:
        Tuple of (list of messages, list of subtopic IDs) - now returns a single message
    """
    gt_index = build_ground_truth_index(ground_truth)

    # Extract ALL notes from the session
    if mode == 'sparkme':
        all_notes = extract_all_notes_interviewer(session_data)
    elif mode == 'storysage':
        all_notes = extract_all_notes_storysage(session_data, snapshot_idx)
    elif mode == 'llmroleplay':
        all_notes = extract_all_notes_llmroleplay(session_data, snapshot_idx)
    elif mode == 'freeform':
        all_notes = extract_all_notes_freeform(session_data, snapshot_idx)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # If no notes, return empty
    if not all_notes or len(all_notes.strip()) == 0:
        return [], []

    # Combine ALL ground truth facts into a single formatted string
    all_gt_facts = []
    for subtopic_id in sorted(gt_index.keys()):
        gt_info = gt_index[subtopic_id]
        all_gt_facts.append(gt_info['formatted'])

    all_gt_formatted = "\n\n".join(all_gt_facts)

    # Create a SINGLE evaluation message comparing ALL notes to ALL ground truth
    msg = [{
        "role": "user",
        "content": JUDGE_PROMPT.format(
            ground_truth=all_gt_formatted,
            all_notes=all_notes
        )
    }]

    # Return single message with a placeholder ID
    return [msg], ['all_subtopics']


def evaluate_emergence_openai(msg: List[Dict]) -> Dict:
    """Evaluate emergence by comparing all notes against all ground truth"""
    response = request_openai_completion(msg)

    if response is None:
        return {
            'emergent_subtopics': '',
            'error': True
        }

    if "</thinking>" in response:
        response = response.split("</thinking>")[-1].strip()
    parsed_response = safe_parse_json(response)

    if parsed_response is None:
        return {
            'emergent_subtopics': parsed_response,
            'error': True
        }

    return {
        'emergent_subtopics': parsed_response,
        'error': False
    }


def prepare_session_data(user_id: str, snapshot_idx: int, ground_truth: List[Dict], config: EvaluationConfig, surgery: bool = False):
    """Prepare session data and paths for emergence evaluation (all notes vs all ground truth)"""
    # Construct paths based on mode
    if config.mode == 'sparkme':
        session_path = f"{config.base_path}/{user_id}/execution_logs/session_0/session_agenda_snap_{snapshot_idx}.json"
        eval_dir = f"{config.base_path}/{user_id}/evaluations_emergence"
    elif config.mode == 'storysage':
        session_path = f"{config.base_path}/{user_id}/memory_bank_content.json"
        eval_dir = f"{config.base_path}/{user_id}/evaluations_emergence"
    elif config.mode == 'llmroleplay':
        session_path = f"{config.base_path}/{user_id}/interview_log.jsonl"
        eval_dir = f"{config.base_path}/{user_id}/evaluations_emergence"
    elif config.mode == 'freeform':
        session_path = f"{config.base_path}/{user_id}/interview_log.jsonl"
        eval_dir = f"{config.base_path}/{user_id}/evaluations_emergence"
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
        # For these modes, load the JSONL file as a list of history items
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

    # Evaluate ALL notes against ALL ground truth in a single call
    if len(session_info['messages']) == 0:
        return

    msg = session_info['messages'][0]  # Now only one message
    result = evaluate_emergence_openai(msg)

    # Save results as a single evaluation
    os.makedirs(session_info['eval_dir'], exist_ok=True)
    with open(session_info['save_path'], 'w') as f:
        json.dump(result, f, indent=2)


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

        # Step 2: Collect ALL messages and metadata (now one message per session)
        all_messages = []
        all_metadata = []  # Track which session each message belongs to

        for session_info in all_session_infos:
            # Now each session has only ONE message (all notes vs all ground truth)
            if len(session_info['messages']) > 0:
                all_messages.append(session_info['messages'][0])
                all_metadata.append({
                    'session_info': session_info
                })

        logging.info(f"Total sessions to evaluate: {len(all_messages)}")

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

        # Step 4: Parse all results and save by session
        logging.info("Parsing results and saving by session...")
        session_results = {}  # {(user_id, snapshot_idx): result}

        for metadata, output in tqdm(zip(all_metadata, outputs), total=len(outputs), desc="Processing outputs"):
            session_info = metadata['session_info']
            session_key = (session_info['user_id'], session_info['snapshot_idx'])

            generated_text = output.outputs[0].text.strip()
            if "</thinking>" in generated_text:
                generated_text = generated_text.split("</thinking>")[-1].strip()
            
            parsed_response = safe_parse_json(generated_text)

            if parsed_response is None:
                result = {
                    "emergent_subtopic": generated_text,
                    'error': True
                }
            else:
                try:
                    for item in parsed_response:
                        required_fields = {"emergent_subtopic", "topic", "rationale"}
                        item_keys = set(item.keys())
                        if item_keys != required_fields:
                            raise ValueError("Missing required fields in parsed response")
                    result = {
                        'emergent_subtopics': parsed_response,
                        'error': False
                    }
                except:
                    result = {
                        'emergent_subtopics': parsed_response,
                        'error': True
                    }

            session_results[session_key] = {
                'result': result,
                'save_path': session_info['save_path'],
                'eval_dir': session_info['eval_dir']
            }

        # Step 5: Write all results to files
        logging.info(f"Writing results to {len(session_results)} files...")
        for session_key, session_data in tqdm(session_results.items(), desc="Writing results"):
            os.makedirs(session_data['eval_dir'], exist_ok=True)
            with open(session_data['save_path'], 'w') as f:
                json.dump(session_data['result'], f, indent=2)

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

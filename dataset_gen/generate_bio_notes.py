import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from llm_client import LLMClient
from prompts import build_distractor_facts_prompt


def load_topics_filled(user_dir: Path, user_id: str) -> List[Dict]:
    """Load topics_filled.json for a user"""
    topics_path = user_dir / f"{user_id}_topics_filled.json"
    if not topics_path.exists():
        return None
    with open(topics_path, 'r') as f:
        return json.load(f)


def extract_all_facts_from_topics_filled(topics_filled: List[Dict]) -> List[str]:
    """Extract all facts from topics_filled.json as flat list"""
    all_facts = []
    for topic in topics_filled:
        for subtopic in topic.get('subtopics', []):
            notes = subtopic.get('notes', [])
            all_facts.extend(notes)
    return all_facts


def get_occupation_from_facts(topics_filled: List[Dict]) -> str:
    """Extract occupation from facts"""
    # Look for occupation in first topic
    for topic in topics_filled:
        if "Introduction" in topic.get("topic", ""):
            for subtopic in topic.get("subtopics", []):
                if "job title" in subtopic.get("subtopic_description", "").lower():
                    notes = subtopic.get("notes", [])
                    if notes:
                        return notes[0]  # Use first fact as occupation
    return "Professional"


def generate_distractor_facts(
    occupation: str,
    core_facts: List[str],
    llm_client: LLMClient
) -> List[str]:
    """Generate 40-50 creative distractor facts"""
    print("    Generating distractor facts...")

    prompt = build_distractor_facts_prompt(occupation, core_facts)

    response = llm_client.call_gpt41(
        prompt=prompt,
        temperature=0.9,  # High temperature for creativity
        max_tokens=2048
    )

    # Parse JSON
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()

    distractor_facts = json.loads(response)
    print(f"    Generated {len(distractor_facts)} distractor facts")
    return distractor_facts


def shuffle_facts(facts: List[str], user_id: str) -> List[str]:
    """Shuffle facts using deterministic seed based on user_id"""
    # Use hash of user_id as seed for reproducibility
    seed = hash(user_id) % (2**32)  # Ensure 32-bit for compatibility

    # Create a copy to avoid mutating original
    shuffled = facts.copy()
    random.Random(seed).shuffle(shuffled)

    return shuffled


def format_as_bio_notes(facts: List[str]) -> str:
    """Format facts as bullet-point markdown"""
    lines = ["# Biographical Notes", ""]
    for fact in facts:
        lines.append(f"- {fact}")
    return "\n".join(lines)


def generate_bio_notes_for_user(
    user_id: str,
    user_dir: Path,
    llm_client: LLMClient
) -> bool:
    """
    Generate shuffled bio notes for a single user

    Args:
        user_id: User ID
        user_dir: Directory containing topics_filled.json
        llm_client: LLM client instance

    Returns:
        True if successful, False otherwise
    """
    output_file = user_dir / f"{user_id}_bio_notes.md"

    # Check if topics_filled exists
    topics_filled_path = user_dir / f"{user_id}_topics_filled.json"
    if not topics_filled_path.exists():
        print(f"  [SKIP] {user_id}: topics_filled.json not found")
        return False

    # Skip if already exists
    if output_file.exists():
        print(f"  [SKIP] {user_id}: {user_id}_bio_notes.md already exists")
        return True

    print(f"  [GENERATING] {user_id}...")

    try:
        # Load topics_filled
        topics_filled = load_topics_filled(user_dir, user_id)

        # Extract core facts
        core_facts = extract_all_facts_from_topics_filled(topics_filled)
        print(f"    Loaded {len(core_facts)} core facts")

        # Extract occupation
        occupation = get_occupation_from_facts(topics_filled)

        # Generate distractor facts
        distractor_facts = generate_distractor_facts(occupation, core_facts, llm_client)

        # Combine all facts
        all_facts = core_facts + distractor_facts
        total_facts = len(all_facts)
        print(f"    Total facts: {total_facts} ({len(core_facts)} core + {len(distractor_facts)} distractors)")

        # Shuffle facts
        shuffled_facts = shuffle_facts(all_facts, user_id)
        print(f"    Facts shuffled (seed: {hash(user_id) % (2**32)})")

        # Format as markdown
        bio_notes = format_as_bio_notes(shuffled_facts)

        # Save
        with open(output_file, 'w') as f:
            f.write(bio_notes)

        print(f"  [SUCCESS] {user_id}: Generated {total_facts} shuffled facts")
        return True

    except Exception as e:
        print(f"  [ERROR] {user_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate shuffled bio notes from topics_filled.json")
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/sample_user_profiles',
        help='Base directory containing topics_filled.json files'
    )
    parser.add_argument(
        '--user-ids',
        nargs='+',
        help='Specific user IDs to process (default: all)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of users to process'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=16,
        help='Number of parallel workers (default: 10)'
    )

    args = parser.parse_args()

    # Initialize paths
    project_root = Path(__file__).parent.parent
    input_base_dir = project_root / args.input_dir

    print("=" * 60)
    print("Generating Shuffled Bio Notes")
    print("=" * 60)
    print(f"Input directory: {input_base_dir}")
    print()

    # Find all user directories
    if args.user_ids:
        user_dirs = [input_base_dir / uid for uid in args.user_ids if (input_base_dir / uid).exists()]
    else:
        user_dirs = [d for d in input_base_dir.iterdir() if d.is_dir()]

    if args.limit:
        user_dirs = user_dirs[:args.limit]

    print(f"Found {len(user_dirs)} user directories")
    print()

    # Process each user in parallel
    print(f"Processing {len(user_dirs)} users with {args.workers} parallel workers...")
    print()

    success_count = 0
    fail_count = 0

    def process_user(user_dir):
        """Wrapper function for parallel processing"""
        user_id = user_dir.name

        # Each thread gets its own LLM client
        thread_llm_client = LLMClient()

        return generate_bio_notes_for_user(user_id, user_dir, thread_llm_client)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_user, user_dir): user_dir.name
                   for user_dir in user_dirs}

        # Process completed tasks with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating bio notes"):
            user_id = futures[future]
            try:
                success = future.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"\n  [ERROR] {user_id}: Unexpected exception: {str(e)}")
                fail_count += 1

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total users processed: {len(user_dirs)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print()

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

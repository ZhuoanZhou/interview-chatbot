import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from llm_client import LLMClient
from prompts import build_fact_generation_prompt

def validate_json_structure(topics_filled: list) -> tuple[bool, str]:
    """
    Validate the structure of topics_filled.json

    Args:
        topics_filled: The generated topics_filled structure

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(topics_filled, list):
        return False, "Root element must be a list"

    for topic_idx, topic in enumerate(topics_filled):
        if not isinstance(topic, dict):
            return False, f"Topic {topic_idx} must be a dictionary"

        if "topic" not in topic:
            return False, f"Topic {topic_idx} missing 'topic' field"

        if "subtopics" not in topic:
            return False, f"Topic {topic_idx} missing 'subtopics' field"

        if not isinstance(topic["subtopics"], list):
            return False, f"Topic {topic_idx} 'subtopics' must be a list"

        for subtopic_idx, subtopic in enumerate(topic["subtopics"]):
            if not isinstance(subtopic, dict):
                return False, f"Topic {topic_idx}, subtopic {subtopic_idx} must be a dictionary"

            required_fields = ["subtopic_id", "subtopic_description", "notes"]
            for field in required_fields:
                if field not in subtopic:
                    return False, f"Topic {topic_idx}, subtopic {subtopic_idx} missing '{field}' field"

            if not isinstance(subtopic["notes"], list):
                return False, f"Topic {topic_idx}, subtopic {subtopic_idx} 'notes' must be a list"

            if len(subtopic["notes"]) < 1:
                return False, f"Topic {topic_idx}, subtopic {subtopic_idx} has no notes (minimum 1 required)"

    return True, ""

def load_topics_schema(topics_path: str) -> list:
    """Load topics.json schema"""
    with open(topics_path, 'r') as f:
        return json.load(f)


def load_csv_data(csv_path: str) -> pd.DataFrame:
    """Load CSV with user metadata"""
    return pd.read_csv(csv_path)


def generate_facts_for_user(
    user_id: str,
    user_data: Dict[str, Any],
    topics: list,
    llm_client: LLMClient,
    output_dir: Path
) -> bool:
    """
    Generate topics_filled.json for a single user

    Args:
        user_id: User ID
        user_data: Dictionary of CSV data for this user
        topics: Topics schema from topics.json
        llm_client: LLM client instance
        output_dir: Output directory for this user

    Returns:
        True if successful, False otherwise
    """
    output_file = output_dir / f"{user_id}_topics_filled.json"

    # Skip if already exists
    if output_file.exists():
        print(f"  [SKIP] {user_id}: topics_filled.json already exists")
        return True

    print(f"  [GENERATING] {user_id}...")

    try:
        # Build prompt
        prompt = build_fact_generation_prompt(user_data, topics)

        # Call GPT-4.1
        response = llm_client.call_gpt41(
            prompt=prompt,
            temperature=0.7,
            max_tokens=8192
        )

        # Parse JSON response
        # Remove markdown code fences if present
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        topics_filled = json.loads(response)

        # Validate structure
        is_valid, error_msg = validate_json_structure(topics_filled)
        if not is_valid:
            print(f"  [ERROR] {user_id}: Invalid JSON structure - {error_msg}")
            return False

        # Count subtopics and facts
        total_subtopics = sum(len(topic.get("subtopics", [])) for topic in topics_filled)
        total_facts = sum(
            len(subtopic.get("notes", []))
            for topic in topics_filled
            for subtopic in topic.get("subtopics", [])
        )

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save
        with open(output_file, 'w') as f:
            json.dump(topics_filled, f, indent=2)

        print(f"  [SUCCESS] {user_id}: Generated {total_facts} facts across {total_subtopics} subtopics")
        return True

    except json.JSONDecodeError as e:
        print(f"  [ERROR] {user_id}: Failed to parse JSON response - {e}")
        return False
    except Exception as e:
        print(f"  [ERROR] {user_id}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate persona facts from CSV data")
    parser.add_argument(
        '--csv-path',
        type=str,
        default='sampled_users.csv',
        help='Path to CSV file with user data'
    )
    parser.add_argument(
        '--topics-path',
        type=str,
        default='data/configs/topics.json',
        help='Path to topics.json schema'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/sample_user_profiles',
        help='Base output directory for generated personas'
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
    csv_path = project_root / args.csv_path
    topics_path = project_root / args.topics_path
    output_base_dir = project_root / args.output_dir

    print("=" * 60)
    print("Generating Persona Facts")
    print("=" * 60)
    print(f"CSV: {csv_path}")
    print(f"Topics schema: {topics_path}")
    print(f"Output directory: {output_base_dir}")
    print()

    # Load data
    print("Loading data...")
    topics = load_topics_schema(str(topics_path))
    df = load_csv_data(str(csv_path))
    print(f"Loaded {len(df)} users from CSV")
    print(f"Loaded {len(topics)} topics from schema")
    print()

    # Filter users if specified
    if args.user_ids:
        df = df[df['User ID'].isin(args.user_ids)]
        print(f"Filtered to {len(df)} specified users")

    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to first {len(df)} users")

    # Process each user in parallel
    print(f"Processing {len(df)} users with {args.workers} parallel workers...")
    print()

    success_count = 0
    fail_count = 0

    def process_user(row):
        """Wrapper function for parallel processing"""
        user_id = row['User ID']
        user_data = row.to_dict()
        user_output_dir = output_base_dir / user_id

        # Each thread gets its own LLM client
        thread_llm_client = LLMClient()

        return generate_facts_for_user(
            user_id=user_id,
            user_data=user_data,
            topics=topics,
            llm_client=thread_llm_client,
            output_dir=user_output_dir
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_user, row): row['User ID']
                   for idx, row in df.iterrows()}

        # Process completed tasks with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating facts"):
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
    print(f"Total users processed: {len(df)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print()

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

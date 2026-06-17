import argparse
import json
import re
from pathlib import Path
import numpy as np
import textstat

def flesch_kincaid_from_log(
    log_path: str | Path,
    role: str = "Interviewer",
    file_format: str = "txt"
) -> dict:
    """
    Compute document-level Flesch–Kincaid readability
    across an entire interview by extracting one speaker's text.

    Args:
        log_path: path to interview log file
        role: speaker to extract (default: Interviewer)
        file_format: "txt" for chat_history.log or "jsonl" for JSONL format

    Returns:
        dict with FK grade, FK reading ease, word count per turn, and sentence count per turn
    """
    log_path = Path(log_path)

    utterances = []

    if file_format == "jsonl":
        # Parse JSONL format
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        assistant_msg = entry.get("assistant_message", "")
                        if assistant_msg:
                            utterances.append(assistant_msg.strip())
                    except json.JSONDecodeError:
                        continue
    else:
        # Parse text format (original)
        pattern = re.compile(rf"{role}:\s*(.*)")
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    utterances.append(match.group(1).strip())

    if not utterances:
        raise ValueError(f"No utterances found for role '{role}' in {file_format} format")

    # Calculate readability on full transcript
    per_turn_fk = []
    per_turn_re = []
    per_turn_words = []
    per_turn_sentences = []

    for u in utterances:
        # Skip very short turns (FK is unstable)
        if textstat.lexicon_count(u) < 5:
            continue

        per_turn_fk.append(textstat.flesch_kincaid_grade(u))
        per_turn_re.append(textstat.flesch_reading_ease(u))
        per_turn_words.append(textstat.lexicon_count(u))
        per_turn_sentences.append(textstat.sentence_count(u))

    return {
        "fk_grade_mean": float(np.mean(per_turn_fk)),
        "fk_grade_median": float(np.median(per_turn_fk)),
        "fk_grade_std": float(np.std(per_turn_fk)),
        "reading_ease_mean": float(np.mean(per_turn_re)),
        "n_turns": len(per_turn_fk),
        "avg_words_per_turn": float(np.mean(per_turn_words)),
        "avg_sentences_per_turn": float(np.mean(per_turn_sentences)),
    }

def calculate_readability_across_users(
    base_dir: str | Path,
    role: str = "Interviewer",
    output_file: str | Path = None,
    log_filename: str = "chat_history.log"
) -> dict:
    """
    Calculate average and standard deviation of readability metrics
    across all user chat histories.

    Args:
        base_dir: Base directory containing user folders
        role: Speaker role to analyze (default: "Interviewer", ignored for jsonl)
        output_file: Optional path to save results as JSON
        log_filename: Name of log file to process (default: "chat_history.log")

    Returns:
        Dictionary containing statistics and per-user results
    """
    base_dir = Path(base_dir)

    # Collect all user directories
    user_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])

    print(f"Found {len(user_dirs)} user directories")

    # Collect results for each user
    results = []
    failed_users = []

    for user_dir in user_dirs:
        user_id = user_dir.name
        if log_filename == "chat_history.log":
            file_format = "txt"
            chat_log = user_dir / "execution_logs" / "session_1" / log_filename
        else:
            file_format = "jsonl"
            chat_log = user_dir / log_filename

        if not chat_log.exists():
            failed_users.append({
                'user_id': user_id,
                'reason': f'{log_filename} not found'
            })
            continue

        try:
            metrics = flesch_kincaid_from_log(chat_log, role=role, file_format=file_format)
            results.append({
                'user_id': user_id,
                **metrics
            })
        except Exception as e:
            failed_users.append({
                'user_id': user_id,
                'reason': str(e)
            })

    print(f"Successfully processed: {len(results)} users")
    print(f"Failed: {len(failed_users)} users")

    if len(results) == 0:
        raise ValueError("No valid results to calculate statistics")

    # Extract metrics for statistics
    fk_grades = [r['fk_grade_mean'] for r in results]
    reading_ease = [r['reading_ease_mean'] for r in results]
    n_turns = [r['n_turns'] for r in results]
    avg_words_per_turn = [r['avg_words_per_turn'] for r in results]
    avg_sentences_per_turn = [r['avg_sentences_per_turn'] for r in results]

    # Calculate statistics
    statistics = {
        'n_users': len(results),
        'n_failed': len(failed_users),
        'flesch_kincaid_grade': {
            'mean': float(np.mean(fk_grades)),
            'std': float(np.std(fk_grades)),
            'median': float(np.median(fk_grades)),
            'min': float(np.min(fk_grades)),
            'max': float(np.max(fk_grades))
        },
        'flesch_reading_ease': {
            'mean': float(np.mean(reading_ease)),
            'std': float(np.std(reading_ease)),
            'median': float(np.median(reading_ease)),
            'min': float(np.min(reading_ease)),
            'max': float(np.max(reading_ease))
        },
        'n_turns': {
            'mean': float(np.mean(n_turns)),
            'std': float(np.std(n_turns)),
            'median': float(np.median(n_turns)),
            'min': float(np.min(n_turns)),
            'max': float(np.max(n_turns))
        },
        'avg_words_per_turn': {
            'mean': float(np.mean(avg_words_per_turn)),
            'std': float(np.std(avg_words_per_turn)),
            'median': float(np.median(avg_words_per_turn)),
            'min': float(np.min(avg_words_per_turn)),
            'max': float(np.max(avg_words_per_turn))
        },
        'avg_sentences_per_turn': {
            'mean': float(np.mean(avg_sentences_per_turn)),
            'std': float(np.std(avg_sentences_per_turn)),
            'median': float(np.median(avg_sentences_per_turn)),
            'min': float(np.min(avg_sentences_per_turn)),
            'max': float(np.max(avg_sentences_per_turn))
        }
    }

    # Print summary
    print("\n" + "="*70)
    speaker = "Assistant" if file_format == "jsonl" else role
    print(f"READABILITY STATISTICS FOR '{speaker}' ACROSS {len(results)} USERS")
    print("="*70)

    print(f"\nFlesch-Kincaid Grade Level:")
    print(f"  Mean:   {statistics['flesch_kincaid_grade']['mean']:.2f}")
    print(f"  Std:    {statistics['flesch_kincaid_grade']['std']:.2f}")
    print(f"  Median: {statistics['flesch_kincaid_grade']['median']:.2f}")
    print(f"  Range:  [{statistics['flesch_kincaid_grade']['min']:.2f}, {statistics['flesch_kincaid_grade']['max']:.2f}]")

    print(f"\nFlesch Reading Ease:")
    print(f"  Mean:   {statistics['flesch_reading_ease']['mean']:.2f}")
    print(f"  Std:    {statistics['flesch_reading_ease']['std']:.2f}")
    print(f"  Median: {statistics['flesch_reading_ease']['median']:.2f}")
    print(f"  Range:  [{statistics['flesch_reading_ease']['min']:.2f}, {statistics['flesch_reading_ease']['max']:.2f}]")

    print(f"\nNumber of Turns:")
    print(f"  Mean:   {statistics['n_turns']['mean']:.1f}")
    print(f"  Std:    {statistics['n_turns']['std']:.1f}")
    print(f"  Median: {statistics['n_turns']['median']:.0f}")
    print(f"  Range:  [{statistics['n_turns']['min']:.0f}, {statistics['n_turns']['max']:.0f}]")

    print(f"\nAverage Words Per Turn:")
    print(f"  Mean:   {statistics['avg_words_per_turn']['mean']:.1f}")
    print(f"  Std:    {statistics['avg_words_per_turn']['std']:.1f}")
    print(f"  Median: {statistics['avg_words_per_turn']['median']:.1f}")
    print(f"  Range:  [{statistics['avg_words_per_turn']['min']:.1f}, {statistics['avg_words_per_turn']['max']:.1f}]")

    print(f"\nAverage Sentences Per Turn:")
    print(f"  Mean:   {statistics['avg_sentences_per_turn']['mean']:.2f}")
    print(f"  Std:    {statistics['avg_sentences_per_turn']['std']:.2f}")
    print(f"  Median: {statistics['avg_sentences_per_turn']['median']:.2f}")
    print(f"  Range:  [{statistics['avg_sentences_per_turn']['min']:.2f}, {statistics['avg_sentences_per_turn']['max']:.2f}]")

    print(f"\n{'='*70}")

    # Interpretation
    print("\nInterpretation:")
    fk_mean = statistics['flesch_kincaid_grade']['mean']
    if fk_mean < 6:
        level = "elementary school"
    elif fk_mean < 9:
        level = "middle school"
    elif fk_mean < 13:
        level = "high school"
    elif fk_mean < 16:
        level = "college"
    else:
        level = "graduate school"

    print(f"  FK Grade Level of {fk_mean:.2f} corresponds to {level} level")

    re_mean = statistics['flesch_reading_ease']['mean']
    if re_mean >= 90:
        ease = "very easy (5th grade)"
    elif re_mean >= 80:
        ease = "easy (6th grade)"
    elif re_mean >= 70:
        ease = "fairly easy (7th grade)"
    elif re_mean >= 60:
        ease = "standard (8th-9th grade)"
    elif re_mean >= 50:
        ease = "fairly difficult (10th-12th grade)"
    elif re_mean >= 30:
        ease = "difficult (college)"
    else:
        ease = "very difficult (graduate)"

    print(f"  Reading Ease of {re_mean:.2f} indicates {ease}")

    # Prepare output
    output = {
        'metadata': {
            'role': role,
            'base_directory': str(base_dir),
            'total_users': len(user_dirs),
            'successful': len(results),
            'failed': len(failed_users)
        },
        'statistics': statistics,
        'per_user_results': results,
        'failed_users': failed_users
    }

    # Save to file if specified
    if output_file:
        output_file = Path(output_file)
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate interview sessions')
    parser.add_argument('--mode', type=str, required=True, choices=['sparkme', 'storysage', 'llmroleplay', 'freeform'],
                        help='Evaluation mode: interviewer, storysage, llmroleplay, or freeform')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base path to logs directory')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output JSON file for readability statistics')
    args = parser.parse_args()
    
    if args.mode == 'sparkme' or args.mode == 'storysage':
        log_filename = "chat_history.log"
    else:
        log_filename = "interview_log.jsonl"

    # Calculate for Interviewer role
    calculate_readability_across_users(
        base_dir=args.base_dir,
        role="Interviewer",
        output_file=args.output_file,
        log_filename=log_filename,
    )


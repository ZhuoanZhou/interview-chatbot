import json
import os
import argparse
import pandas as pd
from typing import List, Dict, Any

def load_json_safely(path: str) -> Any:
    """Load JSON that may be string-encoded or python-literal encoded."""
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            from ast import literal_eval
            data = literal_eval(data)

    return data


def compute_cumulative_emergence(emergence_eval: List[List[Any]]) -> List[int]:
    """Compute cumulative counts of emergence evaluations."""
    cum_sum = []
    cur = 0
    for emerg in emergence_eval:
        cur += len(emerg)
        cum_sum.append(cur)
    return cum_sum


def compute_normalized_coverage(
    list_scores: List[Dict[str, Any]],
    total_subtopic: int,
    binarized: bool
) -> float:
    """Compute normalized coverage score."""
    acc_scores = 0

    for sc in list_scores:
        if isinstance(sc, dict):
            if binarized:
                cur_score = int(sc.get("score", 0))
            else:
                cur_score = int(sc.get("score", 1))

            if binarized and cur_score == 5:
                acc_scores += 1
            elif not binarized and 1 <= cur_score <= 5:
                acc_scores += cur_score

    # Normalize to [0, 1]
    if binarized:
        normalized = acc_scores / total_subtopic
    else:
        # Fill missing subtopics with score=1
        acc_scores += (total_subtopic - len(list_scores))
        normalized = (acc_scores - total_subtopic) / (total_subtopic * 4)
    return normalized

def load_predicted_coverage(
    path: str
) -> Dict[str, int | float] | None:
    """
    Load predicted coverage statistics from session_agenda_snap JSON.
    """
    try:
        with open(path, "r") as f:
            try:
                data = json.load(f)
            except Exception:
                text = f.read()
                try:
                    data = json.loads(text)
                except Exception:
                    from ast import literal_eval
                    data = literal_eval(text)

        coverage_stats = data["interview_topic_manager"]["coverage_stats"]

        req_cov = sum(
            cat["total_required_subtopics_covered"]
            for cat in coverage_stats.values()
        )
        req_tot = sum(
            cat["total_required_subtopics"]
            for cat in coverage_stats.values()
        )
        emg_cov = sum(
            cat["total_emergent_subtopics_covered"]
            for cat in coverage_stats.values()
        )
        emg_tot = sum(
            cat["total_emergent_subtopics"]
            for cat in coverage_stats.values()
        )

        required_coverage = (
            req_cov / req_tot if req_tot > 0 else 0.0
        )

        return {
            "required_covered": req_cov,
            "required_total": req_tot,
            "required_coverage": required_coverage,
            "emergent_covered": emg_cov,
            "emergent_total": emg_tot,
        }

    except Exception as e:
        print(f"[WARN] Failed predicted coverage: {path} ({e})")
        return None

def process_user(
    user: Dict[str, Any],
    base_path: str,
    total_subtopic: int,
    max_steps: int,
    binarized: bool
) -> List[Dict[str, Any]]:
    """Process evaluation files for a single user."""
    user_id = user["User ID"]
    results = []

    folder_path = os.path.join(
        base_path, user_id, "evaluations_all_to_subtopics"
    )
    
    agenda_folder = os.path.join(
        base_path, user_id, "execution_logs", "session_0"
    )

    # Optional emergence eval
    emergence_folder_path = os.path.join(
        base_path, user_id, "evaluations_emergence_coverage"
    )

    error_count = 0

    for step_idx, step in enumerate(range(1, max_steps + 1)):
        json_file = os.path.join(folder_path, f"snap_eval_{step}.json")
        if not os.path.exists(json_file):
            continue

        try:
            list_scores = load_json_safely(json_file)
            coverage = compute_normalized_coverage(
                list_scores, total_subtopic, binarized
            )
        except Exception as e:
            error_count += 1
            continue

        row = {
            "user_id": user_id,
            "step": step,
            "coverage": coverage,
        }

        agenda_path = os.path.join(
            agenda_folder,
            f"session_agenda_snap_{step}.json"
        )

        if os.path.exists(agenda_path):
            pred = load_predicted_coverage(agenda_path)
            if pred:
                row.update(pred)

        emergence_json_file = os.path.join(
            emergence_folder_path,
            f"snap_eval_{step}.json"
        )
        if os.path.exists(emergence_json_file):
            try:
                emergence_eval = load_json_safely(emergence_json_file)
                if "list_subtopic_covered" in emergence_eval:
                    for emerg in emergence_eval["list_subtopic_covered"]:
                        if "subtopic_covered" not in emerg:
                            raise ValueError("Missing 'subtopic_covered' field")
                row["n_emergence"] = len(emergence_eval["list_subtopic_covered"])
            except Exception as e:
                row["n_emergence"] = 0
        else:
            row["n_emergence"] = 0

        results.append(row)
        
    if error_count > 0:
        print(f"[WARN] User {user_id}: {error_count} errors")

    return results

def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation coverage scores"
    )

    parser.add_argument(
        "--users-file",
        type=str,
        required=True,
        help="Path to sampled users JSON file"
    )

    parser.add_argument(
        "--base-path",
        type=str,
        default="logs",
        help="Base logs directory (default: logs)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file"
    )

    parser.add_argument(
        "--num-users",
        type=int,
        default=200,
        help="Number of users to process"
    )

    parser.add_argument(
        "--total-subtopic",
        type=int,
        default=48,
        help="Total number of subtopics"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=72,
        help="Maximum snap_eval steps to process"
    )
    
    parser.add_argument(
        "--binarized",
        action="store_true",
        help="Enable binarization (default: False)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.users_file, "r") as f:
        sample_users = json.load(f)

    all_rows = []

    for user in sample_users[:args.num_users]:
        rows = process_user(
            user=user,
            base_path=args.base_path,
            total_subtopic=args.total_subtopic,
            max_steps=args.max_steps,
            binarized=args.binarized
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)

    print(f"Saved {len(df)} rows → {args.output}")


if __name__ == "__main__":
    main()

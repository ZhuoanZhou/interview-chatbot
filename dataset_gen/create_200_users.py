import argparse
from pathlib import Path
import pandas as pd

NON_LLM_RESPONSES = {
    "No, I have not used them for any work-related activities.",
    "No, I've never heard of them."
}
RANDOM_SEED = 42

def load_required_users(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def filter_llm_users(df):
    return df[~df["LLM Use in Work"].isin(NON_LLM_RESPONSES)]

def sample_one_per_occupation(df, random_state=RANDOM_SEED):
    return (
        df.groupby("Occupation (O*NET-SOC Title)", group_keys=False)
        .apply(lambda g: g.sample(1, random_state=random_state))
        .reset_index(drop=True)
    )

def main():
    parser = argparse.ArgumentParser(
        description="Sample users with occupation diversity and required inclusions."
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="data/workbank_seed/domain_worker_metadata.csv",
        help="Path to domain worker metadata CSV."
    )
    parser.add_argument(
        "--required_users_txt",
        type=str,
        default="data/workbank_seed/sample_users_30.txt",
        help="Path to text file containing required user IDs."
    )

    parser.add_argument(
        "--target_size",
        type=int,
        default=200,
        help="Total number of users to sample."
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/workbank_seed/sampled_users_200.csv",
        help="Output CSV path."
    )
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_csv)
    required_user_ids = load_required_users(args.required_users_txt)
    llm_users = filter_llm_users(df)
    print(f"Total LLM users available: {len(llm_users)}")

    required_users = df[df["User ID"].isin(required_user_ids)]
    print(f"Required users found in dataset: {len(required_users)}")

    # Track occupations already covered
    occupied_jobs = set(required_users["Occupation (O*NET-SOC Title)"].unique())
    print(f"Occupations covered by required users: {len(occupied_jobs)}")

    # Remaining pool excluding required IDs and their occupations
    remaining_llm_users = llm_users[
        (~llm_users["User ID"].isin(required_user_ids)) &
        (~llm_users["Occupation (O*NET-SOC Title)"].isin(occupied_jobs))
    ]

    print(f"Remaining LLM users: {len(remaining_llm_users)}")
    print(f"Unique occupations in remaining pool: "
          f"{remaining_llm_users['Occupation (O*NET-SOC Title)'].nunique()}")

    sampled_from_remaining = sample_one_per_occupation(remaining_llm_users)
    print(f"Sampled (1 per occupation): {len(sampled_from_remaining)}")

    combined = pd.concat([required_users, sampled_from_remaining], ignore_index=True)
    print(f"Combined total so far: {len(combined)}")

    # Fill to target size if needed
    if len(combined) < args.target_size:
        needed = args.target_size - len(combined)
        print(f"Need {needed} additional users to reach {args.target_size}")

        already_included_occupations = set(
            combined["Occupation (O*NET-SOC Title)"].unique()
        )

        additional_pool = llm_users[
            (~llm_users["User ID"].isin(combined["User ID"])) &
            (llm_users["Occupation (O*NET-SOC Title)"]
             .isin(already_included_occupations))
        ]

        print(f"Additional pool size: {len(additional_pool)}")

        if len(additional_pool) >= needed:
            additional_samples = additional_pool.sample(
                n=needed, random_state=RANDOM_SEED
            )
            final_sample = pd.concat(
                [combined, additional_samples],
                ignore_index=True
            )
        else:
            print("Warning: Not enough users to reach target size.")
            final_sample = pd.concat(
                [combined, additional_pool],
                ignore_index=True
            )
    else:
        final_sample = combined.head(args.target_size)

    print(f"Final sample size: {len(final_sample)}")
    print(f"Unique occupations: "
          f"{final_sample['Occupation (O*NET-SOC Title)'].nunique()}")

    # Ensure output directory exists
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_sample.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    print("\nLLM Usage Distribution:")
    print(final_sample["LLM Use in Work"].value_counts())


if __name__ == "__main__":
    main()

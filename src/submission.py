import os
import pandas as pd

def write_submission_files(
    trial_name,
    cv_type,
    preds_df,
    train_trials,
    train_accessions,
    output_root="submission_output"
):
    """
    Write Predictathon submission files into:
        submission_output/{trial}/{cv_type}/
    """

    # Correct directory: include trial_name
    cv_dir = os.path.join(output_root, trial_name, cv_type)
    os.makedirs(cv_dir, exist_ok=True)

    # Sort predictions for consistency
    preds_df = preds_df.sort_values("germplasmName").reset_index(drop=True)

    # Warn if missing predictions
    if preds_df["pred"].isna().any():
        print(f"Warning: {preds_df['pred'].isna().sum()} missing predictions in {trial_name} {cv_type}")

    # Accessions file
    pd.DataFrame({"germplasmName": train_accessions}).drop_duplicates().to_csv(
        os.path.join(cv_dir, f"{cv_type}accessions.csv"), index=False
    )

    # Trials file
    pd.DataFrame({"trial": train_trials}).drop_duplicates().to_csv(
        os.path.join(cv_dir, f"{cv_type}trials.csv"), index=False
    )

    # Predictions file
    preds_df.to_csv(
        os.path.join(cv_dir, f"{cv_type}predictions.csv"), index=False
    )

    print(f"✓ Wrote files to {cv_dir}")
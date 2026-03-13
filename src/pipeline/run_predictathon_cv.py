import argparse
import subprocess
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def banner(msg: str):
    print("")
    print("======================================")
    print(msg)
    print("======================================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    trials = config["focal_trials"]
    paths = config["paths"]

    predictathon_root = Path(paths["predictathon_root"])
    cv0_dir = Path(paths["cv0_predictions"])
    cv00_dir = Path(paths["cv00_predictions"])
    acc_dir = Path(paths["expected_accuracy"])

    cv0_dir.mkdir(parents=True, exist_ok=True)
    cv00_dir.mkdir(parents=True, exist_ok=True)
    acc_dir.mkdir(parents=True, exist_ok=True)

    banner("  T3/Wheat Predictathon Pipeline")

    for trial in trials:
        banner(f"Processing trial: {trial}")

        proc_dir = predictathon_root / trial / "processed"
        grm_path = proc_dir / "GRM.npy"

        # ---------------------------------------------------------
        # Step 1 — Preprocess genotypes (only if needed)
        # ---------------------------------------------------------
        if not grm_path.exists():
            print(f"[pipeline] No GRM found — preprocessing genotypes for {trial}")
            subprocess.run(
                ["python", "-m", "src.genotypes.preprocess_genotypes", trial],
                check=True,
            )
        else:
            print("[pipeline] GRM already exists — skipping genotype preprocessing")

        # ---------------------------------------------------------
        # Step 2 — Train model
        # ---------------------------------------------------------
        print(f"[pipeline] Training model for {trial}")
        subprocess.run(
            ["python", "-m", "src.model.train_model", trial, "--config", args.config],
            check=True,
        )

        # ---------------------------------------------------------
        # Step 3 — CV0
        # ---------------------------------------------------------
        print(f"[pipeline] Running CV0 for {trial}")
        cv0_out = cv0_dir / f"{trial}.csv"
        subprocess.run(
            [
                "python",
                "src/model/cv0_predict.py",
                "--config",
                args.config,
                "--trial",
                trial,
                "--out",
                str(cv0_out),
            ],
            check=True,
        )

        # ---------------------------------------------------------
        # Step 4 — CV00
        # ---------------------------------------------------------
        print(f"[pipeline] Running CV00 for {trial}")
        cv00_out = cv00_dir / f"{trial}.csv"
        subprocess.run(
            [
                "python",
                "src/model/cv00_predict.py",
                "--config",
                args.config,
                "--trial",
                trial,
                "--out",
                str(cv00_out),
            ],
            check=True,
        )

        # ---------------------------------------------------------
        # Step 5 — Expected accuracy
        # ---------------------------------------------------------
        print(f"[pipeline] Computing expected accuracy for {trial}")
        acc_out = acc_dir / f"{trial}.csv"
        subprocess.run(
            [
                "python",
                "src/model/expected_accuracy.py",
                "--config",
                args.config,
                "--trial",
                trial,
                "--out",
                str(acc_out),
            ],
            check=True,
        )

    banner("Pipeline complete.")


if __name__ == "__main__":
    main()
from pathlib import Path
import subprocess
import yaml
import sys

# repo root = directory containing this file
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config("config.yaml")
    paths = config["paths"]

    repo_root = ROOT
    predictathon_root = Path(paths["predictathon_root"])
    submission_root = Path(paths["submission_root"])  # add this to config if needed
    submission_root.mkdir(parents=True, exist_ok=True)

    # List of all 9 Predictathon trials
    trials = [
    "STP1_2025_MCG",
    "TCAP_2025_MANKS",
    "YT_Urb_25",
    "AWY1_DVPWA_2024",
    "25_Big6_SVREC_SVREC",
    "OHRWW_2025_SPO",
    "CornellMaster_2025_McGowan",
    "24Crk_AY2-3",
    "2025_AYT_Aurora",
]

    cv0_script = repo_root / "src" / "model" / "cv0_predict_global.py"
    cv00_script = repo_root / "src" / "model" / "cv00_predict_global.py"

    for trial in trials:
        print(f"\n=== Running CV0 + CV00 for {trial} ===")

        trial_dir = submission_root / trial
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Output paths
        cv0_out = trial_dir / "CV0_Predictions.csv"
        cv00_out = trial_dir / "CV00_Predictions.csv"


        # Run true CV0
        subprocess.run([
            sys.executable, str(cv0_script),
            "--config", "config.yaml",
            "--trial", trial,
            "--out", str(cv0_out)
        ], check=True)

 
        # Run true CV00
        subprocess.run([
            sys.executable, str(cv00_script),
            "--config", "config.yaml",
            "--trial", trial,
            "--out", str(cv00_out)
        ], check=True)

    print("\nAll CV0 and CV00 predictions complete.")
    print(f"Submission folder written to: {submission_root}")

if __name__ == "__main__":
    main()
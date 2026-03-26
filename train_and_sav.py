"""
train_and_save.py
-----------------
Downloads hormones_clean.csv from Hugging Face, trains all models,
and saves .pkl files to the models/ directory.

Run locally:
    python train_and_save.py

Run during Docker build (env vars injected by Render):
    HF_TOKEN=hf_... HF_DATASET=iiqra/cycleai-data python train_and_save.py
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN   = os.getenv("HF_TOKEN")
HF_DATASET = os.getenv("HF_DATASET", "iiqra/cycleai-data")
MODELS_DIR = Path("models")
DATA_DIR   = Path("data/raw")
CSV_NAME   = "hormones_clean.csv"

# Also check EDA folder (local dev convenience)
CSV_SEARCH_PATHS = [
    Path("data/raw/hormones_clean.csv"),
    Path("EDA/hormones_clean.csv"),
    Path("data/hormones_clean.csv"),
]


def download_from_hf() -> Path:
    """Download CSV from private HF dataset. Returns local path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN is not set.")
        sys.exit(1)

    print(f"[INFO] Downloading {CSV_NAME} from {HF_DATASET} ...")

    local_path = hf_hub_download(
        repo_id=HF_DATASET,
        filename=CSV_NAME,
        repo_type="dataset",
        token=HF_TOKEN,
        local_dir=str(DATA_DIR),
    )

    print(f"[INFO] Downloaded to: {local_path}")
    return Path(local_path)


def train(csv_path: Path):
    """Run the full ML pipeline and save models."""
    import pandas as pd

    print(f"[INFO] Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Dataset shape: {df.shape}")

    # Import your existing pipeline
    from backend.ml.features  import engineer_features
    from backend.ml.models    import build_models
    from backend.ml.evaluate  import evaluate_model
    from backend.ml.pipeline  import run_pipeline_with_validation

    print("[INFO] Running pipeline ...")

    # run_pipeline_with_validation() returns (all_results, best_models)
    # best_models = {
    #   "period_start": {"model": <obj>, "model_name": str, "auc": float,
    #                    "X_train": df, "y_train": series},
    #   "lh_surge":     {"model": <obj>, ...},
    # }
    all_results, best_models = run_pipeline_with_validation(df)

    import pickle

    MODELS_DIR.mkdir(exist_ok=True)

    # ── Save models with exact filenames main.py expects ─────────────────────
    # main.py loads: period_model.pkl, ovulation_model.pkl, trained_features.json
    # best_models keys: "period_start" and "lh_surge"

    TARGET_TO_FILENAME = {
        "period_start": "period_model.pkl",
        "lh_surge":     "ovulation_model.pkl",
    }

    period_features = None  # used for trained_features.json

    for target, info in best_models.items():
        model    = info["model"]
        filename = TARGET_TO_FILENAME.get(target, f"{target}_model.pkl")
        out_path = MODELS_DIR / filename

        with open(out_path, "wb") as f:
            pickle.dump(model, f)
        print(f"[INFO] Saved {out_path}  ({info['model_name']}, AUC={info['auc']:.4f})")

        # main.py expects trained_features to be a flat list from period model
        if target == "period_start":
            period_features = list(info["X_train"].columns)

    # ── Save trained_features.json as a flat list (what main.py expects) ─────
    if period_features is None:
        # fallback: use whichever target we have
        period_features = list(next(iter(best_models.values()))["X_train"].columns)

    features_path = MODELS_DIR / "trained_features.json"
    with open(features_path, "w") as f:
        json.dump(period_features, f, indent=2)
    print(f"[INFO] Saved {features_path}  ({len(period_features)} features)")

    print("[SUCCESS] All models trained and saved.")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Search common local paths first (dev convenience)
    csv_path = None
    for candidate in CSV_SEARCH_PATHS:
        if candidate.exists():
            print(f"[INFO] Found local CSV at {candidate}, skipping HF download.")
            csv_path = candidate
            break

    if csv_path is None:
        csv_path = download_from_hf()

    train(csv_path)
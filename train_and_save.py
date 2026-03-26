"""
train_and_save_draft5.py
------------------------
Train models using the same flow as notebooks/draft5.ipynb and save
backend-ready artifacts into the models/ directory.

Run locally:
    python train_and_save_draft5.py

Run during deployment:
    HF_TOKEN=hf_... HF_DATASET=iiqra/cycleai-data python train_and_save_draft5.py
"""

import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv

from backend.ml import HOLDOUT_IDS, LEAKY_FEATURES, run_pipeline_with_validation

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET = os.getenv("HF_DATASET", "iiqra/cycleai-data")
MODELS_DIR = Path("models")
DATA_DIR = Path("data/raw")
CSV_NAME = "hormones_clean.csv"

CSV_SEARCH_PATHS = [
    Path("data/raw/hormones_clean.csv"),
    Path("EDA/hormones_clean.csv"),
    Path("data/hormones_clean.csv"),
]


def download_from_hf() -> Path:
    """Download the training CSV from a Hugging Face dataset."""
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


def resolve_csv_path() -> Path:
    """Use a local CSV if present, otherwise download it."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for candidate in CSV_SEARCH_PATHS:
        if candidate.exists():
            print(f"[INFO] Found local CSV at {candidate}, skipping HF download.")
            return candidate

    return download_from_hf()


def prepare_training_frame(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mirror the filtering logic used in draft5.ipynb."""
    print(f"[INFO] Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Raw dataset shape: {df.shape}")

    df_model = df.drop(columns=[col for col in LEAKY_FEATURES if col in df.columns])
    df_holdout = df_model[df_model["id"].isin(HOLDOUT_IDS)].copy()
    df_train = df_model[~df_model["id"].isin(HOLDOUT_IDS)].copy()

    print(f"[INFO] Training rows: {len(df_train)} | participants: {df_train['id'].nunique()}")
    print(f"[INFO] Holdout rows: {len(df_holdout)} | participants: {df_holdout['id'].nunique()}")
    print(f"[INFO] Holdout period_start positives: {int(df_holdout['period_start'].sum())}")

    return df_train, df_holdout


def save_artifacts(best_models: dict) -> None:
    """Save model files and the feature list in the format expected by backend.main."""
    MODELS_DIR.mkdir(exist_ok=True)

    period_model = best_models["period_start"]["model"]
    joblib.dump(period_model, MODELS_DIR / "period_model.pkl")
    print(f"[INFO] Saved {MODELS_DIR / 'period_model.pkl'}")

    if "lh_surge" in best_models:
        ovulation_model = best_models["lh_surge"]["model"]
        joblib.dump(ovulation_model, MODELS_DIR / "ovulation_model.pkl")
        print(f"[INFO] Saved {MODELS_DIR / 'ovulation_model.pkl'}")

    if hasattr(period_model, "feature_names_in_"):
        trained_features = list(period_model.feature_names_in_)
    else:
        trained_features = list(best_models["period_start"]["X_train"].columns)

    features_path = MODELS_DIR / "trained_features.json"
    with open(features_path, "w") as f:
        json.dump(trained_features, f, indent=2)
    print(f"[INFO] Saved {features_path}")


def main() -> None:
    csv_path = resolve_csv_path()
    df_train, _ = prepare_training_frame(csv_path)

    print("[INFO] Running pipeline ...")
    all_results, best_models = run_pipeline_with_validation(
        df_train,
        output_dir="./pipeline_outputs",
    )

    print(f"[INFO] Targets trained: {list(best_models.keys())}")
    save_artifacts(best_models)
    print("[SUCCESS] Training complete and backend-ready artifacts saved.")


if __name__ == "__main__":
    main()

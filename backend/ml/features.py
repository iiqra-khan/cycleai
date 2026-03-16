# =============================================================================
# features.py — Feature engineering for the menstrual cycle prediction pipeline
# =============================================================================

import numpy as np
import pandas as pd
from .config import ALL_FEATURES, SYMPTOM_FEATURES


def add_cycle_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimates cycle position WITHOUT using period_start or is_bleeding.
    Uses only hormonal patterns observable in real-time.
    """
    df = df.copy()

    df["pdg_above_threshold"] = (df["pdg_imputed"] > 3.0).astype(int)

    df["lh_surge_count"] = (
        df.groupby("id")["lh_surge"]
          .transform(lambda x: x.shift(1).fillna(0).cumsum())
        if "lh_surge" in df.columns
        else 0
    )

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates cycle-aware temporal and hormonal features.

    Feature groups:
      - Rolling means of hormones (3-day, 7-day windows)
      - LH surge momentum (rate of change)
      - Symptom burden score (composite)
      - Lagged features for sequential modeling
      - E2:PDG ratio
    """
    df = df.copy().sort_values(["id", "day_in_study"]).reset_index(drop=True)

    # ── Rolling Hormonal Features ─────────────────────────────────────────────
    for col in ["lh_imputed", "estrogen_imputed", "pdg_imputed"]:
        if col in df.columns:
            df[f"{col}_roll3"] = (
                df.groupby("id")[col]
                  .transform(lambda x: x.rolling(3, min_periods=1).mean())
            )
            df[f"{col}_roll7"] = (
                df.groupby("id")[col]
                  .transform(lambda x: x.rolling(7, min_periods=1).mean())
            )
            df[f"{col}_delta"] = (
                df.groupby("id")[col]
                  .transform(lambda x: x.diff().fillna(0))
            )

    # ── LH Surge Momentum ─────────────────────────────────────────────────────
    if "lh_imputed" in df.columns:
        df["lh_momentum"] = (
            df.groupby("id")["lh_imputed"]
              .transform(lambda x: x.pct_change().fillna(0).clip(-5, 5))
        )
        assert "lh_imputed_delta" in df.columns, \
            "lh_imputed_delta must be computed before lh_rising"
        df["lh_rising"] = (df["lh_imputed_delta"] > 0).astype(int)
        df["lh_consecutive_rise"] = (
            df.groupby("id")["lh_rising"]
              .transform(lambda x: x * (
                  x.groupby((x != x.shift()).cumsum()).cumcount() + 1
              ))
        )

    # ── Symptom Burden Score ──────────────────────────────────────────────────
    symptom_cols = [c for c in SYMPTOM_FEATURES if c in df.columns]
    if symptom_cols:
        df["symptom_burden"] = df[symptom_cols].sum(axis=1)
        df["symptom_burden_roll3"] = (
            df.groupby("id")["symptom_burden"]
              .transform(lambda x: x.rolling(3, min_periods=1).mean())
        )

    # ── Cycle Phase Encoding ──────────────────────────────────────────────────
    if "phase" in df.columns:
        df["is_luteal"]     = (df["phase"] == 4).astype(int)
        df["is_ovulatory"]  = (df["phase"] == 3).astype(int)
        df["is_follicular"] = (df["phase"] == 2).astype(int)
        df["is_menstrual"]  = (df["phase"] == 1).astype(int)

    # ── Lagged Features ───────────────────────────────────────────────────────
    for col in ["cramps_imputed", "lh_imputed", "is_bleeding"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df.groupby("id")[col].shift(1).fillna(0)
            df[f"{col}_lag2"] = df.groupby("id")[col].shift(2).fillna(0)

    # ── E2:PDG Ratio ──────────────────────────────────────────────────────────
    if "estrogen_imputed" in df.columns and "pdg_imputed" in df.columns:
        denom = df["pdg_imputed"].replace(0, np.nan)
        df["e2_pdg_ratio"] = (
            df["estrogen_imputed"] / denom
        ).fillna(0).clip(0, 100)

    print(f"[Feature Engineering] Created {len(df.columns)} total features "
          f"from {len(df)} rows")

    df = add_cycle_proxy_features(df)
    return df


def get_engineered_feature_list(df: pd.DataFrame) -> list:
    """Returns the ordered feature list after engineering (training data only)."""
    engineered = [
        c for c in df.columns
        if any(suffix in c for suffix in [
            "_roll3", "_roll7", "_delta", "_momentum", "consecutive_rise",
            "_burden", "is_luteal", "is_ovulatory", "is_follicular",
            "is_menstrual", "_lag1", "_lag2", "e2_pdg_ratio"
        ])
    ]
    base = [c for c in ALL_FEATURES if c in df.columns]
    return list(dict.fromkeys(base + engineered))
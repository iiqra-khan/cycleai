# =============================================================================
# config.py — Central configuration for the ML pipeline
# =============================================================================

TARGET_PERIOD = "period_start"
TARGET_OVUL   = "lh_surge"

HORMONAL_FEATURES = [
    "lh_imputed", "estrogen_imputed", "pdg_imputed",
    "high_estrogen_flag", "estrogen_capped_flag"
]

CYCLE_FEATURES = [
    "is_weekend",
]

SYMPTOM_FEATURES = [
    "cramps_imputed", "sorebreasts_imputed", "bloating_imputed",
    "moodswing_imputed", "fatigue_imputed", "headaches_imputed",
    "foodcravings_imputed", "indigestion_imputed",
    "exerciselevel_imputed", "stress_imputed",
    "sleepissue_imputed", "appetite_imputed"
]

ALL_FEATURES = HORMONAL_FEATURES + CYCLE_FEATURES + SYMPTOM_FEATURES

LEAKY_FEATURES = [
    "is_menstrual", "is_bleeding", "is_bleeding_lag1", "is_bleeding_lag2",
    "phase", "is_luteal", "is_follicular", "is_ovulatory", "day_in_period",
    "flow_volume_imputed", "flow_grp_Abnormal", "flow_grp_Fresh",
    "flow_grp_Old", "flow_grp_No Flow", "study_interval",
    "lh", "estrogen", "pdg", "cramps", "sorebreasts",
    "bloating", "moodswing", "fatigue", "headaches",
    "foodcravings", "indigestion", "exerciselevel",
    "stress", "sleepissue", "appetite",
    "estrogen_clean", "flow_volume_numeric", "flow_volume",
    "flow_color", "days_since_bleed",
]

HOLDOUT_IDS = [50, 9, 2]
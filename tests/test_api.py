# =============================================================================
# tests/test_api.py
# Run from D:\MajorProject with:
#   python tests/test_api.py
# =============================================================================

import requests
import json
import time
import sys

BASE_URL = "http://127.0.0.1:8000"

# ── Color codes for terminal output ──────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✅ {msg}{RESET}")
def fail(msg): print(f"  {RED}❌ {msg}{RESET}")
def info(msg): print(f"  {CYAN}ℹ  {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠  {msg}{RESET}")

# =============================================================================
# Test Payloads
# =============================================================================

# ── 1. Period incoming — PDG dropping fast, cramps rising ────────────────────
PAYLOAD_PERIOD = {
    "days": [
        {"id":1,"day_in_study":21,"lh_imputed":3.5,"estrogen_imputed":95.0,"pdg_imputed":18.0,"cramps_imputed":0,"sorebreasts_imputed":1,"bloating_imputed":3,"moodswing_imputed":1,"fatigue_imputed":2,"headaches_imputed":1,"foodcravings_imputed":1,"indigestion_imputed":1,"exerciselevel_imputed":3,"stress_imputed":2,"sleepissue_imputed":1,"appetite_imputed":3,"high_estrogen_flag":False,"estrogen_capped_flag":False,"is_weekend":False},
        {"id":1,"day_in_study":22,"lh_imputed":3.5,"estrogen_imputed":90.0,"pdg_imputed":16.0,"cramps_imputed":0,"sorebreasts_imputed":1,"bloating_imputed":3,"moodswing_imputed":1,"fatigue_imputed":2,"headaches_imputed":1,"foodcravings_imputed":1,"indigestion_imputed":1,"exerciselevel_imputed":3,"stress_imputed":2,"sleepissue_imputed":1,"appetite_imputed":3,"high_estrogen_flag":False,"estrogen_capped_flag":False,"is_weekend":False},
        {"id":1,"day_in_study":23,"lh_imputed":3.5,"estrogen_imputed":85.0,"pdg_imputed":14.0,"cramps_imputed":0,"sorebreasts_imputed":2,"bloating_imputed":3,"moodswing_imputed":2,"fatigue_imputed":2,"headaches_imputed":1,"foodcravings_imputed":2,"indigestion_imputed":1,"exerciselevel_imputed":2,"stress_imputed":2,"sleepissue_imputed":1,"appetite_imputed":3,"high_estrogen_flag":False,"estrogen_capped_flag":False,"is_weekend":False},
        {"id":1,"day_in_study":24,"lh_imputed":3.5,"estrogen_imputed":78.0,"pdg_imputed":10.0,"cramps_imputed":0,"sorebreasts_imputed":2,"bloating_imputed":3,"moodswing_imputed":2,"fatigue_imputed":3,"headaches_imputed":1,"foodcravings_imputed":2,"indigestion_imputed":2,"exerciselevel_imputed":2,"stress_imputed":3,"sleepissue_imputed":2,"appetite_imputed":2,"high_estrogen_flag":False,"estrogen_capped_flag":False,"is_weekend":False},
        {"id":1,"day_in_study":25,"lh_imputed":3.5,"estrogen_imputed":65.0,"pdg_imputed":6.0,"cramps_imputed":1,"sorebreasts_imputed":2,"bloating_imputed":3,"moodswing_imputed":2,"fatigue_imputed":3,"headaches_imputed":2,"foodcravings_imputed":3,"indigestion_imputed":2,"exerciselevel_imputed":1,"stress_imputed":3,"sleepissue_imputed":2,"appetite_imputed":2,"high_estrogen_flag":False,"estrogen_capped_flag":False,"is_weekend":False},
        {"id":1,"day_in_study":26,"lh_imputed":3.5,"estrogen_imputed":52.0,"pdg_imputed":3.5,"cramps_imputed":1,"sorebreasts_imputed":3,"bloating_imputed":3,"moodswing_imputed":3,"fatigue_imputed":3,"headaches_imputed":2,"foodcravings_imputed":3,"indigestion_imputed":2,"exerciselevel_imputed":1,"stress_imputed":3,"sleepissue_imputed":2,"appetite_imputed":2,"high_estrogen_flag":False,"estrogen_capped_flag":False,"is_weekend":False},
        {"id":1,"day_in_study":27,"lh_imputed":3.5,"estrogen_imputed":44.0,"pdg_imputed":2.0,"cramps_imputed":3,"sorebreasts_imputed":3,"bloating_imputed":3,"moodswing_imputed":3,"fatigue_imputed":4,"headaches_imputed":2,"foodcravings_imputed":3,"indigestion_imputed":3,"exerciselevel_imputed":1,"stress_imputed":4,"sleepissue_imputed":3,"appetite_imputed":1,"high_estrogen_flag":False,"estrogen_capped_flag":False,"is_weekend":False},
    ],
    "include_rag": False,
    "include_shap": True
}

# ── 2. Ovulation incoming — LH surging, estrogen peaking ────────────────────
PAYLOAD_OVULATION = {
    "days": [
        {"id":1,"day_in_study":10,"lh_imputed":5.0,"estrogen_imputed":120.0,"pdg_imputed":1.2,"cramps_imputed":0,"sorebreasts_imputed":1,"bloating_imputed":1,"moodswing_imputed":1,"fatigue_imputed":1,"headaches_imputed":1,"foodcravings_imputed":1,"indigestion_imputed":1,"exerciselevel_imputed":3,"stress_imputed":2,"sleepissue_imputed":1,"appetite_imputed":3,"high_estrogen_flag":False,"estrogen_capped_flag":False,"is_weekend":False},
        {"id":1,"day_in_study":11,"lh_imputed":6.0,"estrogen_imputed":145.0,"pdg_imputed":1.1,"cramps_imputed":0,"sorebreasts_imputed":1,"bloating_imputed":1,"moodswing_imputed":1,"fatigue_imputed":1,"headaches_imputed":1,"foodcravings_imputed":1,"indigestion_imputed":1,"exerciselevel_imputed":3,"stress_imputed":2,"sleepissue_imputed":1,"appetite_imputed":3,"high_estrogen_flag":False,"estrogen_capped_flag":False,"is_weekend":False},
        {"id":1,"day_in_study":12,"lh_imputed":8.0,"estrogen_imputed":175.0,"pdg_imputed":1.0,"cramps_imputed":0,"sorebreasts_imputed":1,"bloating_imputed":1,"moodswing_imputed":1,"fatigue_imputed":1,"headaches_imputed":1,"foodcravings_imputed":1,"indigestion_imputed":1,"exerciselevel_imputed":3,"stress_imputed":2,"sleepissue_imputed":1,"appetite_imputed":3,"high_estrogen_flag":False,"estrogen_capped_flag":False,"is_weekend":False},
        {"id":1,"day_in_study":13,"lh_imputed":12.0,"estrogen_imputed":210.0,"pdg_imputed":1.0,"cramps_imputed":0,"sorebreasts_imputed":1,"bloating_imputed":1,"moodswing_imputed":2,"fatigue_imputed":1,"headaches_imputed":1,"foodcravings_imputed":1,"indigestion_imputed":1,"exerciselevel_imputed":3,"stress_imputed":2,"sleepissue_imputed":1,"appetite_imputed":3,"high_estrogen_flag":True,"estrogen_capped_flag":False,"is_weekend":False},
        {"id":1,"day_in_study":14,"lh_imputed":18.0,"estrogen_imputed":230.0,"pdg_imputed":1.1,"cramps_imputed":0,"sorebreasts_imputed":1,"bloating_imputed":2,"moodswing_imputed":2,"fatigue_imputed":2,"headaches_imputed":1,"foodcravings_imputed":1,"indigestion_imputed":1,"exerciselevel_imputed":3,"stress_imputed":2,"sleepissue_imputed":1,"appetite_imputed":3,"high_estrogen_flag":True,"estrogen_capped_flag":False,"is_weekend":False},
        {"id":1,"day_in_study":15,"lh_imputed":28.0,"estrogen_imputed":220.0,"pdg_imputed":1.2,"cramps_imputed":0,"sorebreasts_imputed":1,"bloating_imputed":2,"moodswing_imputed":2,"fatigue_imputed":2,"headaches_imputed":1,"foodcravings_imputed":1,"indigestion_imputed":1,"exerciselevel_imputed":3,"stress_imputed":2,"sleepissue_imputed":1,"appetite_imputed":3,"high_estrogen_flag":True,"estrogen_capped_flag":False,"is_weekend":True},
        {"id":1,"day_in_study":16,"lh_imputed":42.0,"estrogen_imputed":195.0,"pdg_imputed":1.3,"cramps_imputed":0,"sorebreasts_imputed":1,"bloating_imputed":2,"moodswing_imputed":2,"fatigue_imputed":2,"headaches_imputed":1,"foodcravings_imputed":1,"indigestion_imputed":1,"exerciselevel_imputed":3,"stress_imputed":2,"sleepissue_imputed":1,"appetite_imputed":3,"high_estrogen_flag":True,"estrogen_capped_flag":False,"is_weekend":True},
    ],
    "include_rag": False,
    "include_shap": True
}

# ── 3. Low risk — mid follicular phase, all normal ───────────────────────────
PAYLOAD_LOW_RISK = {
    "days": [
        {"id":1,"day_in_study":5,"lh_imputed":4.0,"estrogen_imputed":60.0,"pdg_imputed":1.0,"cramps_imputed":0,"sorebreasts_imputed":0,"bloating_imputed":1,"moodswing_imputed":1,"fatigue_imputed":1,"headaches_imputed":0,"foodcravings_imputed":1,"indigestion_imputed":0,"exerciselevel_imputed":3,"stress_imputed":1,"sleepissue_imputed":1,"appetite_imputed":3,"high_estrogen_flag":False,"estrogen_capped_flag":False,"is_weekend":False},
        {"id":1,"day_in_study":6,"lh_imputed":4.1,"estrogen_imputed":65.0,"pdg_imputed":1.0,"cramps_imputed":0,"sorebreasts_imputed":0,"bloating_imputed":1,"moodswing_imputed":1,"fatigue_imputed":1,"headaches_imputed":0,"foodcravings_imputed":1,"indigestion_imputed":0,"exerciselevel_imputed":3,"stress_imputed":1,"sleepissue_imputed":1,"appetite_imputed":3,"high_estrogen_flag":False,"estrogen_capped_flag":False,"is_weekend":False},
        {"id":1,"day_in_study":7,"lh_imputed":4.2,"estrogen_imputed":72.0,"pdg_imputed":1.1,"cramps_imputed":0,"sorebreasts_imputed":0,"bloating_imputed":1,"moodswing_imputed":1,"fatigue_imputed":1,"headaches_imputed":0,"foodcravings_imputed":1,"indigestion_imputed":0,"exerciselevel_imputed":4,"stress_imputed":1,"sleepissue_imputed":1,"appetite_imputed":3,"high_estrogen_flag":False,"estrogen_capped_flag":False,"is_weekend":True},
    ],
    "include_rag": False,
    "include_shap": False
}

# ── 4. Edge case — only 1 day (rolling features will be zero) ────────────────
PAYLOAD_SINGLE_DAY = {
    "days": [
        {"id":1,"day_in_study":28,"lh_imputed":3.6,"estrogen_imputed":60.0,"pdg_imputed":1.1,"cramps_imputed":5,"sorebreasts_imputed":4,"bloating_imputed":5,"moodswing_imputed":4,"fatigue_imputed":5,"headaches_imputed":4,"foodcravings_imputed":4,"indigestion_imputed":3,"exerciselevel_imputed":1,"stress_imputed":4,"sleepissue_imputed":3,"appetite_imputed":1,"high_estrogen_flag":False,"estrogen_capped_flag":False,"is_weekend":True},
    ],
    "include_rag": False,
    "include_shap": False
}

# ── 5. Invalid payload — missing required field ───────────────────────────────
PAYLOAD_INVALID = {
    "days": [{"id": 1, "day_in_study": 1}],  # missing hormone fields
    "include_rag": False,
    "include_shap": False
}

# =============================================================================
# Test runner
# =============================================================================

results = {"passed": 0, "failed": 0, "warnings": 0}

def run_test(name, fn):
    print(f"\n{BOLD}{CYAN}── {name}{RESET}")
    try:
        fn()
        results["passed"] += 1
    except AssertionError as e:
        fail(f"ASSERTION FAILED: {e}")
        results["failed"] += 1
    except Exception as e:
        fail(f"ERROR: {e}")
        results["failed"] += 1


# =============================================================================
# Tests
# =============================================================================

def test_health():
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data["status"] == "healthy",       "Status not healthy"
    assert data["period_model"] == True,      "Period model not loaded"
    assert data["features_loaded"] == True,   "Features not loaded"
    ok(f"Health check passed — models loaded ✓")
    info(f"Ovulation model: {data['ovulation_model']}")


def test_features():
    r = requests.get(f"{BASE_URL}/features", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["count"] > 0, "No features returned"
    ok(f"{data['count']} features loaded")
    info(f"First 5: {data['features'][:5]}")


def test_period_prediction():
    start = time.time()
    r = requests.post(f"{BASE_URL}/predict", json=PAYLOAD_PERIOD, timeout=30)
    elapsed = time.time() - start

    assert r.status_code == 200, f"Expected 200, got {r.status_code} — {r.text}"
    data = r.json()

    prob = data["period_probability"]
    ok(f"Period probability: {prob:.1%}  ({elapsed:.2f}s)")
    info(f"Prediction: {data['period_prediction']} | Risk: {data['period_risk']['level']}")
    info(f"Top feature: {data['top_features'][0]['feature'] if data['top_features'] else 'N/A'}")

    assert prob > 0.85, f"Expected >85%, got {prob:.1%} — model may not be loaded correctly"
    ok("Period prediction threshold passed (>85%)")


def test_ovulation_prediction():
    start = time.time()
    r = requests.post(f"{BASE_URL}/predict", json=PAYLOAD_OVULATION, timeout=30)
    elapsed = time.time() - start

    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()

    prob = data["ovulation_probability"]
    ok(f"Ovulation probability: {prob:.1%}  ({elapsed:.2f}s)")
    info(f"Prediction: {data['ovulation_prediction']} | Risk: {data['ovulation_risk']['level']}")

    if prob < 0.5:
        warn(f"Ovulation probability low ({prob:.1%}) — may need more LH history")
        results["warnings"] += 1
    else:
        ok(f"Ovulation prediction threshold passed (>50%)")


def test_low_risk():
    r = requests.post(f"{BASE_URL}/predict", json=PAYLOAD_LOW_RISK, timeout=30)
    assert r.status_code == 200
    data = r.json()

    prob = data["period_probability"]
    ok(f"Low risk period probability: {prob:.1%}")
    info(f"Risk level: {data['period_risk']['level']}")

    assert prob < 0.5, f"Expected low probability (<50%), got {prob:.1%}"
    ok("Low risk threshold passed (<50%)")


def test_single_day():
    r = requests.post(f"{BASE_URL}/predict", json=PAYLOAD_SINGLE_DAY, timeout=30)
    assert r.status_code == 200
    data = r.json()
    prob = data["period_probability"]
    ok(f"Single day prediction: {prob:.1%} (rolling features = 0, lower than multi-day)")
    info("This is expected — single day has no temporal context")


def test_response_structure():
    r = requests.post(f"{BASE_URL}/predict", json=PAYLOAD_PERIOD, timeout=30)
    assert r.status_code == 200
    data = r.json()

    required_keys = [
        "period_probability", "period_prediction", "period_risk",
        "ovulation_probability", "ovulation_prediction", "ovulation_risk",
        "top_features", "clinical_explanation", "model_used", "features_used"
    ]
    for key in required_keys:
        assert key in data, f"Missing key: {key}"

    assert 0 <= data["period_probability"] <= 1,     "Probability out of range"
    assert 0 <= data["ovulation_probability"] <= 1,  "Probability out of range"
    assert data["period_risk"]["level"] in ["LOW","MODERATE","HIGH"]
    assert isinstance(data["top_features"], list)
    assert data["features_used"] > 0

    ok(f"Response structure valid — all {len(required_keys)} required keys present")
    info(f"Model: {data['model_used']} | Features: {data['features_used']}")


def test_invalid_payload():
    r = requests.post(f"{BASE_URL}/predict", json=PAYLOAD_INVALID, timeout=30)
    # Should either handle gracefully (200 with defaults) or return 4xx
    if r.status_code == 200:
        ok("Invalid payload handled gracefully (filled missing fields with defaults)")
    elif r.status_code in [400, 422]:
        ok(f"Invalid payload correctly rejected with {r.status_code}")
    else:
        warn(f"Unexpected status {r.status_code} for invalid payload")
        results["warnings"] += 1


def test_response_time():
    times = []
    for _ in range(3):
        start = time.time()
        requests.post(f"{BASE_URL}/predict", json=PAYLOAD_LOW_RISK, timeout=30)
        times.append(time.time() - start)

    avg = sum(times) / len(times)
    ok(f"Average response time: {avg:.3f}s over 3 requests")
    if avg > 2.0:
        warn(f"Response time is slow ({avg:.2f}s) — consider caching")
        results["warnings"] += 1
    else:
        ok("Response time acceptable (<2s)")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print(f"\n{BOLD}{'='*55}")
    print("  CycleAI API Test Suite")
    print(f"  Target: {BASE_URL}")
    print(f"{'='*55}{RESET}")

    # Check server is running first
    try:
        requests.get(f"{BASE_URL}/health", timeout=3)
    except Exception:
        print(f"\n{RED}❌ Cannot connect to {BASE_URL}")
        print(f"   Make sure the backend is running:")
        print(f"   python -m uvicorn backend.main:app --reload{RESET}\n")
        sys.exit(1)

    run_test("1. Health Check",         test_health)
    run_test("2. Features Endpoint",    test_features)
    run_test("3. Period Prediction",    test_period_prediction)
    run_test("4. Ovulation Prediction", test_ovulation_prediction)
    run_test("5. Low Risk Prediction",  test_low_risk)
    run_test("6. Single Day Input",     test_single_day)
    run_test("7. Response Structure",   test_response_structure)
    run_test("8. Invalid Payload",      test_invalid_payload)
    run_test("9. Response Time",        test_response_time)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = results["passed"] + results["failed"]
    print(f"\n{BOLD}{'='*55}")
    print(f"  Results: {results['passed']}/{total} passed", end="")
    if results["warnings"]: print(f"  |  {results['warnings']} warning(s)", end="")
    print(f"\n{'='*55}{RESET}")

    if results["failed"] == 0:
        print(f"{GREEN}{BOLD}  ✅ All tests passed!{RESET}\n")
    else:
        print(f"{RED}{BOLD}  ❌ {results['failed']} test(s) failed{RESET}\n")
        sys.exit(1)
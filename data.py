import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib, os, sqlite3, datetime

# ── Synthetic equipment catalogue ────────────────────────────────────────────
EQUIPMENT = [
    {"id": "AHU-01", "type": "AHU",      "location": "Floor 1", "installed": "2018-03-12"},
    {"id": "AHU-02", "type": "AHU",      "location": "Floor 2", "installed": "2019-07-22"},
    {"id": "AHU-03", "type": "AHU",      "location": "Floor 3", "installed": "2020-01-05"},
    {"id": "CHR-01", "type": "Chiller",  "location": "Basement","installed": "2017-11-30"},
    {"id": "CHR-02", "type": "Chiller",  "location": "Roof",    "installed": "2021-06-14"},
    {"id": "ELV-01", "type": "Elevator", "location": "Block A",  "installed": "2016-08-09"},
    {"id": "ELV-02", "type": "Elevator", "location": "Block B",  "installed": "2019-02-28"},
    {"id": "ELV-03", "type": "Elevator", "location": "Block C",  "installed": "2022-04-17"},
]

# ── Feature definitions per equipment type ───────────────────────────────────
FEATURES = {
    "AHU": [
        {"name": "supply_air_temp",   "label": "Supply air temp (°C)",    "min": 10.0, "max": 40.0, "default": 28.0},
        {"name": "return_air_temp",   "label": "Return air temp (°C)",    "min": 15.0, "max": 45.0, "default": 32.0},
        {"name": "filter_dp",         "label": "Filter ΔP (Pa)",          "min": 50.0, "max": 400.0,"default": 180.0},
        {"name": "fan_vibration",     "label": "Fan vibration (mm/s)",    "min": 0.0,  "max": 15.0, "default": 4.0},
        {"name": "coil_fouling",      "label": "Coil fouling index (0–1)","min": 0.0,  "max": 1.0,  "default": 0.3},
        {"name": "runtime_hrs",       "label": "Runtime hours",           "min": 0.0,  "max": 10000.0,"default": 3000.0},
        {"name": "ambient_humidity",  "label": "Ambient humidity (%)",    "min": 20.0, "max": 95.0, "default": 55.0},
    ],
    "Chiller": [
        {"name": "condenser_temp",    "label": "Condenser temp (°C)",     "min": 20.0, "max": 55.0, "default": 35.0},
        {"name": "evaporator_temp",   "label": "Evaporator temp (°C)",    "min": -10.0,"max": 20.0, "default": 6.0},
        {"name": "refrigerant_pres",  "label": "Refrigerant pressure (bar)","min":5.0, "max": 25.0, "default": 14.0},
        {"name": "cop",               "label": "COP",                     "min": 1.0,  "max": 7.0,  "default": 4.5},
        {"name": "compressor_vib",    "label": "Compressor vibration (mm/s)","min":0.0,"max": 12.0, "default": 2.0},
        {"name": "runtime_hrs",       "label": "Runtime hours",           "min": 0.0,  "max": 10000.0,"default": 4000.0},
    ],
    "Elevator": [
        {"name": "door_open_time",    "label": "Door open time (s)",      "min": 1.0,  "max": 10.0, "default": 3.0},
        {"name": "motor_current",     "label": "Motor current (A)",       "min": 5.0,  "max": 50.0, "default": 20.0},
        {"name": "cable_tension",     "label": "Cable tension (kN)",      "min": 1.0,  "max": 20.0, "default": 10.0},
        {"name": "speed_deviation",   "label": "Speed deviation (%)",     "min": 0.0,  "max": 30.0, "default": 2.0},
        {"name": "vibration_rms",     "label": "Vibration RMS (mm/s)",    "min": 0.0,  "max": 10.0, "default": 1.5},
        {"name": "runtime_hrs",       "label": "Runtime hours",           "min": 0.0,  "max": 10000.0,"default": 5000.0},
    ],
}

# ── Thresholds for rule-based diagnosis ─────────────────────────────────────
RULES = {
    "AHU": {
        "filter_dp":        (300, "High filter pressure drop — clogged filter"),
        "fan_vibration":    (10,  "Excessive fan vibration — bearing wear likely"),
        "coil_fouling":     (0.7, "Coil fouling above safe limit — cleaning required"),
        "supply_air_temp":  (35,  "Supply air temp too high — cooling capacity degraded"),
    },
    "Chiller": {
        "condenser_temp":   (48,  "High condenser temp — scaling or fouling suspected"),
        "compressor_vib":   (8,   "Compressor vibration elevated — inspect mounts"),
        "cop":              (2.5, "Low COP — refrigerant charge or compressor issue"),
        "refrigerant_pres": (22,  "High refrigerant pressure — TXV or blockage fault"),
    },
    "Elevator": {
        "door_open_time":   (6,   "Door dwell too long — sensor or mechanism fault"),
        "motor_current":    (40,  "Motor overcurrent — drive or brake issue"),
        "speed_deviation":  (15,  "Speed deviation excessive — governor check needed"),
        "vibration_rms":    (7,   "High vibration — guide rail or counterweight issue"),
    },
}

# ── Synthetic dataset generator ──────────────────────────────────────────────
def make_dataset(equip_type: str, n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    feats = FEATURES[equip_type]
    cols = {f["name"]: [] for f in feats}
    labels = []
    for _ in range(n):
        row = {}
        fail_score = 0.0
        for f in feats:
            mid = (f["max"] - f["min"])
            v = rng.uniform(f["min"], f["max"])
            row[f["name"]] = round(v, 3)
            norm = (v - f["min"]) / (f["max"] - f["min"])
            fail_score += norm
        fail_score /= len(feats)
        noise = rng.uniform(-0.15, 0.15)
        label = 1 if (fail_score + noise) > 0.60 else 0
        for k, v in row.items():
            cols[k].append(v)
        labels.append(label)
    df = pd.DataFrame(cols)
    df["failure"] = labels
    return df

# ── Train / cache models ─────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def get_model(equip_type: str):
    path = os.path.join(MODEL_DIR, f"{equip_type}.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"{equip_type}_scaler.pkl")
    if os.path.exists(path):
        return joblib.load(path), joblib.load(scaler_path)
    df = make_dataset(equip_type)
    feat_cols = [f["name"] for f in FEATURES[equip_type]]
    X, y = df[feat_cols].values, df["failure"].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, class_weight="balanced")
    clf.fit(X_s, y)
    joblib.dump(clf, path)
    joblib.dump(scaler, scaler_path)
    return clf, scaler

def predict(equip_type: str, values: dict):
    clf, scaler = get_model(equip_type)
    feat_cols = [f["name"] for f in FEATURES[equip_type]]
    X = np.array([[values[c] for c in feat_cols]])
    X_s = scaler.transform(X)
    prob = clf.predict_proba(X_s)[0][1]
    importances = clf.feature_importances_
    shap_approx = {feat_cols[i]: round(float(importances[i] * (
        (values[feat_cols[i]] - FEATURES[equip_type][i]["min"]) /
        (FEATURES[equip_type][i]["max"] - FEATURES[equip_type][i]["min"])
    )), 4) for i in range(len(feat_cols))}
    return round(float(prob), 4), shap_approx

def risk_label(prob: float) -> str:
    if prob >= 0.75: return "🔴 Critical"
    if prob >= 0.50: return "🟠 High"
    if prob >= 0.25: return "🟡 Elevated"
    return "🟢 Normal"

def risk_color(prob: float) -> str:
    if prob >= 0.75: return "#dc3545"
    if prob >= 0.50: return "#fd7e14"
    if prob >= 0.25: return "#ffc107"
    return "#28a745"

# ── Rule-based diagnosis ─────────────────────────────────────────────────────
def diagnose(equip_type: str, values: dict) -> list[str]:
    triggered = []
    for feat, (thresh, msg) in RULES.get(equip_type, {}).items():
        if feat in values and values[feat] >= thresh:
            triggered.append(f"⚠️ {msg}")
    return triggered if triggered else ["✅ No rule violations detected"]

# ── SQLite feedback store ─────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "feedback.db")

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, equipment TEXT, outcome TEXT,
            actual_fault TEXT, notes TEXT, prob REAL
        )""")
    con.commit(); con.close()

def save_feedback(equipment, outcome, actual_fault, notes, prob):
    init_db()
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT INTO feedback(ts,equipment,outcome,actual_fault,notes,prob) VALUES(?,?,?,?,?,?)",
                (str(datetime.datetime.now()), equipment, outcome, actual_fault, notes, prob))
    con.commit(); con.close()

def load_feedback() -> pd.DataFrame:
    init_db()
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM feedback ORDER BY id DESC LIMIT 50", con)
    con.close()
    return df

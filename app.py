from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, url_for
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "customer_churn_business_dataset.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "customer_churn_pipeline.joblib"
METRICS_PATH = BASE_DIR / "artifacts" / "test_metrics.json"
COMPARISON_PATH = BASE_DIR / "artifacts" / "validation_model_comparison.csv"
IMPORTANCE_PATH = BASE_DIR / "artifacts" / "permutation_importance.csv"

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")

def load_json(path: Path, fallback: dict | None = None) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback or {}

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)

def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

DATA = load_data()
MODEL = load_model()
REPORT = load_json(METRICS_PATH)
THRESHOLD = float(REPORT.get("decision_threshold", 0.5))
TARGET = str(REPORT.get("target", "churn"))
FEATURE_COLUMNS = REPORT.get("feature_columns", [c for c in DATA.columns if c not in ["churn", "customer_id"]])
CATEGORICAL_COLUMNS = DATA[FEATURE_COLUMNS].select_dtypes(exclude=np.number).columns.tolist() if not DATA.empty else []
NUMERIC_COLUMNS = DATA[FEATURE_COLUMNS].select_dtypes(include=np.number).columns.tolist() if not DATA.empty else []

def json_number(value: Any):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if pd.isna(value): return None
    return value

def probability_for(frame: pd.DataFrame) -> np.ndarray:
    if MODEL is None:
        raise RuntimeError("Trained model artifact is missing. Run the notebook first.")
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))
    clean = frame[FEATURE_COLUMNS].copy()
    probabilities = MODEL.predict_proba(clean)
    model_step = MODEL.named_steps["model"] if hasattr(MODEL, "named_steps") else MODEL
    classes = list(model_step.classes_)
    return probabilities[:, classes.index(1)]

def risk_band(prob: float) -> str:
    if prob >= 0.70: return "High"
    if prob >= THRESHOLD: return "Elevated"
    if prob >= 0.30: return "Watch"
    return "Low"

def explain_row(row: pd.Series) -> list[str]:
    # Transparent heuristic signals for follow-up, not SHAP explanations.
    notes = []
    def val(k, default=None): return row.get(k, default)
    if pd.notna(val("last_login_days_ago")) and float(val("last_login_days_ago")) >= 14:
        notes.append("Recent inactivity: review re-engagement options.")
    if pd.notna(val("usage_growth_rate")) and float(val("usage_growth_rate")) < 0:
        notes.append("Usage is declining: consider a guided product walkthrough.")
    if pd.notna(val("payment_failures")) and float(val("payment_failures")) >= 2:
        notes.append("Repeated payment failures: offer billing support.")
    if pd.notna(val("support_tickets")) and float(val("support_tickets")) >= 3:
        notes.append("Multiple support tickets: review unresolved friction.")
    if pd.notna(val("csat_score")) and float(val("csat_score")) <= 2.5:
        notes.append("Low satisfaction score: prioritize a service recovery follow-up.")
    if pd.notna(val("nps_score")) and float(val("nps_score")) <= 4:
        notes.append("Low NPS: ask for feedback and address key concerns.")
    if str(val("contract_type", "")).strip().lower() in {"monthly", "month-to-month", "monthly contract"}:
        notes.append("Flexible contract: review retention benefits before renewal.")
    if not notes:
        notes.append("No configured rule-based warning detected; use the risk score as a prioritization signal.")
    return notes[:5]

def dashboard_payload():
    data = DATA
    churn_rate = float(data[TARGET].mean()) if not data.empty and TARGET in data else 0.0
    comparison = pd.read_csv(COMPARISON_PATH).to_dict(orient="records") if COMPARISON_PATH.exists() else []
    importance = pd.read_csv(IMPORTANCE_PATH).head(12).to_dict(orient="records") if IMPORTANCE_PATH.exists() else []
    test = REPORT.get("test_metrics_at_selected_threshold", {})
    return {
        "dataset": {"rows": int(len(data)), "columns": int(len(data.columns)), "churn_rate": churn_rate,
                    "features": int(len(FEATURE_COLUMNS)), "target": TARGET},
        "metrics": test,
        "threshold": THRESHOLD,
        "model": REPORT.get("selected_model", "Trained classifier"),
        "cv_average_precision": REPORT.get("best_mean_cv_average_precision"),
        "comparison": comparison,
        "importance": importance,
    }

@app.route("/")
def index():
    payload = dashboard_payload()
    metric = payload["metrics"]
    return render_template(
        "index.html", payload=payload,
        accuracy=float(metric.get("Accuracy", 0)),
        precision=float(metric.get("Precision (churn)", 0)),
        recall=float(metric.get("Recall (churn)", 0)),
        f1=float(metric.get("F1 (churn)", 0)),
        model_ready=MODEL is not None,
    )

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", categorical=CATEGORICAL_COLUMNS, numeric=NUMERIC_COLUMNS,
                               categories={c: sorted(DATA[c].dropna().astype(str).unique().tolist()) for c in CATEGORICAL_COLUMNS},
                               medians=DATA[NUMERIC_COLUMNS].median(numeric_only=True).to_dict() if not DATA.empty else {},
                               threshold=THRESHOLD)
    try:
        record = {}
        for col in NUMERIC_COLUMNS:
            raw = request.form.get(col, "").strip()
            record[col] = float(raw) if raw else float(DATA[col].median())
        for col in CATEGORICAL_COLUMNS:
            raw = request.form.get(col, "").strip()
            if not raw:
                mode = DATA[col].mode(dropna=True)
                raw = str(mode.iloc[0]) if not mode.empty else "Unknown"
            record[col] = raw
        row = pd.DataFrame([record], columns=FEATURE_COLUMNS)
        prob = float(probability_for(row)[0])
        result = {
            "probability": prob, "risk": risk_band(prob),
            "prediction": int(prob >= THRESHOLD), "threshold": THRESHOLD,
            "recommendations": explain_row(row.iloc[0])
        }
        return render_template("result.html", result=result, record=record)
    except Exception as exc:
        app.logger.exception("Single prediction failed")
        flash(f"Could not score this customer: {exc}", "error")
        return redirect(url_for("predict"))

@app.route("/batch", methods=["GET", "POST"])
def batch():
    if request.method == "GET":
        return render_template("batch.html", feature_count=len(FEATURE_COLUMNS))
    uploaded = request.files.get("file")
    if not uploaded or not uploaded.filename.lower().endswith(".csv"):
        flash("Upload a CSV file to score.", "error")
        return redirect(url_for("batch"))
    try:
        incoming = pd.read_csv(uploaded)
        missing = [c for c in FEATURE_COLUMNS if c not in incoming.columns]
        if missing:
            raise ValueError("CSV is missing required feature columns: " + ", ".join(missing))
        probs = probability_for(incoming)
        out = incoming.copy()
        out["churn_probability"] = probs
        out["risk_band"] = [risk_band(float(p)) for p in probs]
        out["predicted_churn"] = (probs >= THRESHOLD).astype(int)
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        return send_file(io.BytesIO(csv_bytes), mimetype="text/csv", as_attachment=True,
                         download_name="churn_predictions.csv")
    except Exception as exc:
        app.logger.exception("Batch prediction failed")
        flash(f"Batch prediction failed: {exc}", "error")
        return redirect(url_for("batch"))

@app.route("/api/dashboard")
def api_dashboard():
    return jsonify(dashboard_payload())

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not request.is_json:
        return jsonify({"error": "Send JSON with a 'customers' array or one customer object."}), 400
    body = request.get_json()
    customers = body.get("customers") if isinstance(body, dict) else None
    if customers is None and isinstance(body, dict):
        customers = [body]
    if not isinstance(customers, list) or not customers:
        return jsonify({"error": "Provide a non-empty customer object or customers array."}), 400
    try:
        frame = pd.DataFrame(customers)
        probs = probability_for(frame)
        results = []
        for i, prob in enumerate(probs):
            results.append({
                "probability": float(prob), "risk_band": risk_band(float(prob)),
                "predicted_churn": int(prob >= THRESHOLD), "threshold": THRESHOLD,
                "recommendations": explain_row(frame.iloc[i])
            })
        return jsonify({"results": results})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

@app.route("/health")
def health():
    return jsonify({
        "status": "ok" if MODEL is not None else "model_missing",
        "model_loaded": MODEL is not None,
        "dataset_loaded": not DATA.empty,
        "feature_count": len(FEATURE_COLUMNS),
        "threshold": THRESHOLD,
    }), (200 if MODEL is not None else 503)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1",
            host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "5000")))

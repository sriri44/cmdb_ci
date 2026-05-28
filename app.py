"""
=============================================================
MODULE 5: Flask Web Application — app.py
AI-Powered Intelligent CMDB Management System
=============================================================
Main web application providing upload, prediction, approval,
ServiceNow insertion, and chatbot routes.
=============================================================
"""

import os
import io
import json
import logging
import pandas as pd
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, session, jsonify
)
from werkzeug.utils import secure_filename

from predict import predict_dataframe, predict_single
from insert_servicenow import bulk_insert, upsert_ci
from chatbot import chat as chatbot_chat

# -----------------------------------------------------------
# App setup
# -----------------------------------------------------------
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER  = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTS   = {"csv", "xlsx", "xls"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "cmdb-ai-secret-2024")
app.config["UPLOAD_FOLDER"]    = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# In-memory store for current session data
_store = {"df": None, "predictions": [], "sn_results": []}


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


def load_file(filepath: str) -> pd.DataFrame:
    ext = filepath.rsplit(".", 1)[1].lower()
    if ext == "csv":
        return pd.read_csv(filepath)
    return pd.read_excel(filepath)


def df_to_records(df: pd.DataFrame) -> list:
    return json.loads(df.to_json(orient="records"))


# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------

@app.route("/")
def index():
    stats = {
        "total_uploaded": len(_store["df"]) if _store["df"] is not None else 0,
        "total_predicted": len(_store["predictions"]),
        "total_inserted":  sum(1 for r in _store["sn_results"] if r.get("status") == "success"),
        "total_errors":    sum(1 for r in _store["sn_results"] if r.get("status") == "error"),
    }
    return render_template("index.html", stats=stats)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part in request.", "danger")
            return redirect(request.url)

        f = request.files["file"]
        if f.filename == "":
            flash("No file selected.", "warning")
            return redirect(request.url)

        if not allowed_file(f.filename):
            flash("Unsupported file type. Please upload CSV or Excel.", "danger")
            return redirect(request.url)

        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        f.save(filepath)

        try:
            df = load_file(filepath)
            # Normalise column names
            df.columns = [c.strip().lower() for c in df.columns]
            _store["df"]          = df
            _store["predictions"] = []
            _store["sn_results"]  = []
            flash(f"✅ File '{filename}' uploaded — {len(df)} records loaded.", "success")
            return redirect(url_for("predict_view"))
        except Exception as e:
            flash(f"Error reading file: {e}", "danger")
            return redirect(request.url)

    return render_template("upload.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_view():
    if _store["df"] is None:
        flash("Please upload a file first.", "warning")
        return redirect(url_for("upload"))

    if request.method == "POST":
        try:
            df_pred = predict_dataframe(_store["df"])
            _store["df"] = df_pred
            records = df_to_records(df_pred)
            _store["predictions"] = records
            flash(f"✅ Predicted CI classes for {len(records)} records.", "success")
        except Exception as e:
            flash(f"Prediction error: {e}", "danger")

    records = _store.get("predictions") or df_to_records(_store["df"])
    columns = list(_store["df"].columns)
    return render_template("prediction.html", records=records, columns=columns)


@app.route("/insert", methods=["POST"])
def insert():
    approved_indices = request.form.getlist("approve")
    if not approved_indices:
        flash("No records approved for insertion.", "warning")
        return redirect(url_for("predict_view"))

    approved_indices = [int(i) for i in approved_indices]
    records = _store.get("predictions", [])
    to_insert = [records[i] for i in approved_indices if i < len(records)]

    if not to_insert:
        flash("No valid records to insert.", "warning")
        return redirect(url_for("predict_view"))

    try:
        summary = bulk_insert(to_insert)
        _store["sn_results"] = summary["results"]
        flash(
            f"✅ Insert complete — {summary['success_count']} succeeded, "
            f"{summary['error_count']} failed.",
            "success" if summary["error_count"] == 0 else "warning"
        )
    except Exception as e:
        flash(f"Insert error: {e}", "danger")

    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    results = _store.get("sn_results", [])
    predictions = _store.get("predictions", [])

    # Class distribution for chart
    class_counts = {}
    for p in predictions:
        cls = p.get("predicted_class", "unknown")
        class_counts[cls] = class_counts.get(cls, 0) + 1

    # Environment distribution
    env_counts = {}
    for p in predictions:
        env = p.get("environment", "unknown")
        env_counts[env] = env_counts.get(env, 0) + 1

    stats = {
        "total":    len(predictions),
        "success":  sum(1 for r in results if r.get("status") == "success"),
        "errors":   sum(1 for r in results if r.get("status") == "error"),
        "class_counts": class_counts,
        "env_counts":   env_counts,
    }
    return render_template("dashboard.html", stats=stats, results=results)


@app.route("/chatbot")
def chatbot_page():
    return render_template("chatbot.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data    = request.get_json(force=True)
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400
    try:
        response = chatbot_chat(message)
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict_single", methods=["POST"])
def api_predict_single():
    data   = request.get_json(force=True)
    result = predict_single(data)
    return jsonify(result)

# -----------------------------------------------------------
# Run
# -----------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

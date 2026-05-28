"""
=============================================================
MODULE 3: Prediction Engine — predict.py
AI-Powered Intelligent CMDB Management System
=============================================================
Loads the trained model and predicts CI class for single or
bulk input records. Returns class label, confidence score,
and the target ServiceNow CMDB table name.
=============================================================
"""

import os
import logging
import pandas as pd
import numpy as np
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cmdb_classifier.pkl")

# -----------------------------------------------------------
# ServiceNow CMDB table mapping
# -----------------------------------------------------------
CMDB_TABLE_MAP = {
    "server":      "cmdb_ci_server",
    "database":    "cmdb_ci_database",
    "network":     "cmdb_ci_netgear",
    "cloud":       "cmdb_ci_vm_instance",
    "application": "cmdb_ci_appl",
    "storage":     "cmdb_ci_storage_device",
}


# -----------------------------------------------------------
# Model loader (singleton pattern)
# -----------------------------------------------------------
_artifact = None

def _load_artifact():
    global _artifact
    if _artifact is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train_model.py first."
            )
        _artifact = joblib.load(MODEL_PATH)
        logger.info("Model artifact loaded successfully.")
    return _artifact


# -----------------------------------------------------------
# Encode a single record
# -----------------------------------------------------------
def _encode_row(row: dict, encoders: dict, feature_cols: list) -> np.ndarray:
    """Transform raw string fields into encoded feature vector."""
    vec = []
    for col in feature_cols:
        val = str(row.get(col, "unknown")).lower()
        le  = encoders[col]
        # Handle unseen labels gracefully
        if val in le.classes_:
            vec.append(le.transform([val])[0])
        else:
            vec.append(0)   # fallback to index 0
    return np.array(vec).reshape(1, -1)


# -----------------------------------------------------------
# Single prediction
# -----------------------------------------------------------
def predict_single(record: dict) -> dict:
    """
    Predict CI class for one record.

    Parameters
    ----------
    record : dict  with keys: os, manufacturer, description, environment

    Returns
    -------
    dict with: predicted_class, confidence, snow_table, all_probabilities
    """
    try:
        artifact     = _load_artifact()
        clf          = artifact["model"]
        encoders     = artifact["encoders"]
        feature_cols = artifact["features"]

        X         = _encode_row(record, encoders, feature_cols)
        pred_idx  = clf.predict(X)[0]
        proba     = clf.predict_proba(X)[0]
        confidence = float(np.max(proba))

        pred_class = encoders["class"].inverse_transform([pred_idx])[0]
        snow_table = CMDB_TABLE_MAP.get(pred_class, "cmdb_ci")

        all_proba = {
            encoders["class"].inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(proba)
        }

        return {
            "predicted_class": pred_class,
            "confidence":      round(confidence, 4),
            "snow_table":      snow_table,
            "all_probabilities": all_proba,
            "status":          "success"
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"status": "error", "error": str(e)}


# -----------------------------------------------------------
# Bulk prediction
# -----------------------------------------------------------
def predict_bulk(records: list) -> list:
    """
    Predict CI class for a list of records.

    Parameters
    ----------
    records : list of dicts

    Returns
    -------
    list of prediction result dicts (same structure as predict_single)
    """
    results = []
    for i, rec in enumerate(records):
        result = predict_single(rec)
        result["record_index"] = i
        result["name"] = rec.get("name", f"record_{i}")
        results.append(result)
        logger.info(
            f"[{i+1}/{len(records)}] {result.get('name')} → "
            f"{result.get('predicted_class')} "
            f"({result.get('confidence', 0)*100:.1f}%)"
        )
    return results


# -----------------------------------------------------------
# DataFrame wrapper (for Flask uploads)
# -----------------------------------------------------------
def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run bulk prediction on a pandas DataFrame.
    Adds columns: predicted_class, confidence, snow_table.
    """
    records = df.to_dict(orient="records")
    preds   = predict_bulk(records)

    df = df.copy()
    df["predicted_class"] = [p.get("predicted_class", "unknown") for p in preds]
    df["confidence"]      = [p.get("confidence", 0.0) for p in preds]
    df["snow_table"]      = [p.get("snow_table", "cmdb_ci") for p in preds]
    return df


# -----------------------------------------------------------
# CLI test
# -----------------------------------------------------------
if __name__ == "__main__":
    test_record = {
        "name":         "test-server-99",
        "os":           "Red Hat Enterprise Linux 8",
        "manufacturer": "Dell",
        "description":  "Apache web server",
        "environment":  "production",
        "ip_address":   "10.0.1.99"
    }
    result = predict_single(test_record)
    print("\n=== Single Prediction Result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

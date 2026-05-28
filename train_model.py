"""
=============================================================
MODULE 2: ML Model Training — train_model.py
AI-Powered Intelligent CMDB Management System
=============================================================
Trains a RandomForestClassifier to predict CI class from
infrastructure attributes. Saves model + encoders via joblib.
=============================================================
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import logging

# -----------------------------------------------------------
# Logging
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "datasets", "cmdb_training_data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "cmdb_classifier.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# Features used for classification
FEATURE_COLS = ["os", "manufacturer", "description", "environment"]
TARGET_COL   = "class"


# -----------------------------------------------------------
# 1. Load & validate dataset
# -----------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} records with columns: {list(df.columns)}")

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Fill any NaN values with 'unknown'
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna("unknown")
    logger.info(f"Class distribution:\n{df[TARGET_COL].value_counts()}")
    return df


# -----------------------------------------------------------
# 2. Feature engineering
# -----------------------------------------------------------
def engineer_features(df: pd.DataFrame):
    """
    Encode categorical text columns to numeric using LabelEncoder.
    Returns encoded feature matrix, target array, and encoders dict.
    """
    logger.info("Engineering features …")
    encoders = {}
    df_enc = df.copy()

    for col in FEATURE_COLS:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str).str.lower())
        encoders[col] = le
        logger.info(f"  Encoded '{col}' → {len(le.classes_)} unique values")

    # Target encoder
    le_target = LabelEncoder()
    df_enc[TARGET_COL] = le_target.fit_transform(df_enc[TARGET_COL].astype(str).str.lower())
    encoders["class"] = le_target

    X = df_enc[FEATURE_COLS].values
    y = df_enc[TARGET_COL].values
    return X, y, encoders


# -----------------------------------------------------------
# 3. Train model
# -----------------------------------------------------------
def train(X, y):
    logger.info("Splitting data 80/20 train/test …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info("Training RandomForestClassifier …")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    logger.info("Training complete.")
    return clf, X_test, y_test


# -----------------------------------------------------------
# 4. Evaluate model
# -----------------------------------------------------------
def evaluate(clf, X_test, y_test, encoders):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"\n{'='*50}")
    logger.info(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    class_names = encoders["class"].classes_
    logger.info(f"\nClassification Report:\n"
                f"{classification_report(y_test, y_pred, target_names=class_names)}")

    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    return acc


# -----------------------------------------------------------
# 5. Save artifacts
# -----------------------------------------------------------
def save_model(clf, encoders):
    artifact = {"model": clf, "encoders": encoders, "features": FEATURE_COLS}
    joblib.dump(artifact, MODEL_PATH)
    logger.info(f"Model saved → {MODEL_PATH}")


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
if __name__ == "__main__":
    df          = load_data(DATA_PATH)
    X, y, encs  = engineer_features(df)
    clf, Xt, yt = train(X, y)
    evaluate(clf, Xt, yt, encs)
    save_model(clf, encs)
    logger.info("✅  Model training pipeline complete.")

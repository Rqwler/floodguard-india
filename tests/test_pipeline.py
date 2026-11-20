"""Offline regression tests for the FloodGuard data and model pipeline.

Run with:
    python -m unittest discover -s tests -v

These tests do not call the network and never send real email. SMTP delivery is
forced into simulation mode so the full alert path can be exercised safely.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.alerts import evaluate_alerts, subscribe, unsubscribe
from app.config import SEQUENCE_LENGTH
from app.emails import render_alert_email
from app.features import (
    FEATURES,
    add_labels,
    build_sequence_dataset,
    compute_thresholds,
    fit_scaler,
    prepare_features,
)
from app.regions import REGIONS


def sample_payload(level: int = 2) -> dict:
    current_level = min(level, 2)
    return {
        "id": "assam-brahmaputra",
        "name": "Assam",
        "area": "Brahmaputra Valley",
        "river": "Brahmaputra",
        "station": "Tezpur",
        "thresholds": {"watch": 100.0, "warning": 200.0, "severe": 300.0, "median": 50.0},
        "current": {
            "date": "2026-11-17",
            "discharge": 215.0,
            "rain_7d": 118.0,
            "observed_level": current_level,
            "observed_score": 78.0,
        },
        "forecast": [
            {
                "date": "2026-11-18",
                "level": level,
                "label": ["Normal", "Watch", "Warning", "Severe"][level],
                "risk_score": 82.0,
                "alert_probability": 91.0,
                "discharge": 245.0,
                "rain": 28.0,
            },
            {
                "date": "2026-11-19",
                "level": 1,
                "label": "Watch",
                "risk_score": 51.0,
                "alert_probability": 35.0,
                "discharge": 180.0,
                "rain": 8.0,
            },
        ],
    }


def sample_frame(days: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    seasonal = np.sin(2 * np.pi * np.arange(days) / 365.25)
    discharge = 900 + 350 * seasonal + rng.normal(0, 45, days)
    return pd.DataFrame(
        {
            "date": dates,
            "discharge": np.clip(discharge, 50, None),
            "discharge_mean": discharge,
            "discharge_median": discharge,
            "discharge_max": discharge * 1.05,
            "discharge_min": discharge * 0.95,
            "precip": rng.gamma(0.5, 8, days),
            "temp_mean": 27 + 3 * seasonal,
            "temp_max": 32 + 3 * seasonal,
            "temp_min": 21 + 3 * seasonal,
            "et0": 4.5 + seasonal,
            "wind_max": 11 + rng.gamma(1.5, 2, days),
            "soil_moisture": 0.28 + 0.06 * seasonal,
        }
    )


class RegionTests(unittest.TestCase):
    def test_exactly_five_unique_basins(self) -> None:
        self.assertEqual(len(REGIONS), 5)
        self.assertEqual(len({region.id for region in REGIONS}), 5)
        self.assertEqual(len({(region.latitude, region.longitude) for region in REGIONS}), 5)
        self.assertEqual(
            {region.name for region in REGIONS},
            {"Assam", "Bihar", "Kerala", "Mumbai", "West Bengal"},
        )


class FeatureTests(unittest.TestCase):
    def test_features_are_finite_and_sequence_dataset_predicts_next_day(self) -> None:
        raw = sample_frame()
        features = prepare_features(raw)
        self.assertEqual(len(features), len(raw))
        self.assertEqual(list(features[FEATURES].columns), FEATURES)
        self.assertFalse(features[FEATURES].isna().any().any())
        self.assertTrue((features["target_discharge"].iloc[:-1].to_numpy()
                         == features["discharge"].shift(-1).iloc[:-1].to_numpy()).all())

        thresholds = compute_thresholds(features["discharge"].iloc[:120])
        labelled = add_labels(features, thresholds)
        scaler = fit_scaler(labelled.iloc[:120])
        x, y, regions, dates = build_sequence_dataset(
            labelled, scaler, 0, sequence_length=SEQUENCE_LENGTH
        )
        self.assertEqual(x.shape[1:], (SEQUENCE_LENGTH, len(FEATURES)))
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(x), len(regions))
        self.assertEqual(len(x), len(dates))
        self.assertTrue(((x >= -10) & (x <= 10)).all())

    def test_features_are_causal(self) -> None:
        raw = sample_frame()
        changed = raw.copy()
        junction = len(changed) - 30
        changed.loc[junction:, "discharge"] *= 4.0
        base_features = prepare_features(raw)
        changed_features = prepare_features(changed)
        pd.testing.assert_frame_equal(
            base_features.loc[: junction - 1, FEATURES],
            changed_features.loc[: junction - 1, FEATURES],
        )

    def test_thresholds_increase_and_labels_follow_them(self) -> None:
        thresholds = compute_thresholds(sample_frame()["discharge"])
        self.assertEqual(thresholds["watch"], thresholds["q80"])
        self.assertEqual(thresholds["warning"], thresholds["q92"])
        self.assertEqual(thresholds["severe"], thresholds["q97"])
        self.assertLess(thresholds["watch"], thresholds["warning"])
        self.assertLess(thresholds["warning"], thresholds["severe"])
        frame = sample_frame().iloc[:4].copy()
        frame["target_discharge"] = [
            thresholds["watch"] * 0.5,
            (thresholds["watch"] + thresholds["warning"]) / 2,
            (thresholds["warning"] + thresholds["severe"]) / 2,
            thresholds["severe"] * 1.1,
        ]
        labels = add_labels(frame, thresholds)["label"].to_numpy()
        self.assertEqual(labels.tolist(), [0, 1, 2, 3])


class EmailTests(unittest.TestCase):
    def test_alert_email_has_recipient_context_and_unsubscribe(self) -> None:
        subject, html_body, text_body = render_alert_email(
            sample_payload(),
            sample_payload()["forecast"][0],
            recipient="person@example.com",
            token="signed-token",
        )
        self.assertIn("WARNING", subject)
        self.assertIn("Assam", subject)
        self.assertIn("Warning flood risk", html_body)
        self.assertIn("api/unsubscribe?token=signed-token", html_body)
        self.assertIn("api/unsubscribe?token=signed-token", text_body)


class AlertPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.db_path = root / "test-floodguard.db"
        self.outbox = root / "outbox"
        self.outbox.mkdir()

        import app.db as db

        self.db_patch = patch.object(db, "DB_PATH", self.db_path)
        self.db_patch.start()
        import app.emails as emails

        self.outbox_patch = patch.object(emails, "OUTBOX_DIR", self.outbox)
        self.outbox_patch.start()
        self.smtp_patch = patch("app.emails.smtp_configured", return_value=False)
        self.smtp_patch.start()
        db.init_db()

    def tearDown(self) -> None:
        self.smtp_patch.stop()
        self.outbox_patch.stop()
        self.db_patch.stop()
        self.temp_dir.cleanup()

    def test_normal_conditions_do_not_trigger_an_alert(self) -> None:
        subscribe("watcher@example.com", ["assam-brahmaputra"])
        payload = sample_payload(level=1)
        produced = evaluate_alerts([payload])
        self.assertEqual(produced, [])
        self.assertEqual(list(self.outbox.glob("alert_*.html")), [])

    def test_warning_sends_email_once_and_marks_subscriber(self) -> None:
        subscription = subscribe("watcher@example.com", ["assam-brahmaputra"])
        self.assertTrue(subscription["active"])

        produced = evaluate_alerts([sample_payload(level=2)])
        self.assertEqual(len(produced), 1)
        record = produced[0]
        self.assertEqual(record["level_label"], "Warning")
        self.assertEqual(record["status"], "simulated")
        self.assertEqual(record["recipients"], 1)

        messages = list(self.outbox.glob("alert_*.html"))
        self.assertEqual(len(messages), 1)
        message = messages[0].read_text(encoding="utf-8")
        self.assertIn("watcher@example.com", message)
        self.assertIn("Warning flood risk", message)
        self.assertIn("api/unsubscribe?token=", message)

        # The same warning must not create a duplicate message on the next cycle.
        self.assertEqual(evaluate_alerts([sample_payload(level=2)]), [])
        self.assertEqual(len(list(self.outbox.glob("alert_*.html"))), 1)

    def test_unsubscribe_uses_signed_token(self) -> None:
        subscription = subscribe("watcher@example.com", ["assam-brahmaputra", "bihar-ganga"])
        import app.db as db

        token = db.get_subscriber_by_email(subscription["email"])["token"]
        result = unsubscribe(token)
        self.assertIsNotNone(result)
        self.assertEqual(result["email"], "watcher@example.com")
        self.assertFalse(unsubscribe(token))

    def test_invalid_addresses_and_empty_region_lists_are_normalised(self) -> None:
        with self.assertRaises(ValueError):
            subscribe("not-an-email", [])
        subscription = subscribe("all@example.com", [])
        self.assertEqual(len(subscription["regions"]), 5)


class ModelArtifactTests(unittest.TestCase):
    def test_saved_model_predicts_four_classes(self) -> None:
        from app.config import METADATA_PATH, MODEL_PATH
        from app.model import load_model_bundle, predict_probabilities

        if not MODEL_PATH.exists() or not METADATA_PATH.exists():
            self.skipTest("trained model artifact is not present")
        bundle = load_model_bundle()
        self.assertIsNotNone(bundle)
        metadata = bundle["metadata"]
        self.assertEqual(metadata["sequence_length"], SEQUENCE_LENGTH)
        self.assertEqual(len(metadata["regions"]), 5)
        self.assertEqual(json.loads(json.dumps(metadata))["model_config"]["num_classes"], 4)
        # Two arbitrary windows for two different basins.
        windows = np.zeros((2, SEQUENCE_LENGTH, len(FEATURES)), dtype=np.float32)
        regions = np.array([0, 4], dtype=np.int64)
        probabilities = predict_probabilities(bundle, windows, regions)
        self.assertEqual(probabilities.shape, (2, 4))
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2), rtol=1e-5)


class LiveInferenceTests(unittest.TestCase):
    def test_cached_live_window_returns_eight_day_outlook(self) -> None:
        from app.config import LIVE_DIR
        from app.inference import FloodPredictor

        csv_path = LIVE_DIR / "assam-brahmaputra.csv"
        if not csv_path.exists():
            self.skipTest("no cached live data on this machine")
        frame = pd.read_csv(csv_path, parse_dates=["date"])
        region = next(region for region in REGIONS if region.id == "assam-brahmaputra")
        prediction = FloodPredictor().predict_region(region, frame)
        self.assertEqual(len(prediction["forecast"]), 8)
        self.assertGreaterEqual(len(prediction["series"]), 60)
        self.assertIn(prediction["active"]["label"], {"Normal", "Watch", "Warning", "Severe"})
        self.assertIn("rain_24h", prediction["current"])
        current_date = pd.Timestamp(prediction["current"]["date"])
        self.assertGreaterEqual(current_date, frame["date"].min())
        self.assertLessEqual(current_date, frame["date"].max())
        self.assertLessEqual(prediction["active"]["level"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

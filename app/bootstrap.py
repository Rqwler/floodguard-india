"""Command-line bootstrap for FloodGuard India.

Typical usage::

    python -m app.bootstrap --check
    python -m app.bootstrap --train
    python -m app.bootstrap --refresh
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _check() -> int:
    from .config import DB_PATH, METADATA_PATH, MODEL_PATH
    from .inference import FloodPredictor

    predictor = FloodPredictor()
    print("FloodGuard India — system check")
    print(f"  model artifact : {'found' if MODEL_PATH.exists() else 'missing'}  ({MODEL_PATH})")
    print(f"  metadata       : {'found' if METADATA_PATH.exists() else 'missing'}  ({METADATA_PATH})")
    print(f"  database       : {'found' if DB_PATH.exists() else 'not created yet'}  ({DB_PATH})")
    print(f"  predictor mode : {'BiLSTM ensemble' if predictor.model_ready else 'hydrology prior'}")
    if predictor.metadata:
        test = predictor.metadata.get("metrics", {}).get("test", {})
        print(
            "  hold-out test  : "
            f"accuracy {test.get('accuracy', 0) * 100:.1f}% · "
            f"macro-F1 {test.get('macro_f1', 0) * 100:.1f}%"
        )
        print(f"  trained at     : {predictor.metadata.get('trained_at')}")
    return 0 if predictor.model_ready else 1


def _train(args: argparse.Namespace) -> int:
    from .model import train_model

    result = train_model(
        force_fetch=args.force_data,
        epochs=args.epochs,
        patience=args.patience,
        verbose=True,
    )
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    return 0


async def _refresh() -> None:
    from .config import ensure_directories
    from .db import init_db
    from .engine import engine

    ensure_directories()
    init_db()
    dashboard = await engine.refresh(force_fetch=True, reason="bootstrap")
    print("\nLive refresh complete")
    print(f"  generated : {dashboard['generated_at']}")
    print(f"  source    : {dashboard['data_source']['status']}")
    for region in dashboard["regions"]:
        active = region["active"]
        print(
            f"  {region['name']:<12} observed {region['current']['observed_label']:<8} "
            f"active {active['label']:<8} score {active['score']:>5} "
            f"data {region['data_status']}"
        )


def _refresh_command(args: argparse.Namespace) -> int:
    asyncio.run(_refresh())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="FloodGuard India bootstrap and maintenance CLI")
    parser.add_argument("--check", action="store_true", help="verify that the model artifacts are present")
    parser.add_argument("--train", action="store_true", help="train and save the BiLSTM model")
    parser.add_argument("--refresh", action="store_true", help="fetch live data, run inference and write the dashboard cache")
    parser.add_argument("--force-data", action="store_true", help="with --train, re-download history even when cached")
    parser.add_argument("--epochs", type=int, default=45, help="maximum training epochs (default: 45)")
    parser.add_argument("--patience", type=int, default=8, help="early-stopping patience (default: 8)")
    parser.add_argument("--json", action="store_true", help="print the full training result as JSON")
    args = parser.parse_args()

    if args.train:
        return _train(args)
    if args.refresh:
        return _refresh_command(args)
    return _check()


if __name__ == "__main__":
    raise SystemExit(main())

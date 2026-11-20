"""BiLSTM + temporal-attention model and training pipeline."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    DROPOUT,
    HIDDEN_SIZE,
    IST,
    METADATA_PATH,
    MODEL_PATH,
    MODEL_DIR,
    NUM_LAYERS,
    REGION_EMBEDDING,
    SEQUENCE_LENGTH,
    ensure_directories,
)
from .data import fetch_history
from .features import (
    FEATURES,
    FEATURE_LABELS,
    add_labels,
    build_sequence_dataset,
    compute_thresholds,
    fit_scaler,
    prepare_features,
    scale_features,
)
from .regions import REGIONS, REGION_IDS, REGION_INDEX


class FloodLSTM(nn.Module):
    """Region-aware bidirectional LSTM with additive temporal attention."""

    def __init__(
        self,
        num_features: int,
        num_regions: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        embedding_size: int = REGION_EMBEDDING,
        dropout: float = DROPOUT,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.region_embedding = nn.Embedding(num_regions, embedding_size)
        self.input_norm = nn.LayerNorm(num_features)
        self.lstm = nn.LSTM(
            input_size=num_features + embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor, region_idx: torch.Tensor) -> torch.Tensor:
        region_vector = self.region_embedding(region_idx).unsqueeze(1)
        region_vector = region_vector.expand(-1, x.size(1), -1)
        x = torch.cat([self.input_norm(x), region_vector], dim=-1)
        outputs, _ = self.lstm(x)
        attention_logits = self.attention(outputs)
        weights = torch.softmax(attention_logits, dim=1)
        context = torch.sum(outputs * weights, dim=1)
        return self.head(context)


def _seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    losses, all_targets, all_predictions, all_probabilities = [], [], [], []
    with torch.no_grad():
        for x, region, y in loader:
            x, region, y = x.to(device), region.to(device), y.to(device)
            logits = model(x, region)
            loss = criterion(logits, y)
            probabilities = torch.softmax(logits, dim=-1)
            losses.append(float(loss.item()))
            all_targets.extend(y.cpu().numpy().tolist())
            all_predictions.extend(probabilities.argmax(dim=-1).cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())
    if not all_targets:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "targets": [],
            "predictions": [],
            "probabilities": [],
        }
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(all_targets, all_predictions)),
        "macro_f1": float(f1_score(all_targets, all_predictions, average="macro", zero_division=0)),
        "targets": all_targets,
        "predictions": all_predictions,
        "probabilities": all_probabilities,
    }


def _metrics_block(result: dict[str, Any]) -> dict[str, Any]:
    targets = result["targets"]
    predictions = result["predictions"]
    labels = [0, 1, 2, 3]
    if targets:
        report = classification_report(
            targets,
            predictions,
            labels=labels,
            target_names=["Normal", "Watch", "Warning", "Severe"],
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(targets, predictions, labels=labels).tolist()
    else:
        report = {}
        matrix = [[0, 0, 0, 0] for _ in labels]
    return {
        "loss": round(result["loss"], 4),
        "accuracy": round(result["accuracy"], 4),
        "macro_f1": round(result["macro_f1"], 4),
        "classification_report": report,
        "confusion_matrix": matrix,
        "samples": len(targets),
    }


def _class_weights(labels: np.ndarray, num_classes: int = 4) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = len(labels) / (num_classes * counts)
    weights = np.clip(weights, 0.4, 10.0)
    return (weights / weights.mean()).astype(np.float32)


def train_model(
    force_fetch: bool = False,
    epochs: int = 45,
    patience: int = 8,
    verbose: bool = True,
) -> dict[str, Any]:
    """Fetch data, engineer features, train and persist the model artifact."""
    ensure_directories()
    _seed_everything(42)
    started = time.time()
    device = torch.device("cpu")

    region_frames: dict[str, pd.DataFrame] = {}
    thresholds: dict[str, dict[str, float]] = {}
    scalers: dict[str, dict[str, dict[str, float]]] = {}
    metadata_sources: dict[str, dict[str, Any]] = {}

    if verbose:
        print("\n[1/5] Fetching historical hydrology and weather…")
    for region in REGIONS:
        raw, meta = fetch_history(region, force=force_fetch)
        frame = prepare_features(raw)
        region_frames[region.id] = frame
        metadata_sources[region.id] = meta
        if verbose:
            print(
                f"  {region.name:<12} {meta.get('status', '?'):<6} "
                f"{len(frame):>5} rows  {meta.get('start')} → {meta.get('end')}"
            )

    all_dates = sorted(
        {
            pd.Timestamp(value)
            for frame in region_frames.values()
            for value in frame["date"].tolist()
        }
    )
    if len(all_dates) < 400:
        raise RuntimeError("Insufficient history to train the model")
    train_end = all_dates[min(len(all_dates) - 1, int(len(all_dates) * 0.80))]
    val_end = all_dates[min(len(all_dates) - 1, int(len(all_dates) * 0.90))]
    if verbose:
        print(f"\n[2/5] Split — train ≤ {train_end.date()}, validation ≤ {val_end.date()}, test after")

    for region in REGIONS:
        frame = region_frames[region.id]
        training_slice = frame[frame["date"] <= train_end]
        thresholds[region.id] = compute_thresholds(training_slice["discharge"])
        scalers[region.id] = fit_scaler(training_slice)
        region_frames[region.id] = add_labels(frame, thresholds[region.id])

    train_x, train_r, train_y, train_dates = [], [], [], []
    val_x, val_r, val_y, val_dates = [], [], [], []
    test_x, test_r, test_y, test_dates = [], [], [], []

    for region in REGIONS:
        x, y, r, target_dates = build_sequence_dataset(
            region_frames[region.id],
            scalers[region.id],
            REGION_INDEX[region.id],
            SEQUENCE_LENGTH,
        )
        for sample_x, sample_y, sample_region, target_date in zip(x, y, r, target_dates):
            if target_date <= train_end:
                train_x.append(sample_x)
                train_r.append(sample_region)
                train_y.append(sample_y)
                train_dates.append(target_date)
            elif target_date <= val_end:
                val_x.append(sample_x)
                val_r.append(sample_region)
                val_y.append(sample_y)
                val_dates.append(target_date)
            else:
                test_x.append(sample_x)
                test_r.append(sample_region)
                test_y.append(sample_y)
                test_dates.append(target_date)

    if not train_x or not val_x or not test_x:
        raise RuntimeError("Chronological split produced an empty train/validation/test set")

    X_train = np.stack(train_x).astype(np.float32)
    R_train = np.asarray(train_r, dtype=np.int64)
    y_train = np.asarray(train_y, dtype=np.int64)
    X_val = np.stack(val_x).astype(np.float32)
    R_val = np.asarray(val_r, dtype=np.int64)
    y_val = np.asarray(val_y, dtype=np.int64)
    X_test = np.stack(test_x).astype(np.float32)
    R_test = np.asarray(test_r, dtype=np.int64)
    y_test = np.asarray(test_y, dtype=np.int64)

    class_counts = {int(label): int(count) for label, count in enumerate(np.bincount(y_train, minlength=4))}
    if verbose:
        print(
            f"\n[3/5] Sequences — train {len(X_train):,}, val {len(X_val):,}, "
            f"test {len(X_test):,}; class counts {class_counts}"
        )

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(R_train),
            torch.from_numpy(y_train),
        ),
        batch_size=256,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(R_val),
            torch.from_numpy(y_val),
        ),
        batch_size=512,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_test),
            torch.from_numpy(R_test),
            torch.from_numpy(y_test),
        ),
        batch_size=512,
        shuffle=False,
    )

    config = {
        "num_features": len(FEATURES),
        "num_regions": len(REGIONS),
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "embedding_size": REGION_EMBEDDING,
        "dropout": DROPOUT,
        "num_classes": 4,
        "bidirectional": True,
    }
    model = FloodLSTM(
        num_features=config["num_features"],
        num_regions=config["num_regions"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        embedding_size=config["embedding_size"],
        dropout=config["dropout"],
    ).to(device)
    weights = _class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_macro_f1": [], "val_accuracy": []}
    best_state = None
    best_epoch = 0
    best_f1 = -1.0
    best_val_loss = float("inf")
    stale_epochs = 0

    if verbose:
        print(f"\n[4/5] Training BiLSTM on {device} for up to {epochs} epochs…")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, region_idx, y in train_loader:
            x, region_idx, y = x.to(device), region_idx.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, region_idx)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.item()) * len(x)
            seen += len(x)
        scheduler.step()
        train_loss = running_loss / max(1, seen)
        val = _evaluate(model, val_loader, criterion, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val["loss"], 4))
        history["val_macro_f1"].append(round(val["macro_f1"], 4))
        history["val_accuracy"].append(round(val["accuracy"], 4))
        improved = val["macro_f1"] > best_f1 + 1e-4 or (
            abs(val["macro_f1"] - best_f1) <= 1e-4 and val["loss"] < best_val_loss
        )
        if improved:
            best_f1 = val["macro_f1"]
            best_val_loss = val["loss"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if verbose and (epoch == 1 or epoch % 5 == 0 or improved or epoch == epochs):
            marker = " *" if improved else ""
            print(
                f"  epoch {epoch:>2}/{epochs}  train {train_loss:.4f}  "
                f"val {val['loss']:.4f}  macro-F1 {val['macro_f1']:.4f}{marker}"
            )
        if stale_epochs >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch}; best epoch {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a model state")
    model.load_state_dict(best_state)
    model.eval()

    if verbose:
        print("\n[5/5] Final evaluation and artifact export…")
    val_result = _evaluate(model, val_loader, criterion, device)
    test_result = _evaluate(model, test_loader, criterion, device)
    val_metrics = _metrics_block(val_result)
    test_metrics = _metrics_block(test_result)

    trained_at = datetime.now(IST).isoformat(timespec="seconds")
    statuses = [meta.get("status", "unknown") for meta in metadata_sources.values()]
    if all(status == "live" for status in statuses):
        data_mode = "live"
    elif any(status == "demo" for status in statuses):
        data_mode = "demo"
    elif any(status == "cached" for status in statuses):
        data_mode = "cached"
    else:
        data_mode = "live"

    metadata = {
        "version": "1.0.0",
        "trained_at": trained_at,
        "training_seconds": round(time.time() - started, 1),
        "data_mode": data_mode,
        "source": "Open-Meteo (GloFAS river discharge + ERA5-Land weather)",
        "regions": [region.to_dict() for region in REGIONS],
        "region_index": REGION_INDEX,
        "feature_names": FEATURES,
        "feature_labels": FEATURE_LABELS,
        "sequence_length": SEQUENCE_LENGTH,
        "model_config": config,
        "thresholds": thresholds,
        "scalers": scalers,
        "split": {
            "train_end": str(train_end.date()),
            "val_end": str(val_end.date()),
            "train_samples": int(len(X_train)),
            "validation_samples": int(len(X_val)),
            "test_samples": int(len(X_test)),
            "train_class_counts": class_counts,
        },
        "metrics": {
            "validation": val_metrics,
            "test": test_metrics,
        },
        "history": history,
        "best_epoch": best_epoch,
        "epochs_run": len(history["epoch"]),
        "data_sources": metadata_sources,
        "class_weights": weights.tolist(),
        "notes": (
            "Labels are derived from region-specific historical river-discharge "
            "quantiles (q80 watch, q92 warning, q97 severe). Validation and test "
            "sets are chronologically later than training data."
        ),
    }

    torch.save({"state_dict": model.state_dict(), "config": config}, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if verbose:
        print(
            f"  Validation accuracy {val_metrics['accuracy']:.3f} | macro-F1 {val_metrics['macro_f1']:.3f}\n"
            f"  Test accuracy       {test_metrics['accuracy']:.3f} | macro-F1 {test_metrics['macro_f1']:.3f}\n"
            f"  Saved {MODEL_PATH} and {METADATA_PATH}\n"
        )
    return metadata


def load_model_bundle(
    model_path=MODEL_PATH,
    metadata_path=METADATA_PATH,
) -> dict[str, Any] | None:
    if not model_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(model_path, map_location="cpu")
    config = checkpoint.get("config") or metadata.get("model_config", {})
    model = FloodLSTM(
        num_features=config["num_features"],
        num_regions=config["num_regions"],
        hidden_size=config.get("hidden_size", HIDDEN_SIZE),
        num_layers=config.get("num_layers", NUM_LAYERS),
        embedding_size=config.get("embedding_size", REGION_EMBEDDING),
        dropout=config.get("dropout", DROPOUT),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {"model": model, "metadata": metadata, "config": config}


def predict_probabilities(
    bundle: dict[str, Any],
    scaled_sequences: np.ndarray,
    region_indices: np.ndarray,
) -> np.ndarray:
    model: FloodLSTM = bundle["model"]
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(scaled_sequences.astype(np.float32)),
            torch.from_numpy(region_indices.astype(np.int64)),
        )
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
    return probabilities


if __name__ == "__main__":
    train_model()

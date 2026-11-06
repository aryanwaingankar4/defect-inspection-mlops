"""Fine-tune a timm ImageNet model on the NEU split (TASK-008).

Full training runs on Kaggle GPU only (ADR-019). On a laptop without a GPU, only
``--smoke`` (a tiny end-to-end check) runs unless ``--allow-cpu`` is passed.

    python -m src.training.train --config configs/params.yaml [--smoke] [--allow-cpu]
        [--raw-dir DIR] [--split-file CSV] [--out-checkpoint PT] [--out-dir DIR] [--num-workers N]

No MLflow yet (TASK-009): metrics, the training curve and provenance (git commit, DVC data
hash, data fingerprint, params) are written to ``--out-dir``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.config import load_params
from src.serving.preprocess import PreprocessConfig
from src.training.augment import AugmentConfig
from src.training.dataset import SplitDataset, class_names
from src.training.metrics import accuracy, macro_f1, per_class_f1

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_FORMAT = 1
EXIT_NEEDS_GPU = 2


# ---------------------------------------------------------------- config + CLI


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, help="override data.raw_dir")
    parser.add_argument("--split-file", type=Path, help="override data.split_file")
    parser.add_argument("--out-checkpoint", type=Path, help="override train.out_checkpoint")
    parser.add_argument("--out-dir", type=Path, help="override train.out_dir")
    parser.add_argument("--num-workers", type=int, help="override train.num_workers")
    parser.add_argument("--smoke", action="store_true", help="tiny run; allowed on CPU")
    parser.add_argument("--allow-cpu", action="store_true", help="allow a full run without GPU")
    return parser.parse_args(argv)


def apply_overrides(params: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Return a copy of params with smoke settings, then CLI overrides, applied."""
    p = copy.deepcopy(params)
    t = p["train"]
    if args.smoke:
        smoke_dir = Path(t["smoke"]["out_dir"])
        t["max_epochs"] = int(t["smoke"]["max_epochs"])
        t["out_checkpoint"] = (smoke_dir / "best.pt").as_posix()
        t["out_dir"] = (smoke_dir / "results").as_posix()
    if args.raw_dir is not None:
        p["data"]["raw_dir"] = args.raw_dir.as_posix()
    if args.split_file is not None:
        p["data"]["split_file"] = args.split_file.as_posix()
    if args.out_checkpoint is not None:
        t["out_checkpoint"] = args.out_checkpoint.as_posix()
    if args.out_dir is not None:
        t["out_dir"] = args.out_dir.as_posix()
    if args.num_workers is not None:
        t["num_workers"] = args.num_workers
    return p


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_device(device: torch.device, *, smoke: bool, allow_cpu: bool) -> bool:
    """True if training may start. Full runs without a GPU need ``--allow-cpu`` (ADR-019)."""
    if device.type == "cuda" or smoke:
        return True
    print(
        "WARNING: no GPU found. Full training runs on Kaggle GPU only (ADR-019); on this "
        "machine it may run out of memory. Use --smoke for a tiny check, or pass --allow-cpu "
        "to force a full CPU run.",
        file=sys.stderr,
    )
    return allow_cpu


# ---------------------------------------------------------------- building blocks


def set_seeds(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_model(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """timm model with a new ``num_classes`` head; the whole network is trainable."""
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def balanced_sampler(labels: Sequence[int], generator: torch.Generator) -> WeightedRandomSampler:
    """Each class is drawn equally often on average (weight = 1 / class count)."""
    counts = np.bincount(np.asarray(labels))
    weights = torch.as_tensor([1.0 / counts[y] for y in labels], dtype=torch.double)
    return WeightedRandomSampler(weights, len(labels), replacement=True, generator=generator)


class EarlyStopping:
    """Track the best score; stop after ``patience`` epochs without a strict improvement."""

    def __init__(self, patience: int) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.patience = patience
        self.best = float("-inf")
        self.bad_epochs = 0

    def update(self, score: float) -> bool:
        """Record a score; return True if it is a new best."""
        if score > self.best:
            self.best, self.bad_epochs = score, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device
) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


@torch.inference_mode()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, np.ndarray, np.ndarray]:
    """Mean cross-entropy, true labels and predicted labels over the loader."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total, y_true, y_pred = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        total += loss_fn(logits, y).item()
        y_true.append(y.cpu().numpy())
        y_pred.append(logits.argmax(dim=1).cpu().numpy())
    true, pred = np.concatenate(y_true), np.concatenate(y_pred)
    return total / len(true), true, pred


# ---------------------------------------------------------------- provenance + outputs


def git_commit(repo_root: Path) -> dict[str, Any]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"sha": "unknown", "dirty": None}
    return {"sha": sha, "dirty": bool(status)}


def dvc_data_md5(repo_root: Path) -> str | None:
    """The DVC hash of ``data/raw`` from the committed ``data/raw.dvc`` (the data version)."""
    dvc_file = repo_root / "data" / "raw.dvc"
    if not dvc_file.exists():
        return None
    outs = yaml.safe_load(dvc_file.read_text(encoding="utf-8")).get("outs", [])
    return outs[0].get("md5") if outs else None


def data_fingerprint(raw_dir: Path, rel_paths: Sequence[str]) -> str:
    """SHA-256 over (path, file SHA-256) of the images actually used, in sorted order.

    Comparable across machines (laptop DVC copy vs Kaggle dataset) for the same split.
    """
    h = hashlib.sha256()
    for rel in sorted(rel_paths):
        h.update(rel.encode("utf-8"))
        h.update(hashlib.sha256((raw_dir / rel).read_bytes()).digest())
    return h.hexdigest()


def write_curve(rows: list[dict[str, float]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (out_dir / "training_curve.csv").open("w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    epochs = [r["epoch"] for r in rows]
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(epochs, [r["train_loss"] for r in rows], label="train")
    ax_loss.plot(epochs, [r["val_loss"] for r in rows], label="val")
    ax_loss.set(xlabel="epoch", ylabel="cross-entropy", title="Loss")
    ax_loss.legend()
    ax_f1.plot(epochs, [r["val_macro_f1"] for r in rows], marker="o")
    ax_f1.set(xlabel="epoch", ylabel="macro-F1", title="Validation macro-F1")
    fig.tight_layout()
    fig.savefig(out_dir / "training_curve.png", dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------- main loop


def run(
    params: dict[str, Any],
    device: torch.device,
    *,
    smoke: bool = False,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Train with early stopping on val macro-F1. Returns the metrics written to JSON."""
    t = params["train"]
    set_seeds(int(t["seed"]))
    raw_dir, split_csv = Path(params["data"]["raw_dir"]), Path(params["data"]["split_file"])
    pre_cfg = PreprocessConfig.from_params(params)
    classes = class_names(split_csv, t["train_splits"])
    per_class = int(t["smoke"]["images_per_class"]) if smoke else None

    common = {"split_csv": split_csv, "raw_dir": raw_dir, "classes": classes, "preprocess": pre_cfg}
    train_ds = SplitDataset(
        splits=t["train_splits"],
        augment_cfg=AugmentConfig.from_params(params),
        images_per_class=per_class,
        **common,
    )
    val_ds = SplitDataset(splits=[t["val_split"]], images_per_class=per_class, **common)

    gen = torch.Generator().manual_seed(int(t["seed"]))
    loader_kw = {
        "batch_size": int(t["batch_size"]),
        "num_workers": int(t["num_workers"]),
        "pin_memory": device.type == "cuda",
        "worker_init_fn": _seed_worker,
        "persistent_workers": int(t["num_workers"]) > 0,
    }
    # Drop a ragged last batch (it could hold 1 image, which BatchNorm cannot train on).
    # The sampler draws with replacement, so no image is systematically left out.
    train_loader = DataLoader(
        train_ds,
        sampler=balanced_sampler(train_ds.labels, gen),
        generator=gen,
        drop_last=len(train_ds) > int(t["batch_size"]),
        **loader_kw,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    model = build_model(t["model"], len(classes), bool(t["pretrained"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(t["lr"]), weight_decay=float(t["weight_decay"])
    )

    ckpt_path, out_dir = Path(t["out_checkpoint"]), Path(t["out_dir"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit(repo_root)
    stopper = EarlyStopping(int(t["patience"]))
    curve: list[dict[str, float]] = []
    best: dict[str, Any] = {}

    for epoch in range(1, int(t["max_epochs"]) + 1):
        start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, y_true, y_pred = predict(model, val_loader, device)
        f1 = macro_f1(y_true, y_pred, len(classes))
        seconds = time.perf_counter() - start
        curve.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": f1,
                "seconds": seconds,
            }
        )
        improved = stopper.update(f1)
        print(
            f"epoch {epoch:3d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
            f"val_macro_f1 {f1:.4f}  {seconds:.1f}s{'  *best*' if improved else ''}",
            flush=True,
        )
        if improved:
            best = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_macro_f1": f1,
                "val_accuracy": accuracy(y_true, y_pred),
                "val_per_class_f1": dict(
                    zip(classes, per_class_f1(y_true, y_pred, len(classes)).tolist(), strict=True)
                ),
            }
            torch.save(
                {
                    "format_version": CHECKPOINT_FORMAT,
                    "model_name": t["model"],
                    "classes": list(classes),
                    "state_dict": model.state_dict(),
                    "preprocess": {
                        k: list(v) if isinstance(v, tuple) else v
                        for k, v in asdict(pre_cfg).items()
                    },
                    "best_epoch": epoch,
                    "val_macro_f1": f1,
                    "seed": int(t["seed"]),
                    "git_commit": commit["sha"],
                },
                ckpt_path,
            )
        if stopper.should_stop:
            break

    metrics = {
        "best_epoch": best["epoch"],
        "val_macro_f1": best["val_macro_f1"],
        "val_accuracy": best["val_accuracy"],
        "val_loss": best["val_loss"],
        "val_per_class_f1": best["val_per_class_f1"],
        "epochs_run": len(curve),
        "stopped_early": stopper.should_stop,
        "classes": list(classes),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "smoke": smoke,
        "device": device.type,
        "checkpoint": ckpt_path.as_posix(),
        "provenance": {
            "git_commit": commit["sha"],
            "git_dirty": commit["dirty"],
            "dvc_data_md5": dvc_data_md5(repo_root),
            "data_fingerprint": data_fingerprint(raw_dir, train_ds.paths + val_ds.paths),
            "torch": torch.__version__,
            "timm": timm.__version__,
        },
        "params": {"data": params["data"], "preprocess": params["preprocess"], "train": t},
    }
    (out_dir / "val_metrics.json").write_bytes(
        (json.dumps(metrics, indent=2) + "\n").encode("utf-8")
    )
    write_curve(curve, out_dir)
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    params = apply_overrides(load_params(args.config), args)
    device = resolve_device()
    if not check_device(device, smoke=args.smoke, allow_cpu=args.allow_cpu):
        return EXIT_NEEDS_GPU
    print(f"device: {device.type}{'  (smoke run)' if args.smoke else ''}", flush=True)
    metrics = run(params, device, smoke=args.smoke)
    print(
        f"best epoch {metrics['best_epoch']}: val macro-F1 {metrics['val_macro_f1']:.4f}; "
        f"outputs in {params['train']['out_dir']} and {metrics['checkpoint']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

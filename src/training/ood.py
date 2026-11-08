"""Energy OOD threshold on val and evaluation on synthetic non-steel images (TASK-017, ADR-027).

    python -m src.training.ood --config configs/params.yaml [--log-mlflow]

CPU only, ONNX Runtime on the serving model (the file named by the ``@champion`` version's
``serving_precision`` tag, checked by SHA-256). ``T`` and the review threshold come from the
version tags. Scores come from the shared ``src/serving/scoring.py``.

Rule (ADR-027): flag the ``ceil(ood.id_flag_budget * n)`` val images with the highest energy;
the threshold is the midpoint to the next one (no labels). Routing = low confidence OR OOD.
Synthetic images (``src/data/synthetic_ood.py``) only evaluate; they choose nothing.

Writes ``ood.json`` and ``ood_score_histogram.png`` into the main model's results_dir;
``--log-mlflow`` adds them to the champion's run and tags the version with
``ood_energy_threshold`` for serving.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from src.config import load_params
from src.data.synthetic_ood import generate
from src.serving.onnx_model import OnnxClassifier
from src.serving.preprocess import PreprocessConfig, load_image, to_model_input
from src.serving.scoring import Thresholds, energy, score
from src.training.compare_val import specs_from_params
from src.training.dataset import PROTECTED_SPLIT, SplitDataset
from src.training.evaluate import (
    PLOT_DIR,
    EvalConfig,
    load_temperature,
    near_copy_groups_within,
    unit_ids,
)
from src.training.export_onnx import ExportConfig, champion
from src.training.metrics import (
    auroc,
    budget_threshold,
    clopper_pearson_lower,
    clopper_pearson_upper,
    fpr_at_tpr,
    grouped_bootstrap_indices,
    percentile_interval,
)
from src.training.train import git_commit, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
PLOT_NAME = "ood_score_histogram.png"  # allowlisted (ADR-018)


class OodError(RuntimeError):
    """The OOD threshold must not be set (wrong model file, held-out split)."""


@dataclass(frozen=True)
class OodConfig:
    id_flag_budget: float
    batch_size: int
    n_resamples: int
    seed: int
    level: float
    detection_level: float
    tpr: float
    synthetic_seed: int
    n_per_family: int
    families: tuple[str, ...]
    out_file: str

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> OodConfig:
        o, b, s = params["ood"], params["ood"]["bootstrap"], params["ood"]["synthetic"]
        return cls(
            id_flag_budget=float(o["id_flag_budget"]),
            batch_size=int(o["batch_size"]),
            n_resamples=int(b["n_resamples"]),
            seed=int(b["seed"]),
            level=float(b["level"]),
            detection_level=float(o["detection_level"]),
            tpr=float(o["fpr_at_tpr"]),
            synthetic_seed=int(s["seed"]),
            n_per_family=int(s["n_per_family"]),
            families=tuple(s["families"]),
            out_file=str(o["out_file"]),
        )


# ---------------------------------------------------------------- inputs


def serving_onnx(params: Mapping[str, Any], champ: Mapping[str, Any]) -> tuple[Path, str]:
    """The local ONNX file named by ``serving_precision``, checked against its version tag."""
    precision = champ["tags"].get("serving_precision")
    if precision == "fp32":
        report, key, tag = ExportConfig.from_params(params).report_file, "onnx_file", "onnx_sha256"
    elif precision == "int8":
        report, key, tag = Path(params["quantize"]["report_file"]), "int8_file", "onnx_int8_sha256"
    else:
        raise OodError(f"@{champ['alias']} has no valid serving_precision tag (ADR-026)")
    path = Path(json.loads(report.read_text(encoding="utf-8"))[key])
    if not path.is_file():
        raise OodError(f"missing {path}: run `make export-onnx` / `make quantize` first")
    if sha256_file(path) != champ["tags"].get(tag):
        raise OodError(f"{path} is not the {precision} file tagged on @{champ['alias']}")
    return path, precision


def threshold_for(energies: np.ndarray, budget: float) -> tuple[float, int]:
    """Flag the ``ceil(budget * n)`` highest energies; midpoint threshold (flag: energy > t)."""
    neg_t, m = budget_threshold(-np.asarray(energies), budget)
    return -neg_t, m


# ---------------------------------------------------------------- reports


def detection(flagged: np.ndarray, routed: np.ndarray, level: float) -> dict[str, Any]:
    n, k = len(flagged), int(flagged.sum())
    return {
        "n": int(n),
        "flagged_ood": k,
        "flagged_share": k / n,
        "flagged_share_lower_bound": clopper_pearson_lower(k, n, level),
        "routed_any_reason": int(routed.sum()),
        "routed_share": float(routed.mean()),
    }


def val_report(
    y: np.ndarray, s: Any, ucb_level: float, threshold: float, n_flagged: int
) -> dict[str, Any]:
    wrong = s.pred != y
    acc_conf, acc_union = ~s.low_confidence, ~s.routed
    k_conf, k_union = int((wrong & acc_conf).sum()), int((wrong & acc_union).sum())
    return {
        "n_images": int(len(y)),
        "n_errors": int(wrong.sum()),
        "ood_threshold": threshold,
        "flagged_ood": int(n_flagged),
        "routed_low_confidence": int(s.low_confidence.sum()),
        "routed_both_reasons": int((s.low_confidence & s.ood).sum()),
        "routed_ood_only": int((s.ood & ~s.low_confidence).sum()),
        "routed_any_reason": int(s.routed.sum()),
        "routed_share": float(s.routed.mean()),
        "errors_routed_any_reason": int((wrong & s.routed).sum()),
        "errors_flagged_ood": int((wrong & s.ood).sum()),
        "low_confidence_only": {
            "n_accepted": int(acc_conf.sum()),
            "errors_accepted": k_conf,
            "risk_ucb": clopper_pearson_upper(k_conf, int(acc_conf.sum()), ucb_level),
        },
        "combined_rule": {
            "n_accepted": int(acc_union.sum()),
            "errors_accepted": k_union,
            "risk_ucb": clopper_pearson_upper(k_union, int(acc_union.sum()), ucb_level),
        },
        "ucb_level": ucb_level,
    }


def separation(
    id_energy: np.ndarray,
    ood_energy: np.ndarray,
    id_conf: np.ndarray,
    ood_conf: np.ndarray,
    id_energy_t1: np.ndarray,
    ood_energy_t1: np.ndarray,
    tpr: float,
) -> dict[str, float]:
    """AUROC (OOD = positive, higher score = more OOD) and FPR at ``tpr`` detection."""
    return {
        "auroc_energy": auroc(id_energy, ood_energy),
        "fpr_at_tpr_energy": fpr_at_tpr(id_energy, ood_energy, tpr),
        "auroc_energy_t1_reference": auroc(id_energy_t1, ood_energy_t1),
        "auroc_confidence_reference": auroc(-id_conf, -ood_conf),
    }


def claims(val: Mapping[str, Any], pooled: Mapping[str, Any], cfg: OodConfig) -> dict:
    return {
        "can_claim": [
            f"On val, {val['flagged_ood']} of {val['n_images']} in-distribution images are "
            f"flagged as OOD by construction ({cfg.id_flag_budget:.0%} budget); with low "
            f"confidence, {val['routed_any_reason']} images ({val['routed_share']:.1%}) go to "
            "review.",
            f"On {pooled['n']} synthetic non-steel images, {pooled['flagged_ood']} are flagged "
            f"as OOD (at least {pooled['flagged_share_lower_bound']:.1%} with "
            f"{cfg.detection_level:.0%} confidence); {pooled['routed_any_reason']} go to review "
            "for either reason.",
        ],
        "cannot_claim": [
            "Anything about unseen steel defect types or other steel lines (near-OOD): the "
            "synthetic images are far from steel. X-SDD evaluates that later (ADR-010).",
            "The false-alarm rate on new in-distribution images: on val it is fixed by the "
            "budget; test_frozen reports it (evaluate.py, report only).",
            f"A precise threshold: it is set by the {val['flagged_ood']} highest-energy val "
            "images; see the bootstrap interval.",
        ],
    }


def plot_histogram(id_e: np.ndarray, ood: Mapping[str, np.ndarray], t: float, path: Path) -> None:
    """Energy histograms: val (in-distribution) vs synthetic families. No images."""
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    allv = np.concatenate([id_e, *ood.values()])
    lo = min(float(id_e.min()), float(np.percentile(allv, 1)))  # keep every val image in view
    hi = float(allv.max())
    off = int((allv < lo).sum())
    bins = np.linspace(lo, hi, 50)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(id_e, bins=bins, alpha=0.6, color="tab:blue", label=f"val (steel), n={len(id_e)}")
    for name, e in ood.items():
        e = e[e >= lo]
        ax.hist(e, bins=bins, histtype="step", lw=1.2, label=f"synthetic: {name}")
    ax.axvline(t, color="tab:red", ls="--", label=f"OOD threshold {t:.2f} (flag if above)")
    ax.set(
        xlabel="energy = -T logsumexp(logits / T)  (higher = less familiar)",
        ylabel="images per bin",
        xlim=(lo, hi + 0.02 * (hi - lo)),
    )
    title = "Energy OOD score: val vs synthetic non-steel images"
    if off:
        title += f"\n{off} synthetic images with energy below {lo:.1f} are off the left edge"
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------- run


def batched_logits(session: OnnxClassifier, x: np.ndarray, batch: int) -> np.ndarray:
    return np.concatenate([session.logits(x[i : i + batch]) for i in range(0, len(x), batch)])


def run(params: Mapping[str, Any], *, log_mlflow: bool = False) -> dict[str, Any]:
    cfg = OodConfig.from_params(params)
    val_split = str(params["train"]["val_split"])
    if val_split == PROTECTED_SPLIT:
        raise OodError(f"the OOD threshold never uses '{PROTECTED_SPLIT}' (ADR-027)")
    champ = champion(params)
    onnx_path, precision = serving_onnx(params, champ)
    spec = specs_from_params(params["compare_val"])[0]
    temperature_cal, cal_path = load_temperature(params, spec)
    temperature = float(champ["tags"]["temperature"])
    if temperature != temperature_cal:
        raise OodError("the champion's temperature tag differs from its calibration.json")
    review_threshold = float(champ["tags"]["review_threshold"])
    session = OnnxClassifier(onnx_path)
    classes = session.classes
    pre = PreprocessConfig.from_params(params)

    split_csv = Path(params["data"]["split_file"])
    ds = SplitDataset(split_csv, Path(params["data"]["raw_dir"]), [val_split], classes, pre)
    batches = [(x.numpy(), y.numpy()) for x, y in DataLoader(ds, batch_size=cfg.batch_size)]
    y = np.concatenate([b[1] for b in batches])
    val_logits = batched_logits(session, np.concatenate([b[0] for b in batches]), cfg.batch_size)
    val_e = energy(val_logits, temperature)
    ood_threshold, n_flagged = threshold_for(val_e, cfg.id_flag_budget)
    th = Thresholds(temperature, review_threshold, ood_threshold)
    s_val = score(val_logits, th)
    ucb_level = float(params["threshold"]["ucb_level"])
    val = val_report(y, s_val, ucb_level, ood_threshold, n_flagged)

    groups = near_copy_groups_within(ds.paths, EvalConfig.from_params(params))
    samples = grouped_bootstrap_indices(y, unit_ids(ds.paths, groups), cfg.n_resamples, cfg.seed)
    q = 1.0 - cfg.id_flag_budget
    lo, hi = percentile_interval(np.array([np.quantile(val_e[i], q) for i in samples]), cfg.level)
    val["ood_threshold_interval"] = {
        "ci_low": float(lo),
        "ci_high": float(hi),
        "level": cfg.level,
        "note": f"{q:.0%} quantile of val energy on grouped bootstrap resamples (approximation)",
    }

    synth = generate(cfg.families, cfg.n_per_family, cfg.synthetic_seed)
    sx = np.stack([to_model_input(load_image(png), pre) for _, png in synth])
    fam = np.array([f for f, _ in synth])
    s_logits = batched_logits(session, sx, cfg.batch_size)
    s_syn = score(s_logits, th)
    s_e_t1, val_e_t1 = energy(s_logits, 1.0), energy(val_logits, 1.0)
    per_family, energies = {}, {}
    for name in cfg.families:
        m = fam == name
        energies[name] = s_syn.energy[m]
        per_family[name] = {
            **detection(s_syn.ood[m], s_syn.routed[m], cfg.detection_level),
            **separation(
                val_e,
                s_syn.energy[m],
                s_val.confidence,
                s_syn.confidence[m],
                val_e_t1,
                s_e_t1[m],
                cfg.tpr,
            ),
            "predicted_as": dict(Counter(classes[i] for i in s_syn.pred[m]).most_common()),
        }
    pooled = {
        **detection(s_syn.ood, s_syn.routed, cfg.detection_level),
        **separation(
            val_e, s_syn.energy, s_val.confidence, s_syn.confidence, val_e_t1, s_e_t1, cfg.tpr
        ),
    }
    commit = git_commit(REPO_ROOT)
    result = {
        "decision_record": "ADR-027",
        "registered_model": champ["name"],
        "version": champ["version"],
        "alias": champ["alias"],
        "run_id": champ["run_id"],
        "ood_energy_threshold": ood_threshold,
        "score": "energy = -T * logsumexp(logits / T); flagged if energy > threshold",
        "route_rule": "review if calibrated confidence < review_threshold OR energy > threshold",
        "rule": {
            "id_flag_budget": cfg.id_flag_budget,
            "choice": "ceil(budget * n) highest val energies flagged; midpoint threshold",
        },
        "temperature": temperature,
        "review_threshold": review_threshold,
        "serving_precision": precision,
        "serving_onnx_file": onnx_path.as_posix(),
        "serving_onnx_sha256": sha256_file(onnx_path),
        "calibration_sha256": sha256_file(cal_path),
        "fitted_on": {
            "split": val_split,
            "split_file": split_csv.name,
            "split_sha256": sha256_file(split_csv),
            "n_images": int(len(y)),
        },
        "val": val,
        "synthetic": {
            "note": "generated in code (src/data/synthetic_ood.py); no NEU pixels; evaluation only",
            "seed": cfg.synthetic_seed,
            "n_per_family": cfg.n_per_family,
            "detection_level": cfg.detection_level,
            "fpr_at_tpr": cfg.tpr,
            "pooled": pooled,
            "per_family": per_family,
        },
        "energy_summary": {
            "val": [float(np.min(val_e)), float(np.median(val_e)), float(np.max(val_e))],
            "synthetic": [
                float(np.min(s_syn.energy)),
                float(np.median(s_syn.energy)),
                float(np.max(s_syn.energy)),
            ],
            "order": "min, median, max",
        },
        "claims": claims(val, pooled, cfg),
        "provenance": {"git_commit": commit["sha"], "git_dirty": commit["dirty"], "device": "cpu"},
    }
    out_file = spec.results_dir / cfg.out_file
    plot = spec.results_dir / PLOT_NAME
    out_file.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    plot_histogram(val_e, energies, ood_threshold, plot)
    out = {"result": result, "out": out_file, "plot": plot}
    if log_mlflow:
        out["mlflow_run"] = log_to_mlflow(params, champ, out_file, plot, result)
    return out


def log_to_mlflow(
    params: Mapping[str, Any],
    champ: Mapping[str, Any],
    out_file: Path,
    plot: Path,
    result: Mapping[str, Any],
) -> str:
    from src.tracking import mlflow_tracking as mt  # only with --log-mlflow

    cfg = mt.TrackingConfig.from_params(params, smoke=False)
    val, pooled = result["val"], result["synthetic"]["pooled"]
    link = mt.add_to_finished_run(
        cfg,
        champ["run_id"],
        metrics={
            "ood_energy_threshold": result["ood_energy_threshold"],
            "val_ood_flagged": float(val["flagged_ood"]),
            "val_routed_any_reason": float(val["routed_any_reason"]),
            "val_combined_risk_ucb": val["combined_rule"]["risk_ucb"],
            "synthetic_ood_flagged_share": pooled["flagged_share"],
            "synthetic_routed_share": pooled["routed_share"],
            "synthetic_auroc_energy": pooled["auroc_energy"],
            "synthetic_auroc_confidence_reference": pooled["auroc_confidence_reference"],
        },
        tags={
            "ood_decision_record": "ADR-027",
            "ood_git_commit": result["provenance"]["git_commit"],
        },
        files=[(plot, PLOT_DIR), (out_file, "ood")],
    )
    mt.set_version_tags(
        cfg, champ["version"], {"ood_energy_threshold": repr(result["ood_energy_threshold"])}
    )
    return link


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="add the result and histogram to the champion's run; tag ood_energy_threshold",
    )
    args = parser.parse_args(argv)
    try:
        out = run(load_params(args.config), log_mlflow=args.log_mlflow)
    except OodError as exc:
        print(f"ood: {exc}", file=sys.stderr)
        return 1
    r = out["result"]
    print(
        f"OOD energy threshold {r['ood_energy_threshold']:.4f} "
        f"(T = {r['temperature']:.4f}, {r['serving_precision']})"
    )
    for line in r["claims"]["can_claim"]:
        print(f"  {line}")
    for name, f in r["synthetic"]["per_family"].items():
        print(
            f"    {name:15s} flagged {f['flagged_ood']:3d}/{f['n']}  "
            f"routed {f['routed_any_reason']:3d}  AUROC energy {f['auroc_energy']:.3f} "
            f"(confidence {f['auroc_confidence_reference']:.3f})"
        )
    print(f"  wrote {out['out'].as_posix()} (+ {PLOT_NAME})")
    if "mlflow_run" in out:
        print(f"  MLflow: {out['mlflow_run']} (+ version tag ood_energy_threshold)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

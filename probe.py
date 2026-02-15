#!/usr/bin/env python3
"""Stage 3a: Train logistic regression probes on cached activations.

Reads a JSONL dataset + matching NPZ activation cache (produced by extract.py),
then runs a leave-one-out logistic regression sweep across layers and K%
truncation points. Outputs timestamped plots + results table.

Usage:
  python probe.py pooh --k-percents 10 25 50 75 100
  python probe.py has_false_starts
  python probe.py has_false_starts --model Qwen/Qwen3-4B
"""

import argparse
from datetime import datetime
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lib import (
    DEFAULT_LAYER_FRACS,
    DEFAULT_MODEL,
    activation_cache_dir,
    dataset_name_from_path,
    infer_num_layers,
    load_activations,
    open_plot,
    read_jsonl,
    resolve_layer_indices,
)


DEFAULT_K_PERCENTS = [10, 25, 50, 75, 90, 100]

# Text-classifier baselines (TF-IDF and embedding-based, from collaborator)
# Format: {dataset_name: {k_percent: (tfidf_acc, tfidf_auc, embed_acc, embed_auc)}}
# Gemini zero-shot baselines (from collaborator)
# Format: {dataset_name: {k_percent: accuracy}}
GEMINI_BASELINES = {
    "explores_hypotheses": {
        10: 0.6833, 25: 0.7033, 50: 0.7100, 75: 0.7267, 90: 0.7733,
    },
    "has_false_starts": {
        10: 0.6150, 25: 0.7025, 50: 0.6800, 75: 0.7375, 90: 0.7600,
    },
}

TEXT_BASELINES = {
    "has_false_starts": {
        10:  (0.590, 0.663, 0.603, 0.667),
        25:  (0.675, 0.734, 0.608, 0.635),
        50:  (0.670, 0.714, 0.585, 0.627),
        75:  (0.678, 0.710, 0.583, 0.604),
        90:  (0.680, 0.722, 0.580, 0.607),
    },
    "explores_hypotheses": {
        10:  (0.707, 0.785, 0.633, 0.657),
        25:  (0.740, 0.807, 0.627, 0.670),
        50:  (0.730, 0.804, 0.630, 0.666),
        75:  (0.720, 0.792, 0.600, 0.626),
        90:  (0.717, 0.794, 0.593, 0.624),
    },
}


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def mean_pool_at_percent(activations: np.ndarray, k_percent: float) -> np.ndarray:
    """Mean-pool the first K% of thinking tokens."""
    seq_len = activations.shape[0]
    n = max(1, ceil(k_percent / 100.0 * seq_len))
    n = min(n, seq_len)
    return activations[:n].mean(axis=0)


def _bootstrap_auc_se(y: np.ndarray, probs: np.ndarray, n_boot: int = 1000) -> float:
    """Bootstrap standard error of AUC from LOO predictions."""
    n = len(y)
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            aucs[b] = np.nan
            continue
        aucs[b] = roc_auc_score(y[idx], probs[idx])
    return float(np.nanstd(aucs))


def loo_cv(X: np.ndarray, y: np.ndarray, C: float) -> tuple[float, float, float, float]:
    """Leave-one-out CV returning (accuracy, AUC, acc_se, auc_se)."""
    n = len(y)
    preds = np.empty(n)
    probs = np.empty(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=C, max_iter=1000, solver="lbfgs")),
        ])
        pipe.fit(X[mask], y[mask])
        preds[i] = pipe.predict(X[i : i + 1])[0]
        probs[i] = pipe.predict_proba(X[i : i + 1])[0, 1]

    acc = (preds == y).mean()
    auc = roc_auc_score(y, probs)
    acc_se = np.sqrt(acc * (1 - acc) / n)
    auc_se = _bootstrap_auc_se(y, probs)
    return acc, auc, acc_se, auc_se


def sweep(
    acts: dict[str, dict[int, np.ndarray]],
    records: list[dict],
    layer_indices: list[int],
    k_percents: list[float],
    C: float,
) -> dict:
    """
    Sweep layers x K% for one dataset. Returns dict with:
      results[(layer_label, k)] = (acc, auc, acc_se, auc_se)
    where layer_label is int (single layer) or "all" (concat).
    """
    valid = [r for r in records if r["id"] in acts]
    if len(valid) < len(records):
        print(f"  Using {len(valid)}/{len(records)} examples (rest have no cached activations)")

    labels = np.array([int(r["label"]) for r in valid])
    results = {}

    total = (len(layer_indices) + 1) * len(k_percents)
    done = 0

    # Per-layer sweeps
    for layer_idx in layer_indices:
        for k in k_percents:
            X = np.stack([
                mean_pool_at_percent(acts[r["id"]][layer_idx].astype(np.float32), k)
                for r in valid
            ])
            acc, auc, acc_se, auc_se = loo_cv(X, labels, C)
            results[(layer_idx, k)] = (acc, auc, acc_se, auc_se)
            done += 1
            print(f"  [{done}/{total}] layer={layer_idx:2d}, K={k:3.0f}% -> acc={acc:.3f}+-{acc_se:.3f}, AUC={auc:.3f}+-{auc_se:.3f}")

    # All layers concatenated
    for k in k_percents:
        X = np.stack([
            np.concatenate([
                mean_pool_at_percent(acts[r["id"]][l].astype(np.float32), k)
                for l in layer_indices
            ])
            for r in valid
        ])
        acc, auc, acc_se, auc_se = loo_cv(X, labels, C)
        results[("all", k)] = (acc, auc, acc_se, auc_se)
        done += 1
        print(f"  [{done}/{total}] layer=all, K={k:3.0f}% -> acc={acc:.3f}+-{acc_se:.3f}, AUC={auc:.3f}+-{auc_se:.3f}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    results: dict, ds_name: str, k_percents: list[float],
    layer_indices: list[int], output_path: Path,
):
    """Single-panel accuracy vs K% plot."""
    # Blue shades for per-layer lines (all activations → same hue family)
    colors = [plt.cm.Blues(0.35 + 0.65 * i / max(1, len(layer_indices) - 1))
              for i in range(len(layer_indices))]

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, layer_idx in enumerate(layer_indices):
        accs = [results[(layer_idx, k)][0] for k in k_percents]
        acc_ses = [results[(layer_idx, k)][2] for k in k_percents]
        ax.errorbar(k_percents, accs, yerr=acc_ses, marker="o", markersize=4,
                    color=colors[i], capsize=3, label=f"Layer {layer_idx}")

    # All layers
    accs = [results[("all", k)][0] for k in k_percents]
    acc_ses = [results[("all", k)][2] for k in k_percents]
    ax.errorbar(k_percents, accs, yerr=acc_ses, marker="s", markersize=5,
                color="red", linewidth=2, capsize=3, label="All layers")

    # Overlay text-classifier baselines if available
    baselines = TEXT_BASELINES.get(ds_name)
    if baselines:
        bk = sorted(baselines)
        tfidf_accs = [baselines[k][0] for k in bk]
        embed_accs = [baselines[k][2] for k in bk]
        ax.plot(bk, tfidf_accs, marker="^", markersize=4, color="orange",
                linestyle=":", linewidth=1.5, alpha=0.8, label="TF-IDF (text)")
        ax.plot(bk, embed_accs, marker="v", markersize=4, color="purple",
                linestyle=":", linewidth=1.5, alpha=0.8, label="Embed (text)")

    # Overlay Gemini zero-shot baseline if available
    gemini = GEMINI_BASELINES.get(ds_name)
    if gemini:
        gk = sorted(gemini)
        gemini_accs = [gemini[k] for k in gk]
        ax.plot(gk, gemini_accs, marker="D", markersize=4, color="green",
                linestyle="--", linewidth=1.5, alpha=0.8, label="Gemini 3 Flash (zero-shot)")

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="Chance")
    ax.set_xlabel("K% of thinking tokens")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"LOO CV Accuracy vs K% — {ds_name}")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    print(f"Plot saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3a: Train LOO probes on cached activations",
    )
    parser.add_argument("name", help="Dataset name (reads data/{name}.jsonl)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model whose activations to load from cache")
    parser.add_argument("--k-percents", nargs="+", type=float, default=DEFAULT_K_PERCENTS,
                        help="Truncation points as %% of thinking tokens")
    parser.add_argument("--layer-fracs", nargs="+", type=float, default=None,
                        help=f"Layer fractions (default: {DEFAULT_LAYER_FRACS})")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Logistic regression regularization")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load JSONL
    jsonl_path = Path("data") / f"{args.name}.jsonl"
    records = read_jsonl(jsonl_path)
    ds_name = args.name
    print(f"Dataset: {ds_name} ({len(records)} examples)")

    n_pos = sum(1 for r in records if r["label"])
    n_neg = len(records) - n_pos
    print(f"  {n_pos} positive / {n_neg} negative")

    # We need num_layers to resolve fracs → indices.
    # Instead of loading the model, peek at an NPZ to figure out which layers exist.
    # Failing that, use a hardcoded heuristic from the model name.
    cache_dir = activation_cache_dir(args.model, ds_name)

    # Try to infer num_layers from the first NPZ file
    fracs = args.layer_fracs or DEFAULT_LAYER_FRACS
    num_layers = infer_num_layers(cache_dir, fracs)
    layer_indices = resolve_layer_indices(num_layers, fracs)
    print(f"Probing layers: {layer_indices} (of {num_layers})")

    # Load cached activations
    acts = {}
    missing = 0
    for rec in records:
        cache_path = cache_dir / f"{rec['id']}.npz"
        loaded = load_activations(cache_path, layer_indices)
        if loaded is not None:
            acts[rec["id"]] = loaded
        else:
            missing += 1

    print(f"Loaded activations: {len(acts)}/{len(records)}", end="")
    if missing:
        print(f" ({missing} missing — run extract.py first)")
    else:
        print()

    if len(acts) < 3:
        print("Not enough examples with cached activations. Run extract.py first.")
        return

    # Sweep
    print(f"\nSweeping {len(layer_indices)} layers x {len(args.k_percents)} K% points + all-layers...")
    results = sweep(acts, records, layer_indices, args.k_percents, args.C)

    # Print results table
    print(f"\n{'Layer':>6} {'K%':>5} {'Accuracy':>10} {'+-':>6}")
    print("-" * 30)
    for layer_idx in layer_indices:
        for k in args.k_percents:
            acc, _, acc_se, _ = results[(layer_idx, k)]
            print(f"{layer_idx:>6} {k:>5.0f} {acc:>10.3f} {acc_se:>6.3f}")
    for k in args.k_percents:
        acc, _, acc_se, _ = results[("all", k)]
        print(f"{'all':>6} {k:>5.0f} {acc:>10.3f} {acc_se:>6.3f}")

    # Plot
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    plot_path = output_dir / f"probe_{ds_name}_{ts}.png"
    plot_results(results, ds_name, args.k_percents, layer_indices, plot_path)
    open_plot(plot_path)


if __name__ == "__main__":
    main()

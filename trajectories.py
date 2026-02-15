#!/usr/bin/env python3
"""Stage 3b: Visualize per-example residual stream trajectories in PCA space.

Reads cached NPZ activations + JSONL labels, projects token-level residuals
through PCA at 3 layers (25%, 50%, 75%), and plots 2D trajectories colored
by label. Interactive command loop for adjusting smoothing and trajectory count.

Usage:
  python trajectories.py pooh                    # first 20 trajectories, no smoothing
  python trajectories.py pooh --first 30         # first 30
  python trajectories.py has_false_starts --smooth 10

Interactive commands after initial plot:
  /smooth <M>   — set trailing moving average window (1 = off)
  /first <K>    — show first K trajectories
  /quit         — exit
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from lib import (
    DEFAULT_LAYER_FRACS,
    DEFAULT_MODEL,
    activation_cache_dir,
    infer_num_layers,
    load_activations,
    open_plot,
    read_jsonl,
    resolve_layer_indices,
)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth_residuals(residuals: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average over the token axis. window=1 is a no-op.

    Args:
        residuals: (T, D) array in any dtype — promoted to float32.
        window: trailing window size (>=1).

    Returns:
        (T, D) float32 array with each row averaged over the preceding `window` rows.
    """
    if window <= 1:
        return residuals.astype(np.float32)
    x = residuals.astype(np.float32)
    cs = np.cumsum(x, axis=0)
    # Insert a row of zeros at the top for the subtraction trick
    cs = np.vstack([np.zeros((1, x.shape[1]), dtype=np.float32), cs])
    # For position t, average over rows max(0, t+1-window) .. t+1
    T = x.shape[0]
    starts = np.maximum(np.arange(T) + 1 - window, 0)
    ends = np.arange(T) + 1
    sums = cs[ends] - cs[starts]
    counts = (ends - starts).astype(np.float32)[:, None]
    return sums / counts


# ---------------------------------------------------------------------------
# PCA projection per layer
# ---------------------------------------------------------------------------

def project_layer(
    all_acts: list[dict[int, np.ndarray]],
    layer_idx: int,
    window: int,
) -> list[np.ndarray]:
    """Smooth, stack, outlier-filter, PCA(2), split back.

    Returns a list (one per example) of (T_i, 2) float32 arrays.
    """
    # Smooth each trajectory
    smoothed = [smooth_residuals(a[layer_idx], window) for a in all_acts]
    lengths = [s.shape[0] for s in smoothed]
    stacked = np.vstack(smoothed)  # (sum(T_i), D)

    # Outlier filter based on norm IQR
    norms = np.linalg.norm(stacked, axis=1)
    q1, q3 = np.percentile(norms, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inlier_mask = (norms >= lo) & (norms <= hi)

    # Fit PCA on inliers, transform all
    pca = PCA(n_components=2)
    pca.fit(stacked[inlier_mask])
    projected = pca.transform(stacked)  # (sum(T_i), 2)

    # Split back
    result = []
    offset = 0
    for length in lengths:
        result.append(projected[offset : offset + length])
        offset += length
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_trajectories(
    records: list[dict],
    all_acts: list[dict[int, np.ndarray]],
    layer_indices: list[int],
    window: int,
    ds_name: str,
    output_path: Path,
):
    """Create a 3-panel PCA trajectory plot, one subplot per layer."""
    n = len(records)
    fig, axes = plt.subplots(1, len(layer_indices), figsize=(6 * len(layer_indices), 6))
    if len(layer_indices) == 1:
        axes = [axes]

    for ax, layer_idx in zip(axes, layer_indices):
        projected = project_layer(all_acts, layer_idx, window)

        for i, (rec, traj) in enumerate(zip(records, projected)):
            label = bool(rec["label"])
            color = "green" if label else "red"
            ls = "-" if label else "--"
            alpha = max(0.3, 1.0 - 0.015 * n)

            ax.plot(traj[:, 0], traj[:, 1], color=color, linestyle=ls,
                    alpha=alpha, linewidth=0.8)
            # Start marker (filled dark dot)
            ax.scatter(traj[0, 0], traj[0, 1], marker="o", s=30,
                       color="black", edgecolors="black", linewidths=0.5,
                       alpha=min(alpha + 0.2, 1.0), zorder=6)
            # End marker (square)
            ax.scatter(traj[-1, 0], traj[-1, 1], marker="s", s=20,
                       color=color, edgecolors="black", linewidths=0.5,
                       alpha=alpha, zorder=5)

        ax.set_title(f"Layer {layer_idx}", fontsize=11)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.3)

    # Shared legend at bottom
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="green", linestyle="-", lw=1.5, label="True"),
        Line2D([0], [0], color="red", linestyle="--", lw=1.5, label="False"),
        Line2D([0], [0], marker="o", color="black", markeredgecolor="black",
               markersize=6, lw=0, label="Start"),
        Line2D([0], [0], marker="s", color="gray", markeredgecolor="black",
               markersize=6, lw=0, label="End"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.02))

    smooth_str = f", smooth={window}" if window > 1 else ""
    fig.suptitle(f"{ds_name} — {n} trajectories{smooth_str}", fontsize=13, y=1.02)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"Plot saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Lazy loader
# ---------------------------------------------------------------------------

def ensure_loaded(
    records: list[dict],
    all_acts: list[dict[int, np.ndarray]],
    cache_dir: Path,
    layer_indices: list[int],
    up_to: int,
) -> tuple[list[dict], list[dict[int, np.ndarray]]]:
    """Extend all_acts up to `up_to` records (lazy). Returns (records_slice, all_acts)."""
    already = len(all_acts)
    if up_to <= already:
        return records[:up_to], all_acts[:up_to]

    for rec in records[already:up_to]:
        cache_path = cache_dir / f"{rec['id']}.npz"
        loaded = load_activations(cache_path, layer_indices)
        if loaded is None:
            print(f"  Warning: no cached activations for {rec['id']}, stopping at {len(all_acts)}")
            break
        all_acts.append(loaded)

    actual = min(up_to, len(all_acts))
    return records[:actual], all_acts[:actual]


# ---------------------------------------------------------------------------
# Replot helper
# ---------------------------------------------------------------------------

def replot(
    records: list[dict],
    all_acts: list[dict[int, np.ndarray]],
    layer_indices: list[int],
    window: int,
    ds_name: str,
    output_dir: Path,
):
    """Slice, plot, open."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"traj_{ds_name}_{ts}.png"
    plot_trajectories(records, all_acts, layer_indices, window, ds_name, output_path)
    open_plot(output_path)


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def interactive_loop(
    all_records: list[dict],
    all_acts: list[dict[int, np.ndarray]],
    cache_dir: Path,
    layer_indices: list[int],
    window: int,
    first_k: int,
    ds_name: str,
    output_dir: Path,
):
    """Command loop: /smooth, /first, /quit."""
    print("\nInteractive mode. Commands: /smooth <M>, /first <K>, /quit")
    while True:
        try:
            cmd = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not cmd:
            continue

        if cmd == "/quit":
            break

        parts = cmd.split()
        if parts[0] == "/smooth" and len(parts) == 2:
            try:
                window = max(1, int(parts[1]))
                print(f"Smoothing window = {window}")
                recs, acts = ensure_loaded(all_records, all_acts, cache_dir,
                                           layer_indices, first_k)
                replot(recs, acts, layer_indices, window, ds_name, output_dir)
            except ValueError:
                print("Usage: /smooth <integer>")
        elif parts[0] == "/first" and len(parts) == 2:
            try:
                first_k = max(1, int(parts[1]))
                print(f"Showing first {first_k} trajectories")
                recs, acts = ensure_loaded(all_records, all_acts, cache_dir,
                                           layer_indices, first_k)
                first_k = len(recs)  # clamp to what's available
                replot(recs, acts, layer_indices, window, ds_name, output_dir)
            except ValueError:
                print("Usage: /first <integer>")
        else:
            print("Unknown command. Try /smooth <M>, /first <K>, or /quit")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3b: Visualize residual stream trajectories in PCA space",
    )
    parser.add_argument("name", help="Dataset name (reads data/{name}.jsonl)")
    parser.add_argument("--first", type=int, default=20,
                        help="Number of trajectories to show (default: 20)")
    parser.add_argument("--smooth", type=int, default=50,
                        help="Trailing moving average window (1 = off)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model whose activations to load from cache")
    parser.add_argument("--layer-fracs", nargs="+", type=float, default=None,
                        help=f"Layer fractions (default: {DEFAULT_LAYER_FRACS})")
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()

    # Load JSONL
    jsonl_path = Path("data") / f"{args.name}.jsonl"
    all_records = read_jsonl(jsonl_path)
    ds_name = args.name
    print(f"Dataset: {ds_name} ({len(all_records)} examples)")

    # Resolve layers
    cache_dir = activation_cache_dir(args.model, ds_name)
    fracs = args.layer_fracs or DEFAULT_LAYER_FRACS
    num_layers = infer_num_layers(cache_dir, fracs)
    layer_indices = resolve_layer_indices(num_layers, fracs)
    print(f"Layers: {layer_indices} (of {num_layers})")

    # Load activations for first K
    all_acts: list[dict[int, np.ndarray]] = []
    first_k = min(args.first, len(all_records))
    recs, all_acts = ensure_loaded(all_records, all_acts, cache_dir,
                                   layer_indices, first_k)
    first_k = len(recs)
    print(f"Loaded {first_k} trajectories")

    if first_k < 2:
        print("Need at least 2 examples with cached activations. Run extract.py first.")
        return

    # Initial plot
    window = max(1, args.smooth)
    output_dir = Path(args.output_dir)
    replot(recs, all_acts, layer_indices, window, ds_name, output_dir)

    # Interactive loop
    interactive_loop(all_records, all_acts, cache_dir, layer_indices,
                     window, first_k, ds_name, output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
idw_record_visualizer.py

Visualize calibration records from calibration.json.

What it shows:
- measured / Apriltag-estimated positions
- true / should-be positions
- arrows from measured -> true
- optional IDW heading-correction heatmap in the background

Usage:
  python idw_record_visualizer.py
  python idw_record_visualizer.py --cal calibration.json
  python idw_record_visualizer.py --no-heatmap
  python idw_record_visualizer.py --annotate
  python idw_record_visualizer.py --save out.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def load_calibration(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def try_get_arena_bounds():
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from config import ARENA_BOUNDS  # type: ignore
        return ARENA_BOUNDS
    except Exception:
        return None


def normalize_bounds(bounds):
    if bounds is None:
        return None
    if isinstance(bounds, dict):
        if {"xmin", "xmax", "ymin", "ymax"}.issubset(bounds):
            return (float(bounds["xmin"]), float(bounds["xmax"]),
                    float(bounds["ymin"]), float(bounds["ymax"]))
        if {"x_min", "x_max", "y_min", "y_max"}.issubset(bounds):
            return (float(bounds["x_min"]), float(bounds["x_max"]),
                    float(bounds["y_min"]), float(bounds["y_max"]))
    if isinstance(bounds, (list, tuple)):
        if len(bounds) == 4 and all(isinstance(v, (int, float)) for v in bounds):
            return tuple(float(v) for v in bounds)
        if len(bounds) == 2 and all(isinstance(v, (list, tuple)) and len(v) == 2 for v in bounds):
            return (float(bounds[0][0]), float(bounds[0][1]),
                    float(bounds[1][0]), float(bounds[1][1]))
    return None


def infer_bounds_from_records(records, padding_mm=120.0):
    xs = []
    ys = []
    for r in records:
        for kx, ky in (("pred_x_mm", "pred_y_mm"), ("true_x_mm", "true_y_mm")):
            if kx in r and ky in r:
                xs.append(float(r[kx]))
                ys.append(float(r[ky]))
    if not xs:
        return (0.0, 1000.0, 0.0, 1000.0)
    return (min(xs) - padding_mm, max(xs) + padding_mm,
            min(ys) - padding_mm, max(ys) + padding_mm)


def idw_grid(points, power, xg, yg):
    if not points:
        return np.full_like(xg, np.nan, dtype=float)

    px = np.array([p["x_mm"] for p in points], dtype=float)
    py = np.array([p["y_mm"] for p in points], dtype=float)
    pv = np.array([p["correction_deg"] for p in points], dtype=float)

    dx = xg[..., None] - px[None, None, :]
    dy = yg[..., None] - py[None, None, :]
    dist2 = dx * dx + dy * dy

    eps = 1e-9
    w = 1.0 / np.power(dist2 + eps, power / 2.0)
    z = np.sum(w * pv[None, None, :], axis=2) / np.sum(w, axis=2)

    near = dist2 <= 1e-9
    exact_any = np.any(near, axis=2)
    if np.any(exact_any):
        idx = np.argmax(near, axis=2)
        z = np.where(exact_any, pv[idx], z)

    return z


def build_heatmap(ax, cal, bounds, grid_res=220):
    points = cal.get("heading_idw_points", [])
    if not points:
        return None

    xmin, xmax, ymin, ymax = bounds
    xs = np.linspace(xmin, xmax, grid_res)
    ys = np.linspace(ymin, ymax, grid_res)
    xg, yg = np.meshgrid(xs, ys)

    power = float(cal.get("heading_idw_power", 2))
    zg = idw_grid(points, power, xg, yg)

    im = ax.imshow(
        zg,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        interpolation="bilinear",
        aspect="equal",
        alpha=0.35,
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Heading correction (deg)")
    return im


def plot_records(ax, records, annotate=False):
    measured_x = []
    measured_y = []
    true_x = []
    true_y = []

    # Filter to records that have both measured and true positions
    usable = []
    for i, r in enumerate(records, start=1):
        if all(k in r for k in ("pred_x_mm", "pred_y_mm", "true_x_mm", "true_y_mm")):
            usable.append((i, r))
            measured_x.append(float(r["pred_x_mm"]))
            measured_y.append(float(r["pred_y_mm"]))
            true_x.append(float(r["true_x_mm"]))
            true_y.append(float(r["true_y_mm"]))

    if not usable:
        raise SystemExit("No usable records found in calibration.json")

    # Measured points
    ax.scatter(
        measured_x, measured_y,
        s=70, facecolors="none", edgecolors="tab:blue",
        linewidths=2.0, label="measured (Apriltag)"
    )

    # True points
    ax.scatter(
        true_x, true_y,
        s=70, marker="x", color="tab:orange",
        linewidths=2.0, label="true / should-be"
    )

    # Arrows and labels
    for idx, r in usable:
        mx = float(r["pred_x_mm"])
        my = float(r["pred_y_mm"])
        tx = float(r["true_x_mm"])
        ty = float(r["true_y_mm"])

        arrow = FancyArrowPatch(
            (mx, my), (tx, ty),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.6,
            color="0.25",
            alpha=0.8,
            zorder=2,
        )
        ax.add_patch(arrow)

        # tiny midpoint marker to make direction obvious
        ax.plot([(mx + tx) / 2], [(my + ty) / 2], marker=".", color="0.25", markersize=4)

        if annotate:
            ax.text(
                tx + 8, ty + 8, f"#{idx}",
                fontsize=8,
                color="black",
                zorder=5,
            )

    return usable


def add_error_vectors_legend(ax):
    # Dummy handles to make the legend clear
    ax.plot([], [], color="tab:blue", marker="o", linestyle="None",
            markerfacecolor="none", markeredgewidth=2.0, label="measured (Apriltag)")
    ax.plot([], [], color="tab:orange", marker="x", linestyle="None",
            markeredgewidth=2.0, label="true / should-be")
    ax.plot([], [], color="0.25", linewidth=1.6, label="measured → true")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal", type=Path, default=Path(__file__).with_name("calibration.json"))
    parser.add_argument("--save", type=Path, default=None, help="Save figure to PNG")
    parser.add_argument("--annotate", action="store_true", help="Label each true point with record number")
    parser.add_argument("--no-heatmap", action="store_true", help="Do not draw the background IDW heatmap")
    parser.add_argument("--grid-res", type=int, default=220)
    args = parser.parse_args()

    cal = load_calibration(args.cal)
    records = cal.get("records", [])

    bounds = normalize_bounds(try_get_arena_bounds())
    if bounds is None:
        bounds = infer_bounds_from_records(records)

    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
    ax.set_title("Calibration record visualizer")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal", adjustable="box")

    xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Faint arena outline
    ax.plot([xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin],
            linewidth=1.2, color="0.5", alpha=0.7)

    if not args.no_heatmap:
        build_heatmap(ax, cal, bounds, grid_res=args.grid_res)

    usable = plot_records(ax, records, annotate=args.annotate)
    add_error_vectors_legend(ax)
    ax.legend(loc="upper left")

    n = len(usable)
    rmsx = cal.get("rms_x_mm", None)
    rmsy = cal.get("rms_y_mm", None)
    fallback = cal.get("heading_idw_fallback_deg", None)
    summary = f"{n} paired records"
    if rmsx is not None and rmsy is not None:
        summary += f" | X RMS {rmsx:.1f} mm | Y RMS {rmsy:.1f} mm"
    if fallback is not None:
        summary += f" | heading fallback {float(fallback):+.2f}°"

    ax.text(
        0.99, 0.01, summary,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    if args.save:
        fig.savefig(args.save, dpi=220, bbox_inches="tight")
        print(f"Saved {args.save}")

    plt.show()


if __name__ == "__main__":
    main()

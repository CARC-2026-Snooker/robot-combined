"""
calibrate_position.py — Interactive robot position calibration.

Usage:
  1. Run new_main3.py (tracking must be live)
  2. Run this script
  3. Place the robot at each listed position, press ENTER to record
  4. The script fits a linear correction and saves calibration.json
  5. new_main3.py auto-loads calibration.json on start

How it works:
  The tracker gives high-precision but potentially offset/scaled readings.
  We record (predicted, true) pairs at known physical positions,
  then fit per-axis linear transforms:

    x_corrected = scale_x * x_raw + offset_x
    y_corrected = scale_y * y_raw + offset_y
    heading_corrected = heading_raw + heading_offset_deg

  These coefficients are saved to calibration.json and loaded by new_main3.py.

Calibration point types:
  "x"    — only X is recorded (robot Y doesn't need to be exact)
  "y"    — only Y is recorded (robot X doesn't need to be exact)
  "xy"   — both X and Y recorded (robot must be precisely placed)

  You need at least 2 "x" (or "xy") points to calibrate X,
  and at least 2 "y" (or "xy") points to calibrate Y.
  With only 1 point per axis, scale is assumed 1.0 (offset-only correction).
"""

import sys, os, json, math, time, socket, threading
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import TELEMETRY_PORT, ARENA_BOUNDS

CAL_FILE = os.path.join(os.path.dirname(__file__), "calibration.json")

# ─────────────────────────────────────────────────────────────────────────────
# DEFINE YOUR CALIBRATION POINTS HERE
# Each entry: (true_x_mm, true_y_mm, axes, label)
#   true_x_mm / true_y_mm: the ACTUAL measured position in mm (world frame)
#   axes: "x", "y", or "xy" — which axes to record at this point
#   label: human-readable description of where to place the robot
#
# Use physical landmarks: table edges, tape marks, pocket centres, etc.
# You only need the axis you're recording to be precise; the other can be approximate.
# ─────────────────────────────────────────────────────────────────────────────
CALIBRATION_POINTS = [
    # (true_x_mm, true_y_mm, axes, description, heading(optional,default=270))

    # assuming wheel to wheel = 28cm > dis to middle = 14
    # straight points
    (140,  3460, "xyh", "bottom left"),
    (600,  3460, "yh",  "bottom just middle somewhere"),
    (1060, 3460, "xyh", "bottom right"),
    (375,  3375, "xh",  "left start tape middle somewhere"),
    (825,  3375, "xh",  "right start tape middle somewhere"),
    (375,  3150, "xyh",  "top left start tape"),
    (600,  3150, "yh",  "top middle start tape somewhere"),
    (825,  3150, "xyh",  "top right start tape"),
    (140,  2640, "xyh", "top inner black left"),
    (600,  2640, "yh",  "top inner black just middle somewhere"),
    (1060, 2640, "xyh",  "top inner black right"),
    (180,  2100, "xyh", "top white left"),
    (600,  2100, "yh",  "top white just middle somewhere"),
    (1020, 2100, "xyh",  "top white right"),

    # angled points
    # (140,  3460, "xyh", "bottom left", 300.0),
    # (1060, 3460, "xyh", "bottom right", 240.0),
    # (375,  3150, "xyh",  "top left start tape, looking left", 255.0),
    # (375,  3150, "xyh",  "top left start tape, looking right", 290.0),
    # (825,  3150, "xyh",  "top right start tape, looking left", 250.0),
    # (825,  3150, "xyh",  "top right start tape, looking right", 285.0),
    # (140,  2640, "xyh", "top inner black left", 300.0),
    # (1060, 2640, "xyh",  "top inner black right", 240.0),
    # (180,  2100, "xyh", "top white left", 300.0),
    # (1020, 2100, "xyh",  "top white right", 240.0),
]

# Optional: record a heading at each point where axes includes "h"
# e.g. (600, 2450, "xyh", "Centre, facing 270°") — robot must face exactly 270°
# heading_true_deg is always the expected heading at that point (usually 270°)
HEADING_POINTS = [
    # (true_heading_deg, description)
    # Add entries here if you want to calibrate heading offset
    # (270.0, "straight, in the middle"),
    # (300.0, "bottom left, looking right"),
    # (300.0, "top left, looking right"),
    # (240.0, "bottom right, looking left"),
    # (240.0, "top right, looking left"),
]


# ─────────────────────────────────────────────────────────────────────────────
# TELEMETRY READER  (same as RobotState in mission_controller)
# ─────────────────────────────────────────────────────────────────────────────
class _TelReader:
    def __init__(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", TELEMETRY_PORT))
        self._sock.settimeout(0.1)
        self._data = {}
        self._lock = threading.Lock()
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self._running:
            try:
                raw, _ = self._sock.recvfrom(4096)
                with self._lock:
                    self._data = json.loads(raw.decode())
            except (socket.timeout, json.JSONDecodeError):
                pass

    def get(self):
        with self._lock:
            return dict(self._data)

    def stop(self):
        self._running = False
        self._sock.close()


def _wait_for_stable(reader, samples=20, interval=0.15):
    """Collect multiple telemetry readings and return their average.
    Waits until tracking is initialised."""
    print("  Waiting for stable tracking...", end="", flush=True)
    while True:
        d = reader.get()
        if d.get("initialized") and d.get("state") in ("STILL", "TRANSITIONING"):
            break
        print(".", end="", flush=True)
        time.sleep(0.2)
    print()

    readings_x, readings_y, readings_hdx, readings_hdy = [], [], [], []
    for _ in range(samples):
        d = reader.get()
        readings_x.append(d.get("x_mm", 0.0))
        readings_y.append(d.get("y_mm", 0.0))
        readings_hdx.append(d.get("heading_dx", 1.0))
        readings_hdy.append(d.get("heading_dy", 0.0))
        time.sleep(interval)

    return {
        "x_mm":  float(np.mean(readings_x)),
        "y_mm":  float(np.mean(readings_y)),
        "x_std": float(np.std(readings_x)),
        "y_std": float(np.std(readings_y)),
        "heading_deg": float((math.degrees(math.atan2(
            np.mean(readings_hdy), np.mean(readings_hdx))) + 360) % 360),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FIT CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
def _fit_axis(pred_vals, true_vals):
    """
    Fit true = scale * pred + offset by least squares.
    With 1 point: assume scale=1, return offset only.
    With 2+ points: full linear regression.
    Returns (scale, offset_m, rms_mm).
    """
    pred = np.array(pred_vals, dtype=float)  # metres
    true = np.array(true_vals, dtype=float)  # metres

    if len(pred) == 1:
        scale  = 1.0
        offset = true[0] - pred[0]
        rms    = 0.0
    else:
        A = np.column_stack([pred, np.ones(len(pred))])
        (scale, offset), *_ = np.linalg.lstsq(A, true, rcond=None)
        residuals = scale * pred + offset - true
        rms = float(np.sqrt(np.mean(residuals**2)))

    return float(scale), float(offset), rms * 1000.0  # rms in mm


def fit_calibration(records):
    """
    Fit calibration from a list of measurement records.
    Returns a calibration dict ready to save as calibration.json.

    Position:
      Step 1 — global linear (scale + offset) via least squares, same as before.
               This handles bulk sensor drift uniformly.
      Step 2 — per-point residual IDW map.
               Each calibration point stores (raw_position, residual_correction)
               where residual = true - linear_corrected.
               At runtime the IDW interpolates the local residuals so that
               each zone of the table gets its own fine correction on top of
               the global one. Points far from any calibration point fall back
               to zero residual (pure global linear).

    Heading:
      IDW correction map — one (x_mm, y_mm, correction_deg) point per 'h' record.
      Identical to the previous behaviour.
    """
    pred_x, true_x = [], []
    pred_y, true_y = [], []

    for r in records:
        axes = r["axes"]
        if "x" in axes:
            pred_x.append(r["pred_x_mm"] / 1000.0)
            true_x.append(r["true_x_mm"] / 1000.0)
        if "y" in axes:
            pred_y.append(r["pred_y_mm"] / 1000.0)
            true_y.append(r["true_y_mm"] / 1000.0)

    cal = {
        "scale_x": 1.0, "offset_x_m": 0.0, "rms_x_mm": 0.0,
        "scale_y": 1.0, "offset_y_m": 0.0, "rms_y_mm": 0.0,
        # --- NEW: per-point residual IDW for X and Y ---
        "position_idw_points": [],
        "position_idw_power": 2,
        # --- Heading IDW (unchanged) ---
        "heading_idw_points": [],
        "heading_idw_power": 2,
        "heading_idw_fallback_deg": 0.0,
        "n_x": len(pred_x), "n_y": len(pred_y), "n_heading": 0,
    }

    if pred_x:
        cal["scale_x"], cal["offset_x_m"], cal["rms_x_mm"] = _fit_axis(pred_x, true_x)
    if pred_y:
        cal["scale_y"], cal["offset_y_m"], cal["rms_y_mm"] = _fit_axis(pred_y, true_y)

    # ── Build position IDW residual map ──────────────────────────────────────
    # For every record that has at least one position axis, compute the
    # residual after the global linear correction.  Points with only one axis
    # ("x" or "y") still contribute a residual for that axis; the other axis
    # residual is set to 0 at that point (no information → IDW will blend
    # it away from other nearby points).
    pos_idw_pts = []
    for r in records:
        if "x" not in r["axes"] and "y" not in r["axes"]:
            continue

        raw_x_m = r["pred_x_mm"] / 1000.0
        raw_y_m = r["pred_y_mm"] / 1000.0

        # Linear-corrected position at this raw reading
        lin_x_m = cal["scale_x"] * raw_x_m + cal["offset_x_m"]
        lin_y_m = cal["scale_y"] * raw_y_m + cal["offset_y_m"]

        # Residual = true - linear_corrected  (zero if axis not recorded here)
        res_x_m = (r["true_x_mm"] / 1000.0 - lin_x_m) if "x" in r["axes"] else 0.0
        res_y_m = (r["true_y_mm"] / 1000.0 - lin_y_m) if "y" in r["axes"] else 0.0

        # Use the TRUE position for the lookup key so the map is indexed in
        # real-world space.  At runtime we look up using the raw reading (which
        # is close enough given the small corrections involved).
        lookup_x_mm = r["true_x_mm"] if "x" in r["axes"] else r["pred_x_mm"]
        lookup_y_mm = r["true_y_mm"] if "y" in r["axes"] else r["pred_y_mm"]

        pos_idw_pts.append({
            "raw_x_mm":      round(raw_x_m * 1000.0, 2),
            "raw_y_mm":      round(raw_y_m * 1000.0, 2),
            "lookup_x_mm":   round(float(lookup_x_mm), 2),
            "lookup_y_mm":   round(float(lookup_y_mm), 2),
            "residual_x_m":  round(res_x_m, 6),
            "residual_y_m":  round(res_y_m, 6),
            "axes":          r["axes"],   # so runtime can weight by axis quality
        })

    cal["position_idw_points"] = pos_idw_pts

    # RMS after IDW residual (should be near zero at calibration points themselves)
    if pos_idw_pts:
        errs_x = [(p["residual_x_m"] * 1000.0) for p in pos_idw_pts if "x" in p["axes"]]
        errs_y = [(p["residual_y_m"] * 1000.0) for p in pos_idw_pts if "y" in p["axes"]]
        cal["rms_x_residual_mm"] = float(np.sqrt(np.mean(np.array(errs_x)**2))) if errs_x else 0.0
        cal["rms_y_residual_mm"] = float(np.sqrt(np.mean(np.array(errs_y)**2))) if errs_y else 0.0

    # ── Build IDW heading map (unchanged) ────────────────────────────────────
    idw_pts = []
    for r in records:
        if 'h' not in r.get('axes', ''):
            continue
        if 'true_heading_deg' not in r:
            continue
        if 'xy' in r['axes']:
            x_mm, y_mm = r['true_x_mm'], r['true_y_mm']
        elif 'y' in r['axes']:
            x_mm, y_mm = r['pred_x_mm'], r['true_y_mm']
        else:
            x_mm, y_mm = r['pred_x_mm'], r['pred_y_mm']
        corr = r['true_heading_deg'] - r['pred_heading_deg']
        while corr >  180: corr -= 360
        while corr < -180: corr += 360
        idw_pts.append({"x_mm": float(x_mm), "y_mm": float(y_mm),
                         "correction_deg": float(corr)})

    if idw_pts:
        cal["heading_idw_points"]       = idw_pts
        cal["heading_idw_fallback_deg"] = float(np.mean([p["correction_deg"] for p in idw_pts]))
        cal["n_heading"]                = len(idw_pts)

    return cal


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CALIBRATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         CARC 2026 — Robot Position Calibration          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("Make sure new_main3.py is running and tracking is active.")
    print()

    reader = _TelReader()
    time.sleep(0.5)   # let socket settle

    records = []

    # ── Position calibration points ──────────────────────────────────────
    i = 0

    while i < len(CALIBRATION_POINTS):
        point = CALIBRATION_POINTS[i]
        true_x, true_y, axes, label, *rest = point
        true_heading = rest[0] if rest else 270.0

        print(f"── Point {i+1}/{len(CALIBRATION_POINTS)} ──────────────────────────────")
        print(f"  Position: X={true_x:.0f}mm, Y={true_y:.0f}mm  (axes: {axes})")
        print(f"  Task    : {label}")
        print()

        # Show which axes matter
        if axes == "x":
            print("  ► X position must be exact. Y can be approximate.")
        elif axes == "y":
            print("  ► Y position must be exact. X can be approximate.")
        elif "xy" in axes:
            print("  ► Both X and Y must be exactly at the stated position.")

        if "h" in axes:
            print(f"  ► heading must be exactly {true_heading} degrees")

        print()

        cmd = input(
            "  Place robot now.\n"
            "  [ENTER=record, S=skip, B=go back one point] "
        ).strip().lower()

        # ── Skip point ───────────────────────────────────────────────────
        if cmd == "s":
            print("  Skipped.\n")
            i += 1
            continue

        # ── Go back one point ────────────────────────────────────────────
        if cmd == "b":
            if i == 0:
                print("  Already at the first point.\n")
                continue

            i -= 1

            if records:
                removed = records.pop()

                removed_desc = (
                    f"({removed['true_x_mm']:.0f}, "
                    f"{removed['true_y_mm']:.0f})"
                )

                print(f"  Removed previous recording {removed_desc}")
                print(f"  Returning to Point {i+1}.\n")
            else:
                print("  No recorded point to remove.\n")

            continue

        # ── Record point ─────────────────────────────────────────────────
        reading = _wait_for_stable(reader)

        print(
            f"  Recorded: pred=({reading['x_mm']:.1f}, "
            f"{reading['y_mm']:.1f})mm  "
            f"heading={reading['heading_deg']:.1f}°  "
            f"std=({reading['x_std']:.1f}, "
            f"{reading['y_std']:.1f})mm"
        )

        if reading["x_std"] > 30 or reading["y_std"] > 30:
            print("  ⚠  High noise (std > 30mm). Consider re-recording.")
            redo = input("  Re-record this point? [y/N] ").strip().lower()

            if redo == "y":
                reading = _wait_for_stable(reader)
                print(
                    f"  Re-recorded: pred=({reading['x_mm']:.1f}, "
                    f"{reading['y_mm']:.1f})mm"
                )

        record = {
            "axes": axes,
            "true_x_mm": float(true_x),
            "true_y_mm": float(true_y),
            "pred_x_mm": reading["x_mm"],
            "pred_y_mm": reading["y_mm"],
            "pred_heading_deg": reading["heading_deg"],
        }

        if "h" in axes:
            record["true_heading_deg"] = true_heading

        records.append(record)

        print()
        i += 1

    # ── Heading calibration points (optional) ────────────────────────────
    i = 0

    while i < len(HEADING_POINTS):
        true_heading, label = HEADING_POINTS[i]

        print(f"── Heading point {i+1}/{len(HEADING_POINTS)} ─────────────────────────")
        print(f"  True heading: {true_heading:.1f}°")
        print(f"  Task        : {label}")
        print()

        cmd = input(
            "  Align robot now.\n"
            "  [ENTER=record, S=skip, B=go back one point] "
        ).strip().lower()

        # ── Skip ─────────────────────────────────────────────────────────
        if cmd == "s":
            print("  Skipped.\n")
            i += 1
            continue

        # ── Go back ──────────────────────────────────────────────────────
        if cmd == "b":
            if i == 0:
                print("  Already at the first heading point.\n")
                continue

            i -= 1

            # Remove last heading record
            for idx in range(len(records) - 1, -1, -1):
                if records[idx].get("axes") == "h":
                    records.pop(idx)
                    print(f"  Removed previous heading recording.")
                    break

            print(f"  Returning to Heading Point {i+1}.\n")
            continue

        # ── Record ───────────────────────────────────────────────────────
        reading = _wait_for_stable(reader)

        print(f"  Recorded heading: {reading['heading_deg']:.2f}°")

        diff = true_heading - reading["heading_deg"]
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360

        records.append({
            "axes": "h",
            "true_heading_deg": true_heading,
            "pred_heading_deg": reading["heading_deg"],
            "true_x_mm": 0,
            "true_y_mm": 0,
            "pred_x_mm": reading["x_mm"],
            "pred_y_mm": reading["y_mm"]
        })

        print()
        i += 1

    # ── Fit calibration ───────────────────────────────────────────────────
    print("═" * 60)
    print("Fitting calibration...")
    cal = fit_calibration(records)

    print()
    print("  X axis (global linear):")
    print(f"    scale  = {cal['scale_x']:.5f}")
    print(f"    offset = {cal['offset_x_m']*1000:+.1f} mm")
    print(f"    RMS    = {cal['rms_x_mm']:.1f} mm  (from {cal['n_x']} points, before IDW)")
    if "rms_x_residual_mm" in cal:
        print(f"    IDW residuals at cal points: ±{cal['rms_x_residual_mm']:.1f} mm")
    print()
    print("  Y axis (global linear):")
    print(f"    scale  = {cal['scale_y']:.5f}")
    print(f"    offset = {cal['offset_y_m']*1000:+.1f} mm")
    print(f"    RMS    = {cal['rms_y_mm']:.1f} mm  (from {cal['n_y']} points, before IDW)")
    if "rms_y_residual_mm" in cal:
        print(f"    IDW residuals at cal points: ±{cal['rms_y_residual_mm']:.1f} mm")
    print()
    print(f"  Position IDW: {len(cal.get('position_idw_points', []))} residual points")
    print()
    if "n_heading" in cal and cal["n_heading"] > 0:
        print("  Heading (IDW position-dependent correction):")
        print(f"    {cal['n_heading']} calibration points")
        print(f"    Fallback (mean) correction: {cal['heading_idw_fallback_deg']:+.3f}°")
        corrs = [p['correction_deg'] for p in cal['heading_idw_points']]
        print(f"    Range: {min(corrs):+.2f}°  to  {max(corrs):+.2f}°")
    else:
        print("  Heading: no calibration points recorded")
    print()

    # Preview correction on each recorded point
    print("  Correction preview (global linear + IDW residual):")
    print(f"  {'axes':<4} {'raw (x,y)':>14} {'linear':>14} {'IDW-corrected':>15} {'true':>14} {'error':>12}")
    for r in records:
        if "x" not in r["axes"] and "y" not in r["axes"]:
            continue
        raw_x_m = r["pred_x_mm"] / 1000.0
        raw_y_m = r["pred_y_mm"] / 1000.0
        # Global linear
        lin_x = cal["scale_x"] * raw_x_m + cal["offset_x_m"]
        lin_y = cal["scale_y"] * raw_y_m + cal["offset_y_m"]
        # IDW residual (at this point it's the stored residual itself)
        idw_x = lin_x
        idw_y = lin_y
        for p in cal["position_idw_points"]:
            pass  # we'll just show linear here; IDW is by definition zero at cal pts
        # At calibration points, IDW residual is essentially the stored residual
        # so the final corrected = true (by construction)
        err_x = lin_x * 1000 - r["true_x_mm"] if "x" in r["axes"] else float("nan")
        err_y = lin_y * 1000 - r["true_y_mm"] if "y" in r["axes"] else float("nan")
        raw_str  = f"({r['pred_x_mm']:.0f}, {r['pred_y_mm']:.0f})"
        lin_str  = f"({lin_x*1000:.0f}, {lin_y*1000:.0f})"
        true_str = f"({r['true_x_mm']:.0f}, {r['true_y_mm']:.0f})"
        err_str  = f"{err_x:+.0f}/{err_y:+.0f}mm"
        print(f"  {r['axes']:<4} {raw_str:>14} {lin_str:>14} {'→ IDW 0 at pt':>15} {true_str:>14} {err_str:>12}")
    print()

    # ── Save ──────────────────────────────────────────────────────────────
    cal["records"] = records
    cal["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(CAL_FILE, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"  Saved to: {CAL_FILE}")
    print()
    print("  new_main3.py will load this automatically on next start.")
    print("  To disable: delete calibration.json or set all scales=1, offsets=0.")
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║                  Calibration complete!                  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    reader.stop()


if __name__ == "__main__":
    main()
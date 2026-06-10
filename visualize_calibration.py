"""
visualize_calibration.py — Visual debugger for calibration.json

Shows 6 panels:
  1. Raw vs True positions  — arrows from predicted to true, coloured by total error
  2. Raw vs Linear-corrected vs True — shows what global linear fixes vs what remains
  3. IDW residual heatmap X  — colour gradient showing X residual across the table
  4. IDW residual heatmap Y  — same for Y
  5. Heading correction heatmap — IDW-interpolated heading error across the table
  6. Summary text panel  — all key numbers

Coordinate convention:
  X=0 left edge → X=1200mm right edge  (horizontal)
  Y=0 top (pockets) → Y=3600mm bottom  (robot start area)
  Calibration points are in the robot operating zone (Y=2000–3500mm)
"""

import json
import math
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe
from scipy.interpolate import griddata

# ── Load calibration ──────────────────────────────────────────────────────────
CAL_FILE = os.path.join(os.path.dirname(__file__), "calibration.json")
if len(sys.argv) > 1:
    CAL_FILE = sys.argv[1]

with open(CAL_FILE) as f:
    cal = json.load(f)

# ── Constants ─────────────────────────────────────────────────────────────────
TABLE_W = 1200   # mm, X axis
TABLE_H = 3600   # mm, Y axis (Y=0 at top/pockets, Y=3600 at robot end)

# Display limits — zoom into the calibrated zone with some margin
MARGIN = 150
Y_MIN_SHOW = 1900   # show from here downward (pockets are at top Y=0)
Y_MAX_SHOW = TABLE_H + MARGIN

scale_x   = cal["scale_x"]
offset_x  = cal["offset_x_m"] * 1000   # convert to mm
scale_y   = cal["scale_y"]
offset_y  = cal["offset_y_m"] * 1000

records     = cal["records"]
pos_pts     = cal["position_idw_points"]
hdg_pts     = cal["heading_idw_points"]
hdg_power   = cal.get("heading_idw_power", 2)
pos_power   = cal.get("position_idw_power", 2)
hdg_fallback= cal.get("heading_idw_fallback_deg", 0.0)

# ── IDW helpers ───────────────────────────────────────────────────────────────
def idw_heading(x_mm, y_mm):
    if not hdg_pts:
        return hdg_fallback
    eps = 10.0**2
    tw = tc = 0.0
    for p in hdg_pts:
        d2 = (x_mm - p["x_mm"])**2 + (y_mm - p["y_mm"])**2
        w  = 1.0 / (d2 + eps)**hdg_power
        tw += w; tc += w * p["correction_deg"]
    return tc / tw

def idw_position(x_mm, y_mm):
    """Returns (res_x_mm, res_y_mm) IDW residuals at (x_mm, y_mm)."""
    if not pos_pts:
        return 0.0, 0.0
    eps = 20.0**2
    tw = sx = sy = 0.0
    for p in pos_pts:
        d2 = (x_mm - p["raw_x_mm"])**2 + (y_mm - p["raw_y_mm"])**2
        w  = 1.0 / (d2 + eps)**pos_power
        tw += w
        sx += w * p["residual_x_m"] * 1000   # → mm
        sy += w * p["residual_y_m"] * 1000
    if tw < 1e-15: return 0.0, 0.0
    return sx/tw, sy/tw

def full_correct(raw_x_mm, raw_y_mm):
    """Apply full calibration: global linear + IDW residual. Returns (true_x, true_y) mm."""
    lin_x = scale_x * raw_x_mm + offset_x
    lin_y = scale_y * raw_y_mm + offset_y
    res_x, res_y = idw_position(lin_x, lin_y)
    return lin_x + res_x, lin_y + res_y

# ── Build dense grids for heatmaps ────────────────────────────────────────────
# Grid covers the robot operating zone in table coordinates
gx = np.linspace(0, TABLE_W, 80)
gy = np.linspace(Y_MIN_SHOW, TABLE_H, 120)
GX, GY = np.meshgrid(gx, gy)

# Heading heatmap
H_grid = np.vectorize(idw_heading)(GX, GY)

# Position IDW residual heatmaps (in raw-sensor space)
RX_grid = np.zeros_like(GX)
RY_grid = np.zeros_like(GY)
for i in range(GX.shape[0]):
    for j in range(GX.shape[1]):
        rx, ry = idw_position(GX[i,j], GY[i,j])
        RX_grid[i,j] = rx
        RY_grid[i,j] = ry

# ── Collect per-record corrected positions ────────────────────────────────────
rec_data = []
for r in records:
    raw_x = r["pred_x_mm"]; raw_y = r["pred_y_mm"]
    true_x = r["true_x_mm"]; true_y = r["true_y_mm"]
    lin_x = scale_x * raw_x + offset_x
    lin_y = scale_y * raw_y + offset_y
    full_x, full_y = full_correct(raw_x, raw_y)
    pred_h = r.get("pred_heading_deg", 270.0)
    true_h = r.get("true_heading_deg", 270.0)
    err_raw   = math.hypot(raw_x  - true_x, raw_y  - true_y)
    err_lin   = math.hypot(lin_x  - true_x, lin_y  - true_y)
    err_full  = math.hypot(full_x - true_x, full_y - true_y)
    rec_data.append({
        "raw": (raw_x, raw_y), "true": (true_x, true_y),
        "lin": (lin_x, lin_y), "full": (full_x, full_y),
        "axes": r["axes"],
        "pred_h": pred_h, "true_h": true_h, "h_err": true_h - pred_h,
        "err_raw": err_raw, "err_lin": err_lin, "err_full": err_full,
    })

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 26), facecolor="#0e0e14")
fig.suptitle(
    f"Calibration Visualiser  —  {cal.get('timestamp','?')}  "
    f"(format v{cal.get('format_version',1)})",
    fontsize=16, color="white", y=0.985, fontweight="bold"
)

# 3-column × 3-row grid.  Last row spans all 3 for the summary.
gs = fig.add_gridspec(3, 3, hspace=0.38, wspace=0.32,
                       left=0.07, right=0.97, top=0.96, bottom=0.04)
ax_raw   = fig.add_subplot(gs[0, 0])   # raw vs true arrows
ax_lin   = fig.add_subplot(gs[0, 1])   # after global linear
ax_full  = fig.add_subplot(gs[0, 2])   # after full correction
ax_hx    = fig.add_subplot(gs[1, 0])   # IDW residual X heatmap
ax_hy    = fig.add_subplot(gs[1, 1])   # IDW residual Y heatmap
ax_hdg   = fig.add_subplot(gs[1, 2])   # heading correction heatmap
ax_sum   = fig.add_subplot(gs[2, :])   # summary

BG = "#1a1a26"
FG = "#e8e8f0"
GRID_C = "#2a2a3a"
MAX_ERR = 90   # mm colour scale ceiling

# ── Helper: draw a table-shaped axes ─────────────────────────────────────────
def setup_table_ax(ax, title):
    ax.set_facecolor(BG)
    ax.set_xlim(-MARGIN, TABLE_W + MARGIN)
    ax.set_ylim(TABLE_H + MARGIN, Y_MIN_SHOW - MARGIN)   # Y inverted: 0=top
    ax.set_xlabel("X  (mm)", color=FG, fontsize=8)
    ax.set_ylabel("Y  (mm)", color=FG, fontsize=8)
    ax.set_title(title, color=FG, fontsize=10, pad=6)
    ax.tick_params(colors=FG, labelsize=7)
    for sp in ax.spines.values(): sp.set_color(GRID_C)
    ax.set_aspect("equal", adjustable="datalim")
    # Table outline
    ax.add_patch(mpatches.Rectangle((0, 0), TABLE_W, TABLE_H,
                 fill=False, edgecolor="#3a3a55", lw=1.5, zorder=0))
    # Pocket wall (y=0)
    ax.axhline(0, color="#555588", lw=1, ls="--", zorder=0)
    ax.text(TABLE_W/2, -80, "Pockets (Y=0)", ha="center", va="center",
            color="#555588", fontsize=7)
    # Robot zone divider
    ax.axhline(2450, color="#334433", lw=0.8, ls=":", zorder=0)
    ax.text(-MARGIN+10, 2450, "shoot Y", color="#446644", fontsize=6, va="center")
    # Grid lines
    for xg in range(0, TABLE_W+1, 300):
        ax.axvline(xg, color=GRID_C, lw=0.4, zorder=0)
    for yg in range(0, TABLE_H+1, 600):
        ax.axhline(yg, color=GRID_C, lw=0.4, zorder=0)
    return ax

def draw_pocket_markers(ax):
    for px_mm, py_mm in [(0,0),(600,0),(1200,0),(0,1050),(1200,1050)]:
        ax.plot(px_mm, py_mm, "D", ms=6, color="#ffcc44", zorder=5)

def draw_arrow(ax, x0, y0, x1, y1, color, alpha=0.9, lw=1.5, hw=8, hl=14):
    dx, dy = x1-x0, y1-y0
    if abs(dx)+abs(dy) < 0.5: return
    ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
                arrowprops=dict(arrowstyle=f"->, head_width={hw/100:.2f}, head_length={hl/100:.2f}",
                                color=color, lw=lw, alpha=alpha),
                zorder=6)

err_cmap = plt.colormaps["RdYlGn_r"]
err_norm = Normalize(vmin=0, vmax=MAX_ERR)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 1 — Raw vs True
# ─────────────────────────────────────────────────────────────────────────────
setup_table_ax(ax_raw, "① Raw  →  True\n(before any correction)")
draw_pocket_markers(ax_raw)

for rd in rec_data:
    rx, ry = rd["raw"]; tx, ty = rd["true"]
    err = rd["err_raw"]
    c = err_cmap(err_norm(err))
    # True position = filled circle (green tint)
    ax_raw.plot(tx, ty, "o", ms=9, color="#44ff88", zorder=7, mec="white", mew=0.8)
    # Raw position = cross (red tint)
    ax_raw.plot(rx, ry, "x", ms=8, color="#ff6666", mew=2, zorder=7)
    # Arrow: raw → true
    draw_arrow(ax_raw, rx, ry, tx, ty, color=c, hw=10, hl=16)
    # Error label
    ax_raw.text((rx+tx)/2 + 10, (ry+ty)/2, f"{err:.0f}mm",
                color=c, fontsize=6, zorder=8,
                path_effects=[pe.withStroke(linewidth=1.5, foreground=BG)])

# Legend
ax_raw.plot([], [], "o", ms=6, color="#44ff88", mec="white", mew=0.8, label="True position")
ax_raw.plot([], [], "x", ms=6, color="#ff6666", mew=2, label="Raw (predicted)")
ax_raw.legend(fontsize=7, facecolor="#222230", labelcolor=FG, loc="lower right",
              edgecolor=GRID_C)
ax_raw.text(TABLE_W/2, TABLE_H+100, "Arrow = error vector (raw→true)",
            ha="center", color="#aaaacc", fontsize=7)

# Colourbar
sm = ScalarMappable(cmap=err_cmap, norm=err_norm)
cb = fig.colorbar(sm, ax=ax_raw, fraction=0.035, pad=0.02)
cb.set_label("Error  (mm)", color=FG, fontsize=7)
cb.ax.yaxis.set_tick_params(color=FG, labelsize=6)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 2 — After global linear correction
# ─────────────────────────────────────────────────────────────────────────────
setup_table_ax(ax_lin, "② After global linear\n(scale + offset only)")
draw_pocket_markers(ax_lin)

for rd in rec_data:
    lx, ly = rd["lin"]; tx, ty = rd["true"]
    err = rd["err_lin"]
    c = err_cmap(err_norm(err))
    ax_lin.plot(tx, ty, "o", ms=9, color="#44ff88", zorder=7, mec="white", mew=0.8)
    ax_lin.plot(lx, ly, "s", ms=7, color="#ffaa33", mew=0, zorder=7, alpha=0.9)
    draw_arrow(ax_lin, lx, ly, tx, ty, color=c, hw=10, hl=16)
    ax_lin.text((lx+tx)/2 + 10, (ly+ty)/2, f"{err:.0f}mm",
                color=c, fontsize=6, zorder=8,
                path_effects=[pe.withStroke(linewidth=1.5, foreground=BG)])

ax_lin.plot([], [], "o", ms=6, color="#44ff88", mec="white", mew=0.8, label="True position")
ax_lin.plot([], [], "s", ms=6, color="#ffaa33", label="Linear-corrected")
ax_lin.legend(fontsize=7, facecolor="#222230", labelcolor=FG, loc="lower right",
              edgecolor=GRID_C)
cb2 = fig.colorbar(ScalarMappable(cmap=err_cmap, norm=err_norm), ax=ax_lin,
                   fraction=0.035, pad=0.02)
cb2.set_label("Residual  (mm)", color=FG, fontsize=7)
cb2.ax.yaxis.set_tick_params(color=FG, labelsize=6)
plt.setp(cb2.ax.yaxis.get_ticklabels(), color=FG)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 3 — After full correction (linear + IDW)
# ─────────────────────────────────────────────────────────────────────────────
setup_table_ax(ax_full, "③ After full correction\n(linear + IDW residuals)")
draw_pocket_markers(ax_full)

for rd in rec_data:
    fx, fy = rd["full"]; tx, ty = rd["true"]
    err = rd["err_full"]
    c = err_cmap(err_norm(err))
    ax_full.plot(tx, ty, "o", ms=9, color="#44ff88", zorder=7, mec="white", mew=0.8)
    ax_full.plot(fx, fy, "^", ms=7, color="#44ccff", mew=0, zorder=7, alpha=0.9)
    draw_arrow(ax_full, fx, fy, tx, ty, color=c, hw=10, hl=16)
    if err > 0.5:
        ax_full.text((fx+tx)/2 + 10, (fy+ty)/2, f"{err:.1f}mm",
                    color=c, fontsize=6, zorder=8,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground=BG)])
    else:
        ax_full.text(tx + 15, ty, "✓", color="#44ff88", fontsize=8, zorder=8)

ax_full.plot([], [], "o", ms=6, color="#44ff88", mec="white", mew=0.8, label="True position")
ax_full.plot([], [], "^", ms=6, color="#44ccff", label="Fully corrected")
ax_full.legend(fontsize=7, facecolor="#222230", labelcolor=FG, loc="lower right",
               edgecolor=GRID_C)
cb3 = fig.colorbar(ScalarMappable(cmap=err_cmap, norm=err_norm), ax=ax_full,
                   fraction=0.035, pad=0.02)
cb3.set_label("Residual  (mm)", color=FG, fontsize=7)
cb3.ax.yaxis.set_tick_params(color=FG, labelsize=6)
plt.setp(cb3.ax.yaxis.get_ticklabels(), color=FG)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 4 — IDW residual X heatmap
# ─────────────────────────────────────────────────────────────────────────────
setup_table_ax(ax_hx, "④ Position IDW residual — X axis\n(local X correction after global linear, mm)")

rx_abs_max = max(abs(RX_grid.min()), abs(RX_grid.max()), 5.0)
rx_norm = TwoSlopeNorm(vmin=-rx_abs_max, vcenter=0, vmax=rx_abs_max)
im_rx = ax_hx.pcolormesh(GX, GY, RX_grid, cmap="RdBu_r", norm=rx_norm,
                          alpha=0.82, zorder=1, shading="gouraud")
draw_pocket_markers(ax_hx)

# Contour lines at ±10, ±20mm
try:
    cs = ax_hx.contour(GX, GY, RX_grid, levels=[-20,-10,0,10,20],
                       colors=["#ff4444","#ff9999","#ffffff","#88ff88","#44cc44"],
                       linewidths=0.8, alpha=0.6, zorder=2)
    ax_hx.clabel(cs, fmt="%+.0fmm", fontsize=6, colors=FG)
except Exception:
    pass

# Plot the actual calibration points with their values
for p in pos_pts:
    if "x" not in p["axes"]: continue
    v = p["residual_x_m"] * 1000
    c = "#ff4444" if v < 0 else "#44cc44"
    ax_hx.plot(p["raw_x_mm"], p["raw_y_mm"], "D", ms=7, color=c,
               mec="white", mew=0.6, zorder=7)
    ax_hx.text(p["raw_x_mm"]+15, p["raw_y_mm"], f"{v:+.0f}",
               color=c, fontsize=6.5, zorder=8, fontweight="bold",
               path_effects=[pe.withStroke(linewidth=1.5, foreground=BG)])

cb_rx = fig.colorbar(im_rx, ax=ax_hx, fraction=0.035, pad=0.02)
cb_rx.set_label("ΔX  (mm)", color=FG, fontsize=7)
cb_rx.ax.yaxis.set_tick_params(color=FG, labelsize=6)
plt.setp(cb_rx.ax.yaxis.get_ticklabels(), color=FG)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 5 — IDW residual Y heatmap
# ─────────────────────────────────────────────────────────────────────────────
setup_table_ax(ax_hy, "⑤ Position IDW residual — Y axis\n(local Y correction after global linear, mm)")

ry_abs_max = max(abs(RY_grid.min()), abs(RY_grid.max()), 5.0)
ry_norm = TwoSlopeNorm(vmin=-ry_abs_max, vcenter=0, vmax=ry_abs_max)
im_ry = ax_hy.pcolormesh(GX, GY, RY_grid, cmap="RdBu_r", norm=ry_norm,
                          alpha=0.82, zorder=1, shading="gouraud")
draw_pocket_markers(ax_hy)

try:
    cs2 = ax_hy.contour(GX, GY, RY_grid, levels=[-20,-10,0,10,20],
                        colors=["#ff4444","#ff9999","#ffffff","#88ff88","#44cc44"],
                        linewidths=0.8, alpha=0.6, zorder=2)
    ax_hy.clabel(cs2, fmt="%+.0fmm", fontsize=6, colors=FG)
except Exception:
    pass

for p in pos_pts:
    if "y" not in p["axes"]: continue
    v = p["residual_y_m"] * 1000
    c = "#ff4444" if v < 0 else "#44cc44"
    ax_hy.plot(p["raw_x_mm"], p["raw_y_mm"], "D", ms=7, color=c,
               mec="white", mew=0.6, zorder=7)
    ax_hy.text(p["raw_x_mm"]+15, p["raw_y_mm"], f"{v:+.0f}",
               color=c, fontsize=6.5, zorder=8, fontweight="bold",
               path_effects=[pe.withStroke(linewidth=1.5, foreground=BG)])

cb_ry = fig.colorbar(im_ry, ax=ax_hy, fraction=0.035, pad=0.02)
cb_ry.set_label("ΔY  (mm)", color=FG, fontsize=7)
cb_ry.ax.yaxis.set_tick_params(color=FG, labelsize=6)
plt.setp(cb_ry.ax.yaxis.get_ticklabels(), color=FG)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 6 — Heading correction heatmap
# ─────────────────────────────────────────────────────────────────────────────
setup_table_ax(ax_hdg, "⑥ Heading correction map\n(correction_deg added to raw heading, IDW-interpolated)")

h_abs_max = max(abs(H_grid.min()), abs(H_grid.max()), 0.5)
h_norm = TwoSlopeNorm(vmin=-h_abs_max, vcenter=0, vmax=h_abs_max)
im_h = ax_hdg.pcolormesh(GX, GY, H_grid, cmap="PiYG", norm=h_norm,
                          alpha=0.85, zorder=1, shading="gouraud")
draw_pocket_markers(ax_hdg)

try:
    cs3 = ax_hdg.contour(GX, GY, H_grid, levels=7, colors="white",
                         linewidths=0.5, alpha=0.35, zorder=2)
    ax_hdg.clabel(cs3, fmt="%+.1f°", fontsize=5.5, colors=FG)
except Exception:
    pass

# Draw heading correction arrows at each calibration point
for p in hdg_pts:
    v = p["correction_deg"]
    c = "#ff44aa" if v < 0 else "#44ffaa"
    ax_hdg.plot(p["x_mm"], p["y_mm"], "o", ms=8, color=c,
                mec="white", mew=0.8, zorder=7)
    ax_hdg.text(p["x_mm"]+15, p["y_mm"], f"{v:+.2f}°",
                color=c, fontsize=6.5, zorder=8,
                path_effects=[pe.withStroke(linewidth=1.5, foreground=BG)])

cb_h = fig.colorbar(im_h, ax=ax_hdg, fraction=0.035, pad=0.02)
cb_h.set_label("Correction  (°)", color=FG, fontsize=7)
cb_h.ax.yaxis.set_tick_params(color=FG, labelsize=6)
plt.setp(cb_h.ax.yaxis.get_ticklabels(), color=FG)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 7 — Summary text
# ─────────────────────────────────────────────────────────────────────────────
ax_sum.set_facecolor(BG)
ax_sum.axis("off")
ax_sum.set_title("Summary & Global Parameters", color=FG, fontsize=10, pad=6)

# Compute summary stats
errs_raw  = [rd["err_raw"]  for rd in rec_data]
errs_lin  = [rd["err_lin"]  for rd in rec_data]
errs_full = [rd["err_full"] for rd in rec_data]
h_errs    = [rd["h_err"]    for rd in rec_data if "h" in rd["axes"]]

def rms(vs): return math.sqrt(sum(v**2 for v in vs)/len(vs)) if vs else 0.0
def pct_improve(a, b): return (1 - b/a)*100 if a > 0 else 0

# Two-column layout
col1_x, col2_x = 0.02, 0.52
row_h = 0.085

def txt(ax, x, y, t, color=FG, size=9, bold=False):
    kw = {"fontweight": "bold"} if bold else {}
    ax.text(x, y, t, transform=ax.transAxes, color=color,
            fontsize=size, va="top", family="monospace", **kw)

# --- Column 1: Global linear ---
y = 0.96
txt(ax_sum, col1_x, y, "GLOBAL LINEAR CORRECTION", "#aaccff", 9, bold=True); y -= row_h
txt(ax_sum, col1_x, y, f"  X:  scale = {scale_x:.5f}   offset = {offset_x:+.1f} mm"); y -= row_h
txt(ax_sum, col1_x, y, f"  Y:  scale = {scale_y:.5f}   offset = {offset_y:+.1f} mm"); y -= row_h
txt(ax_sum, col1_x, y, f"  X RMS (linear only): {cal['rms_x_mm']:.1f} mm  from {cal['n_x']} pts"); y -= row_h
txt(ax_sum, col1_x, y, f"  Y RMS (linear only): {cal['rms_y_mm']:.1f} mm  from {cal['n_y']} pts"); y -= row_h*1.3

txt(ax_sum, col1_x, y, "POSITION IDW RESIDUALS", "#aaccff", 9, bold=True); y -= row_h
txt(ax_sum, col1_x, y, f"  Points: {cal.get('n_position_idw', len(pos_pts))}   Power: {pos_power}"); y -= row_h
txt(ax_sum, col1_x, y, f"  X residual range: {min(p['residual_x_m']*1000 for p in pos_pts):+.1f} to "
                       f"{max(p['residual_x_m']*1000 for p in pos_pts):+.1f} mm"); y -= row_h
txt(ax_sum, col1_x, y, f"  Y residual range: {min(p['residual_y_m']*1000 for p in pos_pts):+.1f} to "
                       f"{max(p['residual_y_m']*1000 for p in pos_pts):+.1f} mm"); y -= row_h*1.3

txt(ax_sum, col1_x, y, "HEADING IDW", "#aaccff", 9, bold=True); y -= row_h
txt(ax_sum, col1_x, y, f"  Points: {cal['n_heading']}   Power: {hdg_power}"); y -= row_h
hvals = [p["correction_deg"] for p in hdg_pts]
txt(ax_sum, col1_x, y, f"  Correction range: {min(hvals):+.3f}° to {max(hvals):+.3f}°"); y -= row_h
txt(ax_sum, col1_x, y, f"  Fallback (mean): {hdg_fallback:+.4f}°"); y -= row_h

# --- Column 2: Error statistics ---
y = 0.96
txt(ax_sum, col2_x, y, "ERROR STATISTICS AT CALIBRATION POINTS", "#aaccff", 9, bold=True); y -= row_h

def stat_line(ax, x, y, label, vals, color):
    txt(ax, x, y,
        f"  {label:<16}  RMS={rms(vals):5.1f}mm  "
        f"max={max(vals):5.1f}mm  mean={sum(vals)/len(vals):5.1f}mm",
        color=color)

stat_line(ax_sum, col2_x, y, "Raw (none)", errs_raw, "#ff8888"); y -= row_h
stat_line(ax_sum, col2_x, y, "After linear", errs_lin, "#ffcc66"); y -= row_h
stat_line(ax_sum, col2_x, y, "After IDW", errs_full, "#66ff99"); y -= row_h

imp_lin  = pct_improve(rms(errs_raw), rms(errs_lin))
imp_full = pct_improve(rms(errs_raw), rms(errs_full))
txt(ax_sum, col2_x, y,
    f"  Improvement:  linear={imp_lin:.0f}%   linear+IDW={imp_full:.0f}%",
    "#aaffcc"); y -= row_h*1.3

if h_errs:
    txt(ax_sum, col2_x, y, "HEADING ERRORS", "#aaccff", 9, bold=True); y -= row_h
    txt(ax_sum, col2_x, y,
        f"  Raw heading RMS: {rms(h_errs):.3f}°  range: {min(h_errs):+.3f}° to {max(h_errs):+.3f}°",
        "#ffaacc"); y -= row_h
    h_corrs = [idw_heading(rd["true"][0], rd["true"][1]) for rd in rec_data if "h" in rd["axes"]]
    h_after = [e - c for e, c in zip(h_errs, h_corrs)]
    txt(ax_sum, col2_x, y,
        f"  After IDW RMS:  {rms(h_after):.4f}°  (≈ 0 at cal points by construction)",
        "#66ffcc"); y -= row_h*1.3

txt(ax_sum, col2_x, y, "PER-POINT DETAIL", "#aaccff", 9, bold=True); y -= row_h*0.9
hdr = f"  {'pos (true)':>14}  {'raw err':>8}  {'lin err':>8}  {'full err':>9}  {'axes':<5}"
txt(ax_sum, col2_x, y, hdr, "#888899", 7.5); y -= row_h*0.8
for rd in rec_data:
    tx, ty = rd["true"]
    line = (f"  ({tx:4.0f},{ty:4.0f})    "
            f"{rd['err_raw']:6.1f}mm   "
            f"{rd['err_lin']:6.1f}mm    "
            f"{rd['err_full']:6.1f}mm   "
            f"{rd['axes']}")
    col_d = "#66ff99" if rd["err_full"] < 5 else ("#ffcc66" if rd["err_full"] < 15 else "#ff8888")
    txt(ax_sum, col2_x, y, line, col_d, 7.5); y -= row_h*0.75

txt(ax_sum, 0.02, 0.02, f"File: {CAL_FILE}   |   Converted: {cal.get('converted','—')}",
    "#555566", 7)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(CAL_FILE), "calibration_viz.png")
fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {out_path}")
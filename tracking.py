"""
tracking.py — Shared position tracking utilities used by both new_main3.py
and sniper_4k.py (HD tracker and 4K pose contribution).

Provides:
  get_global_camera_pose()     — tag pose → world position + heading
  apply_cam_offset()           — camera lens pos → robot centre pos
  fuse_tag_estimates()         — multi-tag consensus + dist^2 weighting
  compute_line_intersections() — ray-intersection extra position estimate
"""

import math
import numpy as np
from config import (
    KNOWN_TAGS, ARENA_BOUNDS,
    HD_MAX_TAG_DIST, HD_OUT_OF_BOUNDS, HD_MAX_POSE_JUMP,
    HD_CONSENSUS_SIGMA, LINE_INTERSECT_BLEND,
)


# ─────────────────────────────────────────────────────────────────────────────
def get_global_camera_pose(tag_id, t, R):
    """
    Convert AprilTag pose (t, R) to world-frame camera position + heading.

    Returns (gx, gy, gdx, gdy) in metres, or None if tag_id unknown.
      gx, gy  = camera optical centre in world coordinates
      gdx,gdy = unit heading vector (which way camera faces)
    """
    if tag_id not in KNOWN_TAGS:
        return None
    cam_in_tag = -np.transpose(R) @ t
    local_x = cam_in_tag[0][0]
    local_z = cam_in_tag[2][0]
    ti    = KNOWN_TAGS[tag_id]
    theta = math.radians(ti["facing_angle"])
    gx  = ti["x"] + local_z * math.cos(theta) - local_x * math.sin(theta)
    gy  = ti["y"] + local_z * math.sin(theta) + local_x * math.cos(theta)
    dx  = R[2, 0]; dz = R[2, 2]
    gdx = dz * math.cos(theta) - dx * math.sin(theta)
    gdy = dz * math.sin(theta) + dx * math.cos(theta)
    return (gx, gy, gdx, gdy)


def apply_cam_offset(gx, gy, gdx, gdy,
                     forward_m: float, sideways_m: float = 0.0):
    """
    Shift from camera-lens world position to robot-centre world position.

    The lens is forward_m metres FORWARD and sideways_m metres to the RIGHT
    of the robot centre (in the robot's own frame). We subtract that offset
    to get the robot centre.

      robot_x = lens_x - forward_m * hdx - sideways_m * perp_x
      robot_y = lens_y - forward_m * hdy - sideways_m * perp_y

    where perp = heading rotated 90° clockwise (robot's right).
    """
    perp_x =  gdy    # right of robot = +90° rotation of heading
    perp_y = -gdx
    rx = gx - forward_m * gdx - sideways_m * perp_x
    ry = gy - forward_m * gdy - sideways_m * perp_y
    return rx, ry


def fuse_tag_estimates(estimates, last_x, last_y, initialized,
                       consensus_sigma=HD_CONSENSUS_SIGMA,
                       max_pose_jump=HD_MAX_POSE_JUMP,
                       out_of_bounds=HD_OUT_OF_BOUNDS):
    """
    Fuse a list of per-tag pose estimates into one position + heading.

    estimates: list of dicts with keys:
        pose  = (gx, gy, gdx, gdy)  robot-centre world coords + heading
        dist  = metres to tag
        w     = distance weight (1/dist^3)

    Bounds/jump checks are done by the caller before appending to the list,
    so this function only applies consensus soft-rejection + weighted average.

    Returns (raw_x, raw_y, raw_dx, raw_dy, sum_dist, n_valid) or None.
    """
    if not estimates:
        return None

    # ── Step 1: distance-weighted initial mean ────────────────────────────
    tw = sum(e["w"] for e in estimates)
    if tw < 1e-12:
        return None
    ix = sum(e["pose"][0] * e["w"] for e in estimates) / tw
    iy = sum(e["pose"][1] * e["w"] for e in estimates) / tw

    # ── Step 2: consensus soft-rejection ─────────────────────────────────
    total_w = raw_x = raw_y = raw_dx = raw_dy = 0.0
    sum_dist = 0.0
    for e in estimates:
        dev = math.hypot(e["pose"][0] - ix, e["pose"][1] - iy)
        cw  = math.exp(-dev**2 / (2 * consensus_sigma**2))
        w   = e["w"] * cw
        raw_x  += e["pose"][0] * w;  raw_y  += e["pose"][1] * w
        raw_dx += e["pose"][2] * w;  raw_dy += e["pose"][3] * w
        total_w += w;  sum_dist += e["dist"]

    if total_w < 1e-12:
        return None

    raw_x /= total_w;  raw_y /= total_w
    raw_dx /= total_w; raw_dy /= total_w
    mag = math.hypot(raw_dx, raw_dy)
    if mag > 0: raw_dx /= mag; raw_dy /= mag

    return raw_x, raw_y, raw_dx, raw_dy, sum_dist, len(estimates)


def _circle_circle_intersect(c1, r1, c2, r2):
    """
    Return up to two intersection points of two circles.
    c1, c2 = (x, y) centres;  r1, r2 = radii.
    Returns list of 0, 1, or 2 (x, y) tuples.
    """
    x1, y1 = c1;  x2, y2 = c2
    d = math.hypot(x2 - x1, y2 - y1)
    if d < 1e-9 or d > r1 + r2 + 1e-6 or d < abs(r1 - r2) - 1e-6:
        return []
    a  = (r1**2 - r2**2 + d**2) / (2.0 * d)
    h2 = r1**2 - a**2
    if h2 < 0:
        return []
    h  = math.sqrt(max(0.0, h2))
    mx = x1 + a * (x2 - x1) / d
    my = y1 + a * (y2 - y1) / d
    px =  h * (y2 - y1) / d
    py = -h * (x2 - x1) / d
    if h < 1e-9:
        return [(mx, my)]
    return [(mx + px, my + py), (mx - px, my - py)]


def compute_distance_circles(estimates):
    """
    Distance-based position estimate using circle intersections.

    Each tag is at a known world position. The detector returns t_z which
    is the measured distance from camera to tag. This places the robot on
    a circle of that radius centred on the tag.

    With two tags we get up to two candidate points; we pick the one inside
    the arena. With more tags we collect all pair-wise candidates, keep only
    arena-valid ones, and return their weighted average (weighted by how
    perpendicular the two circles' normals are — better geometry = lower
    uncertainty).

    Much more reliable than the old ray-intersection method, which was
    sensitive to heading estimation errors that are much larger than
    distance errors.

    Returns (x, y) in metres, or None if not enough data.
    """
    if len(estimates) < 2:
        return None

    # Build (tag_world_pos, measured_dist) pairs
    circles = []
    for e in estimates:
        tid = e["id"]
        if tid not in KNOWN_TAGS:
            continue
        ti = KNOWN_TAGS[tid]
        circles.append({
            "centre": (ti["x"], ti["y"]),
            "radius": e["dist"],          # t_z converted to dist already in estimate
        })

    if len(circles) < 2:
        return None

    candidates = []   # (x, y, weight)
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            c1, r1 = circles[i]["centre"], circles[i]["radius"]
            c2, r2 = circles[j]["centre"], circles[j]["radius"]
            pts = _circle_circle_intersect(c1, r1, c2, r2)
            for (px, py) in pts:
                # Keep only arena-valid candidates
                if not (ARENA_BOUNDS[0] - HD_OUT_OF_BOUNDS <= px <= ARENA_BOUNDS[2] + HD_OUT_OF_BOUNDS and
                        ARENA_BOUNDS[1] - HD_OUT_OF_BOUNDS <= py <= ARENA_BOUNDS[3] + HD_OUT_OF_BOUNDS):
                    continue
                # Weight by perpendicularity of the two tag normals at this point.
                # Each normal points from tag centre toward the intersection point.
                n1x, n1y = px - c1[0], py - c1[1]
                n2x, n2y = px - c2[0], py - c2[1]
                m1 = math.hypot(n1x, n1y); m2 = math.hypot(n2x, n2y)
                if m1 < 1e-9 or m2 < 1e-9:
                    continue
                # |sin θ| = |cross product| of unit normals
                cross = abs(n1x / m1 * n2y / m2 - n1y / m1 * n2x / m2)
                # Extra weight if the candidate is "in front" of both tags (y > tag_y)
                # since robots are always at higher y than the tags (tags at y≈0)
                if py < ARENA_BOUNDS[1]:
                    continue
                candidates.append((px, py, cross))

    if not candidates:
        return None

    total_w = sum(c[2] for c in candidates)
    if total_w < 1e-12:
        return None

    fx = sum(c[0] * c[2] for c in candidates) / total_w
    fy = sum(c[1] * c[2] for c in candidates) / total_w
    return fx, fy


# Keep old name as alias so any remaining callers still work
compute_line_intersections = compute_distance_circles
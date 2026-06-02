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

    Returns (raw_x, raw_y, raw_dx, raw_dy, sum_dist, n_valid)
    or None if no valid estimates.
    """
    valid = []
    for e in estimates:
        gx, gy = e["pose"][0], e["pose"][1]
        if not (ARENA_BOUNDS[0] - out_of_bounds <= gx <= ARENA_BOUNDS[2] + out_of_bounds and
                ARENA_BOUNDS[1] - out_of_bounds <= gy <= ARENA_BOUNDS[3] + out_of_bounds):
            continue
        if initialized and math.hypot(gx - last_x, gy - last_y) > max_pose_jump:
            continue
        valid.append(e)

    if not valid:
        return None

    # ── Step 1: distance-weighted initial mean ────────────────────────────
    tw = sum(e["w"] for e in valid)
    ix = sum(e["pose"][0] * e["w"] for e in valid) / tw
    iy = sum(e["pose"][1] * e["w"] for e in valid) / tw

    # ── Step 2: consensus soft-rejection ─────────────────────────────────
    total_w = raw_x = raw_y = raw_dx = raw_dy = 0.0
    sum_dist = 0.0
    for e in valid:
        dev = math.hypot(e["pose"][0] - ix, e["pose"][1] - iy)
        cw  = math.exp(-dev**2 / (2 * consensus_sigma**2))
        w   = e["w"] * cw
        raw_x  += e["pose"][0] * w;  raw_y  += e["pose"][1] * w
        raw_dx += e["pose"][2] * w;  raw_dy += e["pose"][3] * w
        total_w += w;  sum_dist += e["dist"]

    raw_x /= total_w;  raw_y /= total_w
    raw_dx /= total_w; raw_dy /= total_w
    mag = math.hypot(raw_dx, raw_dy)
    if mag > 0: raw_dx /= mag; raw_dy /= mag

    return raw_x, raw_y, raw_dx, raw_dy, sum_dist, len(valid)


def compute_line_intersections(estimates):
    """
    For each pair of visible tags, compute the ray from the tag toward the
    camera (i.e. the estimated robot-centre position direction from that tag).
    Intersect pairs of rays → collect near-intersection points → average,
    weighted by how perpendicular the rays are.

    Returns (ix, iy) world-frame intersection estimate, or None.

    Why this helps: each tag gives a ray, not just a point. When two rays
    from different tags cross (or nearly cross), their intersection is an
    additional independent position estimate that doesn't depend on per-tag
    pose estimation accuracy.
    """
    if len(estimates) < 2:
        return None

    # Each estimate gives a ray: origin = tag world position, direction = tag→camera
    rays = []
    for e in estimates:
        tag_id = e["id"]
        if tag_id not in KNOWN_TAGS:
            continue
        ti  = KNOWN_TAGS[tag_id]
        ox, oy = ti["x"], ti["y"]          # ray origin = tag world position
        # direction = tag → camera = normalised(camera_pos - tag_pos)
        cx, cy = e["pose"][0], e["pose"][1]
        dx, dy = cx - ox, cy - oy
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            continue
        rays.append({"ox": ox, "oy": oy, "dx": dx/dist, "dy": dy/dist,
                      "dist": e["dist"]})

    intersections = []
    weights       = []

    for i in range(len(rays)):
        for j in range(i+1, len(rays)):
            r1, r2 = rays[i], rays[j]
            # Solve: r1.o + t1*r1.d = r2.o + t2*r2.d
            # [r1.dx, -r2.dx] [t1]   [r2.ox - r1.ox]
            # [r1.dy, -r2.dy] [t2] = [r2.oy - r1.oy]
            a = np.array([[r1["dx"], -r2["dx"]],
                           [r1["dy"], -r2["dy"]]])
            b = np.array([r2["ox"] - r1["ox"],
                           r2["oy"] - r1["oy"]])
            det = a[0,0]*a[1,1] - a[0,1]*a[1,0]
            if abs(det) < 1e-8:
                continue   # rays nearly parallel → unreliable
            t1, t2 = np.linalg.solve(a, b)
            if t1 < 0 or t2 < 0:
                continue   # intersection behind a tag
            ix = r1["ox"] + t1 * r1["dx"]
            iy = r1["oy"] + t1 * r1["dy"]
            # Weight by perpendicularity (|sin θ| = |det / (|d1||d2|)|= |det|)
            perp_w = abs(det)
            intersections.append((ix, iy))
            weights.append(perp_w)

    if not intersections:
        return None

    total_w = sum(weights)
    fx = sum(ix * w for (ix,_), w in zip(intersections, weights)) / total_w
    fy = sum(iy * w for (_,iy), w in zip(intersections, weights)) / total_w
    return fx, fy

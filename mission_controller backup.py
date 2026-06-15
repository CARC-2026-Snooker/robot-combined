"""
mission_controller.py — High-level game loop.

Startup order:
  1. python new_main3.py          (vision + telemetry on port 5005)
  2. python movement_server.py    (motor driver on port 5010)
  3. python mission_controller.py (this file — reads state, sends commands)

All coordinates in mm (world frame, origin = Hole_3 corner).
All headings in degrees (0=right +X, 90=up +Y, 270=down toward robot area).
All movement commands are in ROBOT FRAME (not table frame).
"""

import sys, os, time, json, math, subprocess, socket, threading
from dataclasses import dataclass
import time
from datetime import datetime

from config import (
    TELEMETRY_PORT, SNIPER_TELEMETRY_PORT, SNIPER_FIRE_PORT, MOVE_DEAD_MM, ROT_DEAD_DEG,
    REAL_ZONES, MOVE_POWER_MIN, MOVE_POWER_MAX,
    ROT_POWER_MIN, ROT_POWER_MAX,
    MOVE_GAIN_MM, ROT_GAIN_DEG,
    VISION_CMD_PORT,
)
from movement_client import (
    MovementClient, MoveCmd,
    world_errors_to_omni, omni, move, rotate_by, brake, stop, action,
    compute_move_cmd, _clamp_power  # legacy alias still available
)

sys.path.append("/home/page/robot-combined/robot-vision-and-planning")
from ball_align import BallAlignState, do_ball_align

import signal
import random
import traceback

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

@dataclass
class PID:
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    output_limit: float = 100.0
    integral_limit: float = 1000.0
    deadband: float = 0.0

    integral: float = 0.0
    prev_error: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = None

    def update(self, error: float, dt: float) -> float:
        if abs(error) <= self.deadband:
            error = 0.0

        if dt <= 0.0:
            dt = 1e-3

        self.integral = _clamp(
            self.integral + error * dt,
            -self.integral_limit,
            self.integral_limit,
        )

        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.prev_error = error

        out = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )
        return _clamp(out, -self.output_limit, self.output_limit)

# ─────────────────────────────────────────────────────────────────────────────
# ROBOT STATE — live telemetry subscriber
# ─────────────────────────────────────────────────────────────────────────────
class RobotState:
    """
    Subscribes to new_main3.py's UDP telemetry (port 5005).
    All values are updated in a background thread.

    Key properties:
      .initialized        → bool: tracking active
      .position_mm        → (x_mm, y_mm): robot centre world position
      .heading            → (dx, dy): unit heading vector
      .heading_deg        → float: heading in degrees
      .tracking_state     → "STILL"|"MOVING"|"TRANSITIONING"|"LOST"
      .app_state          → "DRIVING"|"AIMING"
      .get()              → full raw dict for custom access
    """
    def __init__(self, port: int = TELEMETRY_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self._sock.bind(("0.0.0.0", port))
        self._sock.settimeout(0.1)
        self._data: dict = {}
        self._lock = threading.Lock()
        self._running = True
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        while self._running:
            try:
                raw, _ = self._sock.recvfrom(4096)
                data = json.loads(raw.decode())
                with self._lock:
                    self._data = data

            except (socket.timeout, json.JSONDecodeError):
                pass

            except OSError:
                # socket was closed during shutdown
                break

    def get(self) -> dict:
        with self._lock:
            return dict(self._data)

    @property
    def initialized(self) -> bool:
        return self.get().get("initialized", False)

    @property
    def position_mm(self):
        d = self.get()
        return d.get("x_mm", 0.0), d.get("y_mm", 0.0)

    @property
    def heading(self):
        d = self.get()
        return d.get("heading_dx", 1.0), d.get("heading_dy", 0.0)

    @property
    def heading_deg(self) -> float:
        hdx, hdy = self.heading
        return (math.degrees(math.atan2(hdy, hdx)) + 360) % 360

    @property
    def tracking_state(self) -> str:
        return self.get().get("state", "LOST")

    @property
    def app_state(self) -> str:
        return self.get().get("app_state", "DRIVING")

    def stop(self):
        self._running = False
        try: self._sock.close()
        except: pass

STOP_EVENT = threading.Event()
_GLOBAL_MC = None

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def emergency_stop(reason="emergency stop"):
    global _GLOBAL_MC
    STOP_EVENT.set()
    print(f"\n[EMERGENCY STOP] {reason}")

    if _GLOBAL_MC is not None:
        # UDP can drop packets, so send stop commands a few times.
        for _ in range(5):
            try:
                _GLOBAL_MC.brake()
                _GLOBAL_MC.stop()
                _GLOBAL_MC.action("stop_spin_speed")
            except Exception as e:
                print(f"[EMERGENCY STOP] send failed: {e}")
            time.sleep(0.03)

def _signal_emergency_stop(signum, frame):
    emergency_stop("Ctrl+C / signal")
    raise SystemExit(130)

signal.signal(signal.SIGINT, _signal_emergency_stop)
signal.signal(signal.SIGTERM, _signal_emergency_stop)

def load_shooting_plan(path="robot_shooting_plan.json") -> dict:
    """Load the plan produced by snooker_planner.py."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def wait_for_tracking(state: RobotState, timeout=15.0, verbose=True):
    """Block until tracking initialises or raise TimeoutError."""
    t0 = time.time()
    while not state.initialized:
        if time.time() - t0 > timeout:
            raise TimeoutError("Tracking not initialised — make sure tags are visible.")
        if verbose:
            print(f"  waiting for tracking... ({time.time()-t0:.0f}s)", end="\r")
        time.sleep(0.1)
    if verbose:
        print(f"  Tracking OK. Position: {state.position_mm}")


def _ang_diff(a, b):
    """Signed angular difference a - b, normalised to (-180, 180]."""
    d = a - b
    while d >  180: d -= 360
    while d < -180: d += 360
    return d


class SniperState:
    """
    Subscribes to sniper_4k.py's telemetry broadcast (same port as new_main3).
    When sniper_4k.py is running, it takes over the port and sends richer packets.
    This class surfaces all the sniper-specific fields so mission_controller can
    react to the aiming progress without pressing 'f' manually.

    Key properties:
      .active           → bool: sniper is running and broadcasting
      .shot_mode        → "STRAIGHT" | "CUT"
      .target_pocket    → e.g. "Hole_3"
      .target_zone      → e.g. "B"
      .tag_found        → bool: 4K camera sees the target tag
      .ang_correction   → float | None: degrees to rotate (+ = TURN RIGHT)
      .ang_locked       → bool: angle is within ANGLE_LOCK_DEG
      .lateral_mm       → float | None: mm to slide (+ = MOVE RIGHT; STRAIGHT only)
      .lat_locked       → bool: lateral alignment within LATERAL_TOL_PX
      .x_error_mm       → float: HD position X error vs plan
      .y_error_mm       → float: HD position Y error vs plan
      .pos_locked       → bool: position/lateral check passed
      .fully_locked     → bool: both pos_locked AND ang_locked → ready to fire
      .position_mm      → (x_mm, y_mm) same as RobotState
      .tracking_state   → "STILL"|"MOVING"|"LOST"
    """

    def __init__(self, port: int = SNIPER_TELEMETRY_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self._sock.bind(("0.0.0.0", port))
        self._sock.settimeout(0.1)
        self._data: dict = {}
        self._lock = threading.Lock()
        self._running = True
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        while self._running:
            try:
                raw, addr = self._sock.recvfrom(8192)
                data = json.loads(raw.decode())
                with self._lock:
                    self._data = data
                    print(f"[SniperState DEBUG] init={data.get('initialized')} | "
                          f"ang_correction_deg={data.get('ang_correction_deg')} | ang_locked={data.get('ang_locked')} | "
                          f"lateral_px/mm={data.get('lateral_mm')}/{data.get('lateral_error_px')} | lat_locked={data.get('lat_locked')} | "
                          f"state={data.get('state')} | pos_locked={data.get('pos_locked')} | fully_locked={data.get('fully_locked')}")
            except (socket.timeout, json.JSONDecodeError):
                pass
            except Exception as e:
                print(f"[SniperState ERROR] {e}")

    def get(self) -> dict:
        with self._lock:
            return dict(self._data)

    @property
    def active(self) -> bool:
        return self.get().get("sniper_active", False)

    @property
    def shot_mode(self) -> str:
        return self.get().get("shot_mode", "UNKNOWN")

    @property
    def target_pocket(self) -> str:
        return self.get().get("target_pocket", "")

    @property
    def target_zone(self) -> str:
        return self.get().get("target_zone", "")

    @property
    def tag_found(self) -> bool:
        return self.get().get("tag_found_4k", False)

    @property
    def ang_correction(self):
        return self.get().get("ang_correction_deg")

    @property
    def ang_locked(self) -> bool:
        return self.get().get("ang_locked", False)

    @property
    def lateral_mm(self):
        return self.get().get("lateral_mm")

    @property
    def lat_locked(self) -> bool:
        return self.get().get("lat_locked", False)

    @property
    def x_error_mm(self) -> float:
        return self.get().get("x_error_mm", 0.0)

    @property
    def y_error_mm(self) -> float:
        return self.get().get("y_error_mm", 0.0)

    @property
    def pos_locked(self) -> bool:
        return self.get().get("pos_locked", False)

    @property
    def fully_locked(self) -> bool:
        return self.get().get("fully_locked", False)

    @property
    def position_mm(self):
        d = self.get()
        return d.get("x_mm", 0.0), d.get("y_mm", 0.0)

    @property
    def tracking_state(self) -> str:
        return self.get().get("state", "LOST")

    def stop(self):
        self._running = False
        try: self._sock.close()
        except: pass

def _scaled_pulse(error_abs: float,
                  full_error: float,
                  min_s: float = 0.01,
                  max_s: float = 0.08) -> float:
    """
    Small error -> short pulse.
    Large error -> long pulse.
    """
    ratio = _clamp(error_abs / full_error, 0.0, 1.0)
    return min_s + ratio * (max_s - min_s)

def _scaled_settle(error_abs: float,
                   full_error: float,
                   min_s: float = 0.20,
                   max_s: float = 0.60) -> float:
    """
    Near lock -> wait longer for camera to catch up.
    """
    ratio = 1.0 - _clamp(error_abs / full_error, 0.0, 1.0)
    return min_s + ratio * (max_s - min_s)

def wait_for_sniper_lock(sniper: SniperState,
                         mc: MovementClient,
                         timeout: float = 240.0,
                         verbose: bool = True) -> bool:

    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def _scaled_pulse(error_abs: float,
                      full_error: float,
                      min_s: float = 0.01,
                      max_s: float = 0.08) -> float:
        ratio = _clamp(error_abs / full_error, 0.0, 1.0)
        return min_s + ratio * (max_s - min_s)

    def _scaled_settle(error_abs: float,
                       full_error: float,
                       min_s: float = 0.20,
                       max_s: float = 0.60) -> float:
        ratio = 1.0 - _clamp(error_abs / full_error, 0.0, 1.0)
        return min_s + ratio * (max_s - min_s)

    # ─────────────────────────────────────────────────────────────
    # Fine aiming limits
    # ─────────────────────────────────────────────────────────────

    FINE_ROT_MIN = 7
    FINE_ROT_MAX = 20

    FINE_MOVE_MIN = 7
    FINE_MOVE_MAX = 25

    ANG_FULL_DEG = 15.0
    LAT_FULL_MM = 40.0
    POS_FULL_MM = 40.0

    # ─────────────────────────────────────────────────────────────
    # AprilTag trust thresholds
    # ─────────────────────────────────────────────────────────────

    ANGLE_PRIORITY_DEG = 1.25
    CUT_ANGLE_PRIORITY_DEG = 1.25

    t0 = time.time()
    last_print = 0.0

    print("STARTING SNIPER LOCKING")

    while time.time() - t0 < timeout:

        if sniper.fully_locked:
            mc.brake()

            if verbose:
                print(f"[sniper] LOCKED after {time.time()-t0:.1f}s")

            return True

        d = sniper.get()

        if not d:
            time.sleep(0.05)
            continue

        mode        = d.get("shot_mode", "STRAIGHT")
        initialized = d.get("initialized", False)

        ang_corr    = d.get("ang_correction_deg")
        tag_found   = d.get("tag_found_4k", False)
        ang_locked  = d.get("ang_locked", False)

        lat_mm      = d.get("lateral_mm")
        lat_locked  = d.get("lat_locked", False)

        x_err       = d.get("x_error_mm", 0.0)
        y_err       = d.get("y_error_mm", 0.0)

        vX = 0
        vY = 0
        vR = 0

        pulse_error = 0.0
        pulse_full  = 1.0

        # ==========================================================
        # STRAIGHT MODE — three-line alignment, no heading sensor.
        #
        # The 4K camera sees three vertical lines:
        #   GREEN  (AIM)    = camera centre = robot shooting axis
        #   YELLOW (BALL)   = ball centroid column
        #   MAGENTA(TARGET) = tag column ± live corner offset
        #
        # sniper_4k.py computes:
        #   ang_correction_deg = mid_error_deg
        #     = ((ball_col + target_col)/2 - cam_cx) converted to degrees
        #     → drives vR (ROTATE) to centre the midpoint on the camera axis
        #     positive = midpoint right of centre → TURN RIGHT
        #
        #   lateral_mm = diff_mm
        #     = (ball_col - target_col) / lateral_sens_px_per_mm
        #     → drives vX (STRAFE) to collapse the gap between ball and target
        #     positive = ball right of target → MOVE RIGHT
        #
        # Both axes run SIMULTANEOUSLY — no priority switching.
        # Rotating shifts both lines equally; strafing shifts them by different
        # amounts (depth ratio), so the two signals are nearly decoupled.
        #
        # Both zero ↔ ball_col = target_col = cam_cx ← all three lines coincide.
        # ==========================================================

        if mode == "STRAIGHT":

            # Rotate: drive midpoint onto camera centre
            if ang_corr is not None and not ang_locked:

                rot_spd = _clamp_power(
                    abs(ang_corr),
                    ANG_FULL_DEG,
                    FINE_ROT_MIN,
                    FINE_ROT_MAX,
                    ROT_DEAD_DEG,
                )

                vR = rot_spd if ang_corr > 0 else -rot_spd

            # Strafe: drive ball column onto target column
            if lat_mm is not None and not lat_locked:

                lat_spd = _clamp_power(
                    abs(lat_mm),
                    LAT_FULL_MM,
                    FINE_MOVE_MIN,
                    FINE_MOVE_MAX,
                    MOVE_DEAD_MM,
                )

                vX = lat_spd if lat_mm > 0 else -lat_spd

            pulse_error = max(
                abs(ang_corr or 0.0),
                abs(lat_mm or 0.0)
            )

            pulse_full = max(ANG_FULL_DEG, LAT_FULL_MM)

        # ==========================================================
        # CUT MODE — identical servo loop to STRAIGHT.
        # sniper_4k.py computes the same lateral_mm and ang_correction signals;
        # the only difference is target_lateral_px ≠ 0 (ghost-ball offset).
        # mission_controller doesn't need to know which mode it's in — it just
        # follows the same angle + lateral servo loop.
        # ==========================================================

        else:

            angle_large = (
                tag_found and
                ang_corr is not None and
                abs(ang_corr) > ANGLE_PRIORITY_DEG
            )

            # ------------------------------------------------------
            # Large heading error — rotate only
            # ------------------------------------------------------

            if angle_large:

                rot_spd = _clamp_power(
                    abs(ang_corr),
                    ANG_FULL_DEG,
                    FINE_ROT_MIN,
                    FINE_ROT_MAX,
                    ROT_DEAD_DEG,
                )

                vR = rot_spd if ang_corr > 0 else -rot_spd

                pulse_error = abs(ang_corr)
                pulse_full = ANG_FULL_DEG

            # ------------------------------------------------------
            # Small heading error — rotate + strafe together
            # ------------------------------------------------------

            else:

                if tag_found and ang_corr is not None and not ang_locked:

                    rot_spd = _clamp_power(
                        abs(ang_corr),
                        ANG_FULL_DEG,
                        FINE_ROT_MIN,
                        FINE_ROT_MAX,
                        ROT_DEAD_DEG,
                    )

                    vR = rot_spd if ang_corr > 0 else -rot_spd

                if lat_mm is not None and not lat_locked:

                    lat_spd = _clamp_power(
                        abs(lat_mm),
                        LAT_FULL_MM,
                        FINE_MOVE_MIN,
                        FINE_MOVE_MAX,
                        MOVE_DEAD_MM,
                    )

                    vX = lat_spd if lat_mm > 0 else -lat_spd

                pulse_error = max(
                    abs(ang_corr or 0.0),
                    abs(lat_mm or 0.0)
                )

                pulse_full = max(
                    ANG_FULL_DEG,
                    LAT_FULL_MM
                )

        # ==========================================================
        # Debug print
        # ==========================================================

        if verbose and time.time() - last_print > 1.0:

            ang_str = (
                f"{ang_corr:+.2f}°"
                if ang_corr is not None
                else "searching"
            )

            lat_str = (
                f"{lat_mm:+.1f}mm"
                if lat_mm is not None
                else "n/a"
            )

            print(
                f"  sniper: "
                f"mode={mode} "
                f"tag={'FOUND' if tag_found else 'search'} "
                f"ang={'OK' if ang_locked else ang_str} "
                f"lat={'OK' if lat_locked else lat_str} "
                f"x={x_err:+.1f} "
                f"y={y_err:+.1f} "
                f"cmd=({vX:+d},{vY:+d},{vR:+d})"
            )

            last_print = time.time()

        if vX == 0 and vY == 0 and vR == 0:
            time.sleep(0.05)
            continue

        pulse_s = _scaled_pulse(
            pulse_error,
            pulse_full,
            min_s=0.06,
            max_s=0.09,
        )

        settle_s = _scaled_settle(
            pulse_error,
            pulse_full,
            min_s=0.30,
            max_s=0.60,
        )

        mc.send(
            MoveCmd(
                cmd="omni",
                vX=vX,
                vY=vY,
                vR=vR,
            )
        )

        if verbose:
            print(
                f"moving ({vX:+d},{vY:+d},{vR:+d}) "
                f"pulse={pulse_s:.3f}s "
                f"settle={settle_s:.3f}s"
            )

        time.sleep(pulse_s)

        mc.brake()

        time.sleep(settle_s)

    mc.brake()

    if verbose:
        print(
            f"[wait_for_sniper_lock] TIMEOUT after "
            f"{timeout:.0f}s"
        )

    return False

# ─────────────────────────────────────────────────────────────────────────────
# DRIVE TO — simultaneous X+Y+rotation via OmniMovement
# ─────────────────────────────────────────────────────────────────────────────
DRIVE_HEADING    = 270.0   # heading to maintain while translating (all tags visible)
X_FULL_POWER_MM  = 250     # X error that maps to MOVE_POWER_MAX
Y_FULL_POWER_MM  = 450     # Y error that maps to MOVE_POWER_MAX
ROT_FULL_DEG     = 60      # heading error that maps to ROT_POWER_MAX


def drive_to(target_x_mm: float, target_y_mm: float,
             final_heading_deg: float,
             state: RobotState, mc: MovementClient,
             pos_tol_mm:  float = 30.0,
             ang_tol_deg: float = 2.0,
             timeout:     float = 240.0,
             verbose:     bool  = True) -> bool:
    """
    Drive to (target_x_mm, target_y_mm), then rotate to final_heading_deg.

    Phase 1:
      PID on world X and Y position error, plus heading-hold PID at DRIVE_HEADING.
      The world-frame PID outputs are converted into robot-frame vX/vY.

    Phase 2:
      PID on heading only, with vX=vY=0.
    """
    if verbose:
        print(f"[drive_to] target=({target_x_mm:.0f}, {target_y_mm:.0f})mm  "
              f"final={final_heading_deg:.1f}°")

    # --- Tune these starting points on the real robot ---
    # Translation:
    x_pid = PID(kp=0.8, ki=0.001, kd=0.25,
                output_limit=MOVE_POWER_MAX,
                integral_limit=2000.0,
                deadband=5.0)

    y_pid = PID(kp=0.8, ki=0.001, kd=0.25,
                output_limit=MOVE_POWER_MAX,
                integral_limit=2000.0,
                deadband=5.0)

    # Heading hold during translation:
    hold_heading_pid = PID(kp=2.5, ki=0.00, kd=0.08,
                           output_limit=ROT_POWER_MAX,
                           integral_limit=360.0,
                           deadband=0.5)

    # Final rotate-to-heading:
    turn_heading_pid = PID(kp=3.5, ki=0.00, kd=0.10,
                           output_limit=ROT_POWER_MAX,
                           integral_limit=360.0,
                           deadband=0.5)

    t0 = time.time()
    last_t = t0
    _last_print = 0.0

    # ── Phase 1: Translate ────────────────────────────────────────────────────
    if verbose:
        print("  Phase 1: translating...")

    while time.time() - t0 < timeout:
        if STOP_EVENT.is_set():
            mc.brake()
            return False
        if not state.initialized:
            time.sleep(0.02)
            continue

        now = time.time()
        dt = now - last_t
        last_t = now

        x_mm, y_mm = state.position_mm

        # World-frame errors: positive means "need to move toward +axis"
        x_err = target_x_mm - x_mm
        y_err = target_y_mm - y_mm

        if math.hypot(x_err, y_err) < pos_tol_mm:
            break

        # PID outputs in world coordinates
        world_vx = x_pid.update(x_err, dt)
        world_vy = y_pid.update(y_err, dt)

        # Heading-hold PID during translation
        # Keep your old sign convention by using _ang_diff exactly as before.
        ang_err = _ang_diff(state.heading_deg, DRIVE_HEADING)
        vR = int(round(-hold_heading_pid.update(ang_err, dt)))

        # Rotate world velocity into robot frame.
        # Robot frame:
        #   vX > 0 = right
        #   vY > 0 = forward
        h = math.radians(state.heading_deg)
        sin_h = math.sin(h)
        cos_h = math.cos(h)

        vX_raw = -world_vx * sin_h + world_vy * cos_h
        vY_raw =  world_vx * cos_h + world_vy * sin_h

        # Keep combined translation under the motor limit
        mag = math.hypot(vX_raw, vY_raw)
        if mag > MOVE_POWER_MAX and mag > 1e-9:
            scale = MOVE_POWER_MAX / mag
            vX_raw *= scale
            vY_raw *= scale

        cmd = MoveCmd(
            cmd="omni",
            vX=int(round(vX_raw)),
            vY=int(round(vY_raw)),
            vR=vR,
        )
        mc.send(cmd)

        if verbose and time.time() - _last_print > 1.0:
            print(
                f"    pos=({x_mm:.0f},{y_mm:.0f}) "
                f"err=({x_err:+.0f},{y_err:+.0f})mm "
                f"hdg={state.heading_deg:.1f}° "
                f"ang_err={ang_err:+.1f}° → {cmd.describe()}"
            )
            _last_print = time.time()

        time.sleep(0.01)

    else:
        mc.brake()
        if verbose:
            print("  [drive_to] TIMEOUT in phase 1.")
        return False

    mc.brake()
    time.sleep(0.3)

    # ── Phase 2: Rotate to final heading ──────────────────────────────────────
    if verbose:
        print(f"  Phase 2: rotating to {final_heading_deg:.1f}°...")

    turn_heading_pid.reset()
    rot_t0 = time.time()
    last_t = rot_t0

    while time.time() - rot_t0 < 10.0:
        if STOP_EVENT.is_set():
            mc.brake()
            return False
        if not state.initialized:
            time.sleep(0.02)
            continue

        now = time.time()
        dt = now - last_t
        last_t = now

        ang_err = _ang_diff(state.heading_deg, final_heading_deg)
        if abs(ang_err) < ang_tol_deg:
            break

        vR = int(round(-turn_heading_pid.update(ang_err, dt)))
        mc.omni(vR=vR)
        time.sleep(0.01)

    mc.brake()
    time.sleep(0.2)

    if verbose:
        x, y = state.position_mm
        print(f"  Arrived: ({x:.0f}, {y:.0f})mm  heading={state.heading_deg:.1f}°")

    return True

# ─────────────────────────────────────────────────────────────────────────────
# AUTOMATIC SCAN — triggers new_main3 to take a photo and run the planner
# without pressing 's' manually
# ─────────────────────────────────────────────────────────────────────────────
def do_scan(plan_path: str = "robot-vision-and-planning/robot_shooting_plan.json",
            timeout:   float = 20.0) -> dict:
    """
    Trigger new_main3.py to capture a frame and run the planner.
 
    Sends {"cmd":"scan"} to VISION_CMD_PORT and waits for {"status":"done"}.
    new_main3 handles the scan using its own already-open camera handle,
    writes robot_shooting_plan.json, then replies.
 
    This avoids the V4L2 "can't open camera by name" error that happens when
    mission_controller tries to open the camera while new_main3 already has it.
 
    Returns the loaded shooting plan dict (empty dict if scan found nothing).
    """
    try:
        from config import VISION_CMD_PORT
    except ImportError:
        VISION_CMD_PORT_local = 5007
    else:
        VISION_CMD_PORT_local = VISION_CMD_PORT
 
    print("[do_scan] Sending scan command to new_main3...")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    try:
        s.sendto(json.dumps({"cmd": "scan"}).encode(),
                 ("127.0.0.1", VISION_CMD_PORT_local))
        # Wait for new_main3 to finish the scan and reply
        raw, _ = s.recvfrom(1024)
        reply = json.loads(raw.decode())
        if reply.get("status") == "done":
            print("[do_scan] Scan complete.")
        else:
            print(f"[do_scan] Unexpected reply: {reply}")
    except socket.timeout:
        print(f"[do_scan] No reply from new_main3 after {timeout:.0f}s — "
              "is new_main3.py running?")
        return {}
    except Exception as e:
        print(f"[do_scan] Error: {e}")
        return {}
    finally:
        s.close()
 
    return load_shooting_plan(plan_path)


def do_confirm_shot(zone: str,
                    plan: dict,
                    timeout: float = 120.0) -> bool:
    """
    Tell new_main3.py to release its HD camera and launch sniper_4k.py.
    Equivalent to pressing 'c' in new_main3.py.

    Sends {"cmd":"confirm_shot", "target_data": <plan[zone]>} to VISION_CMD_PORT
    and waits for {"status":"sniper_done"} when sniper exits.

    new_main3 handles the actual camera release + subprocess.run(sniper_4k.py)
    so there is no V4L2 conflict — only one process holds the HD camera at a time.

    Returns True when sniper exits normally.
    """
    if zone not in plan:
        print(f"[do_confirm_shot] Zone '{zone}' not in plan.")
        return False

    target_data = plan[zone]
    print(f"[do_confirm_shot] Sending confirm_shot to new_main3 for zone {zone}  "
          f"({target_data['color']} → {target_data['target_pocket']}  "
          f"cut={target_data['cut_angle_deg']}°)")

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    try:
        s.sendto(
            json.dumps({"cmd": "confirm_shot", "target_data": target_data}).encode(),
            ("127.0.0.1", VISION_CMD_PORT)
        )
        # Block until new_main3 replies (sniper has exited)
        raw, _ = s.recvfrom(1024)
        reply = json.loads(raw.decode())
        return reply.get("status") == "sniper_done"
    except socket.timeout:
        print(f"[do_confirm_shot] No reply from new_main3 after {timeout:.0f}s")
        return False
    except Exception as e:
        print(f"[do_confirm_shot] Error: {e}")
        return False
    finally:
        s.close()


# ─────────────────────────────────────────────────────────────────────────────
# BALL PICKUP — back camera alignment + claw
# ─────────────────────────────────────────────────────────────────────────────
def do_pickup_sequence(mc: MovementClient,
                       timeout_align: float = 25.0,
                       verbose: bool = True) -> bool:
    """
    Full ball pickup using back-camera alignment.
    ball_align.py must be running separately (opens back camera).

    Sequence:
      1. Open claw
      2. Align: strafe LEFT/RIGHT and move BACK until ball is centred and close
      3. Close claw

    Usage in competition loop:
        # Drive robot to pickup position (claw facing the ball pedestal)
        drive_to(rx, ry, PICKUP_HEADING, state, mc)

        # Launch ball_align.py if not already running:
        #   import subprocess, sys
        #   subprocess.Popen([sys.executable, "ball_align.py"])
        #   time.sleep(1.0)

        success = do_pickup_sequence(mc)
    """
    if verbose:
        print("[pickup] Opening claw...")
    mc.action("open_claw")
    time.sleep(0.75)

    align = BallAlignState()
    time.sleep(0.5)

    if verbose:
        print("[pickup] Aligning with ball using back camera...")
    success = do_ball_align(mc, align, timeout=timeout_align, verbose=verbose)
    time.sleep(0.5)
    align.stop()

    if not success:
        if verbose: print("[pickup] Alignment failed.")
        mc.brake()
        return False

    mc.brake()
    time.sleep(0.2)
    if verbose: print("[pickup] Closing claw...")
    mc.action("close_claw")
    time.sleep(0.6)
    if verbose: print("[pickup] Done.")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# FULL COMMAND-CHAIN EXAMPLE — tests every available command
# ─────────────────────────────────────────────────────────────────────────────
def demo_all_commands(state: RobotState, mc: MovementClient):
    """
    Runs through every available command so you can see how each one works.
    Comment out sections you don't need.
    """
    print("\n━━━ DEMO: ALL COMMANDS ━━━")

    # ── 1. Read robot state ───────────────────────────────────────────────
    print(f"\n[1] State")
    print(f"    Position : {state.position_mm}")
    print(f"    Heading  : {state.heading_deg:.1f}°  vector={state.heading}")
    print(f"    Tracking : {state.tracking_state}")

    # ── 2. OmniMovement — the main drive command ──────────────────────────
    print("\n[2] OmniMovement (all axes simultaneously):")
    # Forward + slight right + slight CW rotation, all at once
    mc.omni(vX=20, vY=30, vR=0);   time.sleep(0.5); mc.brake(); time.sleep(0.3)
    # Strafe left only
    mc.omni(vX=-25, vY=0, vR=0);   time.sleep(0.5); mc.brake(); time.sleep(0.3)
    # Rotate CW only
    mc.omni(vX=0, vY=0, vR=20);    time.sleep(0.4); mc.brake(); time.sleep(0.3)
    # Rotate CCW only
    mc.omni(vX=0, vY=0, vR=-20);   time.sleep(0.4); mc.brake(); time.sleep(0.3)

    # ── 3. Move with degree convention ────────────────────────────────────
    print("\n[3] Move (degree: 0=fwd 90=right 180=back 270=left):")
    mc.move(speed=25, degree=0);    time.sleep(0.4); mc.brake(); time.sleep(0.3)
    mc.move(speed=25, degree=90);   time.sleep(0.4); mc.brake(); time.sleep(0.3)
    mc.move(speed=25, degree=180);  time.sleep(0.4); mc.brake(); time.sleep(0.3)
    mc.move(speed=25, degree=270);  time.sleep(0.4); mc.brake(); time.sleep(0.3)
    # Move with simultaneous rotation
    mc.move(speed=25, degree=0, rotation=15); time.sleep(0.4); mc.brake(); time.sleep(0.3)

    # ── 6. Actions ────────────────────────────────────────────────────────
    print("\n[6] Actions:")
    mc.action("open_claw");          time.sleep(0.5)
    mc.action("close_claw");         time.sleep(0.5)

    # ── 7. world_errors_to_omni — what drive_to uses internally ──────────
    print("\n[7] world_errors_to_omni:")
    # Robot is 100mm to the right of target at 270° heading
    cmd = world_errors_to_omni(x_err_mm=100, y_err_mm=0, ang_err_deg=0,
                                heading_deg=270.0)
    print(f"    x_err=+100mm at 270° → {cmd.describe()}")
    mc.send(cmd); time.sleep(0.4); mc.brake()

    # Robot needs to go forward (toward lower Y) at 270° heading
    cmd = world_errors_to_omni(x_err_mm=0, y_err_mm=-150, ang_err_deg=0,
                                heading_deg=270.0)
    print(f"    y_err=-150mm at 270° → {cmd.describe()}")
    mc.send(cmd); time.sleep(0.4); mc.brake()

    # Heading 5° off + position error — all corrected simultaneously
    cmd = world_errors_to_omni(x_err_mm=80, y_err_mm=-80, ang_err_deg=5,
                                heading_deg=275.0)
    print(f"    x=+80 y=-80 ang=+5° at 275° → {cmd.describe()}")
    mc.send(cmd); time.sleep(0.4); mc.brake()

    print("\n━━━ DEMO COMPLETE ━━━")


# ─────────────────────────────────────────────────────────────────────────────
# SEND HELP
# ─────────────────────────────────────────────────────────────────────────────
PHOTO_POS_X, PHOTO_POS_Y, PHOTO_HEAD = 475, 2640, 270.0

def fire(mc, shot_time, power = 50):
    # wait until it's been 15s since the last shot
    if shot_time:
        sec_since_shot = (datetime.now() - shot_time).total_seconds()
        while sec_since_shot < 15:
            print(f"waiting 15 seconds after last shot, {sec_since_shot} seconds left")
            sec_since_shot = (datetime.now() - shot_time).total_seconds()


    # Fire mechanism — spin starts AFTER lock, no vibration during aiming
    mc.action("rev_spin_power", power)
    time.sleep(2)
    mc.action("release_trigger")
    time.sleep(2)
    mc.action("reset_trigger")
    mc.action("stop_spin_power")
    print("    Shot fired!")
    shot_time = datetime.now()
    return shot_time

def face_ball_picking(state, mc, white_ball_list, last_side):
    # go to between the two sides, left first, then right
    drive_to(550, 3100, PHOTO_HEAD, state, mc, pos_tol_mm=20.0, ang_tol_deg=1.0)
    mc.brake()
    time.sleep(0.3)
    if white_ball_list[0]: # if still have left do left, else right
        mc.omni(vX=0, vY=0, vR=360)
        time.sleep(1)
        mc.brake()
        last_side = 0
    else:
        mc.omni(vX=0, vY=0, vR=-360)
        time.sleep(1)
        mc.brake()
        last_side = 1
    mc.omni(vX=0, vY=-180, vR=0)
    time.sleep(0.75)
    mc.brake()
    time.sleep(0.5)

    return last_side

def go_home(state, mc):
    drive_to(PHOTO_POS_X, 3000, PHOTO_HEAD, state, mc, pos_tol_mm=40.0, ang_tol_deg=3.0)
    # drive_to(PHOTO_POS_X, 3375, PHOTO_HEAD, state, mc, pos_tol_mm=20.0, ang_tol_deg=1.0) # actual where we want to go
    drive_to(PHOTO_POS_X, 3250, PHOTO_HEAD, state, mc, pos_tol_mm=20.0, ang_tol_deg=1.0) # the robot is weird, so this is what would be the above

def blind_shot(state, mc, shot_time):
    drive_to(300, PHOTO_POS_Y, PHOTO_HEAD, state, mc, pos_tol_mm=40.0, ang_tol_deg=0.85)
    time.sleep(1)
    shot_time = fire(mc, shot_time, 15)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — your game loop goes here
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global _GLOBAL_MC

    state = RobotState(TELEMETRY_PORT)
    mc    = MovementClient()
    _GLOBAL_MC = mc

    white_balls = [2,2]
    last_side = 0 # left:0 right:1

    shot_time = None

    mc.action("close_claw")
    
    # go to left side in front of white balls pedestal
    mc.omni(vX=-80, vY=-160, vR=-190)
    time.sleep(1.9)
    mc.brake()
    # mc.rotate_to(speed=100, degree=-90)

    try:
        # Ball shooting
        for i in range(4):
            if i:
                last_side = face_ball_picking(state, mc, white_balls, last_side)

            result = do_pickup_sequence(mc, verbose=True)
            while not result:
                print("previous pick up attempt failed, trying again")
                result = do_pickup_sequence(mc, verbose=True)

            white_balls[last_side] = white_balls[last_side] - 1

            # go back to middle and align to view point
            if last_side == 0: # Left
                mc.omni(vX=-100, vY=300, vR=-220)
            else: # Right
                mc.omni(vX=100, vY=300, vR=220)
            time.sleep(1.4)
            mc.brake()

            time.sleep(0.5)

            drive_to(PHOTO_POS_X, PHOTO_POS_Y, PHOTO_HEAD, state, mc, pos_tol_mm=40.0, ang_tol_deg=0.85)
            # shot_time = blind_shot(state, mc, shot_time)










            # take a capture of the view and get plan
            plan = do_scan()
            if not plan:
                print("No balls left — done!")
                break
            else:
                print(plan)

            # pick what ball/zone to go for

            # go for red first then blue
            zones = list(plan.values())
            red_zones = [z for z in zones if z["color"] == "Red"]

            if red_zones:
                color_filtered_zones = red_zones
            else:
                color_filtered_zones = zones
            
            # for the picked color: go for middle > corner straights > the rest (back 4)
            color_filtered_zones = [zone["target_ball_zone"] for zone in color_filtered_zones]
            print(color_filtered_zones)

            if 'E' in color_filtered_zones:
                picked_zone = 'E'

            elif 'A' in color_filtered_zones or 'B' in color_filtered_zones:
                ab_zones = [z for z in color_filtered_zones if z in ('A', 'B')]
                picked_zone = random.choice(ab_zones)

            else:
                picked_zone = random.choice(color_filtered_zones) # TODO: need to switch it to pick the lowest/least cut angle

            shot = plan[picked_zone]
            print(f"\n[9] Shooting zone {picked_zone} ({shot['color']} → {shot['target_pocket']})")

            # input("before sniper location aim")

            # Drive to coarse alignment position
            rx, ry = shot["robot_alignment_xy"]
            drive_to(rx, ry, 270.0, state, mc)
            drive_to(rx, ry, shot["firing_angle_deg"], state, mc)
            time.sleep(0.3)

            # input("arrived")

            # Create SniperState BEFORE sending confirm_shot (so socket is bound and ready)
            sniper_st = SniperState(SNIPER_TELEMETRY_PORT)
            time.sleep(0.1)

            # Send confirm_shot to new_main3 in a background thread.
            # new_main3 will: release its HD camera → run sniper_4k.py → reply when done.
            # We run it in a thread so we can simultaneously monitor SniperState telemetry.
            sniper_done = [False]
            def _run_sniper():
                do_confirm_shot(picked_zone, plan)
                sniper_done[0] = True
            threading.Thread(target=_run_sniper, daemon=True).start()

            # Wait for sniper to start broadcasting telemetry (up to 10s)
            print("    Waiting for sniper to initialise...")
            for _ in range(100):
                mc.brake()   # keep it pinned still
                if sniper_st.active: break
                time.sleep(0.1)

            if not sniper_st.active:
                print("    Sniper did not start — aborting.")
                sniper_st.stop()
            else:
                print(f"    Sniper active: mode={sniper_st.shot_mode}  target={sniper_st.target_pocket}")

                # Wait for full lock (angle + position both satisfied)
                locked = wait_for_sniper_lock(sniper_st, mc, timeout=240.0)

                if locked:
                    # Send fire command to sniper_4k — equivalent to pressing 'f'
                    _fire_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    _fire_sock.sendto(
                        json.dumps({"cmd": "fire"}).encode(),
                        ("127.0.0.1", SNIPER_FIRE_PORT)
                    )
                    _fire_sock.close()

                    shot_time = fire(mc, shot_time)
                else:
                    print("    Lock timed out — skipping shot.")
                    mc.action("stop_spin_speed")

            # Wait for sniper to finish (new_main3 reply received) before cleaning up
            for _ in range(50):
                if sniper_done[0]: break
                time.sleep(0.1)
            sniper_st.stop()

            is_all_zero = all(x == 0 for x in white_balls)
            if is_all_zero:
                break











        go_home(state, mc)
        
    except Exception as e:
        print(e)
        traceback.print_exc()
        raise

    finally:
        mc.brake()
        emergency_stop("program exit")
        state.stop()
        mc.close()

if __name__ == "__main__":
    main()

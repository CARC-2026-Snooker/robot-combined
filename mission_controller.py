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
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))

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

sys.path.append("/home/page/Carc2026/robot-vision-and-planning")
from ball_align import BallAlignState, do_ball_align

import signal
import random
import traceback

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
                    print(f"[SniperState DEBUG] Received | init={data.get('initialized')} | "
                          f"x_err={data.get('x_error_mm')} | y_err={data.get('y_error_mm')} | "
                          f"state={data.get('state')} | fully_locked={data.get('fully_locked')}")
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
        return self.get().get("x_error_mm", 999.0)

    @property
    def y_error_mm(self) -> float:
        return self.get().get("y_error_mm", 999.0)

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


def wait_for_sniper_lock(sniper: SniperState,
                         mc: MovementClient,
                         timeout: float = 30.0,
                         verbose: bool = True) -> bool:
    """
    Actively nudge the robot until the sniper reaches full lock.

    Uses a pulse → brake → settle cycle to avoid overshoot:
      1. Read error from sniper telemetry
      2. If correction needed: send a SHORT low-power pulse (PULSE_MS)
      3. Brake immediately
      4. Wait for the robot to stop and the camera/filter to catch up (SETTLE_MS)
      5. Re-read and repeat

    Power is deliberately capped much lower than drive_to (max 20 vs 200)
    because the lock tolerance is 0.3° / ~10px — any fast motion overshoots it.
    """
    # ── Fine-aiming power limits ─────────────────────────────────────────────
    # These override ROT_POWER_MAX/MOVE_POWER_MAX (which are for coarse driving).
    # Keep these LOW — the lock tolerance is tiny (0.3°, ~10px, ~5mm).
    FINE_ROT_MIN   = 5    # minimum rotation power
    FINE_ROT_MAX   = 25    # maximum rotation power — DO NOT raise above ~25
    FINE_MOVE_MIN  = 5    # minimum translation power
    FINE_MOVE_MAX  = 25    # maximum translation power — DO NOT raise above ~25

    ANG_FULL_DEG   = 8.0   # degrees where power reaches FINE_ROT_MAX
    LAT_FULL_MM    = 25.0  # mm where power reaches FINE_MOVE_MAX
    POS_FULL_MM    = 35.0  # mm (CUT mode) where power reaches FINE_MOVE_MAX

    # ── Pulse / settle timing ────────────────────────────────────────────────
    # PULSE_S: how long to run motors before braking.
    # SETTLE_S: how long to wait after braking before reading the next error.
    # Longer settle = more accurate reading but slower convergence.
    PULSE_S        = 0.05   # 80ms pulse — short enough to not overshoot
    SETTLE_S       = 0.20   # 200ms settle — camera updates + robot stops

    t0 = time.time()
    last_print = 0.0

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
        x_err       = d.get("x_error_mm", 999.0)
        y_err       = d.get("y_error_mm", 999.0)

        vX = vY = vR = 0

        if mode == "STRAIGHT":
            if tag_found and not ang_locked and ang_corr is not None:
                rot_spd = _clamp_power(abs(ang_corr), ANG_FULL_DEG,
                                       FINE_ROT_MIN, FINE_ROT_MAX, ROT_DEAD_DEG)
                vR = rot_spd if ang_corr > 0 else -rot_spd

            if ang_locked and lat_mm is not None and not lat_locked:
                lat_spd = _clamp_power(abs(lat_mm), LAT_FULL_MM,
                                       FINE_MOVE_MIN, FINE_MOVE_MAX, MOVE_DEAD_MM)
                vX = lat_spd if lat_mm > 0 else -lat_spd

        else:
            # CUT mode
            if tag_found and ang_corr is not None:
                rot_spd = _clamp_power(abs(ang_corr), ANG_FULL_DEG,
                                       FINE_ROT_MIN, FINE_ROT_MAX, ROT_DEAD_DEG)
                vR = rot_spd if ang_corr > 0 else -rot_spd

            if initialized:
                dist_mm = math.hypot(x_err, y_err)
                if dist_mm > MOVE_DEAD_MM:
                    trans_spd = _clamp_power(dist_mm, POS_FULL_MM,
                                             FINE_MOVE_MIN, FINE_MOVE_MAX, MOVE_DEAD_MM)
                    hd = d.get("heading_dx", 0.0)
                    hy = d.get("heading_dy", -1.0)
                    heading_deg = (math.degrees(math.atan2(hy, hd)) + 360) % 360
                    h = math.radians(heading_deg)
                    raw_vX =  x_err * math.sin(h) - y_err * math.cos(h)
                    raw_vY = -x_err * math.cos(h) - y_err * math.sin(h)
                    mag = math.hypot(raw_vX, raw_vY)
                    if mag > 1e-9:
                        vX = int(round(raw_vX / mag * trans_spd))
                        vY = int(round(raw_vY / mag * trans_spd))

        if verbose and time.time() - last_print > 1.0:
            ang_str = f"{ang_corr:+.2f}°" if ang_corr is not None else "searching"
            lat_str = f"{lat_mm:+.1f}mm" if lat_mm is not None else "n/a"
            print(
                f"  sniper: mode={mode} "
                f"hd={'OK' if initialized else 'wait'} "
                f"tag={'FOUND' if tag_found else 'search'} "
                f"ang={'OK' if ang_locked else ang_str} "
                f"lat={'OK' if lat_locked else lat_str} "
                f"x={x_err:+.1f} y={y_err:+.1f} "
                f"{'LOCKED' if d.get('pos_locked') else 'waiting'} "
                f"cmd=({vX:+d},{vY:+d},{vR:+d})"
            )
            last_print = time.time()

        if vX == 0 and vY == 0 and vR == 0:
            # Nothing to correct — coast and check again soon
            time.sleep(0.05)
            continue

        # ── Pulse → brake → settle ───────────────────────────────────────────
        mc.send(MoveCmd(cmd="omni", vX=vX, vY=vY, vR=vR))
        time.sleep(PULSE_S)    # run for a short burst
        mc.brake()             # stop immediately
        time.sleep(SETTLE_S)   # wait for robot to stop + camera to catch up

    mc.brake()
    if verbose:
        print(f"[wait_for_sniper_lock] TIMEOUT after {timeout:.0f}s")
    return False

    mc.brake()
    if verbose:
        print(f"[wait_for_sniper_lock] TIMEOUT after {timeout:.0f}s")
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
             timeout:     float = 45.0,
             verbose:     bool  = True) -> bool:
    """
    Drive to (target_x_mm, target_y_mm), then rotate to final_heading_deg.

    Phase 1 — translate while keeping heading near DRIVE_HEADING (270°).
      Uses world_errors_to_omni() which computes vX, vY, vR all at once
      and sends a single OmniMovement command per tick. With the new Robot
      API all three axes are corrected simultaneously — no axis priority needed.

    Phase 2 — rotate in place to final_heading_deg (vR only).
    """
    if verbose:
        print(f"[drive_to] target=({target_x_mm:.0f}, {target_y_mm:.0f})mm  "
              f"final={final_heading_deg:.1f}°")

    t0 = time.time()
    _last_print = 0.0

    # ── Phase 1: Translate ────────────────────────────────────────────────────
    if verbose: print("  Phase 1: translating...")
    while time.time() - t0 < timeout:
        if STOP_EVENT.is_set():
            mc.brake(); return False
        if not state.initialized:
            time.sleep(0.02); continue

        x_mm, y_mm = state.position_mm
        x_err = x_mm - target_x_mm
        y_err = y_mm - target_y_mm
        if math.hypot(x_err, y_err) < pos_tol_mm:
            break

        ang_err = _ang_diff(state.heading_deg, DRIVE_HEADING)
        cmd = world_errors_to_omni(
            x_err_mm     = x_err,
            y_err_mm     = y_err,
            ang_err_deg  = ang_err,
            heading_deg  = state.heading_deg,
            x_full_mm    = X_FULL_POWER_MM,
            y_full_mm    = Y_FULL_POWER_MM,
            rot_full_deg = ROT_FULL_DEG,
        )
        mc.send(cmd)
        if verbose and time.time() - _last_print > 1.0:
            print(f"    pos=({x_mm:.0f},{y_mm:.0f}) err=({x_err:+.0f},{y_err:+.0f})mm "
                  f"hdg={state.heading_deg:.1f}° ang_err={ang_err:+.1f}° → {cmd.describe()}")
            _last_print = time.time()
        time.sleep(0.01)

    else:
        mc.brake()
        if verbose: print("  [drive_to] TIMEOUT in phase 1.")
        return False

    mc.brake(); time.sleep(0.3)

    # ── Phase 2: Rotate to final heading ──────────────────────────────────────
    if verbose: print(f"  Phase 2: rotating to {final_heading_deg:.1f}°...")
    from movement_client import _clamp_power
    rot_t0 = time.time()
    while time.time() - rot_t0 < 10.0:
        if STOP_EVENT.is_set():
            mc.brake(); return False
        if not state.initialized:
            time.sleep(0.02); continue
        ang_err = _ang_diff(state.heading_deg, final_heading_deg)
        if abs(ang_err) < ang_tol_deg:
            break
        rot_spd = _clamp_power(abs(ang_err), ROT_FULL_DEG,
                               ROT_POWER_MIN, ROT_POWER_MAX, ROT_DEAD_DEG)
        mc.omni(vR=-rot_spd if ang_err > 0 else rot_spd)
        time.sleep(0.01)

    mc.brake(); time.sleep(0.2)
    if verbose:
        x, y = state.position_mm
        print(f"  Arrived: ({x:.0f}, {y:.0f})mm  heading={state.heading_deg:.1f}°")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# AUTOMATIC SCAN — triggers new_main3 to take a photo and run the planner
# without pressing 's' manually
# ─────────────────────────────────────────────────────────────────────────────
def do_scan(plan_path: str = "robot-vision-and-planning/robot_shooting_plan.json",
            timeout:   float = 15.0) -> dict:
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
    time.sleep(0.5)

    align = BallAlignState()
    time.sleep(0.2)

    if verbose:
        print("[pickup] Aligning with ball using back camera...")
    success = do_ball_align(mc, align, timeout=timeout_align, verbose=verbose)
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
    mc.action("rev_spin_power", 30); time.sleep(1.0)
    mc.action("stop_spin_power");    time.sleep(0.3)

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
# MAIN — your game loop goes here
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global _GLOBAL_MC

    state = RobotState(TELEMETRY_PORT)
    mc    = MovementClient()
    _GLOBAL_MC = mc

    white_balls = [2,2]
    last_side = 0 # left:0 right:1

    # demo_all_commands(state, mc)
    shot_time = None
    
    # go to left side infront of white balls pedestal
    mc.omni(vX=-30, vY=-190, vR=-200)
    time.sleep(1.9)
    mc.brake()

    try:
        while True:

            do_pickup_sequence(mc)
            white_balls[last_side] = white_balls[last_side] - 1

            # go back to middle and align to view point
            if last_side == 0:
                mc.omni(vX=-100, vY=300, vR=-200)
            else:
                mc.omni(vX=100, vY=300, vR=200)
            time.sleep(1.75)
            mc.brake()

            PHOTO_POS_X, PHOTO_POS_Y, PHOTO_HEAD = 600, 2640, 270.0
            drive_to(PHOTO_POS_X, PHOTO_POS_Y, PHOTO_HEAD, state, mc, pos_tol_mm=10.0, ang_tol_deg=1.0)

            # wait until it's been 15s since the last shot
            if shot_time:
                sec_since_shot = (datetime.now() - shot_time).total_seconds()
                while sec_since_shot < 15:
                    print(f"waiting 15 seconds after last shot, {sec_since_shot} seconds left")
                    sec_since_shot = (datetime.now() - shot_time).total_seconds()

            # take a capture of the view and get plan
            plan = do_scan()
            if not plan:
                print("No balls left — done!")
                return
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

            # Drive to coarse alignment position
            rx, ry = shot["robot_alignment_xy"]
            # drive_to(rx, ry, shot["firing_angle_deg"], state, mc)
            drive_to(rx, ry, 270.0, state, mc)
            mc.rotate_by(speed=100, degree=shot["firing_angle_deg"] - 270.0)

            # mc.brake()
            # time.sleep(0.3)

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
                locked = wait_for_sniper_lock(sniper_st, mc, timeout=60.0)

                if locked:
                    # Send fire command to sniper_4k — equivalent to pressing 'f'
                    _fire_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    _fire_sock.sendto(
                        json.dumps({"cmd": "fire"}).encode(),
                        ("127.0.0.1", SNIPER_FIRE_PORT)
                    )
                    _fire_sock.close()

                    # Fire mechanism — spin starts AFTER lock, no vibration during aiming
                    mc.action("rev_spin_power", 100)
                    time.sleep(2)
                    mc.action("release_trigger")
                    time.sleep(2)
                    mc.action("reset_trigger")
                    mc.action("stop_spin_power")
                    print("    Shot fired!")
                    shot_time = datetime.now()
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

            # go to between the two sides, left first, then right
            drive_to(600, 3105, PHOTO_HEAD, state, mc, pos_tol_mm=20.0, ang_tol_deg=1.0)
            if white_balls[0]: # if still have left do left, else right
                mc.rotate_by(speed=200, degree=90)
                last_side = 0
            else:
                mc.rotate_by(speed=200, degree=-90)
                last_side = 1

        drive_to(600, 3375, PHOTO_HEAD, state, mc, pos_tol_mm=20.0, ang_tol_deg=1.0)
        
    except Exception as e:
        print(e)
        traceback.print_exc()
        raise

    finally:
        emergency_stop("program exit")
        state.stop()
        mc.close()

if __name__ == "__main__":
    main()
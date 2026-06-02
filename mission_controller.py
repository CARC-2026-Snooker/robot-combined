"""
mission_controller.py — High-level game loop skeleton.

This is YOUR file to write the competition logic in. All the tools
you need are imported below. The structure shows you what's available.

How to use:
1. Run new_main3.py first (starts the vision + telemetry server).
2. Run movement_server.py (starts the motor driver).
3. Run this script — it reads telemetry and sends movement commands.

Or combine everything: import RobotState, call start_vision_subprocess(), etc.

Available tools/state:
  state = RobotState()          → live robot position, errors, tracking state
  mc    = MovementClient()      → send move/rotate/stop/action commands
  plan  = load_shooting_plan()  → dict of zone → shot info from planner JSON
  compute_move_cmd(...)         → compute the right movement command from errors
"""

import sys, os, time, json, math, subprocess
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    TELEMETRY_PORT, MOVE_DEAD_MM, ROT_DEAD_DEG,
    REAL_ZONES,
)
from movement_client import MovementClient, MoveCmd, compute_move_cmd
import socket, threading


# ─────────────────────────────────────────────────────────────────────────────
# ROBOT STATE — subscribes to telemetry from new_main3.py
# ─────────────────────────────────────────────────────────────────────────────
class RobotState:
    """
    Listens on the telemetry UDP port and exposes the latest robot state.

    Telemetry packet keys (sent by new_main3.py / sniper_4k.py):
      initialized      bool
      x_mm, y_mm       float  — robot centre world position in mm
      heading_dx/dy    float  — unit heading vector
      state            str    — "STILL" | "MOVING" | "TRANSITIONING" | "LOST"
      x_error_mm       float  — position error from plan (AIMING state)
      y_error_mm       float
      ang_error_deg    float
      app_state        str    — "DRIVING" | "AIMING" | "LOCKED"
    """

    def __init__(self, port: int = TELEMETRY_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", port))
        self._sock.settimeout(0.1)
        self._data: dict = {}
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _listen(self):
        while self._running:
            try:
                raw, _ = self._sock.recvfrom(4096)
                data = json.loads(raw.decode())
                with self._lock:
                    self._data = data
            except socket.timeout:
                pass
            except Exception:
                pass

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
    def tracking_state(self) -> str:
        return self.get().get("state", "LOST")

    def stop(self):
        self._running = False
        self._sock.close()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_shooting_plan(path="robot_shooting_plan.json") -> dict:
    """Load the plan produced by snooker_planner.py."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def wait_for_tracking(state: RobotState, timeout=10.0):
    """Block until tracking is initialised or timeout."""
    t0 = time.time()
    while not state.initialized:
        if time.time() - t0 > timeout:
            raise TimeoutError("Tracking not initialised in time.")
        time.sleep(0.1)


def drive_to(target_x_mm: float, target_y_mm: float,
             target_heading_deg: float,
             state: RobotState, mc: MovementClient,
             pos_tol_mm: float = 20.0,
             ang_tol_deg: float = 2.0,
             timeout: float = 30.0):
    """
    Blocking drive loop: move robot to (target_x_mm, target_y_mm) facing
    target_heading_deg. Returns when within tolerances or on timeout.

    All coordinates in mm, heading in degrees (world frame).
    Movement commands are in ROBOT FRAME (FRONT/BACK/LEFT/RIGHT/ROTATE).
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not state.initialized:
            time.sleep(0.05)
            continue

        x_mm, y_mm = state.position_mm
        hdx, hdy   = state.heading
        curr_head  = math.degrees(math.atan2(hdy, hdx)) % 360

        x_err = x_mm - target_x_mm     # positive = too far right
        y_err = y_mm - target_y_mm     # positive = too far forward

        ang_err = curr_head - target_heading_deg
        if ang_err >  180: ang_err -= 360
        if ang_err < -180: ang_err += 360

        dist_err = math.hypot(x_err, y_err)

        if dist_err < pos_tol_mm and abs(ang_err) < ang_tol_deg:
            mc.stop()
            return True

        # Transform x/y errors from world frame into robot frame
        # Robot FRONT = (hdx, hdy), robot RIGHT = (hdy, -hdx)
        fwd_err  =  x_err * hdx + y_err * hdy    # positive = robot behind target
        side_err = -x_err * hdy + y_err * hdx    # positive = robot left of target

        # Remap to movement_client convention:
        #   y_error_mm positive → BACK  (but here fwd_err positive = move FRONT)
        cmd = compute_move_cmd(
            x_error_mm    = side_err,     # positive = move RIGHT
            y_error_mm    = -fwd_err,     # positive = move BACK  (flip sign)
            ang_error_deg = ang_err,
        )
        mc.send(cmd)
        time.sleep(0.05)

    mc.stop()
    return False   # timed out


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE GAME LOOP  — fill this in with your competition logic
# ─────────────────────────────────────────────────────────────────────────────
def main():
    state = RobotState()
    mc    = MovementClient()

    print("Waiting for vision tracking to initialise...")
    wait_for_tracking(state, timeout=15.0)
    print(f"Tracking OK. Robot at: {state.position_mm}")

    # ── EXAMPLE SEQUENCE — replace with your actual game logic ────────────

    # avg_pos = [0, 0]
    # avg_deg = 0
    # for i in range(5):
    #     # print(state.heading)
    #     position = state.position_mm
    #     print(position)
    #     avg_pos[0] += position[0]
    #     avg_pos[1] += position[1]
    #     heading_deg = (math.degrees(math.atan2(state.heading[1], state.heading[0])) + 360) % 360
    #     print(heading_deg)
    #     avg_deg += heading_deg
    #     time.sleep(0.5)
    # avg_pos[0] = avg_pos[0] / 5
    # avg_pos[1] = avg_pos[1] / 5
    # avg_deg /= 5

    # print(avg_pos)
    # print(avg_deg)

    time.sleep(1)

    mc.send(compute_move_cmd(x_error_mm=10, y_error_mm=0, ang_error_deg=0))

    # drive_to(avg_pos[0] + 100, avg_pos[1] - 200, avg_deg, state, mc)

    # while True:
    #     position = state.position_mm
    #     print(position)
    #     heading_deg = (math.degrees(math.atan2(state.heading[1], state.heading[0])) + 360) % 360
    #     print(heading_deg)
    #     time.sleep(0.5)

    # 1. Drive to photo position
    # drive_to(600, 2375, 90, state, mc)
    # mc.action('rev_spin', power=30)
    # time.sleep(1)
    # mc.action('stop_spin')

    # 2. Trigger ball scan (press 's' in new_main3 window, or call via socket)

    # plan = load_shooting_plan()
    # print(plan)
    
    # 3. Choose which ball to go for
    #    target_zone = "E"   # or pick based on plan
    #    shot = plan[target_zone]
    #
    # 4. Drive to alignment position
    #    rx, ry = shot["robot_alignment_xy"]
    #    drive_to(rx, ry, shot["firing_angle_deg"], state, mc)
    #
    # 5. Switch to sniper (press 'c' in new_main3, or trigger programmatically)
    #
    # 6. Fire (press 'f' in sniper window, or detect LOCKED state and auto-fire)
    #
    # 7. Loop back to step 1

    print("\nNo game logic implemented yet — add your steps above.")
    print("Available commands:")
    print("  mc.send(compute_move_cmd(x_error_mm=..., y_error_mm=..., ang_error_deg=...))")
    print("  mc.stop() / mc.brake()")
    print("  mc.action('open_claw') / mc.action('rev_spin', power=30)")
    print("  drive_to(x_mm, y_mm, heading_deg, state, mc)")
    print("  plan = load_shooting_plan()")

    state.stop()
    mc.close()


if __name__ == "__main__":
    main()

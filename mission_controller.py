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
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    TELEMETRY_PORT, MOVE_DEAD_MM, ROT_DEAD_DEG,
    REAL_ZONES, MOVE_POWER_MIN, MOVE_POWER_MAX,
    ROT_POWER_MIN, ROT_POWER_MAX,
    MOVE_GAIN_MM, ROT_GAIN_DEG,
)
from movement_client import MovementClient, MoveCmd, compute_move_cmd


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


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
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


def _proportional_power(err_abs, gain, pmin, pmax):
    return int(max(pmin, min(pmax, err_abs * gain)))


# ─────────────────────────────────────────────────────────────────────────────
# DRIVE TO — improved: lock 270° while driving, then rotate to final heading
# ─────────────────────────────────────────────────────────────────────────────
DRIVE_HEADING = 270.0   # robot always faces this direction while translating
                        # (all tags are ahead at 270°, keeps vision working)


def drive_to(target_x_mm: float, target_y_mm: float,
             final_heading_deg: float,
             state: RobotState, mc: MovementClient,
             pos_tol_mm: float = 30.0,
             ang_tol_deg: float = 2.0,
             timeout: float = 45.0,
             verbose: bool = True) -> bool:
    """
    Drive robot to (target_x_mm, target_y_mm), then rotate to final_heading_deg.

    Phase 1 — TRANSLATE while locked to DRIVE_HEADING (270°):
      - Continuously corrects heading back to 270° (keeps tags in view)
      - Moves FRONT/BACK (Y) and LEFT/RIGHT (X) based on position error
      - Stops when within pos_tol_mm

    Phase 2 — ROTATE to final_heading_deg:
      - Rotates in place until within ang_tol_deg

    Commands are in ROBOT FRAME. Sends a stop after each phase.
    Returns True on success, False on timeout.
    """
    if verbose:
        print(f"[drive_to] target=({target_x_mm:.0f}, {target_y_mm:.0f}) heading={final_heading_deg:.1f}°")

    t0 = time.time()

    # ── Phase 1: Translate (locked to 270°) ──────────────────────────────
    if verbose: print("  Phase 1: translating...")
    while time.time() - t0 < timeout:
        if not state.initialized:
            time.sleep(0.05)
            continue

        x_mm, y_mm = state.position_mm
        x_err = x_mm - target_x_mm   # + = too far right
        y_err = y_mm - target_y_mm   # + = too far forward (past target)

        if math.hypot(x_err, y_err) < pos_tol_mm:
            break

        # Keep heading locked to DRIVE_HEADING while translating
        head_err = _ang_diff(state.heading_deg, DRIVE_HEADING)

        if abs(head_err) > 3.0:
            # Correct heading first, then translate
            pwr = _proportional_power(abs(head_err), ROT_GAIN_DEG,
                                      ROT_POWER_MIN, ROT_POWER_MAX)
            d = "RIGHT" if head_err > 0 else "LEFT"
            mc.send(MoveCmd(cmd="rotate", direction=d, power=pwr))
        else:
            # Translate: transform world errors to robot frame (robot faces 270° = -Y)
            # At 270°: robot FRONT = world -Y direction
            #          robot RIGHT = world +X direction
            # So: x_err (world right) → robot right error
            #     y_err (world forward) → robot front error (positive = ahead = behind us)
            fwd_err  = -y_err   # + = need to move FRONT (-Y world)
            side_err =  x_err   # + = too far right = need to move LEFT

            if abs(fwd_err) >= abs(side_err):
                pwr = _proportional_power(abs(fwd_err), MOVE_GAIN_MM,
                                          MOVE_POWER_MIN, MOVE_POWER_MAX)
                d = "FRONT" if fwd_err > 0 else "BACK"
                mc.send(MoveCmd(cmd="move", direction=d, power=pwr))
            else:
                pwr = _proportional_power(abs(side_err), MOVE_GAIN_MM,
                                          MOVE_POWER_MIN, MOVE_POWER_MAX)
                d = "LEFT" if side_err > 0 else "RIGHT"
                mc.send(MoveCmd(cmd="move", direction=d, power=pwr))

        time.sleep(0.04)
    else:
        mc.stop()
        if verbose: print("  [drive_to] TIMEOUT in phase 1.")
        return False

    mc.send(MoveCmd(cmd="brake"))
    time.sleep(0.3)

    # ── Phase 2: Rotate to final heading ─────────────────────────────────
    if verbose: print(f"  Phase 2: rotating to {final_heading_deg:.1f}°...")
    rot_t0 = time.time()
    while time.time() - rot_t0 < 10.0:
        if not state.initialized:
            time.sleep(0.05)
            continue
        ang_err = _ang_diff(state.heading_deg, final_heading_deg)
        if abs(ang_err) < ang_tol_deg:
            break
        pwr = _proportional_power(abs(ang_err), ROT_GAIN_DEG,
                                  ROT_POWER_MIN, ROT_POWER_MAX)
        d = "RIGHT" if ang_err > 0 else "LEFT"
        mc.send(MoveCmd(cmd="rotate", direction=d, power=pwr))
        time.sleep(0.04)

    mc.send(MoveCmd(cmd="brake"))
    time.sleep(0.2)

    if verbose:
        x, y = state.position_mm
        print(f"  Arrived: ({x:.0f}, {y:.0f}) heading={state.heading_deg:.1f}°")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# AUTOMATIC SCAN — triggers new_main3 to take a photo and run the planner
# without pressing 's' manually
# ─────────────────────────────────────────────────────────────────────────────
def do_scan(image_path: str = "current_scan.jpg",
            plan_path:  str = "robot_shooting_plan.json") -> dict:
    """
    Capture a frame and run the planner automatically.
    Equivalent to pressing 's' in new_main3.py.
    Returns the loaded shooting plan dict (may be empty if no balls found).
    """
    print("[do_scan] Capturing image and running planner...")
    # Import here to avoid circular deps
    import cv2
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robot-vision-and-planning"))
    from config import CAM_HD_INDEX, HD_SCAN_RESOLUTION
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(CAM_HD_INDEX, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(CAM_HD_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  HD_SCAN_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HD_SCAN_RESOLUTION[1])
    for _ in range(5): cap.read(); time.sleep(0.05)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(image_path, frame)
        print(f"[do_scan] Saved {image_path}")
    else:
        print("[do_scan] WARNING: failed to capture frame.")
        return {}

    vision_dir = os.path.join(os.path.dirname(__file__), "robot-vision-and-planning")
    planner = os.path.join(vision_dir, "snooker_planner.py")
    subprocess.run([sys.executable, planner, os.path.abspath(image_path)],
                   cwd=vision_dir)
    return load_shooting_plan(plan_path)


def do_confirm_shot(zone: str,
                    plan: dict,
                    vision_dir: str = None) -> bool:
    """
    Automatically write selected_target.json and launch sniper_4k.py.
    Equivalent to pressing 'c' in new_main3.py.
    Returns True when sniper exits (shot fired or ESC).
    """
    if zone not in plan:
        print(f"[do_confirm_shot] Zone '{zone}' not in plan.")
        return False
    with open("selected_target.json", "w") as f:
        json.dump(plan[zone], f, indent=2)
    print(f"[do_confirm_shot] Launching sniper for zone {zone}...")
    if vision_dir is None:
        vision_dir = os.path.join(os.path.dirname(__file__), "robot-vision-and-planning")
    sniper = os.path.join(vision_dir, "sniper_4k.py")
    result = subprocess.run([sys.executable, sniper], cwd=vision_dir)
    return result.returncode == 0


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
    print(f"\n[1] Position:    {state.position_mm}")
    print(f"    Heading:     {state.heading_deg:.1f}°  (vector {state.heading})")
    print(f"    Tracking:    {state.tracking_state}")
    print(f"    App state:   {state.app_state}")

    # ── 2. Raw movement commands ──────────────────────────────────────────
    print("\n[2] Movement commands:")
    mc.send(MoveCmd(cmd="move", direction="FRONT", power=20)); time.sleep(0.5)
    mc.send(MoveCmd(cmd="brake")); time.sleep(0.3)
    mc.send(MoveCmd(cmd="move", direction="BACK",  power=20)); time.sleep(0.5)
    mc.send(MoveCmd(cmd="brake")); time.sleep(0.3)
    mc.send(MoveCmd(cmd="move", direction="LEFT",  power=20)); time.sleep(0.5)
    mc.send(MoveCmd(cmd="brake")); time.sleep(0.3)
    mc.send(MoveCmd(cmd="move", direction="RIGHT", power=20)); time.sleep(0.5)
    mc.send(MoveCmd(cmd="brake")); time.sleep(0.3)

    # ── 3. Rotation commands ──────────────────────────────────────────────
    print("\n[3] Rotation commands:")
    mc.send(MoveCmd(cmd="rotate", direction="LEFT",  power=15)); time.sleep(0.4)
    mc.send(MoveCmd(cmd="brake")); time.sleep(0.3)
    mc.send(MoveCmd(cmd="rotate", direction="RIGHT", power=15)); time.sleep(0.4)
    mc.send(MoveCmd(cmd="brake")); time.sleep(0.3)

    # ── 4. Stop variants ──────────────────────────────────────────────────
    print("\n[4] Stop commands:")
    mc.stop()    # soft stop
    mc.brake()   # hard brake (motor lock)

    # ── 5. Actions — claw, trigger, spin wheel ────────────────────────────
    print("\n[5] Actions:")
    mc.action("open_claw");     time.sleep(0.5)
    mc.action("close_claw");    time.sleep(0.5)
    mc.action("rev_spin", power=25); time.sleep(1.0)
    mc.action("stop_spin");     time.sleep(0.3)
    mc.action("release_trigger"); time.sleep(0.5)
    mc.action("reset_trigger");   time.sleep(0.5)

    # ── 6. Proportional move commands via compute_move_cmd ────────────────
    print("\n[6] compute_move_cmd:")
    # Positive x_error → robot is too far RIGHT → command LEFT
    cmd = compute_move_cmd(x_error_mm=80, y_error_mm=0, ang_error_deg=0)
    print(f"    x=+80mm → {cmd.describe()}")
    mc.send(cmd); time.sleep(0.3); mc.brake()

    cmd = compute_move_cmd(x_error_mm=0, y_error_mm=-120, ang_error_deg=0)
    print(f"    y=-120mm → {cmd.describe()}")
    mc.send(cmd); time.sleep(0.3); mc.brake()

    cmd = compute_move_cmd(x_error_mm=0, y_error_mm=0, ang_error_deg=15)
    print(f"    ang=+15° → {cmd.describe()}")
    mc.send(cmd); time.sleep(0.3); mc.brake()

    # ── 7. drive_to ───────────────────────────────────────────────────────
    print("\n[7] drive_to (example — target = current position + 100mm forward):")
    x, y = state.position_mm
    drive_to(x, y - 100, 270.0, state, mc, pos_tol_mm=30)

    # ── 8. Scan table ─────────────────────────────────────────────────────
    print("\n[8] Scan table:")
    plan = do_scan()
    if plan:
        print(f"    Found zones: {list(plan.keys())}")
        for zone, info in plan.items():
            print(f"    {zone}: {info['color']} → {info['target_pocket']}  cut={info['cut_angle_deg']}°")
    else:
        print("    No balls detected.")

    # ── 9. Confirm shot (if plan found) ───────────────────────────────────
    if plan:
        zone = list(plan.keys())[0]
        print(f"\n[9] Confirming shot for zone {zone}...")
        rx, ry = plan[zone]["robot_alignment_xy"]
        drive_to(rx, ry, plan[zone]["firing_angle_deg"], state, mc)
        do_confirm_shot(zone, plan)   # launches sniper_4k.py

    print("\n━━━ DEMO COMPLETE ━━━")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — your game loop goes here
# ─────────────────────────────────────────────────────────────────────────────
def main():
    state = RobotState()
    mc    = MovementClient()

    print("Waiting for vision tracking...")
    wait_for_tracking(state, timeout=15.0)

    # ── Run the demo to test every command ───────────────────────────────
    # demo_all_commands(state, mc)

    # ── Example competition loop ─────────────────────────────────────────
    #
    # PHOTO_POS_X = 600    # mm — centre of table width
    # PHOTO_POS_Y = 2375   # mm — robot shoot line
    # PHOTO_HEAD  = 270.0  # degrees — facing toward all tags
    #
    # while True:
    #     # 1. Drive to photo position
    #     drive_to(PHOTO_POS_X, PHOTO_POS_Y, PHOTO_HEAD, state, mc)
    #
    #     # 2. Scan table and get plan
    #     plan = do_scan()
    #     if not plan:
    #         print("No balls left — done!")
    #         break
    #
    #     # 3. Pick which ball to go for (choose yourself or pick first one)
    #     zone = list(plan.keys())[0]
    #     shot = plan[zone]
    #
    #     # 4. Rev up spin wheel while navigating
    #     mc.action("rev_spin", power=25)
    #
    #     # 5. Drive to alignment position
    #     rx, ry = shot["robot_alignment_xy"]
    #     drive_to(rx, ry, shot["firing_angle_deg"], state, mc)
    #
    #     # 6. Launch sniper — blocks until shot fired or ESC
    #     do_confirm_shot(zone, plan)
    #     mc.action("stop_spin")
    #
    #     # 7. Loop back

    mc.stop()
    state.stop()
    mc.close()


if __name__ == "__main__":
    main()
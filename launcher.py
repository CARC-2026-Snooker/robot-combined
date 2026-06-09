"""
launcher.py — Start all robot processes in the correct order.

Startup sequence:
  1. movement_server  — robot hardware + IMU (non-blocking IMU calibration)
  2. new_main3        — HD camera + AprilTag tracking
  3. ball_align       — back-camera pickup alignment service
  4. Wait for IMU ready signal on IMU_TELEMETRY_PORT (max 15s)
  5. mission_controller — competition logic

Shutdown: Ctrl-C terminates all child processes cleanly.
"""

import subprocess
import socket
import signal
import sys
import time
import json
from config import DISABLE_IMU

IMU_TELEMETRY_PORT = 5009   # matches config.py IMU_TELEMETRY_PORT
IMU_READY_TIMEOUT  = 30.0   # max seconds to wait for IMU calibration
STARTUP_PAUSE      = 5.0    # seconds to let movement_server/new_main3 initialise
                             # before checking IMU (robot + camera need time)

processes = []


def _wait_for_imu(timeout: float = IMU_READY_TIMEOUT) -> bool:
    """
    Listen on IMU_TELEMETRY_PORT until movement_server broadcasts is_ready=True.
    Returns True when ready, False on timeout.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind(("0.0.0.0", IMU_TELEMETRY_PORT))
    sock.settimeout(1.0)
    t0 = time.time()
    try:
        while time.time() - t0 < timeout:
            try:
                data, _ = sock.recvfrom(512)
                msg = json.loads(data.decode())
                if msg.get("is_ready") or msg.get("calibrated"):
                    return True
            except socket.timeout:
                elapsed = time.time() - t0
                print(f"  [launcher] Waiting for IMU... ({elapsed:.0f}s / {timeout:.0f}s)")
            except Exception:
                pass
    finally:
        sock.close()
    return False


try:
    # ── Step 1: movement_server (robot + IMU) ────────────────────────────────
    processes.append(subprocess.Popen(
        ["uv", "run", "movement_server.py"],
        cwd="/home/page/Carc2026"
    ))

    # ── Step 2: new_main3 (HD camera + tracking) ─────────────────────────────
    processes.append(subprocess.Popen(
        ["uv", "run", "new_main3.py"],
        cwd="/home/page/Carc2026/robot-vision-and-planning"
    ))

    # ── Step 3: ball_align ───────────────────────────────────────────────────
    processes.append(subprocess.Popen(
        ["uv", "run", "ball_align.py"],
        cwd="/home/page/Carc2026/robot-vision-and-planning"
    ))

    # ── Step 4: wait for basic init then for IMU ready ───────────────────────
    print(f"[launcher] Waiting {STARTUP_PAUSE:.0f}s for basic startup...")
    time.sleep(STARTUP_PAUSE)

    if not DISABLE_IMU:
        print("[launcher] Waiting for IMU calibration to complete...")
        imu_ok = _wait_for_imu(IMU_READY_TIMEOUT)

        if imu_ok:
            print("[launcher] IMU ready ✓")
        else:
            print(f"[launcher] WARNING: IMU not ready after {IMU_READY_TIMEOUT:.0f}s — starting mission anyway")

    # ── Step 5: mission_controller ───────────────────────────────────────────
    print("[launcher] Starting mission_controller...")
    mission = subprocess.Popen(
        ["uv", "run", "mission_controller.py"],
        cwd="/home/page/Carc2026"
    )
    mission.wait()

finally:
    print("[launcher] Stopping all processes...")
    for p in processes:
        if p.poll() is None:
            p.terminate()
    for p in processes:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    print("[launcher] Done.")
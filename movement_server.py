"""
movement_server.py — UDP listener that drives the robot via Robot.py.

Run from the Carc2026/ root:
    uv run robot-movement/library/movement_server.py

Protocol (JSON over UDP on MOVEMENT_CMD_PORT):

  { "cmd": "omni", "vX": int, "vY": int, "vR": int }
      OmniMovement — simultaneous translation + rotation. Range: -720..720.
        vX > 0 = strafe RIGHT      vX < 0 = strafe LEFT
        vY > 0 = move FORWARD      vY < 0 = move BACK
        vR > 0 = rotate CW/RIGHT   vR < 0 = rotate CCW/LEFT
      This is the main command used by drive_to in mission_controller.

  { "cmd": "move", "speed": int, "degree": int, "rotation": int }
      robot.Move(speed, degree, rotation).
      Degree convention: 0=forward, 90=right, 180=back, 270=left (robot frame).
      rotation: simultaneous angular speed (-720..720), 0 for straight movement.

  { "cmd": "rotate_by", "speed": int, "degree": int }
      robot.Rotate(speed, degree). BLOCKING — waits until encoder target reached.
      Use for precise snap rotations only (e.g. aiming). Not for the control loop.

  { "cmd": "brake" }   — lock motors (hold in place)
  { "cmd": "stop"  }   — release motors / close serial

  { "cmd": "action", "action": str, "power": int }
      Actions: "open_claw" | "close_claw" | "release_trigger" | "reset_trigger"
               "rev_spin_power"  | "stop_spin_power"   (power-based spin wheel)
               "rev_spin_speed"  | "stop_spin_speed"   (speed-based spin wheel)
"""

import sys, os, socket, json, time
from pathlib import Path

# Locate Robot.py and config.py regardless of where this is run from
sys.path.append("/home/page/Carc2026/robot-movement/library")
sys.path.insert(0, str(Path(__file__).parent))               # Robot.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent)) # config.py (project root)

from config import SERIAL_PORT, SERIAL_BAUD, MOVEMENT_CMD_PORT
from Robot import Robot


def run_server():
    robot = Robot(serial_port=SERIAL_PORT, baudrate=SERIAL_BAUD)
    print(f"[movement_server] Robot connected on {SERIAL_PORT}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", MOVEMENT_CMD_PORT))
    sock.settimeout(0.5)

    last_cmd_time = time.time()
    # IDLE_TIMEOUT  = 0.5   # brake automatically if no command for this long
    IDLE_TIMEOUT  = 60   # brake automatically if no command for this long
    moving        = False
    print(f"[movement_server] Listening on UDP port {MOVEMENT_CMD_PORT}")

    try:
        while True:
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                # Safety: brake if the control loop has gone silent
                if moving and time.time() - last_cmd_time > IDLE_TIMEOUT:
                    robot.Break()
                    moving = False
                continue

            try:
                msg = json.loads(data.decode("utf-8"))
            except json.JSONDecodeError:
                print(f"[movement_server] Bad JSON: {data}")
                continue

            cmd           = msg.get("cmd", "")
            last_cmd_time = time.time()

            # ── OmniMovement — main drive command ────────────────────────────
            if cmd == "omni":
                vX = int(msg.get("vX", 0))
                vY = int(msg.get("vY", 0))
                vR = int(msg.get("vR", 0))
                robot.OmniMovement(vX, vY, vR)
                moving = (vX != 0 or vY != 0 or vR != 0)

            # ── Move with degree + optional rotation ─────────────────────────
            elif cmd == "move":
                speed    = int(msg.get("speed",    0))
                degree   = int(msg.get("degree",   0))
                rotation = int(msg.get("rotation", 0))
                robot.Move(speed, degree, rotation)
                moving = (speed != 0 or rotation != 0)

            # ── Blocking precise rotation (encoder-based) ────────────────────
            elif cmd == "rotate_by":
                speed  = int(msg.get("speed",  50))
                degree = int(msg.get("degree",  0))
                robot.Rotate(speed, degree)   # blocks until done
                moving = False

            # ── Stop commands ────────────────────────────────────────────────
            elif cmd == "brake":
                robot.Break()
                moving = False

            elif cmd == "stop":
                robot.Break()
                moving = False

            # ── Mechanism actions ────────────────────────────────────────────
            elif cmd == "action":
                act   = msg.get("action", "")
                power = int(msg.get("power", 20))
                if   act == "open_claw":        robot.OpenClaw()
                elif act == "close_claw":       robot.CloseClaw()
                elif act == "release_trigger":  robot.ReleaseTrigger()
                elif act == "reset_trigger":    robot.ResetTrigger()
                elif act == "rev_spin_power":   robot.RevUpSpinWheelPower(power)
                elif act == "stop_spin_power":  robot.StopSpinWheelPower()
                elif act == "rev_spin_speed":   robot.RevUpSpinWheelSpeed(power)
                elif act == "stop_spin_speed":  robot.StopSpinWheelSpeed()
                else:
                    print(f"[movement_server] Unknown action: {act}")
            else:
                print(f"[movement_server] Unknown cmd: {cmd}")

    except KeyboardInterrupt:
        print("\n[movement_server] Shutting down.")
    finally:
        try: robot.Break()
        except Exception: pass
        sock.close()


if __name__ == "__main__":
    run_server()
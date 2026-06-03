"""
movement_server.py — Listens on UDP for drive commands from the vision system
and executes them on the robot hardware via Robot.py.

Run this on the Raspberry Pi alongside the vision code:
    cd Carc2026/robot-movement/library
    python movement_server.py

Protocol (JSON over UDP):
  { "cmd": "move",   "direction": "FRONT"|"BACK"|"LEFT"|"RIGHT", "power": 0-100 }
  { "cmd": "rotate", "direction": "LEFT"|"RIGHT",                "power": 0-100 }
  { "cmd": "stop"  }
  { "cmd": "brake" }
  { "cmd": "action", "action": "open_claw"|"close_claw"|"release_trigger"|
                                "reset_trigger"|"rev_spin"|"stop_spin",
                     "power": 20 }  # power only used for rev_spin
"""

import sys, os
from pathlib import Path
import time

sys.path.append(
    str(Path(__file__).parent / "robot-movement" / "library")
)

# Allow importing from parent folder so movement_server can share config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import socket
import json
import traceback
from config import (
    SERIAL_PORT, SERIAL_BAUD,
    MOVEMENT_CMD_PORT,
)
from Robot import Robot, Direction, Rotation


def run_server():
    robot = Robot(serial_port=SERIAL_PORT, baudrate=SERIAL_BAUD)
    print(f"[movement_server] Robot connected on {SERIAL_PORT}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", MOVEMENT_CMD_PORT))
    sock.settimeout(0.5)   # non-blocking so we can handle KeyboardInterrupt
    last_cmd_time = time.time()
    COMMAND_TIMEOUT = 0.35
    moving = False
    print(f"[movement_server] Listening on UDP port {MOVEMENT_CMD_PORT}")

    try:
        while True:
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                if moving and time.time() - last_cmd_time > COMMAND_TIMEOUT:
                    robot.Break()
                    robot.Stop()
                    # robot = Robot(serial_port=SERIAL_PORT, baudrate=SERIAL_BAUD)
                    moving = False
                continue

            try:
                msg = json.loads(data.decode("utf-8"))
            except json.JSONDecodeError:
                print(f"[movement_server] Bad JSON: {data}")
                continue

            cmd = msg.get("cmd", "")

            if cmd == "move":
                direction_str = msg.get("direction", "FRONT").upper()
                power = int(msg.get("power", 20))
                direction = Direction[direction_str]
                robot.Move(power, direction)
                moving = True

            elif cmd == "rotate":
                direction_str = msg.get("direction", "LEFT").upper()
                power = int(msg.get("power", 15))
                rotation = Rotation[direction_str]
                # Use raw SetMotorPowers for continuous rotation (no encoder blocking)
                from Robot import MapRotationToWheel
                multiplier = MapRotationToWheel(rotation)
                robot.left_wheel_driver.SetMotorPowers(
                    power * multiplier[0], power * multiplier[2])
                robot.right_wheel_driver.SetMotorPowers(
                    power * multiplier[1], power * multiplier[3])
                moving = True

            elif cmd in ("stop", "brake"):
                robot.Break()
                robot.Stop()
                moving = False

            elif cmd == "action":
                action = msg.get("action", "")
                power  = int(msg.get("power", 20))
                if action == "open_claw":        robot.OpenClaw()
                elif action == "close_claw":     robot.CloseClaw()
                elif action == "release_trigger":robot.ReleaseTrigger()
                elif action == "reset_trigger":  robot.ResetTrigger()
                elif action == "rev_spin":       robot.RevUpSpinWheel(power)
                elif action == "stop_spin":      robot.StopSpinWheel()
                else:
                    print(f"[movement_server] Unknown action: {action}")
            else:
                print(f"[movement_server] Unknown cmd: {cmd}")

    except KeyboardInterrupt:
        print("\n[movement_server] Shutting down.")
    finally:
        try:
            robot.Break()
            robot.Stop()
        except Exception:
            pass
        sock.close()


if __name__ == "__main__":
    run_server()

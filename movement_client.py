"""
movement_client.py — Helper used by vision code to send drive commands.

Import this anywhere you need to move the robot:
    from movement_client import MovementClient, MoveCmd

The client computes the correct command from the current position/angle errors
using proportional control with min/max power limits and dead zones.
All directions are in ROBOT FRAME (not table frame):
  - FRONT/BACK = along robot's own facing direction
  - LEFT/RIGHT = perpendicular to robot's facing direction (strafe)
  - ROTATE LEFT/RIGHT = spin about robot centre
"""

import socket
import json
import math
from dataclasses import dataclass, field
from typing import Optional

from config import (
    MOVEMENT_HOST, MOVEMENT_CMD_PORT,
    MOVE_POWER_MIN, MOVE_POWER_MAX, MOVE_GAIN_MM, MOVE_DEAD_MM,
    ROT_POWER_MIN,  ROT_POWER_MAX,  ROT_GAIN_DEG,  ROT_DEAD_DEG,
)


@dataclass
class MoveCmd:
    """Represents a single computed movement command."""
    cmd:       str            # "stop" | "move" | "rotate" | "brake" | "action"
    direction: str  = ""     # "FRONT"|"BACK"|"LEFT"|"RIGHT"|"LEFT"|"RIGHT"
    power:     int  = 0
    action:    str  = ""     # for cmd="action"

    def to_dict(self) -> dict:
        d = {"cmd": self.cmd}
        if self.direction: d["direction"] = self.direction
        if self.power:     d["power"]     = self.power
        if self.action:    d["action"]    = self.action
        return d

    def describe(self) -> str:
        if self.cmd == "stop":   return "STOP"
        if self.cmd == "move":   return f"MOVE {self.direction} pwr={self.power}"
        if self.cmd == "rotate": return f"ROTATE {self.direction} pwr={self.power}"
        if self.cmd == "action": return f"ACTION {self.action}"
        return self.cmd


def proportional_power(error_abs: float, gain: float,
                        pmin: int, pmax: int) -> int:
    """Compute a clamped proportional power from an error magnitude."""
    return int(max(pmin, min(pmax, error_abs * gain)))


def compute_move_cmd(x_error_mm:   Optional[float] = None,
                     y_error_mm:   Optional[float] = None,
                     ang_error_deg: Optional[float] = None,
                     priority: str = "rotate_first") -> MoveCmd:
    """
    Compute a single movement command from current errors.

    Args:
        x_error_mm:   Positive → robot too far RIGHT → must move LEFT.
        y_error_mm:   Positive → robot too far forward → must move BACK.
        ang_error_deg: Positive → rotate RIGHT to reduce error.
        priority:     "rotate_first" | "translate_first" | "simultaneous"
                      With "rotate_first", rotation is issued before translation.

    Returns:
        MoveCmd — send via MovementClient.send().
    """
    ang_ok  = ang_error_deg is None or abs(ang_error_deg) < ROT_DEAD_DEG
    x_ok    = x_error_mm   is None or abs(x_error_mm)    < MOVE_DEAD_MM
    y_ok    = y_error_mm   is None or abs(y_error_mm)    < MOVE_DEAD_MM

    # ── Rotation ────────────────────────────────────────────────────────────
    if not ang_ok:
        if priority == "rotate_first" or (x_ok and y_ok):
            pwr = proportional_power(abs(ang_error_deg), ROT_GAIN_DEG,
                                     ROT_POWER_MIN, ROT_POWER_MAX)
            d = "RIGHT" if ang_error_deg > 0 else "LEFT"
            return MoveCmd(cmd="rotate", direction=d, power=pwr)

    # ── Translation ─────────────────────────────────────────────────────────
    # Prioritise the larger error axis
    errors = []
    if not x_ok and x_error_mm is not None:
        errors.append(("x", x_error_mm))
    if not y_ok and y_error_mm is not None:
        errors.append(("y", y_error_mm))

    if errors:
        axis, err = max(errors, key=lambda e: abs(e[1]))
        pwr = proportional_power(abs(err), MOVE_GAIN_MM, MOVE_POWER_MIN, MOVE_POWER_MAX)
        if axis == "x":
            # positive x_error = robot too far RIGHT = move LEFT
            d = "LEFT" if err > 0 else "RIGHT"
        else:
            # positive y_error = robot too far FORWARD = move BACK
            d = "BACK" if err > 0 else "FRONT"
        return MoveCmd(cmd="move", direction=d, power=pwr)

    return MoveCmd(cmd="stop")


class MovementClient:
    """
    Sends movement commands over UDP to movement_server.py.
    Safe to create once and reuse across the whole program lifetime.
    """

    def __init__(self,
                 host: str = MOVEMENT_HOST,
                 port: int = MOVEMENT_CMD_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (host, port)

    def send(self, cmd: MoveCmd) -> None:
        try:
            payload = json.dumps(cmd.to_dict()).encode()
            self._sock.sendto(payload, self._addr)
        except Exception as e:
            print(f"[MovementClient] send error: {e}")

    def send_raw(self, msg: dict) -> None:
        try:
            payload = json.dumps(msg).encode()
            self._sock.sendto(payload, self._addr)
        except Exception as e:
            print(f"[MovementClient] send_raw error: {e}")

    def stop(self):
        self.send(MoveCmd(cmd="stop"))

    def brake(self):
        self.send(MoveCmd(cmd="brake"))

    def action(self, action_name: str, power: int = 20):
        self.send(MoveCmd(cmd="action", action=action_name, power=power))

    def close(self):
        self._sock.close()

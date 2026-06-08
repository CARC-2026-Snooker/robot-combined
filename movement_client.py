"""
movement_client.py — Build and send drive commands to movement_server.py.

The robot now uses OmniMovement(vX, vY, vR) which handles simultaneous
translation and rotation in one call. This is the only movement command
needed for drive_to — one UDP packet per tick, all three axes together.

Robot.OmniMovement conventions (robot frame, all in range -720..720):
  vX > 0  = strafe RIGHT      vX < 0  = strafe LEFT
  vY > 0  = move FORWARD      vY < 0  = move BACK
  vR > 0  = rotate CW/RIGHT   vR < 0  = rotate CCW/LEFT

Key function:  world_errors_to_omni()
  Given position + heading errors in world frame and current heading,
  returns a single MoveCmd("omni") that corrects all three axes at once.
"""

import socket, json, math
from dataclasses import dataclass
from typing import Optional
from config import (
    MOVEMENT_HOST, MOVEMENT_CMD_PORT,
    MOVE_POWER_MIN, MOVE_POWER_MAX,
    ROT_POWER_MIN,  ROT_POWER_MAX,
    MOVE_DEAD_MM, ROT_DEAD_DEG,
)


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND DATACLASS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MoveCmd:
    """One movement command to send to movement_server."""
    cmd:      str        # "omni" | "move" | "rotate_by" | "brake" | "stop" | "action"
    vX:       int = 0   # omni: strafe  (+right, -left)
    vY:       int = 0   # omni: forward (+fwd,   -back)
    vR:       int = 0   # omni: rotate  (+CW,    -CCW)
    speed:    int = 0   # move / rotate_by
    degree:   int = 0   # move / rotate_by
    rotation: int = 0   # move simultaneous rotation
    action:   str = ""  # action name
    power:    int = 20  # action power

    def to_dict(self) -> dict:
        if self.cmd == "omni":
            return {"cmd": "omni", "vX": self.vX, "vY": self.vY, "vR": self.vR}
        if self.cmd == "move":
            return {"cmd": "move", "speed": self.speed,
                    "degree": self.degree, "rotation": self.rotation}
        if self.cmd == "rotate_by":
            return {"cmd": "rotate_by", "speed": self.speed, "degree": self.degree}
        if self.cmd == "action":
            return {"cmd": "action", "action": self.action, "power": self.power}
        return {"cmd": self.cmd}   # brake / stop

    def describe(self) -> str:
        if self.cmd == "omni":
            return f"OMNI vX={self.vX:+d} vY={self.vY:+d} vR={self.vR:+d}"
        if self.cmd == "move":
            return f"MOVE speed={self.speed} deg={self.degree} rot={self.rotation}"
        if self.cmd == "rotate_by":
            return f"ROTATE_BY speed={self.speed} deg={self.degree} [BLOCKING]"
        if self.cmd == "action":
            return f"ACTION {self.action} pwr={self.power}"
        return self.cmd.upper()


# ─────────────────────────────────────────────────────────────────────────────
# PROPORTIONAL SPEED HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _clamp_power(err_abs: float, full_range: float,
                 pmin: int, pmax: int, dead: float) -> int:
    """Linear proportional speed. 0 in dead zone, pmin–pmax above it."""
    if err_abs < dead:
        return 0
    frac = min(1.0, (err_abs - dead) / max(full_range - dead, 1.0))
    return int(pmin + frac * (pmax - pmin))


# ─────────────────────────────────────────────────────────────────────────────
# WORLD → ROBOT FRAME CONVERSION  (the core of drive_to)
# ─────────────────────────────────────────────────────────────────────────────
def world_errors_to_omni(
        x_err_mm:    float,
        y_err_mm:    float,
        ang_err_deg: float,
        heading_deg: float,
        x_full_mm:    float = 250.0,
        y_full_mm:    float = 450.0,
        rot_full_deg: float = 60.0,
) -> MoveCmd:
    """
    Convert world-frame position and heading errors into one OmniMovement command.

    x_err_mm    = robot_x - target_x  (mm).  + means robot too far RIGHT in world.
    y_err_mm    = robot_y - target_y  (mm).  + means robot too far toward high-Y.
    ang_err_deg = current_heading - target_heading (°). + means rotate CW to fix.
    heading_deg = robot's current world-frame heading.

    Robot frame (from Robot.py OmniMovement convention):
      vX > 0 = strafe RIGHT    vX < 0 = strafe LEFT
      vY > 0 = move FORWARD    vY < 0 = move BACK
      vR > 0 = rotate CW       vR < 0 = rotate CCW

    World → robot frame transform (standard 2D rotation, robot right = CCW perp of forward):
      robot forward  = (cosθ, sinθ)
      robot right    = (-sinθ, cosθ)    ← CCW perpendicular

    To reach target we need velocity toward (−x_err, −y_err) in world.
    Project onto robot axes:
      vY_raw = (-x_err)·cosθ + (-y_err)·sinθ  = -x_err·cosθ - y_err·sinθ
      vX_raw = (-x_err)·(-sinθ) + (-y_err)·cosθ = x_err·sinθ - y_err·cosθ

    At heading 270° (robot faces AprilTags, sinθ=-1, cosθ=0):
      vX = x_err·(-1) - y_err·0 = -x_err   (too far right → vX<0 → LEFT ✓)
      vY = -x_err·0 - y_err·(-1) = y_err    (too far +Y → vY>0 → FORWARD ✓)
    """
    h     = math.radians(heading_deg)
    sin_h = math.sin(h)
    cos_h = math.cos(h)

    # ── Rotation (vR) ────────────────────────────────────────────────────────
    rot_spd = _clamp_power(abs(ang_err_deg), rot_full_deg,
                           ROT_POWER_MIN, ROT_POWER_MAX, ROT_DEAD_DEG)
    vR = -rot_spd if ang_err_deg > 0 else rot_spd

    # ── Translation (vX, vY) ─────────────────────────────────────────────────
    dist_mm = math.hypot(x_err_mm, y_err_mm)
    trans_spd = _clamp_power(dist_mm,
                             max(x_full_mm, y_full_mm),
                             MOVE_POWER_MIN, MOVE_POWER_MAX, MOVE_DEAD_MM)

    if trans_spd > 0 and dist_mm > 0:
        # Project world error direction into robot frame (note: CCW right)
        raw_vX =  x_err_mm * sin_h - y_err_mm * cos_h
        raw_vY = -x_err_mm * cos_h - y_err_mm * sin_h
        mag = math.hypot(raw_vX, raw_vY)
        if mag > 1e-9:
            vX = int(round(raw_vX / mag * trans_spd))
            vY = int(round(raw_vY / mag * trans_spd))
        else:
            vX = vY = 0
    else:
        vX = vY = 0

    return MoveCmd(cmd="omni", vX=vX, vY=vY, vR=vR)


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FACTORIES  (for manual use in mission_controller)
# ─────────────────────────────────────────────────────────────────────────────
def omni(vX: int = 0, vY: int = 0, vR: int = 0) -> MoveCmd:
    """Direct OmniMovement. All three axes, all at once."""
    return MoveCmd(cmd="omni", vX=vX, vY=vY, vR=vR)

def move(speed: int, degree: int, rotation: int = 0) -> MoveCmd:
    """robot.Move(speed, degree, rotation).
    Degree: 0=forward, 90=right, 180=back, 270=left (robot frame)."""
    return MoveCmd(cmd="move", speed=speed, degree=degree, rotation=rotation)

def rotate_by(speed: int, degree: int) -> MoveCmd:
    """robot.RotateStack(speed, degree). BLOCKING — encoder-based precise snap."""
    return MoveCmd(cmd="rotate_by", speed=speed, degree=degree)

# def rotate_to(speed: int, degree: int) -> MoveCmd:
#     """robot.Rotate(speed, degree). BLOCKING — encoder-based precise snap."""
#     return MoveCmd(cmd="rotate_to", speed=speed, degree=degree)

def brake() -> MoveCmd:
    return MoveCmd(cmd="brake")

def stop() -> MoveCmd:
    return MoveCmd(cmd="stop")

def action(name: str, power: int = 20) -> MoveCmd:
    return MoveCmd(cmd="action", action=name, power=power)


# ─────────────────────────────────────────────────────────────────────────────
# CLIENT
# ─────────────────────────────────────────────────────────────────────────────
class MovementClient:
    """UDP client — sends MoveCmd packets to movement_server.py."""

    def __init__(self, host: str = MOVEMENT_HOST, port: int = MOVEMENT_CMD_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (host, port)

    def send(self, cmd: MoveCmd) -> None:
        try:
            self._sock.sendto(json.dumps(cmd.to_dict()).encode(), self._addr)
        except Exception as e:
            print(f"[MovementClient] send error: {e}")

    # ── Shorthand methods ────────────────────────────────────────────────────
    def omni(self, vX: int = 0, vY: int = 0, vR: int = 0):
        """OmniMovement: move and rotate simultaneously."""
        self.send(omni(vX, vY, vR))

    def move(self, speed: int, degree: int, rotation: int = 0):
        """robot.Move: degree 0=fwd 90=right 180=back 270=left, optional rotation."""
        self.send(move(speed, degree, rotation))

    def rotate_by(self, speed: int, degree: int):
        """Blocking precise rotation. Positive degree = CW."""
        self.send(rotate_by(speed, degree))

    # def rotate_to(self, speed: int, degree: int):
    #     """Blocking precise rotation. Positive degree = CW."""
    #     self.send(rotate_to(speed, degree))

    def brake(self):
        """Hold motors in place."""
        self.send(brake())

    def stop(self):
        """Release motors."""
        self.send(stop())

    def action(self, name: str, power: int = 20):
        """
        Mechanism actions:
          open_claw / close_claw
          release_trigger / reset_trigger
          rev_spin_power(power) / stop_spin_power
          rev_spin_speed(power) / stop_spin_speed
        """
        self.send(action(name, power))

    def close(self):
        self._sock.close()


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY ALIAS  (old code that imports compute_move_cmd still works)
# ─────────────────────────────────────────────────────────────────────────────
def compute_move_cmd(x_error_mm=None, y_error_mm=None,
                     ang_error_deg=None, heading_deg=270.0, **_) -> MoveCmd:
    """Legacy wrapper. Use world_errors_to_omni() for new code."""
    return world_errors_to_omni(
        x_err_mm    = x_error_mm    or 0.0,
        y_err_mm    = y_error_mm    or 0.0,
        ang_err_deg = ang_error_deg or 0.0,
        heading_deg = heading_deg,
    )
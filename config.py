"""
config.py — Single source of truth for the entire CARC 2026 Snooker Robot.

Every tunable value, threshold, camera parameter, and physical constant lives
here. All other files import from this module. To tune something, edit it once
here — no hunting across multiple files.

Structure:
  HARDWARE       — serial port, camera indices
  NETWORK        — socket ports and addresses
  FIELD          — table dimensions, pocket positions, tag positions
  ROBOT_GEOMETRY — physical offsets between cameras and robot centre
  HD_CAMERA      — intrinsics, tracking thresholds, performance knobs
  CAM_4K         — intrinsics, distortion, ball HSV ranges
  MOVEMENT       — power limits, proportional gains, dead zones
  PLANNER        — shot geometry constants, corner offset
  DISPLAY        — map scale, colours
"""

import numpy as np

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HARDWARE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 1_000_000

CAM_HD_INDEX  = 2   # HD webcam (tracking + ball scan)
CAM_4K_INDEX  = 1   # 4K sniper camera

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NETWORK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VISION_HOST        = "127.0.0.1"   # where vision code runs
MOVEMENT_HOST      = "127.0.0.1"   # where movement_server runs (same Pi)

TELEMETRY_PORT     = 5005   # vision → anyone   (robot state, position, errors)
MOVEMENT_CMD_PORT  = 5010   # vision → movement (drive commands)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIELD GEOMETRY  (all in metres, origin = Tag 3 / Hole_3 corner)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARENA_BOUNDS = (0.0, 0.0, 1.2, 3.6)   # (x_min, y_min, x_max, y_max) metres

# AprilTag positions and facing angle (degrees).
# facing_angle = world-frame direction the tag's front face points.
# 270° = pointing in the -Y world direction (toward the robot side).
KNOWN_TAGS = {
    1: {"x": 1.262, "y": 1.05,  "facing_angle": 270},
    2: {"x":-0.062, "y": 1.05,  "facing_angle": 270},
    3: {"x": 0.0,   "y": 0.0,   "facing_angle": 270},
    4: {"x": 0.6,   "y": 0.0,   "facing_angle": 270},
    5: {"x": 1.2,   "y": 0.0,   "facing_angle": 270},
}
TAG_SIZE_M = 0.11   # physical tag size in metres (used by pose estimator)

# Pocket positions in mm (x=width axis, y=length axis, origin=Hole_3 corner)
POCKETS_MM = {
    "Hole_3": (0,    0),
    "Hole_4": (600,  0),
    "Hole_5": (1200, 0),
    "Hole_2": (0,    1050),
    "Hole_1": (1200, 1050),
}
POCKET_TO_TAG = {
    "Hole_3": 3, "Hole_4": 4, "Hole_5": 5, "Hole_2": 2, "Hole_1": 1,
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROBOT GEOMETRY  (offsets from robot centre to camera lens, in metres)
# Positive = forward along robot's heading direction.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HD_CAM_FORWARD_M  = 0.05    # HD lens is 5cm ahead of robot centre
HD_CAM_SIDEWAYS_M = 0.00    # lateral offset (positive = robot's right); 0 for now

CAM_4K_FORWARD_M  = 0.20    # 4K lens tip is 20cm ahead of robot centre
CAM_4K_SIDEWAYS_M = 0.00    # lateral offset; 0 for now

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HD CAMERA — TRACKING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HD_RESOLUTION   = (1280, 720)    # width × height in pixels
HD_FPS          = 30
HD_CAM_PARAMS   = [1924.70, 1924.67, 907.46, 483.50]  # fx, fy, cx, cy at 1080p
# Scale factor: actual_h / 1080.0 applied at runtime

# Crop strip removed from top/bottom before detection (fraction of frame height).
# Reduces area the detector has to search → faster.
HD_CROP_TOP     = 0.20
HD_CROP_BOT     = 0.20

# Tracking filter constants — DRIVING mode
HD_MAX_TAG_DIST     = 10.0    # ignore tags farther than this (metres)
HD_OUT_OF_BOUNDS    = 0.15    # metres outside arena boundary still accepted
HD_MAX_POSE_JUMP    = 0.50    # metres; larger jumps discarded as outliers
HD_TRACKING_TIMEOUT = 0.30    # seconds without tags → lose tracking

HD_STILL_ALPHA  = 0.005   # low-pass alpha when robot is stationary
HD_MOVE_ALPHA   = 0.25    # low-pass alpha when robot is moving
HD_STILL_THR    = 0.005   # raw-reading std (m) below which = STILL
HD_MOVE_THR     = 0.030   # raw-reading std (m) above which = MOVING
HD_HISTORY_LEN  = 15      # frames of raw readings kept for std calculation

# Consensus soft-rejection sigma (metres).
# Tag estimates further than this from the distance-weighted mean are down-weighted.
HD_CONSENSUS_SIGMA = 0.045

# Tracking filter constants — SNIPER mode (robot is stationary, tighter)
SN_STILL_ALPHA  = 0.005
SN_MOVE_ALPHA   = 0.30
SN_STILL_THR    = 0.003
SN_MOVE_THR     = 0.020

# Line-intersection fusion weight (0 = ignore intersection estimate, 1 = use only intersection)
# The intersection estimate is blended with the pose estimate: final = blend * intersection + (1-blend) * pose
LINE_INTERSECT_BLEND = 0.0   # 30% from ray-intersection, 70% from per-tag pose

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HD CAMERA — BALL DETECTION (main.py, used for scanning)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HD_SCAN_RESOLUTION = (1920, 1080)   # higher res for ball scan snapshot

# Zones (x, y, w, h) in pixels at 1080p
HD_DETECTION_ZONES = {
    "B": (360, 560, 170, 170),
    "C": (610, 500, 130, 130),
    "D": (790, 500, 130, 130),
    "E": (960, 520, 130, 130),
    "F": (1130,500, 130, 130),
    "G": (1320,500, 130, 130),
    "A": (1550,560, 170, 170),
}

HD_BALL_RED_LO1  = (0,   50,  20)
HD_BALL_RED_HI1  = (15,  255, 255)
HD_BALL_RED_LO2  = (165, 50,  20)
HD_BALL_RED_HI2  = (180, 255, 255)
HD_BALL_BLUE_LO  = (95,  50,  30)
HD_BALL_BLUE_HI  = (150, 255, 255)
HD_BALL_WHITE_LO = (180, 75,  255)
HD_BALL_WHITE_HI = (180, 75,  255)
HD_BALL_PERCENT_THRESHOLD = 0.05

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4K CAMERA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CAM_4K_RESOLUTION = (3840, 2160)   # set lower for Pi performance; was 3840×2160
CAM_4K_PARAMS     = [4534.22, 4528.08, 979.27, 775.85]  # fx, fy, cx, cy at 1080p
CAM_4K_DIST_COEFFS = np.array(
    [[-1.23467127e+00, 1.37940482e+01,
      -2.47010796e-02, 6.62621201e-03, -1.51346017e+02]]
)

# Ball colour masks used in the 4K sniper view
CAM_4K_BALL_RED = [
    (np.array([0,   50,  20]),  np.array([15,  255, 255])),
    (np.array([165, 50,  20]),  np.array([180, 255, 255])),
]
CAM_4K_BALL_BLUE = [
    (np.array([65,  10,  10]),  np.array([170, 255, 255])),
]

# Lateral lock tolerance (pixels at full capture resolution)
LATERAL_TOL_PX = 10

# Angle lock tolerance (degrees)
ANGLE_LOCK_DEG = 0.3

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RASPBERRY PI PERFORMANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DETECT_EVERY_N_FRAMES = 1    # AprilTag detection runs every N HD frames (1=every frame)
DISPLAY_SCALE         = 0.5 # Scale display windows (1.0=full, 0.5=half)

HD_BRIGHTNESS_BOOST   = 1.0  # 1.0=off; increase if image is dark after low-exposure tuning

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MOVEMENT — proportional speed control
# Power range: 0–100 (passed directly to Robot.Move / Robot.Rotate)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MOVE_POWER_MIN  = 15    # slowest movement power (near target)
MOVE_POWER_MAX  = 60    # fastest movement power (far from target)

ROT_POWER_MIN   = 10    # slowest rotation power
ROT_POWER_MAX   = 40    # fastest rotation power

# Proportional gain: power = clamp(|error| * gain, MIN, MAX)
MOVE_GAIN_MM    = 0.4   # power per mm of position error   (e.g. 50mm → power 20)
ROT_GAIN_DEG    = 1.0   # power per degree of angle error  (e.g. 10° → power 10)

# Dead zones — stop commanding movement below these errors
MOVE_DEAD_MM    = 5.0   # mm  — don't move if closer than this
ROT_DEAD_DEG    = 0.3   # °   — don't rotate if heading error is smaller than this

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLANNER — shot geometry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BALL_DIAMETER_MM   = 52.5
TABLE_WIDTH_MM     = 1200
TABLE_LENGTH_MM    = 3600

ROBOT_SIZE_MM      = 450
ROBOT_HALF_MM      = ROBOT_SIZE_MM / 2.0
ROBOT_X_MIN        = ROBOT_HALF_MM + 20
ROBOT_X_MAX        = TABLE_WIDTH_MM - ROBOT_HALF_MM - 20
ROBOT_SHOOT_Y      = 2100 + ROBOT_HALF_MM + 50   # mm

# Corner pocket offset — shifts aim point inward to avoid clipping the side wall.
# Must match in both snooker_planner.py and sniper_4k.py.
# Set to 0 to disable.
CORNER_OFFSET_MM   = 25.0

CORNER_EFFECTIVE_POCKETS = {
    "Hole_3": (CORNER_OFFSET_MM,              0),
    "Hole_5": (TABLE_WIDTH_MM - CORNER_OFFSET_MM, 0),
}

# Ball zone world coordinates (mm)
REAL_ZONES = {
    "A": (1050, 1050),
    "B": (150,  1050),
    "C": (200,  300),
    "D": (400,  300),
    "E": (600,  500),
    "F": (800,  300),
    "G": (1000, 300),
}

# Shot scoring weights
SHOT_WEIGHT_DISTANCE = 1.0
SHOT_WEIGHT_ANGLE    = 25.0
SHOT_MAX_CUT_ANGLE   = 55.0
SHOT_STRAIGHT_THR    = 5.0    # cut angles below this → STRAIGHT shot mode

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DISPLAY — 2D map drawing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAP_SCALE   = 200    # pixels per metre
MAP_PADDING = 60     # pixels of border

TAG_COLORS = {
    1: (255, 105, 180),  # hot pink
    2: (0,   255, 255),  # yellow
    3: (0,   165, 255),  # orange
    4: (255, 255,   0),  # cyan
    5: (255,   0, 255),  # magenta
}

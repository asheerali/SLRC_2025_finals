"""REAL ROBOT: detect blue/green caps and the white boxes: blue -> left box, green -> right box.

Same logic as test_cap_color.py (the simulation script); only the camera topics differ and depth
is required. Keep the emergency stop within reach and the workspace clear (rules.md.txt, Rule 5).
"""

import math
import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from olo import Client, OloError, OloTimeoutError, OloUnavailable
from olo.core.images import Image, compressed_message_to_frame
from olo.kinematics import JointTarget
from olo.spatial import Point, Pose, Quaternion
from olo.vision import Pixel, camera_info_to_intrinsics, deproject_to_frame, project, sample_depth_point

# Real robot camera topics (the simulation uses wrist_rgbd/...). CAMERA_INFO_TOPIC and DEPTH_TOPIC
# are ASSUMED names: check them in your topic list and change them if they differ.
RGB_TOPIC = "camera/color/image_raw/compressed"
CAMERA_INFO_TOPIC = "camera/color/camera_info"
DEPTH_TOPIC = "camera/depth/image_raw"   # raw depth (not the colourised stream); gives each cap's real height
OUTPUT_TOPIC = "camera/annotated"
REQUIRE_DEPTH = True         # real robot: never guess heights; skip the cap/box if depth is unavailable
MAX_FPS = 10
NO_FRAMES_WARNING_S = 5.0   # warn if the camera topic stays silent this long

# ---------------- YOLO detection (optional) ----------------
# Set YOLO_MODEL_PATH to your trained .pt file to detect with YOLO instead of colour ranges.
# Leave it as None to keep the colour method below.
YOLO_MODEL_PATH = None            # e.g. r"C:\models\caps_and_boxes.pt"
YOLO_CONFIDENCE = 0.5
# Your model's class name -> what it means here: "blue", "green" (caps) or "box".
YOLO_CLASS_NAMES = {"blue_cap": "blue", "green_cap": "green", "box": "box"}

# ---------------- Cap / box detection settings ----------------
# HSV ranges (OpenCV: H 0-179, S/V 0-255). Tune these under your lighting.
BLUE_HSV = ((100, 120, 50), (130, 255, 255))
GREEN_HSV = ((40, 100, 50), (85, 255, 255))
WHITE_HSV = ((0, 0, 190), (179, 50, 255))
MIN_CAP_AREA_PX = 150        # smaller coloured blobs are noise
MIN_BOX_AREA_PX = 4000       # white blobs at least this big count as boxes

# Box search: the boxes sit left and right, outside the camera view at the detection
# pose, so the arm turns joint1 (base rotation) through these angles, pausing at each.
# Everything else stays at the detection pose. Positive joint1 = turn left (assumed).
SEARCH_JOINT1_RAD = [0.5, -0.5, 1.0, -1.0]   # left and right first, then further out
SEARCH_DWELL_S = 2.0         # wait at each search window before reading the boxes
SEARCH_MAX_S = 30.0          # give up searching after this long
CAMERA_HFOV_RAD = 1.0        # assumed horizontal field of view; tune for your camera
BOX_MERGE_RAD = 0.2          # sightings closer than this in bearing are the same box
BOX_MIN_SEPARATION_RAD = 0.5 # left and right boxes must be at least this far apart

# Pick and place. Heights are z in the planning frame (assumed: +Z up, base on the table).
# TABLE_Z_M, CAP_HEIGHT_M and BOX_RIM_Z_M are ASSUMED values: measure them in your scene.
TABLE_Z_M = 0.0              # table surface height
CAP_HEIGHT_M = 0.03          # cap height; only used if the depth image is unavailable
BOX_RIM_Z_M = 0.08           # top rim of the white boxes
PRE_HOVER_Z_M = 0.25         # first move: straight above the target, arm orientation unchanged
ALIGN_STANDOFF_M = 0.05      # stop this far above the top of a cap before descending
LIFT_HEIGHT_M = 0.12         # lift a grasped cap this far
MAX_ITERATIONS = 15
POSITION_TOLERANCE_M = 0.005
ORIENTATION_TOLERANCE_RAD = math.radians(5)
MAX_TRANSLATION_STEP_M = 0.1                # alignment moves at most this far per step
MAX_ROTATION_STEP_RAD = math.radians(15)    # ...and turns the wrist at most this much
ALIGN_PLANNING_TIMEOUT_S = 1.5              # per attempt; unreachable steps fail fast
EXECUTION_TIMEOUT_S = 30.0                  # slow (speed 0.1) moves need longer than the 10 s default
GRASP_BELOW_TOP_M = (0.010, 0.005, 0.0)   # grasp this far below the measured cap top; later ones are retries
CLOSE_ANYWAY_ABOVE_TOP_M = 0.02   # if the descent errors out within this of the cap top, still close
PICK_GAP_M = 0.10            # gripper opening (gap logic) before a grasp and when releasing
RELEASE_ABOVE_RIM_M = 0.06   # open the gripper this far above the box rim
RELEASE_APPROACH_M = 0.04    # hover this far above the release point before lowering
RELEASE_PULL_IN_M = (0.0, 0.04, 0.08, 0.12)   # if the box centre is out of reach, aim this much nearer the base
RELEASE_NEAR_BOX_M = 0.12    # if the gripper is this close to the box centre (sideways), just drop the cap
CAP_MERGE_M = 0.03           # cap sightings closer than this are the same cap
CAP_SCAN_S = 2.0             # watch the caps this long before picking
CAP_IGNORE_NEAR_BOX_M = 0.12 # a cap this close to a box is already sorted; do not pick it again
PLANNING_TIMEOUT_S = 2.0   # per attempt; unreachable poses fail fast
SETTLE_S = 0.5
CAP_DESTINATION = {"blue": "left", "green": "right"}   # blue caps -> left box, green -> right box
DRAW_COLOR = {"blue": (255, 0, 0), "green": (0, 200, 0), "box": (255, 255, 255)}  # BGR


# ---------------- Arm positions and rules (see rules.md.txt) ----------------
INITIAL_JOINTS = {
    "joint1": -0.0040,
    "joint2": -0.0013,
    "joint3": -0.0055,
    "joint4": 0.0078,
    "joint5": -0.0227,
    "joint6": -0.0090,
}
INITIAL_FINGER_GAP_M = 0.0004   # recorded: finger_left 0.0002, finger_right -0.0002

DETECTION_JOINTS = {
    "joint1": 0.0000,
    "joint2": -1.5708,
    "joint3": -1.5708,
    "joint4": 1.0472,
    "joint5": 0.0000,
    "joint6": 0.0000,
}
DETECTION_FINGER_GAP_M = 0.10   # 10 cm between the fingers

MAX_FINGER_GAP_M = 0.10         # RULE: finger gap must never exceed 10 cm
JOINT_TOLERANCE_RAD = 0.01      # how close counts as "already there" (~0.6 deg)
SPEED = 0.1                     # 0-1, lower is slower
GRIPPER_TIMEOUT_S = 15.0
GAP_TOLERANCE_M = 0.002         # 2 mm counts as "already there"
CLOSED_THRESHOLD_M = 0.002      # gaps below 2 mm are treated as "closed"


def arm_target(arm, joints):
    """Build a JointTarget for the arm joints."""
    group = arm.model.group()
    group_joints = tuple(group.joints)

    if group_joints:
        names = tuple(name for name in group_joints if name in joints)
        missing = [name for name in group_joints if name not in joints]
        if missing:
            raise ValueError(f"Target is missing arm joints: {missing}")
    else:
        names = tuple(joints.keys())

    if not names:
        raise ValueError("No joints to move")
    return JointTarget(names=names, positions=tuple(joints[n] for n in names))


def check_joint_limits(arm, joints):
    """Refuse to move if any target joint is outside the robot model's limits."""
    limits = {j.name: j for j in arm.model.joints}
    for name, value in joints.items():
        j = limits.get(name)
        if j and j.lower_limit < j.upper_limit and not (j.lower_limit <= value <= j.upper_limit):
            raise ValueError(
                f"{name}={value:.4f} is outside its limits [{j.lower_limit:.4f}, {j.upper_limit:.4f}]"
            )


def is_at(arm, joints, tolerance=JOINT_TOLERANCE_RAD):
    """Return True if every arm joint is within tolerance of the target."""
    current = arm.joints().position_by_name
    off = {
        name: current.get(name, float("nan")) - target
        for name, target in joints.items()
        if not abs(current.get(name, float("inf")) - target) <= tolerance
    }
    for name, err in off.items():
        print(f"  {name} is off by {err:+.4f} rad")
    return not off


def move_to(arm, joints, label):
    """Move the arm to a joint position, skipping if already there."""
    if is_at(arm, joints):
        print(f"Already at the {label} position")
        return
    check_joint_limits(arm, joints)
    print(f"Moving to {label} position...")
    arm.move_joints(arm_target(arm, joints), speed_scaling=SPEED)
    print(f"Reached {label} position")


def measured_gap(arm):
    """Return the current finger gap in metres, or None if fingers aren't in joint states."""
    current = arm.joints().position_by_name
    if "finger_left" in current and "finger_right" in current:
        return abs(current["finger_left"] - current["finger_right"])
    return None


def set_finger_gap(arm, gap_m):
    """Set the total finger gap (metres), enforcing the max-gap rule."""
    gripper = arm.ee
    if gripper is None:
        print("No gripper found; skipping fingers")
        return

    limit_cm = MAX_FINGER_GAP_M * 100

    # RULE: never exceed the maximum gap.
    if gap_m > MAX_FINGER_GAP_M:
        print(f"Requested gap {gap_m * 100:.1f} cm exceeds the {limit_cm:.0f} cm limit; clamping")
        gap_m = MAX_FINGER_GAP_M
    gap_m = max(gap_m, 0.0)

    before = measured_gap(arm)
    if before is not None and abs(before - gap_m) <= GAP_TOLERANCE_M:
        print(f"Fingers already at {before * 100:.1f} cm; no command needed")
        return

    try:
        if gap_m <= CLOSED_THRESHOLD_M:
            result = gripper.close(timeout=GRIPPER_TIMEOUT_S)
        else:
            per_finger = gap_m / 2.0  # mirrored fingers: each moves half the gap
            info = arm.model.end_effector_for(arm.model.group().name).gripper
            lo, hi = sorted((info.min_position, info.max_position))
            if hi > lo and not (lo <= per_finger <= hi):
                per_finger = min(max(per_finger, lo), hi)
                print(f"Clamped to gripper range: gap {per_finger * 200:.1f} cm")
            result = gripper.goal(per_finger, timeout=GRIPPER_TIMEOUT_S)
        print(f"Gripper command done (reached_goal={result.reached_goal}, stalled={result.stalled})")
    except OloTimeoutError:
        print("Gripper command timed out; checking where the fingers actually are...")

    after = measured_gap(arm)
    if after is None:
        print("Finger positions not reported; cannot verify gap")
        return
    print(f"Measured finger gap: {after * 100:.1f} cm (target {gap_m * 100:.1f} cm)")
    if after > MAX_FINGER_GAP_M + GAP_TOLERANCE_M:
        gripper.goal(MAX_FINGER_GAP_M / 2.0, timeout=GRIPPER_TIMEOUT_S)
        raise RuntimeError(f"Finger gap {after * 100:.1f} cm exceeded the {limit_cm:.0f} cm limit; corrected")
    if abs(after - gap_m) > 0.01:
        print("Warning: fingers did not reach the target gap (more than 1 cm off)")


def find_blobs(hsv, bounds, min_area, skip_border=False):
    """Return (centre_xy, bbox, area) for every blob inside the HSV range.

    skip_border drops blobs touching the image edge (e.g. the table, which runs off the frame).
    """
    height, width = hsv.shape[:2]
    mask = cv2.inRange(hsv, np.array(bounds[0], np.uint8), np.array(bounds[1], np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if skip_border and (x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2):
            continue
        blobs.append(((x + w / 2, y + h / 2), (x, y, w, h), area))
    return blobs


_yolo = None


def load_yolo():
    """Load the YOLO model once (needs `pip install ultralytics`)."""
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO(YOLO_MODEL_PATH)
        print(f"YOLO model loaded: {YOLO_MODEL_PATH} (classes: {list(_yolo.names.values())})")
    return _yolo


def detect_with_yolo(bgr):
    """Detect caps and boxes with the YOLO model, in the same format as the colour method."""
    result = load_yolo()(bgr, conf=YOLO_CONFIDENCE, verbose=False)[0]
    caps, boxes = [], []
    for xyxy, cls in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
        kind = YOLO_CLASS_NAMES.get(result.names[int(cls)])
        if kind is None:
            continue
        x1, y1, x2, y2 = xyxy
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        centre, area = (x1 + w / 2, y1 + h / 2), w * h
        if kind == "box":
            boxes.append((centre, (x, y, w, h), area))
        else:
            caps.append({"colour": kind, "centre": centre, "bbox": (x, y, w, h), "area": area})
    return caps, boxes


def detect_caps_and_boxes(bgr):
    """Find blue/green caps and white boxes in a BGR frame (YOLO if a model is set, else colour)."""
    if YOLO_MODEL_PATH:
        return detect_with_yolo(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    caps = []
    for colour, bounds in (("blue", BLUE_HSV), ("green", GREEN_HSV)):
        for centre, bbox, area in find_blobs(hsv, bounds, MIN_CAP_AREA_PX):
            caps.append({"colour": colour, "centre": centre, "bbox": bbox, "area": area})
    boxes = find_blobs(hsv, WHITE_HSV, MIN_BOX_AREA_PX, skip_border=True)
    return caps, boxes


class BoxMemory:
    """Boxes found during the search, stored as a bearing (rad, positive = left).

    A box's bearing is the joint1 angle the arm was at, corrected for where the
    box appeared in the image.
    """

    def __init__(self):
        self.boxes = []   # {"bearing": rad, "joint1": rad, "pixel": (x, y)}

    def add(self, boxes, rgb, joint1, locate):
        width = rgb.array.shape[1]
        for (cx, cy), _bbox, _area in boxes:
            bearing = joint1 - (cx / width - 0.5) * CAMERA_HFOV_RAD
            if any(abs(bearing - old["bearing"]) < BOX_MERGE_RAD for old in self.boxes):
                continue
            try:
                position = locate(rgb, (cx, cy), BOX_RIM_Z_M)
            except Exception as exc:
                position = None
                print(f"  could not locate box in 3D: {exc}")
            self.boxes.append({"bearing": bearing, "joint1": joint1, "pixel": (cx, cy), "position": position})
            where = f", at {fmt(position)}" if position is not None else ""
            print(f"  box found: bearing {bearing:+.2f} rad (arm joint1 {joint1:+.2f}, pixel {cx:.0f},{cy:.0f}){where}")

    @property
    def saved(self):
        """{"left": box, "right": box} once two well-separated boxes are known."""
        if len(self.boxes) < 2:
            return None
        left = max(self.boxes, key=lambda box: box["bearing"])
        right = min(self.boxes, key=lambda box: box["bearing"])
        if left["bearing"] - right["bearing"] < BOX_MIN_SEPARATION_RAD:
            return None
        return {"left": left, "right": right}


def pump_frames(frames, errors, publisher, memory, loop, seconds=None):
    """Detect and publish annotated frames for `seconds` (forever if None).

    Returns (caps, boxes, rgb) from the last frame processed (rgb is None if none arrived).
    """
    end = None if seconds is None else time.monotonic() + seconds
    last = ([], [], None)
    while end is None or time.monotonic() < end:
        try:
            message = frames.get(timeout=0.1)
        except queue.Empty:
            if not errors.empty():
                raise errors.get()
            if not loop["warned"] and time.monotonic() - loop["last_frame_time"] > NO_FRAMES_WARNING_S:
                print(f"WARNING: no frames on '{RGB_TOPIC}' for {NO_FRAMES_WARNING_S:.0f}s; "
                      "check the topic name.")
                loop["warned"] = True
            continue
        loop["last_frame_time"] = time.monotonic()
        loop["warned"] = False

        now = time.monotonic()
        if now < loop["next_frame"]:
            continue
        loop["next_frame"] = now + 1 / MAX_FPS

        rgb = compressed_message_to_frame(message)
        caps, boxes = detect_caps_and_boxes(rgb.array)
        last = (caps, boxes, rgb)
        state = len(caps)
        if state != loop["last_count"]:
            print(f"{len(caps)} cap(s)")
            for cap in caps:
                cx, cy = cap["centre"]
                print(f"  {cap['colour']} cap at ({cx:.0f}, {cy:.0f}) px -> "
                      f"{CAP_DESTINATION[cap['colour']]} box")
            loop["last_count"] = state
        publisher.publish(annotate(rgb, caps, boxes, memory))
    return last


def search_for_boxes(arm, memory, pump, locate):
    """Turn joint1 through the search windows, 2 s each, until both boxes are found (max 30 s)."""
    start = time.monotonic()
    for angle in SEARCH_JOINT1_RAD:
        if memory.saved:
            break
        if time.monotonic() - start >= SEARCH_MAX_S:
            print(f"Search time limit ({SEARCH_MAX_S:.0f} s) reached")
            break
        target = {**DETECTION_JOINTS, "joint1": angle}
        try:
            move_to(arm, target, f"search window (joint1 {angle:+.2f})")
        except ValueError as exc:
            print(f"Skipping search window: {exc}")
            continue
        _caps, boxes, rgb = pump(SEARCH_DWELL_S)
        if rgb is not None:
            memory.add(boxes, rgb, angle, locate)
    if memory.saved:
        print("Boxes saved: "
              f"left bearing {memory.saved['left']['bearing']:+.2f} rad, "
              f"right bearing {memory.saved['right']['bearing']:+.2f} rad")
    else:
        print(f"Search finished with {len(memory.boxes)} box(es); need a left and a right box")


def annotate(rgb: Image, caps, boxes, memory) -> Image:
    canvas = rgb.array.copy()
    for centre, (x, y, w, h), _ in boxes:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), DRAW_COLOR["box"], 2)
    for cap in caps:
        x, y, w, h = cap["bbox"]
        colour = DRAW_COLOR[cap["colour"]]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, 2)
        label = f"{cap['colour']} -> {CAP_DESTINATION[cap['colour']]}"
        cv2.putText(canvas, label, (x, max(y - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    status = "boxes saved" if memory.saved else f"{len(memory.boxes)} box(es) found"
    cv2.putText(canvas, f"{len(caps)} cap(s) | {status}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return Image(array=canvas, encoding="bgr8", header=rgb.header)


def fmt(p):
    return f"({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"


def pinhole(intrinsics):
    """(cx, cy, fx, fy) read back through project(), so no intrinsics field names are assumed."""
    origin = project(Point(x=0.0, y=0.0, z=1.0), intrinsics)
    right = project(Point(x=1.0, y=0.0, z=1.0), intrinsics)
    down = project(Point(x=0.0, y=1.0, z=1.0), intrinsics)
    return origin.x, origin.y, right.x - origin.x, down.y - origin.y


def pixel_to_plane(robot, arm, intrinsics, rgb, pixel, plane_z):
    """3D point (planning frame) where the camera ray through `pixel` meets z = plane_z."""
    if intrinsics is None:
        raise RuntimeError("no camera intrinsics (camera_info missing)")
    cx, cy, fx, fy = pinhole(intrinsics)
    header = rgb.header
    frame_id = header["frame_id"] if isinstance(header, dict) else header.frame_id
    camera = robot.frames.lookup(arm.planning_frame, frame_id)
    origin = camera.apply(Point(x=0.0, y=0.0, z=0.0))
    through = camera.apply(Point(x=(pixel[0] - cx) / fx, y=(pixel[1] - cy) / fy, z=1.0))
    direction = through - origin
    if direction.z >= -1e-6:
        raise RuntimeError("camera ray does not point down at the table")
    return origin + direction * ((plane_z - origin.z) / direction.z)


def locate_with_depth(robot, arm, intrinsics, depth_topic, pixel):
    """3D point (planning frame) under `pixel`, measured from the depth image."""
    if intrinsics is None:
        raise RuntimeError("no camera intrinsics (camera_info missing)")
    depth = robot.core.get_image(depth_topic, timeout=3.0)
    observation = sample_depth_point(depth, Pixel(x=float(pixel[0]), y=float(pixel[1])), intrinsics,
                                     window_size=5, min_depth_metres=0.05, max_depth_metres=3.0)
    if observation is None:
        raise RuntimeError(f"no valid depth at pixel ({pixel[0]:.0f}, {pixel[1]:.0f})")
    return deproject_to_frame(observation, frames=robot.frames, target_frame=arm.planning_frame).point


def locate_caps(locate, rgb, caps):
    """{"blue": [Point, ...], "green": [...]} cap-top centres in the planning frame, nearest first.

    point.z is the height of the cap's top (measured by depth when available).
    """
    found = {"blue": [], "green": []}
    plane_z = TABLE_Z_M + CAP_HEIGHT_M    # fallback height of the cap top
    for cap in caps:
        try:
            point = locate(rgb, cap["centre"], plane_z)
        except Exception as exc:
            print(f"  could not locate {cap['colour']} cap in 3D: {exc}")
            continue
        same = found[cap["colour"]]
        if any(math.hypot(point.x - q.x, point.y - q.y) < CAP_MERGE_M for q in same):
            continue
        same.append(point)
    for points in found.values():
        points.sort(key=lambda q: math.hypot(q.x, q.y))
    return found


# ---------------- Pick and place ----------------
class PickFailed(Exception):
    """The cap could not be picked; nothing is being held."""


@dataclass
class Grasp:
    pick_pose: Pose    # end-effector pose when the gripper closed
    lift_pose: Pose    # pick pose raised by LIFT_HEIGHT_M


class ReachFailed(Exception):
    """No bounded alignment step could be planned towards the target."""


def quat_dot(a, b):
    return sum(x * y for x, y in zip(a.array(), b.array()))


def quat_angle(a, b):
    """Rotation angle between two orientations, in radians."""
    return 2 * math.acos(min(1.0, abs(quat_dot(a, b))))


def slerp(q0, q1, t):
    """Orientation a fraction t of the way from q0 to q1 (shortest path)."""
    a, b = np.asarray(q0.array(), float), np.asarray(q1.array(), float)
    d = float(np.dot(a, b))
    if d < 0.0:
        b, d = -b, -d
    if d > 0.9995:
        r = a + t * (b - a)
    else:
        theta = math.acos(d)
        r = (math.sin((1 - t) * theta) * a + math.sin(t * theta) * b) / math.sin(theta)
    return Quaternion.from_array(r / np.linalg.norm(r))


def call_arm(method, pose, **kwargs):
    """Run an arm motion at SPEED (Rule 5)."""
    return method(pose, speed_scaling=SPEED, execution_timeout=EXECUTION_TIMEOUT_S, **kwargs)


def move_to_pose(arm, pose, avoid_obstacles, planning_timeout=PLANNING_TIMEOUT_S):
    """move_pose, retrying once without collision checking. Returns the error or None."""
    error = None
    for check in dict.fromkeys((avoid_obstacles, False)):
        try:
            call_arm(arm.move_pose, pose, avoid_obstacles=check, planning_timeout=planning_timeout)
            return None
        except Exception as exc:
            error = exc
    return error


def tool_down_orientations(point):
    """Gripper-pointing-down orientations, 90 degrees apart in yaw, facing the base first."""
    base = math.atan2(point.y, point.x)
    flip = Quaternion.from_rpy(math.pi, 0.0, 0.0)
    return [Quaternion.from_rpy(0.0, 0.0, base + k * math.pi / 2) * flip for k in range(4)]


def step_towards(arm, ee_pose, target_position, orientations):
    """Take one bounded step; shrink it or change yaw if it is unreachable.

    Tries each candidate orientation (closest first) with a full, half and quarter
    step, turning at most MAX_ROTATION_STEP_RAD. As a last resort it moves without
    turning. Returns False only if nothing at all could be planned.
    """
    error = target_position - ee_pose.position
    distance = error.magnitude()
    step = error if distance <= MAX_TRANSLATION_STEP_M else error * (MAX_TRANSLATION_STEP_M / distance)
    last_error = None
    for yaw_index, orientation in enumerate(orientations):
        angle = quat_angle(orientation, ee_pose.orientation)
        for fraction in (1.0, 0.5, 0.25):
            turn = min(1.0, fraction * MAX_ROTATION_STEP_RAD / angle) if angle > 1e-6 else 1.0
            pose = Pose(
                position=ee_pose.position + step * fraction,
                orientation=slerp(ee_pose.orientation, orientation, turn),
            )
            last_error = move_to_pose(arm, pose, True, ALIGN_PLANNING_TIMEOUT_S)
            if last_error is None:
                return True
            print(f"  step unreachable (grip yaw {yaw_index + 1}/{len(orientations)}, {fraction:.0%} step): {last_error}")
    last_error = move_to_pose(
        arm, Pose(position=ee_pose.position + step * 0.5, orientation=ee_pose.orientation),
        True, ALIGN_PLANNING_TIMEOUT_S)
    if last_error is None:
        return True
    print(f"Alignment step unreachable: {last_error}")
    return False


def align_to(arm, target_position, orientations):
    """Walk the gripper to `target_position` in bounded steps, turning towards the closest of
    `orientations`. Returns the orientation it settled on."""
    for _ in range(MAX_ITERATIONS):
        ee_pose = arm.pose()
        ordered = sorted(orientations, key=lambda q: -abs(quat_dot(q, ee_pose.orientation)))
        distance = (target_position - ee_pose.position).magnitude()
        angle = quat_angle(ordered[0], ee_pose.orientation)
        print(f"  position error {distance:.3f} m, orientation error {math.degrees(angle):.1f} deg")
        if distance <= POSITION_TOLERANCE_M and angle <= ORIENTATION_TOLERANCE_RAD:
            return ordered[0]
        if not step_towards(arm, ee_pose, target_position, ordered):
            raise ReachFailed("no reachable step towards the target (out of the arm's reach?)")
        time.sleep(SETTLE_S)
    raise ReachFailed(f"did not converge within {MAX_ITERATIONS} steps")


def grip(arm, closing):
    """Close with plain gripper.close(); open with the gap logic (set_finger_gap).

    Returns the close() result, or None when opening or on timeout.
    """
    if not closing:
        set_finger_gap(arm, PICK_GAP_M)
        return None
    try:
        return arm.ee.close(timeout=GRIPPER_TIMEOUT_S)
    except OloTimeoutError:
        print("  gripper close timed out")
        return None


def finger_position(arm):
    """Opening of the gripper joint in metres (0 = closed), or None if it is not reported."""
    try:
        info = arm.model.end_effector_for(arm.model.group().name).gripper
        current = arm.joints().position_by_name
        for name in info.joints:
            if name in current:
                return current[name]
    except Exception:
        pass
    return None


def open_gripper(arm):
    """Open the fingers, check that they really opened, and force a full open if they did not.

    Used to let go of a cap: the gap command can leave the fingers clamped on it.
    """
    grip(arm, closing=False)
    time.sleep(SETTLE_S)
    wanted = PICK_GAP_M / 2 - 0.01
    position = finger_position(arm)
    if position is not None and position < wanted:
        print(f"  fingers only {position * 1000:.0f} mm open; forcing a full open")
        try:
            arm.ee.open(timeout=GRIPPER_TIMEOUT_S)
        except Exception as exc:
            print(f"  open() reported: {exc}")
        time.sleep(SETTLE_S)
        position = finger_position(arm)
        if position is not None and position < wanted:
            try:
                info = arm.model.end_effector_for(arm.model.group().name).gripper
                print(f"  still only {position * 1000:.0f} mm; commanding the open position {info.open_position}")
                arm.ee.goal(info.open_position, timeout=GRIPPER_TIMEOUT_S)
            except Exception as exc:
                print(f"  goal(open position) reported: {exc}")
            time.sleep(SETTLE_S)
            position = finger_position(arm)
    if position is not None:
        print(f"  fingers now {position * 1000:.0f} mm open")


def pick_cap(arm, point):
    """Hover over the cap, align on it, descend, grasp and lift. Returns the Grasp.

    point.z is the height of the cap's top.
    """
    grip(arm, closing=False)
    time.sleep(SETTLE_S)

    # Start straight above the cap with the arm's current orientation: that is reachable.
    start_orientation = arm.pose().orientation
    above = Pose(position=Point(x=point.x, y=point.y, z=PRE_HOVER_Z_M), orientation=start_orientation)
    if move_to_pose(arm, above, True) is None:
        time.sleep(SETTLE_S)

    standoff = Point(x=point.x, y=point.y, z=point.z + ALIGN_STANDOFF_M)
    try:
        orientation = align_to(arm, standoff, tool_down_orientations(point))
    except ReachFailed as exc:
        raise PickFailed(f"cannot reach the cap: {exc}") from exc

    pick_pose = None
    for below_top in GRASP_BELOW_TOP_M:
        target = Pose(position=Point(x=point.x, y=point.y, z=point.z - below_top), orientation=orientation)
        try:
            call_arm(arm.move_linear, target)
            pick_pose = target
            break
        except Exception as exc:
            print(f"  cannot lower to z={target.position.z:.3f}: {exc}")
    if pick_pose is None:
        # The controller often reports an error when the fingers touch the cap or table. If the
        # gripper is already down around the cap, that is low enough: close on it anyway.
        now = arm.pose()
        if now.position.z <= point.z + CLOSE_ANYWAY_ABOVE_TOP_M:
            print(f"  controller stopped at z={now.position.z:.3f} (cap top {point.z:.3f}); closing anyway")
            pick_pose = Pose(position=now.position, orientation=orientation)
        else:
            raise PickFailed(f"cannot lower onto the cap (stopped at z={now.position.z:.3f}, cap top {point.z:.3f})")
    time.sleep(SETTLE_S)
    result = grip(arm, closing=True)
    time.sleep(SETTLE_S)
    if result is not None and not result.stalled:
        print("  warning: the gripper closed fully; the cap may have been missed")

    lift_pose = Pose(position=Point(x=point.x, y=point.y, z=pick_pose.position.z + LIFT_HEIGHT_M),
                     orientation=orientation)
    call_arm(arm.move_linear, lift_pose)
    time.sleep(SETTLE_S)
    return Grasp(pick_pose=pick_pose, lift_pose=lift_pose)


def swing_towards(arm, point):
    """Turn only the base joint (joint1) so the gripper ends up facing `point`.

    A joint-space move needs no inverse kinematics, so it works where stepped Cartesian moves
    fail. The gripper keeps its height and distance from the base; the tool stays pointing down.
    """
    ee = arm.pose().position
    delta = math.atan2(point.y, point.x) - math.atan2(ee.y, ee.x)
    current = arm.joints().position_by_name
    joints = {name: current[name] for name in arm.model.group().joints}
    joints["joint1"] = joints["joint1"] + delta
    check_joint_limits(arm, joints)
    print(f"  swinging the base by {math.degrees(delta):+.0f} deg towards the box")
    arm.move_joints(arm_target(arm, joints), speed_scaling=SPEED)
    time.sleep(SETTLE_S)


def release_cap(arm, point, grasp):
    """Release over the box centre; if that is out of reach, aim a little nearer the base.

    Boxes are wide, so a point a few cm towards the base is still inside the box.
    """
    radius = math.hypot(point.x, point.y)
    for pull_in in RELEASE_PULL_IN_M:
        scale = max(radius - pull_in, 0.15) / radius
        aim = Point(x=point.x * scale, y=point.y * scale, z=point.z)
        if pull_in:
            print(f"  box centre out of reach; trying {pull_in * 100:.0f} cm nearer the base")
        if release_at(arm, aim, grasp):
            return True
    return False


def release_at(arm, point, grasp):
    """Carry the held cap over `point`, lower to just above the rim, open, and back off."""
    release_z = BOX_RIM_Z_M + RELEASE_ABOVE_RIM_M
    above_position = Point(x=point.x, y=point.y, z=release_z + RELEASE_APPROACH_M)
    try:
        swing_towards(arm, point)
    except Exception as exc:
        print(f"  could not swing the base towards the box: {exc}")
    held = grasp.lift_pose.orientation   # also allow the grasp orientation (or a 90-degree turn of it)
    orientations = (tool_down_orientations(point)
                    + [held * Quaternion.from_rpy(0.0, 0.0, k * math.pi / 2) for k in range(4)])
    try:
        orientation = align_to(arm, above_position, orientations)
    except ReachFailed as exc:
        here = arm.pose()
        sideways = math.hypot(here.position.x - point.x, here.position.y - point.y)
        if sideways <= RELEASE_NEAR_BOX_M:
            # The arm is already over the box; do not insist on the exact spot, just drop the cap.
            print(f"  {exc}; but the gripper is {sideways * 100:.0f} cm from the box centre, so dropping here")
            orientation = here.orientation
        else:
            print(f"  stepped approach failed ({exc}); moving straight above the box")
            orientation = None
            for candidate in orientations:
                if move_to_pose(arm, Pose(position=above_position, orientation=candidate), False, 5.0) is None:
                    orientation = candidate
                    break
            if orientation is None:
                print(f"  cannot reach the box (gripper is {sideways * 100:.0f} cm from its centre)")
                return False
    time.sleep(SETTLE_S)
    print(f"  above the box at {fmt(arm.pose().position)}; lowering to z={release_z:.3f}")
    try:
        call_arm(arm.move_linear, Pose(position=Point(x=point.x, y=point.y, z=release_z), orientation=orientation))
    except Exception as exc:
        # We are already above the box: let go from here, the cap just drops in.
        print(f"  cannot lower into the box ({exc}); dropping from here")
    time.sleep(SETTLE_S)
    print(f"  releasing at {fmt(arm.pose().position)}")
    open_gripper(arm)
    time.sleep(SETTLE_S)
    try:
        call_arm(arm.move_linear, Pose(position=above_position, orientation=orientation))
    except Exception as exc:
        print(f"  cap released; could not back off straight up ({exc})")
    time.sleep(SETTLE_S)
    return True


def put_back(arm, grasp):
    """Return the held cap to exactly where it was picked up."""
    print("  putting the cap back where it was picked up")
    move_to_pose(arm, grasp.lift_pose, False)
    time.sleep(SETTLE_S)
    call_arm(arm.move_linear, grasp.pick_pose)
    time.sleep(SETTLE_S)
    open_gripper(arm)
    time.sleep(SETTLE_S)
    call_arm(arm.move_linear, grasp.lift_pose)
    time.sleep(SETTLE_S)


def describe_arm(arm):
    """Print what the planner needs to see, to help diagnose unreachable poses."""
    try:
        pose = arm.pose()
        print(f"Arm now: end effector {fmt(pose.position)}, planning frame '{arm.planning_frame}', "
              f"link '{arm.end_effector_link}'")
        gripper = arm.model.end_effector_for(arm.model.group().name).gripper
        print(f"Gripper joints {tuple(gripper.joints)}, open {gripper.open_position}, "
              f"closed {gripper.closed_position}")
    except Exception as exc:
        print(f"(could not describe the arm: {exc})")


def sort_caps(arm, memory, pump, locate):
    """Pick every blue cap into the left box, then every green cap into the right box.

    After every drop (or failed pick) the arm goes back to the detection pose and looks again,
    so each cap is found from a fresh view and the arm never drives straight from a box to a cap.
    """
    describe_arm(arm)
    boxes = memory.saved
    plan = (("blue", boxes["left"], "left"), ("green", boxes["right"], "right"))
    box_points = [box["position"] for box in boxes.values() if box["position"] is not None]
    moved = skipped = 0
    failed = []   # positions of caps that could not be picked; ignored from then on

    def is_ignored(point):
        near_box = any(math.hypot(point.x - b.x, point.y - b.y) < CAP_IGNORE_NEAR_BOX_M for b in box_points)
        return near_box or any(math.hypot(point.x - f.x, point.y - f.y) < CAP_MERGE_M for f in failed)

    for colour, box, side in plan:
        target = box["position"]
        if target is None:
            print(f"No 3D position for the {side} box; skipping {colour} caps")
            continue
        while True:
            move_to(arm, DETECTION_JOINTS, "detection")
            caps, _boxes, rgb = pump(CAP_SCAN_S)
            if rgb is None:
                break
            remaining = [p for p in locate_caps(locate, rgb, caps)[colour] if not is_ignored(p)]
            if not remaining:
                print(f"No more {colour} caps to pick")
                break
            point = remaining[0]
            print(f"{colour} cap ({len(remaining)} left) at {fmt(point)} -> {side} box {fmt(target)}")
            try:
                grasp = pick_cap(arm, point)
            except PickFailed as exc:
                print(f"  skipped: {exc}")
                failed.append(point)
                skipped += 1
                continue
            if release_cap(arm, target, grasp):
                moved += 1
            else:
                put_back(arm, grasp)
                failed.append(point)
                skipped += 1
    print(f"Sorting finished: {moved} cap(s) moved, {skipped} skipped")


def receive_latest(robot, topic, frames, errors):
    try:
        with robot.core.subscribe(topic) as sub:
            for message in sub:
                try:
                    frames.get_nowait()
                except queue.Empty:
                    pass
                frames.put_nowait(message)
    except Exception as error:
        errors.put(error)


try:
    with Client() as client:
        robot = client.robot()
        rgb_topic = robot.core.topic(RGB_TOPIC, "sensor_msgs/msg/CompressedImage")
        info_topic = robot.core.topic(CAMERA_INFO_TOPIC, "sensor_msgs/msg/CameraInfo")
        output_topic = robot.core.topic(OUTPUT_TOPIC, "sensor_msgs/msg/Image")
        depth_topic = robot.core.topic(DEPTH_TOPIC, "sensor_msgs/msg/Image")
        publisher = robot.core.publisher(output_topic)
        base_frame = robot.frames.base_frame() or "base_link"
        arm = robot.kinematics

        # RULE 1: start from the initial position.
        print("[1] Initial position")
        move_to(arm, INITIAL_JOINTS, "initial")
        set_finger_gap(arm, INITIAL_FINGER_GAP_M)

        try:
            # RULE 2: detection is only done at the detection position.
            print("[2] Detection position")
            move_to(arm, DETECTION_JOINTS, "detection")
            set_finger_gap(arm, DETECTION_FINGER_GAP_M)

            # Camera intrinsics are read as in test_shape_color.py; they will be
            # needed to turn cap pixels into 3D positions for the pick-and-place step.
            try:
                intrinsics = camera_info_to_intrinsics(robot.core.get_latest(info_topic, timeout=3.0))
            except OloError as exc:
                intrinsics = None
                print(f"No camera_info ({exc}); continuing without 3D positions.")

            frames = queue.Queue(maxsize=1)
            errors = queue.Queue(maxsize=1)
            threading.Thread(target=receive_latest,
                             args=(robot, rgb_topic, frames, errors),
                             daemon=True).start()

            memory = BoxMemory()
            loop = {"next_frame": time.monotonic(), "last_count": None,
                    "last_frame_time": time.monotonic(), "warned": False}
            print(f"Publishing annotated frames to {OUTPUT_TOPIC} - add it as a camera panel.")

            def pump(seconds=None):
                return pump_frames(frames, errors, publisher, memory, loop, seconds)

            depth_warned = []

            def locate(rgb, pixel, plane_z):
                """Depth-measured 3D point; falls back to the assumed table plane if depth fails."""
                try:
                    return locate_with_depth(robot, arm, intrinsics, depth_topic, pixel)
                except Exception as exc:
                    if REQUIRE_DEPTH:
                        raise RuntimeError(f"depth unavailable ({exc}); not guessing heights on the real robot") from exc
                    if not depth_warned:
                        print(f"  depth not usable ({exc}); using the assumed table plane instead")
                        depth_warned.append(True)
                    return pixel_to_plane(robot, arm, intrinsics, rgb, pixel, plane_z)

            # Search for the left and right boxes (joint1 sweep, 2 s per window, max 30 s).
            print("[2b] Searching for boxes")
            search_for_boxes(arm, memory, pump, locate)

            if memory.saved:
                print("[2c] Sorting caps")
                sort_caps(arm, memory, pump, locate)
                move_to(arm, DETECTION_JOINTS, "detection")
            else:
                print("Not sorting: need both boxes saved. "
                      "Showing the live detection instead (Ctrl+C to stop).")
                move_to(arm, DETECTION_JOINTS, "detection")
                pump()
        finally:
            # RULE 4: always return to the initial position, even on Ctrl+C or error.
            print("[3] Back to initial position")
            move_to(arm, INITIAL_JOINTS, "initial")
            set_finger_gap(arm, INITIAL_FINGER_GAP_M)

except KeyboardInterrupt:
    print("Stopped.")
except OloTimeoutError as exc:
    print(f"The appliance did not respond in time: {exc}")
except OloUnavailable as exc:
    print(f"Could not connect to the appliance: {exc}")
except OloError as exc:
    print(f"SDK call failed: {exc}")

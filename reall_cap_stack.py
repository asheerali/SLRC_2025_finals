"""Find the round, colourless bottle caps at the detection position and stack one on top of another."""

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from olo import Client, OloError, OloTimeoutError, OloUnavailable
from olo.core.images import Image
from olo.kinematics import JointTarget
from olo.spatial import Point, Pose, Quaternion
from olo.vision import Pixel, camera_info_to_intrinsics, deproject_to_frame, project, sample_depth_point

RGB_TOPIC = "wrist_rgbd/image"
CAMERA_INFO_TOPIC = "wrist_rgbd/camera_info"
DEPTH_TOPIC = "wrist_rgbd/depth_image"   # depth gives the real height of each cap
OUTPUT_TOPIC = "wrist_rgbd/annotated"    # add this as a camera panel: it shows what was found / rejected
MAX_FPS = 10

# ---------------- YOLO detection (optional) ----------------
# Set YOLO_MODEL_PATH to your trained .pt file to detect with YOLO instead of the shape method.
YOLO_MODEL_PATH = None            # e.g. r"C:\models\bottle_caps.pt"
YOLO_CONFIDENCE = 0.5
YOLO_CAP_CLASSES = {"bottle_cap", "cap"}   # class names that mean "bottle cap"; None = every class

# ---------------- Shape detection (used when YOLO_MODEL_PATH is None) ----------------
# Caps are round and have no colour (grey / white / black): look for low-saturation blobs that are round.
CAP_MAX_SATURATION = 70       # above this a pixel is "coloured" (the wooden table) and is ignored
CAP_MIN_BRIGHTNESS = 40       # ignore very dark shadows
CAP_MIN_AREA_PX = 200
CAP_MAX_AREA_FRACTION = 0.15  # a blob bigger than this share of the image is background, not a cap
CAP_MIN_ROUNDNESS = 0.6       # short axis / long axis of the fitted ellipse (oblique views look oval)
CAP_MIN_FILL = 0.75           # blob area / fitted ellipse area
CAP_WIDTH_M = 0.08            # simulator caps: 8 cm across
CAP_WIDTH_TOLERANCE_M = 0.03  # accepted real size: width +- 3 cm (checked with the depth image)

# ---------------- Stacking ----------------
# TABLE_Z_M is only used if the depth image cannot be read. Cap height is real (2 cm in the simulator).
TABLE_Z_M = 0.0
CAP_HEIGHT_M = 0.02
PRE_HOVER_Z_M = 0.25         # first move: straight above the cap, arm orientation unchanged
ALIGN_STANDOFF_M = 0.05      # stop this far above the top of a cap before descending
LIFT_HEIGHT_M = 0.12         # lift a grasped cap this far
MAX_ITERATIONS = 15
POSITION_TOLERANCE_M = 0.005
ORIENTATION_TOLERANCE_RAD = math.radians(5)
MAX_TRANSLATION_STEP_M = 0.1                # alignment moves at most this far per step
MAX_ROTATION_STEP_RAD = math.radians(15)    # ...and turns the wrist at most this much
ALIGN_PLANNING_TIMEOUT_S = 1.5
EXECUTION_TIMEOUT_S = 30.0                  # slow (speed 0.1) moves need longer than the 10 s default
GRASP_BELOW_TOP_M = (0.010, 0.005, 0.0)     # grasp this far below the measured cap top; later ones are retries
CLOSE_ANYWAY_ABOVE_TOP_M = 0.02   # if the descent errors out within this of the cap top, still close
PICK_GAP_M = 0.10            # gripper opening before a grasp and when releasing (an 8 cm cap fits)
PLACE_DROP_GAP_M = 0.005     # release the cap this far above the cap below
RELEASE_APPROACH_M = 0.08    # hover this far above the release height before lowering
CAP_MERGE_M = 0.03           # cap sightings closer than this are the same cap
CAP_MIN_SEPARATION_M = 0.12  # the two caps should be at least this far apart (open fingers are 10 cm wide)
CAP_SCAN_S = 2.0             # watch the caps this long at the detection position before picking
PLANNING_TIMEOUT_S = 2.0
SETTLE_S = 0.5

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


# ---------------- Cap detection ----------------
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
    """Caps found by the YOLO model, as ([cap dicts], [])."""
    result = load_yolo()(bgr, conf=YOLO_CONFIDENCE, verbose=False)[0]
    caps = []
    for xyxy, cls in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
        if YOLO_CAP_CLASSES is not None and result.names[int(cls)] not in YOLO_CAP_CLASSES:
            continue
        x1, y1, x2, y2 = xyxy
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        caps.append({"centre": (x1 + w / 2, y1 + h / 2), "bbox": (x, y, w, h), "area": w * h})
    return caps, []


def detect_by_shape(bgr):
    """Round, colourless blobs. Returns ([cap dicts], [rejected blobs as (bbox, reason)])."""
    height, width = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] <= CAP_MAX_SATURATION) & (hsv[..., 2] >= CAP_MIN_BRIGHTNESS)).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    caps, rejected = [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < CAP_MIN_AREA_PX or len(cnt) < 5:
            continue   # tiny specks: not worth drawing
        x, y, w, h = cv2.boundingRect(cnt)
        if area > CAP_MAX_AREA_FRACTION * width * height:
            rejected.append(((x, y, w, h), "too big"))
            continue
        (cx, cy), (ax1, ax2), _ = cv2.fitEllipse(cnt)
        long_ax, short_ax = max(ax1, ax2), min(ax1, ax2)
        if long_ax <= 0:
            continue
        roundness = short_ax / long_ax
        fill = area / (math.pi * long_ax * short_ax / 4.0)
        if roundness < CAP_MIN_ROUNDNESS or fill < CAP_MIN_FILL:
            rejected.append(((x, y, w, h), f"not round {roundness:.2f}/{fill:.2f}"))
            continue
        caps.append({"centre": (cx, cy), "bbox": (x, y, w, h), "area": area})
    return caps, rejected


def detect_caps(bgr):
    """Find the caps in a BGR frame (YOLO if a model is set, else by shape)."""
    return detect_with_yolo(bgr) if YOLO_MODEL_PATH else detect_by_shape(bgr)


def annotate(rgb: Image, caps, rejected) -> Image:
    canvas = np.array(rgb.array() if callable(rgb.array) else rgb.array)
    for (x, y, w, h), reason in rejected:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(canvas, reason, (x, max(y - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    for cap in caps:
        x, y, w, h = cap["bbox"]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(canvas, "cap", (x, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{len(caps)} cap(s) | {len(rejected)} rejected (red)", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return Image(array=canvas, encoding="bgr8", header=rgb.header)


def grab_bgr(robot, rgb_topic):
    """Latest camera image as (Image, BGR array). Raw images may be RGB: convert so the colour code is right."""
    img = robot.core.get_image(rgb_topic, timeout=3.0)
    arr = np.asarray(img.array() if callable(img.array) else img.array)[..., :3].astype(np.uint8)
    if "bgr" not in str(getattr(img, "encoding", "rgb8")).lower():
        arr = np.ascontiguousarray(arr[..., ::-1])
    return img, arr


def pump_frames(grab, publisher, loop, seconds):
    """Detect and publish annotated frames for `seconds`.

    Returns (caps, bgr_image, rgb) from the last frame processed (rgb is None if no frame could be read).
    """
    end = time.monotonic() + seconds
    last = ([], None, None)
    while time.monotonic() < end:
        try:
            img, bgr = grab()
        except Exception as exc:
            print(f"Could not read a camera frame: {exc}")
            time.sleep(0.5)
            continue
        caps, rejected = detect_caps(bgr)
        last = (caps, bgr, img)
        state = (len(caps), tuple(reason for _, reason in rejected))
        if state != loop["last_state"]:
            print(f"{len(caps)} cap(s) found, {len(rejected)} blob(s) rejected: "
                  f"{[reason for _, reason in rejected] or 'none'}")
            loop["last_state"] = state
        try:
            publisher.publish(annotate(Image(array=bgr, encoding="bgr8", header=img.header), caps, rejected))
        except Exception as exc:
            print(f"Could not publish the annotated frame: {exc}")
        time.sleep(1 / MAX_FPS)
    return last


# ---------------- 3D positions ----------------
def fmt(p):
    return f"({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"


def pinhole(intrinsics):
    """(cx, cy, fx, fy) read back through project(), so no intrinsics field names are assumed."""
    origin = project(Point(x=0.0, y=0.0, z=1.0), intrinsics)
    right = project(Point(x=1.0, y=0.0, z=1.0), intrinsics)
    down = project(Point(x=0.0, y=1.0, z=1.0), intrinsics)
    return origin.x, origin.y, right.x - origin.x, down.y - origin.y


def camera_pose(robot, arm, rgb):
    header = rgb.header
    frame_id = header["frame_id"] if isinstance(header, dict) else header.frame_id
    return robot.frames.lookup(arm.planning_frame, frame_id)


def pixel_to_plane(robot, arm, intrinsics, rgb, pixel, plane_z):
    """3D point (planning frame) where the camera ray through `pixel` meets z = plane_z."""
    if intrinsics is None:
        raise RuntimeError("no camera intrinsics (camera_info missing)")
    cx, cy, fx, fy = pinhole(intrinsics)
    camera = camera_pose(robot, arm, rgb)
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


def locate_caps(robot, arm, intrinsics, locate, rgb, caps):
    """Cap-top centres in the planning frame, nearest the base first. point.z is the cap's top height.

    Round blobs whose real size (from the distance to the camera) is not a cap are dropped.
    """
    found = []
    plane_z = TABLE_Z_M + CAP_HEIGHT_M    # fallback height of the cap top
    fx = pinhole(intrinsics)[2] if intrinsics is not None else None
    origin = None
    if fx:
        try:
            origin = camera_pose(robot, arm, rgb).apply(Point(x=0.0, y=0.0, z=0.0))
        except Exception as exc:
            print(f"  camera pose unavailable, skipping the size check: {exc}")
    for cap in caps:
        try:
            point = locate(rgb, cap["centre"], plane_z)
        except Exception as exc:
            print(f"  could not locate a cap in 3D: {exc}")
            continue
        if origin is not None:
            width_m = max(cap["bbox"][2], cap["bbox"][3]) * (point - origin).magnitude() / fx
            if abs(width_m - CAP_WIDTH_M) > CAP_WIDTH_TOLERANCE_M:
                print(f"  ignored a round blob {width_m * 100:.1f} cm wide at {fmt(point)} "
                      f"(caps are {CAP_WIDTH_M * 100:.0f} cm)")
                continue
        if any(math.hypot(point.x - q.x, point.y - q.y) < CAP_MERGE_M for q in found):
            continue
        found.append(point)
    found.sort(key=lambda q: math.hypot(q.x, q.y))
    return found


# ---------------- Pick and place ----------------
class PickFailed(Exception):
    """The cap could not be picked; nothing is being held."""


@dataclass
class Grasp:
    pick_pose: Pose      # end-effector pose when the gripper closed
    lift_pose: Pose      # pick pose raised by LIFT_HEIGHT_M
    cap_top_z: float     # height of the picked cap's top before it was lifted


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
    """Open the fingers, check that they really opened, and force a full open if they did not."""
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

    lift_pose = Pose(position=Point(x=pick_pose.position.x, y=pick_pose.position.y,
                                    z=pick_pose.position.z + LIFT_HEIGHT_M),
                     orientation=orientation)
    call_arm(arm.move_linear, lift_pose)
    time.sleep(SETTLE_S)
    return Grasp(pick_pose=pick_pose, lift_pose=lift_pose, cap_top_z=point.z)


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
    print(f"  swinging the base by {math.degrees(delta):+.0f} deg towards the other cap")
    arm.move_joints(arm_target(arm, joints), speed_scaling=SPEED)
    time.sleep(SETTLE_S)


def place_on_cap(arm, base_cap, grasp):
    """Carry the held cap over `base_cap`, lower it onto that cap, open, and back off straight up.

    base_cap.z is the top of the cap below. Returns False if the stack position cannot be reached.
    """
    # The held cap's top sits (cap_top_z - pick z) above the gripper. Its bottom must end up
    # PLACE_DROP_GAP_M above the top of the cap below, so its top is CAP_HEIGHT_M higher than that.
    held_top_above_ee = grasp.cap_top_z - grasp.pick_pose.position.z
    release_z = base_cap.z + PLACE_DROP_GAP_M + CAP_HEIGHT_M - held_top_above_ee
    above_position = Point(x=base_cap.x, y=base_cap.y, z=release_z + RELEASE_APPROACH_M)
    try:
        swing_towards(arm, base_cap)
    except Exception as exc:
        print(f"  could not swing the base towards the cap: {exc}")
    held = grasp.lift_pose.orientation   # also allow the grasp orientation (or a 90-degree turn of it)
    orientations = (tool_down_orientations(base_cap)
                    + [held * Quaternion.from_rpy(0.0, 0.0, k * math.pi / 2) for k in range(4)])
    try:
        orientation = align_to(arm, above_position, orientations)
    except ReachFailed as exc:
        print(f"  stepped approach failed ({exc}); moving straight above the cap")
        orientation = None
        for candidate in orientations:
            if move_to_pose(arm, Pose(position=above_position, orientation=candidate), False, 5.0) is None:
                orientation = candidate
                break
        if orientation is None:
            print("  cannot reach the cap to stack on")
            return False
    time.sleep(SETTLE_S)
    print(f"  above the cap at {fmt(arm.pose().position)}; lowering to z={release_z:.3f}")
    try:
        call_arm(arm.move_linear, Pose(position=Point(x=base_cap.x, y=base_cap.y, z=release_z), orientation=orientation))
    except Exception as exc:
        print(f"  cannot lower onto the cap: {exc}")
        return False
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


def stack_caps(arm, pump, locate_all):
    """Look at the detection position, then pick the cap nearest the base and stack it on the other."""
    caps, _bgr, rgb = pump(CAP_SCAN_S)
    if rgb is None:
        print("No camera frames; cannot look for caps")
        return False
    points = locate_all(rgb, caps)
    print(f"{len(points)} cap(s) located: {[fmt(p) for p in points]}")
    if len(points) < 2:
        print("Need two caps to stack. Check the annotated camera view: red boxes are rejected blobs.")
        return False
    pick_point, base_point = points[0], points[1]
    separation = math.hypot(pick_point.x - base_point.x, pick_point.y - base_point.y)
    if separation < CAP_MIN_SEPARATION_M:
        print(f"Warning: the caps are only {separation * 100:.0f} cm apart; the open fingers may touch the other cap")
    print(f"Picking the cap at {fmt(pick_point)} and stacking it on the cap at {fmt(base_point)}")
    try:
        grasp = pick_cap(arm, pick_point)
    except PickFailed as exc:
        print(f"Could not pick the cap: {exc}")
        return False
    if place_on_cap(arm, base_point, grasp):
        print("Done: the cap is stacked")
        return True
    put_back(arm, grasp)
    return False


try:
    with Client() as client:
        robot = client.robot()
        rgb_topic = robot.core.topic(RGB_TOPIC, "sensor_msgs/msg/Image")
        info_topic = robot.core.topic(CAMERA_INFO_TOPIC, "sensor_msgs/msg/CameraInfo")
        output_topic = robot.core.topic(OUTPUT_TOPIC, "sensor_msgs/msg/Image")
        depth_topic = robot.core.topic(DEPTH_TOPIC, "sensor_msgs/msg/Image")
        publisher = robot.core.publisher(output_topic)
        arm = robot.kinematics

        if YOLO_MODEL_PATH:
            load_yolo()   # load before the arm moves, so a wrong path fails early

        # RULE 1: start from the initial position.
        print("[1] Initial position")
        move_to(arm, INITIAL_JOINTS, "initial")
        set_finger_gap(arm, INITIAL_FINGER_GAP_M)

        try:
            # RULE 2: detection is only done at the detection position.
            print("[2] Detection position")
            move_to(arm, DETECTION_JOINTS, "detection")
            set_finger_gap(arm, DETECTION_FINGER_GAP_M)

            try:
                intrinsics = camera_info_to_intrinsics(robot.core.get_latest(info_topic, timeout=3.0))
            except OloError as exc:
                intrinsics = None
                print(f"No camera_info ({exc}); continuing without 3D positions.")

            loop = {"last_state": None}
            print(f"Publishing annotated frames to {OUTPUT_TOPIC} - add it as a camera panel.")

            def pump(seconds=None):
                return pump_frames(lambda: grab_bgr(robot, rgb_topic), publisher, loop, seconds)

            depth_warned = []

            def locate(rgb, pixel, plane_z):
                """Depth-measured 3D point; falls back to the assumed table plane if depth fails."""
                try:
                    return locate_with_depth(robot, arm, intrinsics, depth_topic, pixel)
                except Exception as exc:
                    if not depth_warned:
                        print(f"  depth not usable ({exc}); using the assumed table plane instead")
                        depth_warned.append(True)
                    return pixel_to_plane(robot, arm, intrinsics, rgb, pixel, plane_z)

            def locate_all(rgb, caps):
                return locate_caps(robot, arm, intrinsics, locate, rgb, caps)

            print("[3] Stacking")
            stack_caps(arm, pump, locate_all)
            move_to(arm, DETECTION_JOINTS, "detection")
        finally:
            # RULE 4: always return to the initial position, even on Ctrl+C or error.
            print("[4] Back to initial position")
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

"""Detect the box (cube) from the detection position and print its coordinates.

Follows rules.md.txt: start at the initial position, detect only at the detection position,
finger gap never above 10 cm, always return to the initial position, speed 0.1.
No pick-up: the arm only goes to the detection position and back.

Coordinates are in the robot planning frame (printed at run time): x forward, y left, z up, in metres,
with the origin at the robot base. They come from the pose of the ArUco tag on the top face of the box.
"""

import math
import time

import cv2
import numpy as np

from olo import Client
from olo.core.images import Image
from olo.errors import OloTimeoutError
from olo.kinematics import JointTarget
from olo.spatial import Point
from olo.vision import camera_info_to_intrinsics

RGB_TOPIC = "wrist_rgbd/image"
CAMERA_INFO_TOPIC = "wrist_rgbd/camera_info"
ANNOTATED_TOPIC = "wrist_rgbd/annotated"   # add this as a camera panel to see the boxes and coordinates
TAG_FAMILY = "aruco_4x4_50"
MARKER_SIZE_M = 0.036
BOX_SIZE_M = 0.05
BOX_FACE_SCALE = BOX_SIZE_M / MARKER_SIZE_M   # marker outline -> whole top face (white margin)
DETECT_FRAMES = 15             # frames used; the coordinates are the median over them
FRAME_PERIOD_S = 0.15


# ---------------- Positions (radians) ----------------
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

# ---------------- Rules / settings ----------------
MAX_FINGER_GAP_M = 0.10         # RULE: finger gap must never exceed 10 cm
JOINT_TOLERANCE_RAD = 0.01      # how close counts as "already there" (~0.6 deg)
SPEED = 0.1                     # 0-1, lower is slower
GRIPPER_TIMEOUT_S = 15.0
GAP_TOLERANCE_M = 0.002         # 2 mm counts as "already there"
CLOSED_THRESHOLD_M = 0.002      # gaps below 2 mm are treated as "closed"


# ---------------- Arm helpers ----------------
def arm_target(arm, joints):
    """Build a JointTarget for the arm joints."""
    group = arm.model.group()
    group_joints = tuple(group.joints)

    if group_joints:
        # Group lists its joints: send only those, and make sure none are missing.
        names = tuple(name for name in group_joints if name in joints)
        missing = [name for name in group_joints if name not in joints]
        if missing:
            raise ValueError(f"Target is missing arm joints: {missing}")
    else:
        # Group doesn't list joints (e.g. chain-defined): use the target's joint names.
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


# ---------------- Finger helpers ----------------
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

    # Skip if the fingers are already at the requested gap.
    before = measured_gap(arm)
    if before is not None and abs(before - gap_m) <= GAP_TOLERANCE_M:
        print(f"Fingers already at {before * 100:.1f} cm; no command needed")
        return

    try:
        if gap_m <= CLOSED_THRESHOLD_M:
            # Nearly closed: close() treats the fingers touching as success.
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

    # Verify the result and enforce the max-gap rule.
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


def locate_cube(robot, arm, tag):
    """Top-face centre (x, y, z) and edge yaw of the cube a tag sits on, in the planning frame."""
    tp = tag.to_frame(robot.frames, arm.planning_frame)
    top = tp.apply(Point())
    yaw = None
    for axis in (Point(x=1.0), Point(y=1.0)):  # the tag axes lying on the top face
        p = tp.apply(axis)
        dx, dy = p.x - top.x, p.y - top.y
        if math.hypot(dx, dy) > 0.7:
            yaw = math.atan2(dy, dx)
            break
    return (top.x, top.y, top.z), (0.0 if yaw is None else yaw)


def average_yaw(yaws):
    """Mean of cube yaws; cube edges repeat every 90 degrees, so average 4x the angle."""
    return math.atan2(sum(math.sin(4 * y) for y in yaws), sum(math.cos(4 * y) for y in yaws)) / 4


def box_from_tag(robot, arm, tag):
    """Box coordinates from one top-face tag: centre, top-face centre, yaw, camera-frame position."""
    top, yaw = locate_cube(robot, arm, tag)
    camera = tag.transform.apply(Point())  # top-face centre in the camera optical frame
    centre = (top[0], top[1], top[2] - BOX_SIZE_M / 2)  # the tag is on the top face; the centre is half a box below
    return {"centre": centre, "top": top, "yaw": yaw, "camera": (camera.x, camera.y, camera.z)}


def detect_boxes(robot, arm, rgb_topic, detector, intrinsics, publisher):
    """Look for DETECT_FRAMES frames and return the boxes (left to right in the image) with median coordinates."""
    per_frame = []
    for _ in range(DETECT_FRAMES):
        rgb = robot.core.get_image(rgb_topic)
        frame = detector.detect(rgb, intrinsics=intrinsics)
        posed = [t for t in frame.tags if t.transform is not None]
        posed.sort(key=lambda t: (np.mean([c.x for c in t.corners]), np.mean([c.y for c in t.corners])))
        boxes = [box_from_tag(robot, arm, t) for t in posed]
        pixels = [np.array([(c.x, c.y) for c in t.corners], np.float32) for t in posed]
        per_frame.append((boxes, pixels, rgb))
        time.sleep(FRAME_PERIOD_S)

    counts = [len(boxes) for boxes, _, _ in per_frame if boxes]
    if not counts:
        return [], None
    n = max(set(counts), key=counts.count)  # the number of boxes seen in most frames
    good = [item for item in per_frame if len(item[0]) == n]
    result = []
    for i in range(n):
        samples = [boxes[i] for boxes, _, _ in good]
        result.append({
            "centre": tuple(float(np.median([s["centre"][k] for s in samples])) for k in range(3)),
            "top": tuple(float(np.median([s["top"][k] for s in samples])) for k in range(3)),
            "camera": tuple(float(np.median([s["camera"][k] for s in samples])) for k in range(3)),
            "yaw": average_yaw([s["yaw"] for s in samples]),
            "pixels": good[-1][1][i],
        })
    publisher.publish(annotate(good[-1][2], result))
    return result, n


def annotate(rgb, boxes):
    """Draw each box's top face with its planning-frame coordinates."""
    canvas = np.ascontiguousarray(rgb.array).copy()
    for number, box in enumerate(boxes, start=1):
        marker = box["pixels"]
        centre = marker.mean(axis=0)
        pts = ((marker - centre) * BOX_FACE_SCALE + centre).astype(np.int32)
        cv2.polylines(canvas, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(canvas, (int(centre[0]), int(centre[1])), 3, (0, 0, 255), -1)
        x, y, z = box["centre"]
        caption = f"box {number} ({x:.3f}, {y:.3f}, {z:.3f}) m"
        (text_w, _), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        text_x = int(min(max(pts[:, 0].min(), 4), canvas.shape[1] - text_w - 4))
        cv2.putText(canvas, caption, (text_x, max(int(pts[:, 1].min()) - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{len(boxes)} box(es)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    return Image(array=canvas, encoding=rgb.encoding, header=rgb.header)


# ---------------- Main sequence ----------------
with Client() as client:
    robot = client.robot()
    arm = robot.kinematics

    # RULE 1: always start from the initial position.
    print("[1] Initial position")
    move_to(arm, INITIAL_JOINTS, "initial")
    set_finger_gap(arm, INITIAL_FINGER_GAP_M)

    try:
        # RULE 2: detection only happens at the detection position.
        print("[2] Detection position")
        move_to(arm, DETECTION_JOINTS, "detection")
        set_finger_gap(arm, DETECTION_FINGER_GAP_M)

        # ---- detection happens here ----
        rgb_topic = robot.core.topic(RGB_TOPIC, "sensor_msgs/msg/Image")
        info_topic = robot.core.topic(CAMERA_INFO_TOPIC, "sensor_msgs/msg/CameraInfo")
        intrinsics = camera_info_to_intrinsics(robot.core.get_latest(info_topic))
        detector = robot.vision.tags(TAG_FAMILY, size_m=MARKER_SIZE_M)
        publisher = robot.core.publisher(robot.core.topic(ANNOTATED_TOPIC, "sensor_msgs/msg/Image"))

        print(f"[3] Detecting the box ({DETECT_FRAMES} frames); publishing to {ANNOTATED_TOPIC}")
        boxes, _ = detect_boxes(robot, arm, rgb_topic, detector, intrinsics, publisher)
        if not boxes:
            print("No box with a pose was detected.")
        frame_name = arm.planning_frame
        for number, box in enumerate(boxes, start=1):
            x, y, z = box["centre"]
            tx, ty, tz = box["top"]
            cx, cy, cz = box["camera"]
            print(f"\nBox {number}:")
            print(f"  centre (frame '{frame_name}'):  x={x:.3f}  y={y:.3f}  z={z:.3f} m")
            print(f"  top face centre:                x={tx:.3f}  y={ty:.3f}  z={tz:.3f} m")
            print(f"  distance from the base:         {math.hypot(x, y):.3f} m (horizontal)")
            print(f"  rotation about the vertical:    {math.degrees(box['yaw']):+.1f} deg")
            print(f"  top face in the camera frame:   x={cx:.3f}  y={cy:.3f}  z={cz:.3f} m")
    finally:
        # RULE 4: always return to the initial position, even if a step failed.
        print("\n[4] Back to initial position")
        move_to(arm, INITIAL_JOINTS, "initial")
        set_finger_gap(arm, INITIAL_FINGER_GAP_M)

    print("Done: back at the initial position")

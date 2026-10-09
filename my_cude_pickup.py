"""Go to the detection position and detect the cubes (ArUco tags) from there."""

import time

import cv2
import numpy as np

from olo import Client
from olo.core.images import Image
from olo.errors import OloTimeoutError
from olo.kinematics import JointTarget
from olo.vision import camera_info_to_intrinsics

RGB_TOPIC = "wrist_rgbd/image"
CAMERA_INFO_TOPIC = "wrist_rgbd/camera_info"
ANNOTATED_TOPIC = "wrist_rgbd/annotated"   # add this as a camera panel to see the detections
TAG_FAMILY = "aruco_4x4_50"
MARKER_SIZE_M = 0.036
MAX_FPS = 8


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


def annotate(rgb, frame):
    """Draw each detected tag (outline, centre, id) on a copy of the camera image."""
    canvas = np.ascontiguousarray(rgb.array).copy()
    for tag in frame.tags:
        pts = np.array([(c.x, c.y) for c in tag.corners], np.int32)
        cv2.polylines(canvas, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.circle(canvas, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        cv2.putText(canvas, f"id {tag.id}", (int(pts[:, 0].min()), max(int(pts[:, 1].min()) - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{len(frame.tags)} tag(s)", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
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
        publisher = robot.core.publisher(
            robot.core.topic(ANNOTATED_TOPIC, "sensor_msgs/msg/Image")
        )

        print(f"Detecting cubes; publishing to {ANNOTATED_TOPIC}. Press Ctrl+C to stop.")
        last_seen = None
        try:
            while True:
                rgb = robot.core.get_image(rgb_topic)
                frame = detector.detect(rgb, intrinsics=intrinsics)
                publisher.publish(annotate(rgb, frame))
                seen = sorted(t.id for t in frame.tags)
                if seen != last_seen:
                    print(f"Cubes seen: {len(seen)} tag(s) {seen}")
                    last_seen = seen
                time.sleep(1.0 / MAX_FPS)
        except KeyboardInterrupt:
            print("Stopping...")
    finally:
        # RULE 4: always return to the initial position, even if a step failed.
        print("[3] Back to initial position")
        move_to(arm, INITIAL_JOINTS, "initial")
        set_finger_gap(arm, INITIAL_FINGER_GAP_M)

    print("Done: back at the initial position")

"""Detect the cubes from the detection position, pick cube 1 up from the side and put it back."""

import math
import time

import cv2
import numpy as np

from olo import Client
from olo.core.images import Image
from olo.errors import OloTimeoutError
from olo.kinematics import JointTarget
from olo.spatial import Point, Pose, Quaternion
from olo.vision import camera_info_to_intrinsics

RGB_TOPIC = "wrist_rgbd/image"
DEPTH_TOPIC = "wrist_rgbd/depth_image"
CAMERA_INFO_TOPIC = "wrist_rgbd/camera_info"
DEPTH_TIMEOUT_S = 2.0
ANNOTATED_TOPIC = "wrist_rgbd/annotated"   # add this as a camera panel to see the detections
TAG_FAMILY = "aruco_4x4_50"
MARKER_SIZE_M = 0.036
CUBE_SIZE_M = 0.05
TOP_FACE_SCALE = CUBE_SIZE_M / MARKER_SIZE_M   # marker outline -> whole top face (white margin)
MAX_FPS = 8
DETECT_FRAMES = 10             # frames averaged to locate the cube before picking

# ---------------- Cube pick and put-back (Rule 5 cube exception) ----------------
PICK_GAP_M = min(CUBE_SIZE_M + 0.04, 0.10)  # opening while approaching: cube + 4 cm, never above the 10 cm rule
SIDE_STANDOFF_M = 0.08         # stop this far in front of the cube's near face before going in
SIDE_PENETRATION_M = (0.02, 0.01, 0.0)   # how far the fingertips go past the near face, tried in this order
SIDE_HEIGHT_OFFSET_M = 0.01    # grasp this far above the cube centre (keeps the fingers off the table)
LIFT_HEIGHT_M = 0.10           # how far the cube is lifted
PLACE_CONTACT_GAP_M = 0.002    # release this far above where the cube was picked from
# Tool tilt (degrees) to try for the side grasp, in order: 90 = tool horizontal, 180 = straight down.
# A fully horizontal tool pushes joint4 past its limit at this distance, so it comes in from the base
# side tilted down by 45, 60 or 30 degrees; the fingers still close on two opposite side faces.
SIDE_PITCH_CANDIDATES_DEG = (135.0, 150.0, 120.0)
# Roll about the tool z axis. With 0 / 180 deg the fingers open sideways (horizontally) when the tool is
# horizontal; use +-90 deg instead if the fingers end up above and below the cube.
SIDE_ROLLS_RAD = (0.0, math.pi)
SETTLE_S = 0.5


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


def top_depth_m(depth, pts, rgb_shape):
    """Median depth (metres) over the cube's top-face outline `pts` (RGB pixels), or None."""
    arr = np.asarray(depth.array)
    if arr.ndim == 3:
        arr = arr[..., 0]
    scale = [arr.shape[1] / rgb_shape[1], arr.shape[0] / rgb_shape[0]]
    mask = np.zeros(arr.shape, np.uint8)
    cv2.fillPoly(mask, [(pts * scale).astype(np.int32)], 1)
    values = arr[mask.astype(bool)].astype(np.float64)
    if arr.dtype == np.uint16:
        values /= 1000.0  # millimetres -> metres
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.median(values)) if values.size else None


def annotate(rgb, depth, frame):
    """Draw each cube's top-face outline with its depth on a copy of the camera image."""
    canvas = np.ascontiguousarray(rgb.array).copy()
    tags = sorted(frame.tags, key=lambda t: (np.mean([c.x for c in t.corners]), np.mean([c.y for c in t.corners])))
    cubes = []
    for number, tag in enumerate(tags, start=1):
        marker = np.array([(c.x, c.y) for c in tag.corners], np.float32)
        centre = marker.mean(axis=0)
        pts = ((marker - centre) * TOP_FACE_SCALE + centre).astype(np.int32)  # whole top face
        top_m = top_depth_m(depth, pts, canvas.shape) if depth is not None else None
        # The depth camera sees the top face; the cube centre is half a cube further down.
        centre_m = None if top_m is None else top_m + CUBE_SIZE_M / 2
        caption = f"cube {number}"
        if top_m is not None:
            caption += f" top {top_m:.3f}m centre {centre_m:.3f}m"
        cv2.polylines(canvas, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.circle(canvas, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        (text_w, _), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        text_x = int(min(max(pts[:, 0].min(), 4), canvas.shape[1] - text_w - 4))  # keep the label in view
        cv2.putText(canvas, caption, (text_x, max(int(pts[:, 1].min()) - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cubes.append((number, top_m, centre_m))
    cv2.putText(canvas, f"{len(cubes)} cube(s)", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    return Image(array=canvas, encoding=rgb.encoding, header=rgb.header), cubes


def quat_dot(a, b):
    return sum(x * y for x, y in zip(a.array(), b.array()))


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


def rotate(q, v):
    """Rotate the 3-vector `v` by the quaternion `q` (x, y, z, w order)."""
    x, y, z, w = q.array()
    vx, vy, vz = v
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def side_orientation(heading, pitch, roll):
    """Tool orientation for a side grasp.

    The tool z axis (the approach direction) points along `heading` (an angle in the horizontal
    plane); pitch = 90 deg is horizontal. The roll about the tool z axis puts the
    finger opening horizontal, so the fingers close on two opposite side faces.
    """
    return (
        Quaternion.from_rpy(0.0, 0.0, heading)
        * Quaternion.from_rpy(0.0, pitch, 0.0)
        * Quaternion.from_rpy(0.0, 0.0, roll)
    )


def side_candidates(x, y, yaw, reference):
    """(tilt in degrees, orientation) to try: approach the cube's side face that faces the base first."""
    azimuth = math.atan2(y, x)
    headings = [yaw + k * math.pi / 2 for k in range(4)]  # directions perpendicular to a cube face
    headings.sort(key=lambda h: abs(math.atan2(math.sin(h - azimuth), math.cos(h - azimuth))))
    result = []
    for pitch_deg in SIDE_PITCH_CANDIDATES_DEG:
        for heading in headings[:2]:
            options = [side_orientation(heading, math.radians(pitch_deg), roll) for roll in SIDE_ROLLS_RAD]
            options.sort(key=lambda q: -abs(quat_dot(q, reference)))
            result.extend((pitch_deg, q) for q in options)
    return result


def joints_outside_limits(arm):
    """Names of arm joints that are currently outside the robot model's limits."""
    current = arm.joints().position_by_name
    bad = []
    for j in arm.model.joints:
        if j.name in current and j.lower_limit < j.upper_limit and not (j.lower_limit <= current[j.name] <= j.upper_limit):
            bad.append(f"{j.name}={current[j.name]:.3f} (limits {j.lower_limit:.3f}..{j.upper_limit:.3f})")
    return bad


def move_pose_slow(arm, pose):
    """move_pose at SPEED, with collision checking first, then without."""
    last_error = None
    for check in (True, False):
        try:
            arm.move_pose(pose, speed_scaling=SPEED, avoid_obstacles=check)
            return
        except Exception as exc:
            last_error = exc
    raise last_error


def move_straight(arm, point, orientation):
    """Straight-line move to `point` (x, y, z) at SPEED; returns the pose."""
    pose = Pose(position=Point(x=point[0], y=point[1], z=point[2]), orientation=orientation)
    arm.move_linear(pose, speed_scaling=SPEED)
    return pose


def pick_cube(arm, top, yaw, holder):
    """Go in from the side, grasp two opposite faces of the cube and lift it.

    `top` is the top-face centre; the cube centre is half a cube below it. Fills `holder` with
    [(grasp_point, standoff_point, orientation)] once the cube may be in the gripper.
    """
    half = CUBE_SIZE_M / 2
    centre = (top[0], top[1], top[2] - half + SIDE_HEIGHT_OFFSET_M)
    set_finger_gap(arm, PICK_GAP_M)
    time.sleep(SETTLE_S)

    # Pre-grasp: in front of the cube's near face, tool pointing at the cube.
    last_error = None
    chosen = None
    for pitch_deg, orientation in side_candidates(centre[0], centre[1], yaw, arm.pose().orientation):
        d = rotate(orientation, (0.0, 0.0, 1.0))  # approach direction (tool z)
        standoff = tuple(centre[i] - d[i] * (half + SIDE_STANDOFF_M) for i in range(3))
        try:
            move_pose_slow(arm, Pose(position=Point(x=standoff[0], y=standoff[1], z=standoff[2]), orientation=orientation))
            print(f"In front of the cube with tool tilt {pitch_deg:.0f} deg")
            chosen = (orientation, d, standoff)
            break
        except Exception as exc:
            last_error = exc
            bad = joints_outside_limits(arm)
            if bad:  # the planner refuses to plan from here, so trying more poses is pointless
                raise RuntimeError(
                    f"A failed move left joints outside their limits: {bad}. Reset the simulation / "
                    f"move the joints back, then run again. Last error: {exc}"
                ) from exc
    if chosen is None:
        raise RuntimeError(
            f"Cannot reach the side of the cube at radius {math.hypot(centre[0], centre[1]):.3f} m from the base: {last_error}"
        )
    orientation, d, standoff = chosen
    time.sleep(SETTLE_S)

    # Go straight in until the fingertips are past the near face.
    grasp = None
    for penetration in SIDE_PENETRATION_M:
        point = tuple(centre[i] - d[i] * (half - penetration) for i in range(3))
        try:
            move_straight(arm, point, orientation)
            grasp = point
            break
        except Exception as exc:
            print(f"Cannot go in {penetration * 1000:.0f} mm past the cube face: {exc}")
    if grasp is None:
        raise RuntimeError("Cannot move in to the cube")
    time.sleep(SETTLE_S)

    # From here on the cube may be in the gripper, so record it for the put-back in `finally`.
    holder[:] = [(grasp, standoff, orientation)]
    try:
        arm.ee.close(timeout=GRIPPER_TIMEOUT_S)  # plain close(): the fingers stop on the cube
    except OloTimeoutError:
        gap = measured_gap(arm)
        print(f"Gripper close timed out; measured finger gap: {(gap or 0) * 100:.1f} cm")
        if gap is not None and gap > MAX_FINGER_GAP_M + GAP_TOLERANCE_M:  # Rule 3 check after a timeout
            arm.ee.goal(MAX_FINGER_GAP_M / 2.0, timeout=GRIPPER_TIMEOUT_S)
            raise RuntimeError(f"Finger gap {gap * 100:.1f} cm exceeded the {MAX_FINGER_GAP_M * 100:.0f} cm limit; corrected")
    time.sleep(SETTLE_S)
    move_straight(arm, (grasp[0], grasp[1], grasp[2] + LIFT_HEIGHT_M), orientation)
    time.sleep(SETTLE_S)


def put_cube_back(arm, grasp, standoff, orientation):
    """Lower the held cube to where it was, let go, and back straight out of the way."""
    print("Putting the cube back...")
    move_straight(arm, (grasp[0], grasp[1], grasp[2] + PLACE_CONTACT_GAP_M), orientation)
    time.sleep(SETTLE_S)
    set_finger_gap(arm, PICK_GAP_M)  # open, never more than 10 cm
    time.sleep(SETTLE_S)
    move_straight(arm, standoff, orientation)  # back out along the approach line
    time.sleep(SETTLE_S)


# ---------------- Main sequence ----------------
with Client() as client:
    robot = client.robot()
    arm = robot.kinematics
    held = []  # [(grasp_point, standoff_point, orientation)] while a cube may be in the gripper

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
        depth_topic = robot.core.topic(DEPTH_TOPIC, "sensor_msgs/msg/Image")
        info_topic = robot.core.topic(CAMERA_INFO_TOPIC, "sensor_msgs/msg/CameraInfo")
        intrinsics = camera_info_to_intrinsics(robot.core.get_latest(info_topic))
        detector = robot.vision.tags(TAG_FAMILY, size_m=MARKER_SIZE_M)
        publisher = robot.core.publisher(
            robot.core.topic(ANNOTATED_TOPIC, "sensor_msgs/msg/Image")
        )

        print(f"Detecting cubes ({DETECT_FRAMES} frames); publishing to {ANNOTATED_TOPIC}")
        positions = []
        yaws = []
        for _ in range(DETECT_FRAMES):
            rgb = robot.core.get_image(rgb_topic)
            try:
                depth = robot.core.get_image(depth_topic, timeout=DEPTH_TIMEOUT_S)
            except OloTimeoutError:
                depth = None
            frame = detector.detect(rgb, intrinsics=intrinsics)
            image, cubes = annotate(rgb, depth, frame)
            publisher.publish(image)
            # Cube 1 = the left-most cube in the image, the same numbering as the overlay.
            posed = [t for t in frame.tags if t.transform is not None]
            if posed:
                # With one cube it is the only candidate; with several, the left-most one is picked.
                first = min(posed, key=lambda t: (np.mean([c.x for c in t.corners]), np.mean([c.y for c in t.corners])))
                top, yaw = locate_cube(robot, arm, first)
                positions.append(top)
                yaws.append(yaw)
            time.sleep(1.0 / MAX_FPS)
        if not positions:
            raise RuntimeError("No cube with a pose was detected; nothing to pick")

        top = tuple(float(np.median([p[i] for p in positions])) for i in range(3))
        yaw = average_yaw(yaws)
        print(f"Cube 1 top face at ({top[0]:.3f}, {top[1]:.3f}, {top[2]:.3f}) m, yaw {math.degrees(yaw):+.0f} deg")

        # ---- pick the cube up from the side, then put it back ----
        print("[3] Picking cube 1 from the side")
        pick_cube(arm, top, yaw, held)
        put_cube_back(arm, *held[0])
        held.clear()
    finally:
        # RULE 4: always return to the initial position, even if a step failed.
        cube_in_gripper = False
        if held:
            try:  # do not squeeze the cube shut: put it down and let go first
                put_cube_back(arm, *held[0])
            except Exception as exc:
                cube_in_gripper = True
                print(f"WARNING: could not put the cube back ({exc}); it may still be in the gripper")
        print("[4] Back to initial position")
        try:
            move_to(arm, INITIAL_JOINTS, "initial")
        except Exception:
            print(f"Could not return to the initial position. Joints outside their limits: {joints_outside_limits(arm)}. "
                  "Reset the simulation (or move the joints back inside their limits) and run again.")
            raise
        if cube_in_gripper:
            print("WARNING: not closing the fingers because the cube may still be held")
        else:
            set_finger_gap(arm, INITIAL_FINGER_GAP_M)

    print("Done: back at the initial position")

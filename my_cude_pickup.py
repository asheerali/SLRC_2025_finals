"""Detect the cubes from the detection position, pick cube 1 up from the side, tip it over so the face
that was towards the base (its back face) ends up on top, and set it down again.

Before the side grasp the cube is first lifted by its top face and set down at SIDE_PICK_POSITION_M (a spot the
arm can reach from the side), then found again there and picked from the side."""

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
SIDE_PENETRATION_M = (0.03, 0.02, 0.01)  # how far the fingertips go past the near face (tried in this order): 3 cm
                                         # puts them just past the middle of the 5 cm cube, so the pads grip its middle
SIDE_HEIGHT_OFFSET_M = 0.0     # grasp height above the cube centre (0 = the middle of the back face)
# Shift of the grasp point along the tool x and y axes (metres). Tune it if the fingers land beside the middle
# of the face: the run prints "Cube centre in the tool frame", and the x / y values there are the error.
GRASP_SHIFT_TOOL_M = (0.0, 0.0)
LIFT_HEIGHT_M = 0.10           # how far the cube is lifted
PLACE_CONTACT_GAP_M = 0.005    # release this far above the table, so the cube is not pressed into it and jammed
# Gentle release so the cube does not stay stuck to the fingers or get dragged along:
RELEASE_GAP_M = 0.10           # open fully (the 10 cm limit) so there is clearance on both sides of the cube
RELEASE_SETTLE_S = 1.0         # wait for the fingers to be fully open before moving away
GENTLE_SPEED = 0.1             # speed for the last part of lowering and the first part of backing out
BACK_OUT_SLOW_M = 0.03         # back out this far slowly, then the rest at the normal speed
# Tool tilt (degrees) to try for the side grasp, in this order: 90 = tool horizontal, pointing straight at the
# middle of the back face (the fingers then meet the face square on, not at its top edge); 180 = straight down.
# The tilt is raised in 5 degree steps (the wrist can usually bend a little further than the next big step),
# then 135 / 150 as last resorts. Larger tilts are fall-backs that come in from above-behind and grip nearer the top edge. If the wrist
# (joint4 / joint5) cannot bend far enough for the horizontal approach at this distance, the arm is moved
# back inside its joint limits and the next tilt is tried, so the grasp falls back by itself.
SIDE_PITCH_CANDIDATES_DEG = (90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 135.0, 150.0)
SIDE_PLANNING_TIMEOUT_S = 3.0   # per attempt while searching for the flattest reachable tilt
SIDE_TILT_RESOLUTION_DEG = 1.0  # once a tilt works, the tilt is narrowed down to this resolution (flatter = better)
# Roll about the tool z axis. With 0 / 180 deg the fingers open sideways (horizontally) when the tool is
# horizontal; use +-90 deg instead if the fingers end up above and below the cube.
SIDE_ROLLS_RAD = (0.0, math.pi)
# The side grasp is done at this spot (x, y in the planning frame, metres; about 0.443 m from the base, where the
# arm can reach the back face with a flat tool). If the cube is not already there, it is first picked up by its
# top face and set down here, then found again and picked from the side.
SIDE_PICK_POSITION_M = (0.443, -0.003)
RELOCATE_TOLERANCE_M = 0.02    # a cube already within this distance of SIDE_PICK_POSITION_M is not moved
APPROACH_HEIGHT_M = 0.10       # top grasp: hover this far above the cube top before going down
GRASP_BELOW_TOP_M = (0.01, 0.005, 0.0)   # top grasp: tool height under the cube top, tried in this order
TOP_PITCH_CANDIDATES_DEG = (180.0, 165.0, 150.0, 135.0)   # top grasp tool tilt; 180 = straight down
# After the side grasp the cube is tipped over by turning the wrist about the finger axis, so the face that
# pointed towards the base (the back face) ends up on top. Set FLIP_BACK_FACE_UP = False to just put it back.
FLIP_BACK_FACE_UP = True
FLIP_STEP_DEG = 30.0           # the 90 degree turn is made in steps of this size
# The tipped-over cube is always set down here (planning frame, metres): the box centre, resting on the table,
# with its edges lined up with the x / y axes (rotation about the vertical = 0 deg).
FLIP_PLACE_CENTRE_M = (0.313, 0.013, 0.012)
# At the end location the cube is let go this far above the table (the bottom of the cube above the table), so it
# drops free of the fingers instead of being pushed down between them. The drop is small, so it lands flat.
FLIP_DROP_HEIGHT_M = 0.03
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
SPEED = 0.9                   # 0-1, lower is slower
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


def rotate_xyzw(array, v):
    """Rotate the 3-vector `v` by the quaternion `array` (x, y, z, w order)."""
    x, y, z, w = array
    vx, vy, vz = v
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def rotate(q, v):
    """Rotate the 3-vector `v` by the quaternion `q`."""
    return rotate_xyzw(q.array(), v)


def rotate_inverse(q, v):
    """Express the world-frame vector `v` in the frame of the quaternion `q` (the inverse rotation)."""
    x, y, z, w = q.array()
    return rotate_xyzw((-x, -y, -z, w), v)


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
    """(tilt in degrees, orientation, heading, roll) to try: approach the cube's face towards the base first."""
    azimuth = math.atan2(y, x)
    headings = [yaw + k * math.pi / 2 for k in range(4)]  # directions perpendicular to a cube face
    headings.sort(key=lambda h: abs(math.atan2(math.sin(h - azimuth), math.cos(h - azimuth))))
    result = []
    for pitch_deg in SIDE_PITCH_CANDIDATES_DEG:
        heading = headings[0]  # the face towards the base
        options = [(side_orientation(heading, math.radians(pitch_deg), roll), roll) for roll in SIDE_ROLLS_RAD]
        options.sort(key=lambda item: -abs(quat_dot(item[0], reference)))
        result.extend((pitch_deg, q, heading, roll) for q, roll in options)
    return result


def side_frame(centre, orientation, half):
    """Approach direction, aim point and stand-off point for a side grasp with this tool orientation."""
    d = rotate(orientation, (0.0, 0.0, 1.0))  # approach direction (tool z)
    shift = rotate(orientation, (GRASP_SHIFT_TOOL_M[0], GRASP_SHIFT_TOOL_M[1], 0.0))
    aim = tuple(centre[i] + shift[i] for i in range(3))  # the point the tool is aimed at
    standoff = tuple(aim[i] - d[i] * (half + SIDE_STANDOFF_M) for i in range(3))
    return d, aim, standoff


def joints_outside_limits(arm):
    """Names of arm joints that are currently outside the robot model's limits."""
    current = arm.joints().position_by_name
    bad = []
    for j in arm.model.joints:
        if j.name in current and j.lower_limit < j.upper_limit and not (j.lower_limit <= current[j.name] <= j.upper_limit):
            bad.append(f"{j.name}={current[j.name]:.3f} (limits {j.lower_limit:.3f}..{j.upper_limit:.3f})")
    return bad


def recover_joint_limits(arm):
    """After a failed move, bring any joint that is past its limit back inside it. Returns True if all are inside.

    The planner refuses to plan from a pose outside the joint limits, so the next fall-back would fail
    too. The joints are first sent just inside their limits (a small joint move), then, if that is
    refused, back to the detection position.
    """
    if not joints_outside_limits(arm):
        return True
    print(f"Joints past their limits: {joints_outside_limits(arm)}; moving them back inside...")
    limits = {j.name: j for j in arm.model.joints}
    current = arm.joints().position_by_name
    inside = {}
    for name in arm.model.group().joints:
        j, value = limits.get(name), current.get(name)
        if value is None:
            continue
        if j is not None and j.lower_limit < j.upper_limit:
            margin = 0.02 * (j.upper_limit - j.lower_limit)
            value = min(max(value, j.lower_limit + margin), j.upper_limit - margin)
        inside[name] = value
    for target in (inside, DETECTION_JOINTS):
        try:
            arm.move_joints(arm_target(arm, target), speed_scaling=SPEED)
            time.sleep(SETTLE_S)
            if not joints_outside_limits(arm):
                print("Joints are back inside their limits")
                return True
        except Exception as exc:
            print(f"Could not move the joints back ({exc})")
    return not joints_outside_limits(arm)


def move_pose_slow(arm, pose, planning_timeout=None):
    """move_pose at SPEED, with collision checking first, then without."""
    last_error = None
    extra = {} if planning_timeout is None else {"planning_timeout": planning_timeout}
    for check in (True, False):
        try:
            arm.move_pose(pose, speed_scaling=SPEED, avoid_obstacles=check, **extra)
            return
        except Exception as exc:
            last_error = exc
    raise last_error


def move_straight(arm, point, orientation, speed=None):
    """Straight-line move to `point` (x, y, z) at SPEED (or `speed`); returns the pose."""
    pose = Pose(position=Point(x=point[0], y=point[1], z=point[2]), orientation=orientation)
    arm.move_linear(pose, speed_scaling=SPEED if speed is None else speed)
    return pose


def close_on_cube(arm):
    """Plain close(): the fingers stop on the cube. On a timeout the real gap is checked against Rule 3."""
    try:
        arm.ee.close(timeout=GRIPPER_TIMEOUT_S)
    except OloTimeoutError:
        gap = measured_gap(arm)
        print(f"Gripper close timed out; measured finger gap: {(gap or 0) * 100:.1f} cm")
        if gap is not None and gap > MAX_FINGER_GAP_M + GAP_TOLERANCE_M:  # Rule 3 check after a timeout
            arm.ee.goal(MAX_FINGER_GAP_M / 2.0, timeout=GRIPPER_TIMEOUT_S)
            raise RuntimeError(f"Finger gap {gap * 100:.1f} cm exceeded the {MAX_FINGER_GAP_M * 100:.0f} cm limit; corrected")
    time.sleep(SETTLE_S)


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
    failed_pitch = None   # the flattest tilt tried so far that was not reachable
    for pitch_deg, orientation, heading, roll in side_candidates(centre[0], centre[1], yaw, arm.pose().orientation):
        d, aim, standoff = side_frame(centre, orientation, half)
        try:
            move_pose_slow(arm, Pose(position=Point(x=standoff[0], y=standoff[1], z=standoff[2]), orientation=orientation),
                           planning_timeout=SIDE_PLANNING_TIMEOUT_S)
            print(f"In front of the cube with tool tilt {pitch_deg:.0f} deg")
            chosen = (orientation, d, standoff, aim)
            break
        except Exception as exc:
            last_error = exc
            failed_pitch = pitch_deg
            print(f"Tool tilt {pitch_deg:.0f} deg not reachable ({exc}); trying the next tilt")
            if not recover_joint_limits(arm):  # the planner refuses to plan from here, so stop
                raise RuntimeError(
                    f"A failed move left joints outside their limits: {joints_outside_limits(arm)}. Reset the "
                    f"simulation / move the joints back, then run again. Last error: {exc}"
                ) from exc
    if chosen is None:
        raise RuntimeError(
            f"Cannot reach the side of the cube at radius {math.hypot(centre[0], centre[1]):.3f} m from the base: {last_error}"
        )

    # Narrow the tilt down: the flattest reachable tilt lies between the last failure and this success.
    if failed_pitch is not None:
        good_pitch, bad_pitch, arm_at_good = pitch_deg, failed_pitch, True
        while good_pitch - bad_pitch > SIDE_TILT_RESOLUTION_DEG:
            mid = (good_pitch + bad_pitch) / 2.0
            trial = side_orientation(heading, math.radians(mid), roll)
            t_d, t_aim, t_standoff = side_frame(centre, trial, half)
            try:
                move_pose_slow(arm, Pose(position=Point(x=t_standoff[0], y=t_standoff[1], z=t_standoff[2]), orientation=trial),
                               planning_timeout=SIDE_PLANNING_TIMEOUT_S)
                good_pitch, chosen, arm_at_good = mid, (trial, t_d, t_standoff, t_aim), True
            except Exception:
                bad_pitch, arm_at_good = mid, False
                if not recover_joint_limits(arm):
                    raise RuntimeError(
                        f"A failed move left joints outside their limits: {joints_outside_limits(arm)}. "
                        "Reset the simulation / move the joints back, then run again."
                    )
        if not arm_at_good:  # the last trial failed, so go back to the flattest tilt that worked
            o, _, so, _ = chosen
            move_pose_slow(arm, Pose(position=Point(x=so[0], y=so[1], z=so[2]), orientation=o),
                           planning_timeout=SIDE_PLANNING_TIMEOUT_S)
        print(f"Flattest reachable tool tilt: {good_pitch:.1f} deg (90 = horizontal)")
    orientation, d, standoff, aim = chosen
    time.sleep(SETTLE_S)

    # Go straight in until the fingertips are past the near face.
    grasp = None
    for penetration in SIDE_PENETRATION_M:
        point = tuple(aim[i] - d[i] * (half - penetration) for i in range(3))
        try:
            move_straight(arm, point, orientation)
            grasp = point
            break
        except Exception as exc:
            print(f"Cannot go in {penetration * 1000:.0f} mm past the cube face: {exc}")
    if grasp is None:
        raise RuntimeError("Cannot move in to the cube")
    time.sleep(SETTLE_S)
    # Where is the cube centre relative to the tool? x / y are sideways errors (tune GRASP_SHIFT_TOOL_M with them);
    # z is how far the cube centre is ahead (+) or behind (-) the tool point along the approach direction.
    here = arm.pose().position
    err = rotate_inverse(orientation, (centre[0] - here.x, centre[1] - here.y, centre[2] - here.z))
    print(f"Cube centre in the tool frame: x={err[0] * 1000:+.0f} mm  y={err[1] * 1000:+.0f} mm  z={err[2] * 1000:+.0f} mm")

    # From here on the cube may be in the gripper, so record it for the put-back in `finally`.
    holder[:] = [(grasp, standoff, orientation)]
    close_on_cube(arm)
    tcp_lifted = (grasp[0], grasp[1], grasp[2] + LIFT_HEIGHT_M)
    move_straight(arm, tcp_lifted, orientation)
    time.sleep(SETTLE_S)
    # What the flip needs: the cube centre (resting on the table, then lifted), the tool point, the tool direction.
    cube_centre_lifted = (top[0], top[1], top[2] - half + LIFT_HEIGHT_M)
    return cube_centre_lifted, tcp_lifted, orientation, d


def put_cube_back(arm, grasp, standoff, orientation, drop_height=None):
    """Lower the held cube gently to where it should rest, let go, and back out slowly.

    With `drop_height` the cube is let go that far above the table (its bottom above the table) and drops the rest.

    The last part of the lowering is slow, the fingers open fully and settle, and the first centimetres of
    backing out are slow too, so the cube is not pressed into the table, pinched or dragged along.
    """
    print("Putting the cube down...")
    release_gap = PLACE_CONTACT_GAP_M if drop_height is None else drop_height
    final = (grasp[0], grasp[1], grasp[2] + release_gap)
    near = (final[0], final[1], final[2] + 0.03)
    move_straight(arm, near, orientation)               # most of the way down at the normal speed
    move_straight(arm, final, orientation, GENTLE_SPEED)  # the last 3 cm slowly
    time.sleep(SETTLE_S)
    set_finger_gap(arm, RELEASE_GAP_M)  # open fully (never more than 10 cm)
    time.sleep(RELEASE_SETTLE_S)
    # Back out along the approach line: slowly at first, while the fingers are still next to the cube.
    d = rotate(orientation, (0.0, 0.0, 1.0))
    first = tuple(final[i] - d[i] * BACK_OUT_SLOW_M for i in range(3))
    move_straight(arm, first, orientation, GENTLE_SPEED)
    move_straight(arm, standoff, orientation)
    time.sleep(SETTLE_S)


def flip_back_face_up(arm, pick_info, yaw, holder):
    """Carry the held cube to FLIP_PLACE_CENTRE_M, tip it over so the face that was on the gripper side
    (towards the base) is on top, and set it down there.

    First the cube is moved over the place position and turned about the vertical so its edges line up
    with the x / y axes. Then the wrist turns it by 90 degrees about the horizontal axis through the two
    fingertips, pivoting about the cube centre so the cube stays over the place position. Then it is
    lowered and released.
    """
    centre, tcp, orientation, d = pick_info
    place = (FLIP_PLACE_CENTRE_M[0], FLIP_PLACE_CENTRE_M[1], FLIP_PLACE_CENTRE_M[2] + LIFT_HEIGHT_M)  # lifted
    to_centre = tuple(tcp[i] - centre[i] for i in range(3))

    # Which way to turn about the vertical so the cube edges line up with x / y (edges repeat every 90 deg).
    delta = -(((yaw + math.pi / 4) % (math.pi / 2)) - math.pi / 4)
    align = Quaternion.from_rpy(0.0, 0.0, delta)
    orientation_now = align * orientation
    offset = rotate(align, to_centre)
    tcp_now = tuple(place[i] + offset[i] for i in range(3))

    def remember(tcp_point, tool_orientation_now):
        """If a later step fails, the put-back in `finally` lowers the cube from where it is now."""
        d_now = rotate(tool_orientation_now, (0.0, 0.0, 1.0))
        grasp = (tcp_point[0], tcp_point[1], tcp_point[2] - LIFT_HEIGHT_M)
        standoff = tuple(grasp[i] - d_now[i] * (CUBE_SIZE_M / 2 + SIDE_STANDOFF_M) for i in range(3))
        holder[:] = [(grasp, standoff, tool_orientation_now)]

    print(f"Carrying the cube to ({FLIP_PLACE_CENTRE_M[0]:.3f}, {FLIP_PLACE_CENTRE_M[1]:.3f}), "
          f"turning it {math.degrees(delta):+.0f} deg about the vertical")
    move_pose_slow(arm, Pose(position=Point(x=tcp_now[0], y=tcp_now[1], z=tcp_now[2]), orientation=orientation_now))
    time.sleep(SETTLE_S)
    remember(tcp_now, orientation_now)

    # The finger axis is perpendicular to the tool direction; turn about it so the gripper-side face goes up.
    d_aligned = rotate(orientation_now, (0.0, 0.0, 1.0))
    heading = math.atan2(d_aligned[1], d_aligned[0])
    steps = max(1, int(round(90.0 / FLIP_STEP_DEG)))
    print(f"Tipping the cube over in {steps} steps")
    start_offset, start_orientation = offset, orientation_now
    for k in range(1, steps + 1):
        turn = (
            Quaternion.from_rpy(0.0, 0.0, heading)
            * Quaternion.from_rpy(0.0, math.radians(90.0) * k / steps, 0.0)
            * Quaternion.from_rpy(0.0, 0.0, -heading)
        )
        step_offset = rotate(turn, start_offset)
        tcp_now = tuple(place[i] + step_offset[i] for i in range(3))
        orientation_now = turn * start_orientation
        move_pose_slow(arm, Pose(position=Point(x=tcp_now[0], y=tcp_now[1], z=tcp_now[2]), orientation=orientation_now))
        time.sleep(SETTLE_S)
        remember(tcp_now, orientation_now)

    put_cube_back(arm, *holder[0], drop_height=FLIP_DROP_HEIGHT_M)  # let go a little above the table, back out


def tool_orientation(azimuth, cube_yaw, pitch, turn):
    """Tool orientation for a top grasp: pitch = 180 deg is straight down, smaller tilts the tool outward."""
    k = azimuth - cube_yaw - turn * math.pi / 2
    return (
        Quaternion.from_rpy(0.0, 0.0, azimuth)
        * Quaternion.from_rpy(0.0, pitch, 0.0)
        * Quaternion.from_rpy(0.0, 0.0, k)
    )


def top_candidates(x, y, yaw, reference):
    """(tilt in degrees, orientation) for a top grasp, most vertical first; the 2 turns closest to `reference`."""
    azimuth = math.atan2(y, x)
    result = []
    for pitch_deg in TOP_PITCH_CANDIDATES_DEG:
        options = [tool_orientation(azimuth, yaw, math.radians(pitch_deg), turn) for turn in range(4)]
        options.sort(key=lambda q: -abs(quat_dot(q, reference)))
        result.extend((pitch_deg, q) for q in options[:2])
    return result


def move_cube_to(arm, top, yaw, holder, target_xy):
    """Lift the cube by its top face and set it down at `target_xy` (planning frame, metres)."""
    x, y, top_z = top
    set_finger_gap(arm, PICK_GAP_M)
    time.sleep(SETTLE_S)

    # The tool keeps the same orientation while it carries the cube, so choose a tilt that is reachable over the
    # cube AND over the target (a straight-down tool cannot reach as far out as a slightly tilted one). The target
    # is tried first, with nothing in the gripper, so a bad tilt costs nothing.
    nx, ny = target_xy
    carry_z = top_z - GRASP_BELOW_TOP_M[0] + LIFT_HEIGHT_M
    last_error = None
    hover = None
    for pitch_deg, orientation in top_candidates(x, y, yaw, arm.pose().orientation):
        try:
            hover = Pose(position=Point(x=x, y=y, z=top_z + APPROACH_HEIGHT_M), orientation=orientation)
            move_pose_slow(arm, hover)
            move_pose_slow(arm, Pose(position=Point(x=nx, y=ny, z=carry_z), orientation=orientation))  # probe the target
            move_pose_slow(arm, hover)  # and back over the cube
            print(f"Tool tilt {pitch_deg:.0f} deg reaches both the cube and the target")
            break
        except Exception as exc:
            last_error = exc
            hover = None
            print(f"Tool tilt {pitch_deg:.0f} deg does not reach both ({exc}); trying the next tilt")
            if not recover_joint_limits(arm):
                raise RuntimeError(
                    f"A failed move left joints outside their limits: {joints_outside_limits(arm)}. "
                    "Reset the simulation, then run again."
                ) from exc
    if hover is None:
        raise RuntimeError(f"No tool tilt reaches both the cube and ({nx:.3f}, {ny:.3f}): {last_error}")
    orientation = hover.orientation
    time.sleep(SETTLE_S)

    grasp_z = None
    for below in GRASP_BELOW_TOP_M:
        try:
            move_straight(arm, (x, y, top_z - below), orientation)
            grasp_z = top_z - below
            break
        except Exception as exc:
            print(f"Cannot lower to {below * 1000:.0f} mm under the cube top: {exc}")
    if grasp_z is None:
        raise RuntimeError("Cannot lower onto the cube")
    time.sleep(SETTLE_S)

    holder[:] = [((x, y, grasp_z), (x, y, grasp_z + APPROACH_HEIGHT_M), orientation)]
    close_on_cube(arm)
    move_straight(arm, (x, y, grasp_z + LIFT_HEIGHT_M), orientation)
    time.sleep(SETTLE_S)

    # Carry it (tool orientation unchanged, so the cube keeps its turn) to the target and set it down.
    print(f"Moving the cube to ({nx:.3f}, {ny:.3f}), {math.hypot(nx, ny):.3f} m from the base")
    move_pose_slow(arm, Pose(position=Point(x=nx, y=ny, z=grasp_z + LIFT_HEIGHT_M), orientation=orientation))
    time.sleep(SETTLE_S)
    holder[:] = [((nx, ny, grasp_z), (nx, ny, grasp_z + APPROACH_HEIGHT_M), orientation)]
    put_cube_back(arm, *holder[0])  # lower it, let go, back off upwards
    holder.clear()


def detect_cube(robot, arm, rgb_topic, depth_topic, detector, intrinsics, publisher):
    """Look at the table for DETECT_FRAMES frames; return the top-face centre and yaw of cube 1."""
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
    print(f"Cube 1 top face at ({top[0]:.3f}, {top[1]:.3f}, {top[2]:.3f}) m, "
          f"{math.hypot(top[0], top[1]):.3f} m from the base, yaw {math.degrees(yaw):+.0f} deg")
    return top, yaw


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

        top, yaw = detect_cube(robot, arm, rgb_topic, depth_topic, detector, intrinsics, publisher)

        if math.dist(top[:2], SIDE_PICK_POSITION_M) > RELOCATE_TOLERANCE_M:
            # Not at the spot the side grasp works from: lift it by the top face and set it down there.
            print(f"[3] Moving the cube to the side-pick position {SIDE_PICK_POSITION_M}")
            move_cube_to(arm, top, yaw, held, SIDE_PICK_POSITION_M)
            # RULE 2: look for it again from the detection position.
            print("[3b] Back to the detection position to find the cube again")
            move_to(arm, DETECTION_JOINTS, "detection")
            set_finger_gap(arm, DETECTION_FINGER_GAP_M)
            top, yaw = detect_cube(robot, arm, rgb_topic, depth_topic, detector, intrinsics, publisher)

        # ---- pick the cube up from the side, then put it back ----
        print("[4] Picking cube 1 from the side")
        pick_info = pick_cube(arm, top, yaw, held)
        if FLIP_BACK_FACE_UP:
            print("[4b] Tipping the cube over so its back face is on top")
            flip_back_face_up(arm, pick_info, yaw, held)
        else:
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
        print("[5] Back to initial position")
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

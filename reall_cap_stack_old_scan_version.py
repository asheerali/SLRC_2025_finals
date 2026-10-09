import math
import sys
import time
 
import cv2
import numpy as np
from olo import Client
from olo.errors import OloTimeoutError
from olo.kinematics import JointTarget
from olo.spatial import Point, Pose, Quaternion
from olo.vision import camera_info_to_intrinsics
 
# @olo-playground: autofill-sim-topics
 
# =====================================================================================
#  THE ONE SWITCH: what are we stacking?     "cube"   or   "bottle_cap"
# =====================================================================================
OBJECT_TYPE = "bottle_cap"
 
# Everything that differs between the two object types lives in this table. The pick / park /
# stack logic below is the same for both.
PROFILES = {
    "cube": {
        "label": "cube",
        "prefix": "box",
        "detection": "tags",             # found by ArUco markers on its faces
        "height_m": 0.05,                # cube edge (the cube is 5 cm)
        "width_m": 0.05,                 # width the fingers must fit around
        "tag_family": "aruco_4x4_50",
        "marker_size_m": 0.036,
        "same_distance_m": 0.10,         # tags this close to a known cube are faces of that same cube
        "target_min_distance_m": 0.10,   # stacking target must be this far from the picked object's old spot
        "held_exclusion_m": 0.10,        # detections this close to the gripper while carrying are the held object
        "grasp_clearance_m": 0.01,       # grasp this far below the object's top
    },
    "bottle_cap": {
        "label": "bottle cap",
        "prefix": "cap",
        "detection": "shape",            # no markers: found by SHAPE (grey, round) + the depth image
        "height_m": 0.02,                # simulator caps: 2 cm high
        "width_m": 0.08,                 # simulator caps: 8 cm across
        "same_distance_m": 0.06,
        "target_min_distance_m": 0.15,   # caps are wide: keep more room between them
        "held_exclusion_m": 0.10,
        "grasp_clearance_m": 0.008,      # fingers grip the middle of the cap wall
    },
}
if OBJECT_TYPE not in PROFILES:
    raise ValueError(f"OBJECT_TYPE must be one of {sorted(PROFILES)}, not {OBJECT_TYPE!r}")
P = PROFILES[OBJECT_TYPE]
 
OBJ_LABEL = P["label"]
OBJ_PREFIX = P["prefix"]
OBJ_HEIGHT_M = P["height_m"]
OBJ_HALF_M = OBJ_HEIGHT_M / 2          # half the object height = centre height above the table
OBJ_WIDTH_M = P["width_m"]
SAME_OBJECT_DISTANCE_M = P["same_distance_m"]
TARGET_MIN_DISTANCE_M = P["target_min_distance_m"]
HELD_EXCLUSION_M = P["held_exclusion_m"]
GRASP_CLEARANCE_M = P["grasp_clearance_m"]
 
# ---------------- Camera ----------------
RGB_TOPIC = "wrist_rgbd/image"
DEPTH_TOPIC = "wrist_rgbd/depth_image"
CAMERA_INFO_TOPIC = "wrist_rgbd/camera_info"
 
# ---------------- YOLO detection (bottle_cap only) ----------------
# Path to your trained .pt file (needs `pip install ultralytics`). None = fall back to the grey-blob
# method below. YOLO finds the caps in the image; the depth image still gives their position and size.
YOLO_MODEL_PATH = None            # e.g. r"C:\models\bottle_caps.pt"
YOLO_CONFIDENCE = 0.5
# Class names in your model that mean "bottle cap". None = accept every class.
YOLO_CAP_CLASSES = {"bottle_cap", "cap"}

# ---------------- Shape detection (bottle_cap only; used when YOLO_MODEL_PATH is None) ----------------
CAP_DIAMETER_TOLERANCE_M = 0.03      # accepted size range: width +- 3 cm
CAP_MAX_SATURATION = 70              # caps are grey (low colour saturation); the wooden table is strongly coloured
CAP_MIN_BRIGHTNESS = 70              # ignore dark shadows
CAP_MIN_AREA_PX = 300
CAP_MIN_ROUNDNESS = 0.5              # short axis / long axis of the fitted ellipse (oblique views look oval)
CAP_MIN_FILL = 0.75                  # blob area / fitted ellipse area
# Only needed if the depth image cannot be read: table height in the planning frame.
TABLE_Z_M = None
 
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
DETECTION_FINGER_GAP_M = 0.15   # open gap used for searching, approaching and releasing
 
# ---------------- Rules / settings ----------------
MAX_FINGER_GAP_M = 0.15         # RULE: finger gap must never exceed this (15 cm)
JOINT_TOLERANCE_RAD = 0.01
SPEED = 0.1                     # 0-1, speed of joint moves (initial/detection) for normal, short moves
# Long joint moves (e.g. the arm was left in a far-away pose by an earlier run) would take so long at
# SPEED that the controller gives up with a timeout, so the speed is raised with the distance, up to this.
SPEED_MAX_FOR_LONG_MOVES = 0.5
LONG_MOVE_RAD = 0.5                  # joint moves longer than this (largest single joint) get a higher speed
MOVE_RETRIES = 2                     # extra attempts when a joint move times out
GRIPPER_TIMEOUT_S = 15.0
GAP_TOLERANCE_M = 0.002
CLOSED_THRESHOLD_M = 0.002
 
# ---------------- Search / pick / place settings ----------------
SCAN_ANGLE_MIN_DEG = -90.0
SCAN_ANGLE_MAX_DEG = 90.0
SCAN_ANGLE_STEP_DEG = 20.0
SCAN_HOVER_Z_M = 0.25
SCAN_RADII_M = (0.30, 0.40, 0.50)
SCAN_RADIUS_OFFSETS_M = (0.0, -0.06, 0.05, -0.12, 0.10)
SCAN_GRID_STEP_M = 0.10
SCAN_GRID_RADIUS_M = 0.30
SCAN_SETTLE_S = 0.7
TARGET_SEARCH_TIMEOUT_S = 90.0
 
PITCH_CANDIDATES_DEG = (180.0, 165.0, 150.0, 135.0)
 
VIEW_HEIGHT_M = 0.20
APPROACH_HEIGHT_M = 0.10
GRASP_SHIFT_TOOL_M = (0.0, 0.0)
STOP_AFTER_APPROACH = False
REFINE_ITERATIONS = 2
# Fine-tune the grasp height (metres, + = higher, - = lower). The tool reference point is often
# NOT at the fingertips, so the fingers can close above a thin object. Try -0.005, -0.01, ...
GRASP_Z_OFFSET_M = 0.0
GRASP_HEIGHT_RETRIES_M = (0.0, 0.01, 0.02, 0.03)
# Set to the object CENTRE height in the planning frame (table height + half the object height) if
# known; tag/depth height estimates are noisy by about 1 cm. None = use the estimate.
OBJECT_CENTRE_Z_M = None
LINEAR_SPEED_SCALES = (0.5, 0.25, 0.1)
COMFORTABLE_RADIUS_M = 0.50
PARK_RADIUS_M = 0.35
PARK_ANGLE_DEG = 50.0
LIFT_HEIGHT_M = 0.12
PLACE_DROP_GAP_M = 0.005
CARRY_CLEARANCE_M = 0.05
SETTLE_S = 0.5
 
 
# ---------------- Plain-language output ----------------
def say(msg):
    print(f"\n>>> {msg}")
 
 
def explain_failure(exc):
    text = str(exc)
    hints = []
    if "NO_IK_SOLUTION" in text or "No reachable tool orientation" in text:
        hints.append("The arm cannot reach that position (too far, too high or too low for it).")
    if "timeout" in text.lower() or "DEADLINE" in text:
        hints.append("The robot took longer than the allowed time to finish a move (the simulation may be slow, "
                     "stuck, or the arm was left far from its start pose).")
    if "Motion planning failed" in text or "FAILURE" in text:
        hints.append(f"The motion planner found no safe path (blocked by the table, a {OBJ_LABEL} or the arm itself).")
    if "collision" in text.lower():
        hints.append("The planner reported a collision.")
    if "Could not descend" in text:
        hints.append(f"The gripper could not be lowered to the {OBJ_LABEL}; the height estimate is probably wrong.")
    if "missed" in text:
        hints.append(f"The fingers closed on nothing: the {OBJ_LABEL} was not between them.")
    if "slipped" in text:
        hints.append(f"The {OBJ_LABEL} slipped out of the gripper while lifting.")
    if "exceeded" in text and "limit" in text:
        hints.append("The fingers opened wider than the allowed maximum gap.")
    return " ".join(hints) or "See the error text below for details."
 
 
def _excepthook(exc_type, exc, tb):
    say("STOPPED WITH AN ERROR. In plain words: " + explain_failure(exc))
    say("The robot stays where it is. Reset the simulation before running again.")
    sys.__excepthook__(exc_type, exc, tb)
 
 
sys.excepthook = _excepthook
 
 
# ---------------- Math helpers ----------------
def quat_dot(a, b):
    return sum(x * y for x, y in zip(a.array(), b.array()))
 
 
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
 
 
# ---------------- Arm helpers ----------------
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
 
 
def joint_distance(arm, joints):
    """Largest single-joint distance (rad) between the current pose and `joints`."""
    current = arm.joints().position_by_name
    return max((abs(current.get(n, t) - t) for n, t in joints.items()), default=0.0)
 
 
def move_to(arm, joints, label):
    """Move the arm to a joint position, skipping if already there.
 
    The speed grows with the distance to travel, and a timed-out move is retried (faster), because
    the arm may already be close to the target when the controller's time limit runs out.
    """
    if is_at(arm, joints):
        print(f"Already at the {label} position")
        return
    check_joint_limits(arm, joints)
    target = arm_target(arm, joints)
    for attempt in range(MOVE_RETRIES + 1):
        dist = joint_distance(arm, joints)
        speed = SPEED
        if dist > LONG_MOVE_RAD:
            speed = min(SPEED_MAX_FOR_LONG_MOVES, SPEED * dist / LONG_MOVE_RAD)
        speed = min(1.0, speed * (1.5 ** attempt))
        say(f"Moving to the {label} position (largest joint has {dist:.2f} rad to go, speed {speed:.2f}).")
        try:
            arm.move_joints(target, speed_scaling=speed)
            print(f"Reached {label} position")
            return
        except OloTimeoutError as exc:
            print(f"Move to {label} timed out: {exc}")
            time.sleep(1.0)
            if is_at(arm, joints):
                say(f"The move timed out, but the arm is at the {label} position anyway.")
                return
            say(f"The move timed out before the arm arrived ({joint_distance(arm, joints):.2f} rad still to go). "
                f"{'Retrying faster.' if attempt < MOVE_RETRIES else 'Giving up.'}")
    raise RuntimeError(
        f"The arm could not reach the {label} position: the moves kept timing out. Reset the simulation "
        "(the arm may be stuck or in a far-away pose) and run again."
    )
 
 
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
            per_finger = gap_m / 2.0
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
 
 
# ---------------- Scan poses ----------------
def hover_poses(origin):
    count = int(round((SCAN_ANGLE_MAX_DEG - SCAN_ANGLE_MIN_DEG) / SCAN_ANGLE_STEP_DEG)) + 1
    angles = [math.radians(SCAN_ANGLE_MIN_DEG + i * SCAN_ANGLE_STEP_DEG) for i in range(count)]
    angles.sort(key=abs)
    a0 = math.atan2(origin.position.y, origin.position.x)
    return [
        (
            f"hover r={r:.2f} m, {math.degrees(a):+.0f} deg",
            Pose(
                position=Point(x=r * math.cos(a), y=r * math.sin(a), z=SCAN_HOVER_Z_M),
                orientation=Quaternion.from_rpy(0.0, 0.0, a - a0) * origin.orientation,
            ),
        )
        for a in angles
        for r in SCAN_RADII_M
    ]
 
 
def scan_poses(origin):
    ox, oy, oz = origin.position.x, origin.position.y, origin.position.z
    r0 = math.hypot(ox, oy)
    a0 = math.atan2(oy, ox)
    count = int(round((SCAN_ANGLE_MAX_DEG - SCAN_ANGLE_MIN_DEG) / SCAN_ANGLE_STEP_DEG)) + 1
    angles = [math.radians(SCAN_ANGLE_MIN_DEG + i * SCAN_ANGLE_STEP_DEG) for i in range(count)]
    angles.sort(key=lambda a: abs(a - a0))
 
    poses = []
    for angle in angles:
        yaw = Quaternion.from_rpy(0.0, 0.0, angle - a0)
        for dr in SCAN_RADIUS_OFFSETS_M:
            r = r0 + dr
            if r < 0.15:
                continue
            poses.append((
                f"arc r={r:.2f} m, {math.degrees(angle):+.0f} deg",
                Pose(
                    position=Point(x=r * math.cos(angle), y=r * math.sin(angle), z=oz),
                    orientation=yaw * origin.orientation,
                ),
            ))
 
    n = int(SCAN_GRID_RADIUS_M / SCAN_GRID_STEP_M)
    grid = [(i * SCAN_GRID_STEP_M, j * SCAN_GRID_STEP_M) for i in range(-n, n + 1) for j in range(-n, n + 1)]
    grid = sorted((p for p in grid if math.hypot(*p) <= SCAN_GRID_RADIUS_M + 1e-9), key=lambda p: math.hypot(*p))
    for dx, dy in grid:
        poses.append((
            f"grid ({dx:+.2f}, {dy:+.2f})",
            Pose(position=Point(x=ox + dx, y=oy + dy, z=oz), orientation=origin.orientation),
        ))
    return poses
 
 
def tool_orientation(azimuth, obj_yaw, pitch, turn):
    """Tool orientation for a top grasp.
 
    `pitch` = 180 deg is straight down; smaller values tilt the tool outward, away from the base
    (azimuth = direction from the base to the object). The tool x axis is lined up with an object
    edge; `turn` picks one of the four 90-degree equivalents.
    """
    k = azimuth - obj_yaw - turn * math.pi / 2
    return (
        Quaternion.from_rpy(0.0, 0.0, azimuth)
        * Quaternion.from_rpy(0.0, pitch, 0.0)
        * Quaternion.from_rpy(0.0, 0.0, k)
    )
 
 
def orientation_candidates(position, obj_yaw, reference):
    """Orientations to try, most vertical first; the two turns closest to `reference` for each tilt."""
    azimuth = math.atan2(position.y, position.x)
    result = []
    for pitch_deg in PITCH_CANDIDATES_DEG:
        pitch = math.radians(pitch_deg)
        options = [tool_orientation(azimuth, obj_yaw, pitch, turn) for turn in range(4)]
        options.sort(key=lambda q: -abs(quat_dot(q, reference)))
        result.extend((pitch_deg, q) for q in options[:2])
    return result
 
 
# ---------------- Main sequence ----------------
with Client() as client:
    robot = client.robot()
    arm = robot.kinematics
    gripper = arm.ee
    if gripper is None:
        raise RuntimeError("This script requires a gripper end effector")
 
    say(f"Object type: {OBJ_LABEL} ({P['detection']} detection), {OBJ_WIDTH_M * 100:.0f} cm wide, "
        f"{OBJ_HEIGHT_M * 100:.0f} cm high.")
    print("Default group:", arm.model.default_group)
    for g in arm.model.groups:
        print(f"  group {g.name}: joints={g.joints}")
 
    rgb_topic = robot.core.topic(RGB_TOPIC, "sensor_msgs/msg/Image")
    info_topic = robot.core.topic(CAMERA_INFO_TOPIC, "sensor_msgs/msg/CameraInfo")
    intrinsics = camera_info_to_intrinsics(robot.core.get_latest(info_topic))
    depth_topic = None
    detector = None
    if P["detection"] == "shape":
        depth_topic = robot.core.topic(DEPTH_TOPIC, "sensor_msgs/msg/Image")
    else:
        detector = robot.vision.tags(P["tag_family"], size_m=P["marker_size_m"])
 
    # object name -> {"centre": Point, "yaw": float} in the planning frame (latest estimate)
    known = {}
    frozen = set()          # objects whose estimate must not change any more (the one we are holding)
    holding = {"v": False}  # True while an object is in the gripper
 
    def current_pose():
        return robot.frames.lookup(arm.planning_frame, arm.end_effector_link).as_pose()
 
    def open_fingers():
        set_finger_gap(arm, DETECTION_FINGER_GAP_M)
 
    def new_name():
        return f"{OBJ_PREFIX}_{len(known) + 1}"
 
    # ------------------------------------------------------------------
    # Detection, variant 1 (cube): ArUco markers, possibly several faces per cube.
    # ------------------------------------------------------------------
    def tag_centre(tag):
        return tag.to_frame(robot.frames, arm.planning_frame).apply(Point(z=-OBJ_HALF_M))
 
    def object_of(tag, c):
        """Which object a tag is on: the nearest known one within SAME_OBJECT_DISTANCE_M, else a new one."""
        best, best_d = None, SAME_OBJECT_DISTANCE_M
        for n, est in known.items():
            k = est["centre"]
            d = math.hypot(c.x - k.x, c.y - k.y)
            if d < best_d:
                best, best_d = n, d
        if best is not None:
            return best
        name = new_name()
        known[name] = {"centre": c, "yaw": 0.0}  # placeholder, replaced by update_estimate_from_tags
        print(f"New {OBJ_LABEL} found: {name} (tag {tag.id})")
        say(f"Found a NEW {OBJ_LABEL}, calling it {name} (marker id {tag.id}).")
        return name
 
    def update_estimate_from_tags(name, tags):
        """Estimate the object centre and yaw from ALL tags currently seen on it."""
        centres = []
        yaw_samples = []
        for t in tags:
            tp = t.to_frame(robot.frames, arm.planning_frame)
            centres.append(tp.apply(Point(z=-OBJ_HALF_M)))
            origin = tp.apply(Point())
            for axis in (Point(x=1.0), Point(y=1.0), Point(z=1.0)):
                p = tp.apply(axis)
                dx, dy = p.x - origin.x, p.y - origin.y
                if math.hypot(dx, dy) > 0.7:  # this tag axis lies roughly horizontal
                    yaw_samples.append(math.atan2(dy, dx))
        n = len(centres)
        centre = Point(
            x=sum(c.x for c in centres) / n,
            y=sum(c.y for c in centres) / n,
            z=OBJECT_CENTRE_Z_M if OBJECT_CENTRE_Z_M is not None else sum(c.z for c in centres) / n,
        )
        if yaw_samples:
            # cube edges repeat every 90 degrees: average the angle multiplied by 4
            s = sum(math.sin(4 * a) for a in yaw_samples)
            c = sum(math.cos(4 * a) for a in yaw_samples)
            yaw = math.atan2(s, c) / 4
        else:
            yaw = known.get(name, {}).get("yaw", 0.0)
        known[name] = {"centre": centre, "yaw": yaw}
        print(
            f"{name}: centre ({centre.x:.3f}, {centre.y:.3f}, {centre.z:.3f}), "
            f"yaw {math.degrees(yaw):+.0f} deg, seen by {n} tag(s)"
        )
        say(f"{name} is at x={centre.x:.2f}, y={centre.y:.2f}, centre height z={centre.z:.3f} m, "
            f"rotated {math.degrees(yaw):+.0f} deg.")
 
    def detect_by_tags():
        """Return {name: [tags]} and update the position estimate of every visible object."""
        frame = detector.detect(robot.core.get_image(rgb_topic), intrinsics=intrinsics)
        if frame.tags:
            print("Tag IDs in view:", sorted(t.id for t in frame.tags))
        ee = current_pose().position if holding["v"] else None
        result = {}
        for tag in frame.tags:
            if tag.transform is None:
                continue
            c = tag_centre(tag)
            if ee is not None and math.dist((c.x, c.y, c.z), (ee.x, ee.y, ee.z)) < HELD_EXCLUSION_M:
                continue  # that is the object in the gripper
            name = object_of(tag, c)
            result.setdefault(name, []).append(tag)
        for name, tags in result.items():
            if name not in frozen:
                update_estimate_from_tags(name, tags)
        return result
 
    # ------------------------------------------------------------------
    # Detection, variant 2 (bottle cap): no markers; grey round blobs + the depth image.
    # ------------------------------------------------------------------
    def image_array(img):
        arr = img.array
        arr = arr() if callable(arr) else arr
        return np.asarray(arr)
 
    def camera_to_planning(p_cam):
        """Convert a point (x, y, z) in the camera frame to the planning frame."""
        cam = robot.frames.lookup(arm.planning_frame, intrinsics.frame_id).as_pose()
        dx, dy, dz = rotate(cam.orientation, p_cam)
        return Point(x=cam.position.x + dx, y=cam.position.y + dy, z=cam.position.z + dz)
 
    def read_depth_m():
        """Depth image in metres (NaN where invalid), or None if it cannot be read."""
        try:
            d = image_array(robot.core.get_image(depth_topic))
            if d.ndim == 3:
                d = d[..., 0]
            d = d.astype(np.float32)
            if np.nanmax(d) > 100:  # millimetres
                d = d / 1000.0
            d[(d <= 0.05) | ~np.isfinite(d)] = np.nan
            return d
        except Exception as exc:
            print(f"Could not read the depth image: {exc}")
            return None
 
    _yolo_model = {"v": None}

    def load_yolo():
        """Load the YOLO model once (needs `pip install ultralytics`)."""
        if _yolo_model["v"] is None:
            from ultralytics import YOLO
            _yolo_model["v"] = YOLO(YOLO_MODEL_PATH)
            print(f"YOLO model loaded: {YOLO_MODEL_PATH} (classes: {list(_yolo_model['v'].names.values())})")
        return _yolo_model["v"]

    def yolo_candidates(rgb, rejected):
        """Caps found by YOLO as (cx, cy, long_ax, short_ax, outline points). The outline is the
        ellipse inscribed in the detection box, which follows the rim of the cap."""
        height, width = rgb.shape[:2]
        result = load_yolo()(rgb[..., ::-1], conf=YOLO_CONFIDENCE, verbose=False)[0]  # ultralytics wants BGR
        candidates = []
        for xyxy, cls in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
            if YOLO_CAP_CLASSES is not None and result.names[int(cls)] not in YOLO_CAP_CLASSES:
                continue
            x1, y1, x2, y2 = xyxy
            if x1 <= 2 or y1 <= 2 or x2 >= width - 2 or y2 >= height - 2:
                rejected["touches image edge"] += 1
                continue
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w / 2, y1 + h / 2
            t = np.linspace(0.0, 2 * math.pi, 48, endpoint=False)
            pts = np.stack([cx + 0.45 * w * np.cos(t), cy + 0.45 * h * np.sin(t)], axis=1).astype(int)
            candidates.append((cx, cy, max(w, h), min(w, h), pts))
        return candidates

    def colour_candidates(rgb, rejected):
        """Grey round blobs as (cx, cy, long_ax, short_ax, outline points) (the non-YOLO method)."""
        height, width = rgb.shape[:2]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = ((hsv[..., 1] <= CAP_MAX_SATURATION) & (hsv[..., 2] >= CAP_MIN_BRIGHTNESS)).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < CAP_MIN_AREA_PX or len(cnt) < 5:
                rejected["too small"] += 1
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2:
                rejected["touches image edge"] += 1
                continue
            (cx, cy), (ax1, ax2), _ = cv2.fitEllipse(cnt)
            long_ax, short_ax = max(ax1, ax2), min(ax1, ax2)
            fill = area / (math.pi * long_ax * short_ax / 4.0)
            if short_ax / long_ax < CAP_MIN_ROUNDNESS or fill < CAP_MIN_FILL:
                rejected["not round"] += 1
                continue
            candidates.append((cx, cy, long_ax, short_ax, cnt[:, 0, :]))
        return candidates

    def find_round_blobs():
        """Look for bottle caps (YOLO if YOLO_MODEL_PATH is set, else grey round blobs) of the right
        size. Returns a list of (centre Point, info text)."""
        rgb = image_array(robot.core.get_image(rgb_topic))[..., :3].astype(np.uint8)
        depth = read_depth_m()
        if depth is not None and depth.shape[:2] != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        height, width = rgb.shape[:2]
        rejected = {"too small": 0, "touches image edge": 0, "not round": 0, "wrong size": 0, "no depth": 0}
        found = []
        candidates = (yolo_candidates if YOLO_MODEL_PATH else colour_candidates)(rgb, rejected)
        for cx, cy, long_ax, short_ax, pts in candidates:
            # Depth along the outline = the rim of the cap (top edge, OBJ_HEIGHT_M above the table).
            rim_depth = None
            if depth is not None:
                vals = depth[pts[:, 1].clip(0, height - 1), pts[:, 0].clip(0, width - 1)]
                vals = vals[np.isfinite(vals)]
                if len(vals) >= 5:
                    rim_depth = float(np.median(vals))
            if rim_depth is None and TABLE_Z_M is not None:
                # Fallback: intersect the viewing ray of the blob centre with the rim plane.
                ray = ((cx - intrinsics.cx) / intrinsics.fx, (cy - intrinsics.cy) / intrinsics.fy, 1.0)
                cam = robot.frames.lookup(arm.planning_frame, intrinsics.frame_id).as_pose()
                _, _, rz = rotate(cam.orientation, ray)
                if abs(rz) > 1e-6:
                    t = (TABLE_Z_M + OBJ_HEIGHT_M - cam.position.z) / rz
                    if t > 0:
                        rim_depth = t  # the camera-frame ray has z = 1, so t is the depth
            if rim_depth is None:
                rejected["no depth"] += 1
                continue
 
            diameter_m = (long_ax + short_ax) / 2.0 * rim_depth / intrinsics.fx
            if abs(diameter_m - OBJ_WIDTH_M) > CAP_DIAMETER_TOLERANCE_M:
                rejected["wrong size"] += 1
                print(f"  Ignored a round blob of {diameter_m * 100:.1f} cm (caps are {OBJ_WIDTH_M * 100:.0f} cm)")
                continue
 
            p_cam = ((cx - intrinsics.cx) * rim_depth / intrinsics.fx,
                     (cy - intrinsics.cy) * rim_depth / intrinsics.fy, rim_depth)
            rim_point = camera_to_planning(p_cam)
            z_centre = OBJECT_CENTRE_Z_M if OBJECT_CENTRE_Z_M is not None else rim_point.z - OBJ_HEIGHT_M / 2
            centre = Point(x=rim_point.x, y=rim_point.y, z=z_centre)
            found.append((centre, f"pixel ({cx:.0f}, {cy:.0f}), {diameter_m * 100:.1f} cm wide"))
        rej = ", ".join(f"{n} {k}" for k, n in rejected.items() if n)
        say(f"Camera check: {len(found)} {OBJ_LABEL}(s) found" + (f" (rejected: {rej})" if rej else "") + ".")
        return found
 
    def detect_by_shape():
        """Return {name: centre} for visible objects, and update the position estimates."""
        ee = current_pose().position if holding["v"] else None
        result = {}
        for centre, info in find_round_blobs():
            if ee is not None and math.dist((centre.x, centre.y, centre.z), (ee.x, ee.y, ee.z)) < HELD_EXCLUSION_M:
                continue  # that is the object in the gripper
            name, best = None, SAME_OBJECT_DISTANCE_M
            for n, est in known.items():
                d = math.hypot(centre.x - est["centre"].x, centre.y - est["centre"].y)
                if d < best:
                    name, best = n, d
            if name is None:
                name = new_name()
                known[name] = {"centre": centre, "yaw": 0.0}
                say(f"Found a NEW {OBJ_LABEL}, calling it {name}: {info}.")
            result[name] = centre
        for name, centre in result.items():
            if name not in frozen:
                known[name] = {"centre": centre, "yaw": 0.0}  # round: yaw does not matter
                print(f"{name}: centre ({centre.x:.3f}, {centre.y:.3f}, {centre.z:.3f})")
                say(f"{name} is at x={centre.x:.2f}, y={centre.y:.2f}, centre height z={centre.z:.3f} m.")
        return result
 
    # The one detection function the rest of the script uses.
    detect_objects = detect_by_tags if P["detection"] == "tags" else detect_by_shape
 
    # ------------------------------------------------------------------
    # Moving
    # ------------------------------------------------------------------
    def scan_for(allowed, origin, avoid_obstacles, timeout=None):
        """Move the camera around `origin` until an object accepted by `allowed(name)` is visible.
 
        Returns None if every scan pose was tried, or `timeout` seconds passed, without seeing one.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        o = origin.position
        print(f"Scan origin: ({o.x:.3f}, {o.y:.3f}, {o.z:.3f}), radius from base {math.hypot(o.x, o.y):.3f} m")
        for label, pose in hover_poses(origin) + scan_poses(origin):
            if deadline is not None and time.monotonic() >= deadline:
                print(f"Search timed out after {timeout:.0f} s")
                return None
            moved = False
            error = None
            t0 = time.monotonic()
            print(f"Moving to {label} ...")
            say(f"No {OBJ_LABEL} in view yet. Moving the camera to a new viewpoint to keep searching.")
            # Planning can fail because of collision checking, so retry once without it.
            # An IK failure means the pose is unreachable: retrying would only waste time.
            for check_obstacles in dict.fromkeys((avoid_obstacles, False)):
                try:
                    arm.move_pose(pose, avoid_obstacles=check_obstacles)
                    moved = True
                    break
                except Exception as exc:
                    error = exc
                    if "NO_IK_SOLUTION" in str(exc):
                        break
            if not moved:
                print(f"Skipping {label} after {time.monotonic() - t0:.1f} s: {error}")
                continue
            print(f"Scanning at {label} (move took {time.monotonic() - t0:.1f} s)")
            time.sleep(SCAN_SETTLE_S)
            seen = detect_objects()
            hits = [name for name in seen if allowed(name)]
            if hits:
                return hits[0]
        return None
 
    def move_over(position, obj_yaw, avoid_obstacles, shift=False):
        """Move the tool to `position` with the most vertical orientation the arm can reach.
 
        With shift=True the target is moved by GRASP_SHIFT_TOOL_M along the tool x/y axes
        (used for grasping/placing, not for the camera view). Returns the pose reached.
        """
        last_error = None
        for pitch_deg, orientation in orientation_candidates(position, obj_yaw, current_pose().orientation):
            target = position
            if shift:
                sx, sy, _ = rotate(orientation, (GRASP_SHIFT_TOOL_M[0], GRASP_SHIFT_TOOL_M[1], 0.0))
                target = Point(x=position.x + sx, y=position.y + sy, z=position.z)
            pose = Pose(position=target, orientation=orientation)
            for check_obstacles in dict.fromkeys((avoid_obstacles, False)):
                try:
                    arm.move_pose(pose, avoid_obstacles=check_obstacles)
                    print(f"Reached ({position.x:.3f}, {position.y:.3f}, {position.z:.3f}) with tool tilt {pitch_deg:.0f}")
                    return pose
                except Exception as exc:
                    last_error = exc
        raise RuntimeError(
            f"No reachable tool orientation at ({position.x:.3f}, {position.y:.3f}, {position.z:.3f}), "
            f"radius {math.hypot(position.x, position.y):.3f} m from the base. "
            f"The position may be outside the arm's reach. Last error: {last_error}"
        )
 
    def move_linear_safe(pose):
        """Straight-line move that slows down when the planner rejects the path as too fast."""
        last_error = None
        for scale in LINEAR_SPEED_SCALES:
            try:
                return arm.move_linear(pose, speed_scaling=scale)
            except TypeError:
                print("move_linear has no speed_scaling argument in this SDK version")
                return arm.move_linear(pose)
            except Exception as exc:
                last_error = exc
                print(f"Straight-line move failed at speed scaling {scale}: {exc}")
        raise last_error
 
    home_pose = None  # end-effector pose at the detection position
 
    def go_home(avoid_obstacles):
        """Move to the DETECTION joint position (fallback: its Cartesian pose); return the pose reached.
 
        Only the arm joints are commanded, so the fingers keep their state (closed while carrying).
        """
        errors = []
        say("Returning the arm to the detection position.")
        try:
            move_to(arm, DETECTION_JOINTS, "detection")
            time.sleep(SETTLE_S)
            return current_pose()
        except Exception as exc:
            errors.append(f"joint target: {exc}")
        if home_pose is not None:
            for check_obstacles in dict.fromkeys((avoid_obstacles, False)):
                try:
                    arm.move_pose(home_pose, avoid_obstacles=check_obstacles)
                    time.sleep(SETTLE_S)
                    print("At detection position (Cartesian pose)")
                    return home_pose
                except Exception as exc:
                    errors.append(f"home pose: {exc}")
        raise RuntimeError(
            "Could not reach the detection position (" + "; ".join(errors) + "). "
            "The arm may be in a bad state from a previous run: reset the simulation and retry."
        )
 
    if YOLO_MODEL_PATH:
        load_yolo()  # load before the arm moves, so a wrong path fails early
 
    # 1. Initial position, then the detection position with the open finger gap.
    say("Step 1: moving the arm to its initial (folded) position with the gripper closed.")
    print("\n[1] Initial position")
    move_to(arm, INITIAL_JOINTS, "initial")
    set_finger_gap(arm, INITIAL_FINGER_GAP_M)
 
    print("\n[2] Detection position")
    say(f"Step 2: moving to the detection position and opening the gripper wide "
        f"(up to {MAX_FINGER_GAP_M * 100:.0f} cm) so it fits around the {OBJ_LABEL}.")
    home_pose = go_home(avoid_obstacles=True)
    open_fingers()
    time.sleep(SETTLE_S)
    hp = home_pose.position
    print(f"Detection pose: ({hp.x:.3f}, {hp.y:.3f}, {hp.z:.3f}), orientation {home_pose.orientation.array()}")
 
    # Check: can the open gripper actually fit around the object?
    open_gap = measured_gap(arm)
    if open_gap is not None:
        if open_gap < OBJ_WIDTH_M + 0.01:
            say(f"WARNING: the open gripper is only {open_gap * 100:.1f} cm wide but the {OBJ_LABEL} is "
                f"{OBJ_WIDTH_M * 100:.0f} cm wide. The fingers will land on top of it. Check "
                f"DETECTION_FINGER_GAP_M / MAX_FINGER_GAP_M and the gripper's own maximum opening.")
        else:
            say(f"Gripper opens {open_gap * 100:.1f} cm: enough for a {OBJ_WIDTH_M * 100:.0f} cm {OBJ_LABEL}.")
 
    # ------------------------------------------------------------------
    # Pick an object up / put the carried object down (on the table or on another object).
    # ------------------------------------------------------------------
    def pick_object(name):
        """Look closer at `name`, approach it from above, grasp it and lift it.
 
        Returns (pick_pose, lift_pose, centre), centre being the final centre estimate.
        """
        say(f"Picking up {name}: first taking a closer look from above to improve its position.")
        # Refine the position from a closer view (if the object is still visible).
        for _ in range(REFINE_ITERATIONS):
            est = known[name]
            c = est["centre"]
            moved_closer = False
            for view_h in dict.fromkeys((VIEW_HEIGHT_M, 0.15, 0.10, APPROACH_HEIGHT_M)):
                view_position = Point(x=c.x, y=c.y, z=c.z + OBJ_HALF_M + view_h)
                try:
                    move_over(view_position, est["yaw"], avoid_obstacles=True)
                    moved_closer = True
                    break
                except Exception as exc:
                    print(f"Closer view at {view_h:.2f} m above the {OBJ_LABEL} not reachable: {exc}")
            if not moved_closer:
                print("No reachable closer view; using the previous estimate")
                break
            time.sleep(SCAN_SETTLE_S)
            if name not in detect_objects():
                print(f"{OBJ_LABEL.capitalize()} not visible from the closer view; using the previous estimate")
                break
 
        est = known[name]
        c = est["centre"]
        grasp_z = c.z + OBJ_HALF_M - GRASP_CLEARANCE_M + GRASP_Z_OFFSET_M
        print(f"Final estimate: ({c.x:.3f}, {c.y:.3f}, {c.z:.3f}), yaw {math.degrees(est['yaw']):+.0f} deg")
        radius = math.hypot(c.x, c.y)
        if radius > COMFORTABLE_RADIUS_M:
            print(
                f"WARNING: the {OBJ_LABEL} is {radius:.2f} m from the base, near the edge of the arm's reach. "
                f"Straight-line moves may fail; moving the {OBJ_LABEL} to 0.30-0.40 m is recommended."
            )
 
        say(f"Moving the open gripper to a point {APPROACH_HEIGHT_M * 100:.0f} cm above the {OBJ_LABEL} "
            f"(centre x={c.x:.2f}, y={c.y:.2f}, z={c.z:.3f} m).")
        approach_pose = move_over(
            Point(x=c.x, y=c.y, z=grasp_z + APPROACH_HEIGHT_M), est["yaw"], avoid_obstacles=True, shift=True
        )
        time.sleep(SETTLE_S)
        ap = approach_pose.position
        print(f"Above the {OBJ_LABEL}: EE ({ap.x:.3f}, {ap.y:.3f}, {ap.z:.3f}), centre ({c.x:.3f}, {c.y:.3f}, {c.z:.3f})")
        if STOP_AFTER_APPROACH:
            raise SystemExit(f"Stopped above the {OBJ_LABEL} (STOP_AFTER_APPROACH). Check it is between the fingers.")
        gx, gy = approach_pose.position.x, approach_pose.position.y  # descend straight down from here
        say(f"Lowering the gripper straight down around the {OBJ_LABEL} (target height z={grasp_z:.3f} m).")
 
        # Grasp.
        pick_pose = None
        for extra in GRASP_HEIGHT_RETRIES_M:
            candidate = Pose(
                position=Point(x=gx, y=gy, z=grasp_z + extra), orientation=approach_pose.orientation
            )
            try:
                move_linear_safe(candidate)
                pick_pose = candidate
                break
            except Exception as exc:
                print(f"Descent to grasp height z={grasp_z + extra:.3f} m failed ({exc}); retrying higher")
                try:  # go back above the object before the next try
                    arm.move_pose(approach_pose, avoid_obstacles=False)
                except Exception as back_exc:
                    print(f"Could not return to the approach pose: {back_exc}")
                time.sleep(SETTLE_S)
        if pick_pose is None:
            print("Straight-line descent failed at every height; trying a normal (not perfectly straight) move")
            candidate = Pose(
                position=Point(x=gx, y=gy, z=grasp_z + 0.01), orientation=approach_pose.orientation
            )
            try:
                arm.move_pose(candidate, avoid_obstacles=False)
                pick_pose = candidate
            except Exception as exc:
                print(f"That failed too: {exc}")
        if pick_pose is None:
            raise RuntimeError(
                f"Could not descend to the {OBJ_LABEL} at any grasp height. The height estimate may be "
                f"wrong (estimated centre z = {c.z:.3f} m): check the table height in the planning frame "
                "and set OBJECT_CENTRE_Z_M"
            )
        ep = current_pose().position
        print(f"Grasping at EE ({ep.x:.3f}, {ep.y:.3f}, {ep.z:.3f})")
        time.sleep(SETTLE_S)
        say(f"Closing the gripper on the {OBJ_LABEL}.")
        gripper.close(timeout=GRIPPER_TIMEOUT_S)
        time.sleep(SETTLE_S)
 
        # Check: did the fingers close on something?
        gap_after = measured_gap(arm)
        if gap_after is not None:
            print(f"Finger gap after closing: {gap_after * 100:.1f} cm")
            if gap_after <= CLOSED_THRESHOLD_M:
                open_fingers()
                raise RuntimeError(
                    f"Fingers closed fully: nothing is between them (missed the {OBJ_LABEL}). Check the grasp "
                    "height (GRASP_Z_OFFSET_M / OBJECT_CENTRE_Z_M) and the sideways alignment "
                    "(GRASP_SHIFT_TOOL_M) using STOP_AFTER_APPROACH = True."
                )
            say(f"Good: the fingers stopped at {gap_after * 100:.1f} cm, so something is gripped "
                f"(a {OBJ_WIDTH_M * 100:.0f} cm {OBJ_LABEL} should read about that).")
        frozen.add(name)  # the object moves with the gripper now: stop updating its position
        holding["v"] = True
 
        # Lift straight up. Try the full lift height first; near the edge of the arm's reach a
        # high pose may have no IK solution, so fall back to lower ones. The approach height is
        # known to be reachable (the arm was just there).
        say(f"Lifting the {OBJ_LABEL} straight up.")
        lift_heights_z = [pick_pose.position.z + LIFT_HEIGHT_M, approach_pose.position.z,
                          pick_pose.position.z + 0.07, pick_pose.position.z + 0.04]
        lift_pose = None
        for lift_z in lift_heights_z:
            candidate = Pose(
                position=Point(x=pick_pose.position.x, y=pick_pose.position.y, z=lift_z),
                orientation=pick_pose.orientation,
            )
            try:
                move_linear_safe(candidate)
                lift_pose = candidate
                break
            except Exception as exc:
                print(f"Lift to z={lift_z:.3f} m failed ({exc}); trying a lower lift")
        if lift_pose is None:
            raise RuntimeError(
                f"Could not lift the {OBJ_LABEL}. It is probably at the edge of the arm's reach "
                f"(radius {math.hypot(c.x, c.y):.2f} m from the base): move it closer to the robot"
            )
        time.sleep(SETTLE_S)
 
        # Check: is it still held after the lift?
        gap_lifted = measured_gap(arm)
        if gap_lifted is not None and gap_lifted <= CLOSED_THRESHOLD_M:
            holding["v"] = False
            frozen.discard(name)
            open_fingers()
            raise RuntimeError(f"The {OBJ_LABEL} slipped out of the gripper while lifting (fingers are fully closed).")
        return pick_pose, lift_pose, c
 
    def place_object(x, y, base_centre_z, yaw, levels, lift_z, label):
        """Put the carried object down at (x, y).
 
        levels = 0: on the table (base_centre_z = centre height of an object lying on the table).
        levels = 1: on top of an object lying on the table.
        """
        place_ee_z = (
            base_centre_z + 2 * OBJ_HALF_M * levels
            + OBJ_HALF_M - GRASP_CLEARANCE_M + PLACE_DROP_GAP_M
        )
        # high enough that the carried object clears whatever is below
        carry_z = max(lift_z, place_ee_z + CARRY_CLEARANCE_M)
        say(f"Carrying the {OBJ_LABEL} over to x={x:.2f}, y={y:.2f} ({label}).")
        above = move_over(Point(x=x, y=y, z=carry_z), yaw, avoid_obstacles=False, shift=True)
        time.sleep(SETTLE_S)
        place_pose = Pose(
            position=Point(x=above.position.x, y=above.position.y, z=place_ee_z),
            orientation=above.orientation,
        )
        try:
            move_linear_safe(place_pose)
        except Exception as exc:
            print(f"Straight-line lowering failed ({exc}); using a normal move")
            arm.move_pose(place_pose, avoid_obstacles=False)
        time.sleep(SETTLE_S)
        say(f"Opening the gripper to release the {OBJ_LABEL}.")
        open_fingers()
        holding["v"] = False
        time.sleep(SETTLE_S)
        say("Moving the gripper straight up and away.")
        try:
            move_linear_safe(above)
        except Exception as exc:
            print(f"Straight-line retreat failed ({exc}); using a normal move")
            try:
                arm.move_pose(above, avoid_obstacles=False)
            except Exception as exc2:
                print(f"Retreat failed too: {exc2}")
        print(f"Placed the {OBJ_LABEL} ({label})")
 
    def park_spots(first, centre):
        """Candidate spots to put the first object aside, best first.
 
        Many spots (several angles and radii on both sides) are tried, so a single unreachable
        spot does not stop the run. A spot must be far enough from every other known object and
        from the pick-up point; among those, the spots closest to the pick-up point come first
        (the arm is already near there, so they are the most likely to be reachable).
        """
        others = [k["centre"] for n, k in known.items() if n != first]
        min_gap = TARGET_MIN_DISTANCE_M + 0.05
 
        def clearance(p):
            return min([math.hypot(p.x - o.x, p.y - o.y) for o in others] + [1.0])
 
        spots = []
        for radius in (PARK_RADIUS_M, PARK_RADIUS_M - 0.05, PARK_RADIUS_M + 0.05):
            for angle_deg in (PARK_ANGLE_DEG, 30.0, 70.0, 10.0, 90.0):
                for side in (1, -1):
                    a = side * math.radians(angle_deg)
                    p = Point(x=radius * math.cos(a), y=radius * math.sin(a), z=centre.z)
                    if clearance(p) >= min_gap and math.hypot(p.x - centre.x, p.y - centre.y) >= min_gap:
                        spots.append(p)
        spots.sort(key=lambda p: math.hypot(p.x - centre.x, p.y - centre.y))
        return spots
 
    # ------------------------------------------------------------------
    # 3. Find the first object and pick it up.
    # ------------------------------------------------------------------
    print(f"\n[3] Searching for a {OBJ_LABEL} to pick...")
    say(f"Step 3: looking through the camera for a {OBJ_LABEL}.")
    time.sleep(SCAN_SETTLE_S)
    visible_now = list(detect_objects())  # maybe one is already in view from the detection position
    first = visible_now[0] if visible_now else scan_for(lambda name: True, home_pose, avoid_obstacles=True)
    if first is None:
        raise RuntimeError(f"No {OBJ_LABEL} found while scanning; widen the scan range")
    print(f"Picking the first {OBJ_LABEL} ({first}) to put it aside")
    say(f"The first {OBJ_LABEL} ({first}) will be moved aside to make room.")
    pick_pose, lift_pose, first_centre = pick_object(first)
 
    # ------------------------------------------------------------------
    # 4. Put the first object aside.
    # ------------------------------------------------------------------
    parked_at = None
    for spot in park_spots(first, first_centre):
        print(f"Putting the first {OBJ_LABEL} aside at ({spot.x:.2f}, {spot.y:.2f})")
        say(f"Trying to set the {OBJ_LABEL} down at x={spot.x:.2f}, y={spot.y:.2f}.")
        try:
            place_object(spot.x, spot.y, first_centre.z, 0.0, 0, lift_pose.position.z, "put aside")
            parked_at = spot
            break
        except Exception as exc:
            print(f"Could not use ({spot.x:.2f}, {spot.y:.2f}): {exc}")
            say(f"Spot x={spot.x:.2f}, y={spot.y:.2f} did not work: {explain_failure(exc)} Trying the next free spot.")
 
    if parked_at is None:
        print(f"Could not put the {OBJ_LABEL} aside; putting it back where it was picked up")
        say(f"No parking spot worked. Putting the {OBJ_LABEL} back where it was picked up.")
        try:
            arm.move_pose(lift_pose, avoid_obstacles=False)
            move_linear_safe(pick_pose)
            open_fingers()
            holding["v"] = False
            time.sleep(SETTLE_S)
            move_linear_safe(lift_pose)
        except Exception as exc:
            print(f"Putting it back failed: {exc}")
        go_home(avoid_obstacles=True)
        raise RuntimeError(f"Could not put the first {OBJ_LABEL} aside: check that the parking spots are reachable")
 
    # From now on the first object is a normal object again, at its new position.
    frozen.discard(first)
    known[first] = {"centre": Point(x=parked_at.x, y=parked_at.y, z=first_centre.z), "yaw": 0.0}
    home_pose = go_home(avoid_obstacles=True)
 
    # ------------------------------------------------------------------
    # 5. Find the second object (any object that is not the first one).
    # ------------------------------------------------------------------
    def is_other(name):
        k = known.get(name)
        f = known[first]["centre"]
        return (
            name != first
            and k is not None
            and math.hypot(k["centre"].x - f.x, k["centre"].y - f.y) > TARGET_MIN_DISTANCE_M
        )
 
    second = next((name for name in known if is_other(name)), None)
    if second is None:
        print(f"Searching for the second {OBJ_LABEL} (up to {TARGET_SEARCH_TIMEOUT_S:.0f} s)...")
        say(f"Looking for the second {OBJ_LABEL}.")
        time.sleep(SCAN_SETTLE_S)
        visible_now = [name for name in detect_objects() if is_other(name)]
        second = visible_now[0] if visible_now else scan_for(
            is_other, home_pose, avoid_obstacles=True, timeout=TARGET_SEARCH_TIMEOUT_S
        )
 
    if second is None:
        print(f"No second {OBJ_LABEL} found. The first one stays where it was put.")
        say(f"No second {OBJ_LABEL} found. The first one stays where it was put.")
    else:
        # --------------------------------------------------------------
        # 6. Pick the second object up.
        # --------------------------------------------------------------
        print(f"Picking the second {OBJ_LABEL} ({second})")
        say(f"Now picking the second {OBJ_LABEL} ({second}), which will be stacked on top of the first.")
        pick_pose2, lift_pose2, _ = pick_object(second)
 
        # --------------------------------------------------------------
        # 7. Carry it over the first object (take a closer look first), lower it, release.
        # --------------------------------------------------------------
        for _ in range(REFINE_ITERATIONS):
            fe = known[first]
            fc = fe["centre"]
            try:
                move_over(
                    Point(x=fc.x, y=fc.y, z=fc.z + OBJ_HALF_M + VIEW_HEIGHT_M),
                    fe["yaw"], avoid_obstacles=False,
                )
            except Exception as exc:
                print(f"Could not get a closer view of the first {OBJ_LABEL}: {exc}")
                break
            time.sleep(SCAN_SETTLE_S)
            if first not in detect_objects():
                print(f"First {OBJ_LABEL} not visible from the closer view; using the previous estimate")
                break
 
        fe = known[first]
        fc = fe["centre"]
        print(f"Placing the second {OBJ_LABEL} on the first at ({fc.x:.3f}, {fc.y:.3f})")
        say(f"Stacking: placing the second {OBJ_LABEL} on top of the first one.")
        place_object(fc.x, fc.y, first_centre.z, fe["yaw"], 1, lift_pose2.position.z,
                     f"on top of the first {OBJ_LABEL}")
        go_home(avoid_obstacles=False)
        print(f"Done: the second {OBJ_LABEL} is on top of the first")
        say(f"Done: the second {OBJ_LABEL} is on top of the first.")
 
    # ------------------------------------------------------------------
    # 8. Return to the initial position.
    # ------------------------------------------------------------------
    print("\n[8] Back to initial position")
    say("Finished. Returning the arm to its initial position.")
    move_to(arm, INITIAL_JOINTS, "initial")
    set_finger_gap(arm, INITIAL_FINGER_GAP_M)
    print("\nDone: back at the initial position")
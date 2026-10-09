import math
import random
import statistics
import threading
import time

import cv2
import numpy as np

from olo import Client
from olo.core.images import Image
from olo.kinematics import JointTarget
from olo.spatial import Point, Pose, Quaternion
from olo.vision import camera_info_to_intrinsics

# @olo-playground: autofill-sim-topics

RGB_TOPIC = "wrist_rgbd/image"
CAMERA_INFO_TOPIC = "wrist_rgbd/camera_info"
OUTPUT_TOPIC = "wrist_rgbd/annotated"   # add this as a camera panel to see what is detected
ANNOTATE_MAX_FPS = 8
TAG_FAMILY = "aruco_4x4_50"
MARKER_SIZE_M = 0.036
CUBE_HALF_SIZE_M = 0.025
BOX_SIZE_M = 2 * CUBE_HALF_SIZE_M


# ---------------- Positions (radians) ----------------
# Initial position: the robot starts here (moved there if it is not) and returns here at the end.
INITIAL_JOINTS = {
    "joint1": -0.0040,
    "joint2": -0.0013,
    "joint3": -0.0055,
    "joint4": 0.0078,
    "joint5": -0.0227,
    "joint6": -0.0090,
}
INITIAL_FINGER_GAP_M = 0.0004   # recorded: finger_left 0.0002, finger_right -0.0002

# Detection position: the camera looks over the table from here. It is also the "home" the
# robot returns to between cycles, and where every search starts.
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
OPEN_FINGER_GAP_M = 0.10        # gap used whenever the gripper opens (grasping / releasing)
JOINT_TOLERANCE_RAD = 0.01      # how close counts as "already there" (~0.6 deg)
JOINT_SPEED = 0.9               # speed for joint moves (initial/detection), 0-1; 1.0 = full speed

# ----------------------------------------------------------------------
# Endless sorting loop
# ----------------------------------------------------------------------
NUM_BOXES = 2                  # how many boxes are on the table (the code also works for 3+)
START_SIDE = "left"            # first side to stack on; after that it alternates left/right
MAX_CYCLES = None              # None = run forever (stop with Ctrl+C); or a number of stack moves
PAUSE_BETWEEN_CYCLES_S = 0.5
MAX_CONSECUTIVE_FAILURES = 3   # stop if this many cycles in a row fail even after recovery
INITIAL_SEARCH_TIMEOUT_S = 180.0

# Stack spots: this far from the base, this many degrees left (+Y) and right (-Y) of straight
# ahead. ASSUMES the planning frame origin is the robot base, +X forward, +Y to the left.
STACK_RADIUS_M = 0.35
STACK_ANGLE_DEG = 50.0
STACK_MATCH_RADIUS_M = 0.04    # a box this close to the spot (or to the box below) counts as stacked
STACK_BLOCK_RADIUS_M = 0.09    # a box this close to an empty spot is in the way: it is used as the base
STACK_TOLERANCE_M = 0.010      # top box more than this off-centre from the box below -> place it again
MAX_REPLACE_ATTEMPTS = 1

# Optional: before every sort (except the first), put the boxes down at random spots in
# front of the robot, so each cycle is "random -> left", "random -> right", ...
# False: "random (as you set them up) -> left -> right -> left -> ..."
SCATTER_BEFORE_EACH_SORT = False
SCATTER_ANGLE_DEG = 30.0       # random spots between -30 and +30 degrees (away from the stack spots)
SCATTER_RADII_M = (0.28, 0.42)
SCATTER_MIN_SPACING_M = 0.12

# Search pattern: sweep an arc around the robot base.
SCAN_ANGLE_MIN_DEG = -90.0
SCAN_ANGLE_MAX_DEG = 90.0
SCAN_ANGLE_STEP_DEG = 20.0
SCAN_HOVER_Z_M = 0.25
SCAN_RADII_M = (0.30, 0.40, 0.50)
SCAN_RADIUS_OFFSETS_M = (0.0, -0.06, 0.05, -0.12, 0.10)
SCAN_GRID_STEP_M = 0.10
SCAN_GRID_RADIUS_M = 0.30
SCAN_SETTLE_S = 0.4

SAME_CUBE_DISTANCE_M = 0.10    # tags this close (horizontally) to a known box are faces of that box...
SAME_CUBE_DZ_M = 0.025         # ...if also this close in height (keeps stacked boxes apart)
REASSIGN_MAX_M = 0.30          # a box that seems to have moved is matched only within this distance
HELD_EXCLUSION_M = 0.10        # tags this close to the gripper while carrying belong to the held cube

# Tool tilt (degrees) to try, in order. 180 = tool pointing straight down.
PITCH_CANDIDATES_DEG = (180.0, 165.0, 150.0, 135.0)

VIEW_HEIGHT_M = 0.20        # camera height above the cube top while refining its position
APPROACH_HEIGHT_M = 0.10    # height above the grasp pose to approach from
# Shift of the grasp point along the tool's x and y axes (metres). Tune it if the fingers
# land beside the cube (see STOP_AFTER_APPROACH). This directly affects stacking precision.
GRASP_SHIFT_TOOL_M = (0.0, 0.0)
STOP_AFTER_APPROACH = False
REFINE_ITERATIONS = 2
# A second close look is skipped when the first one moved the estimate by less than this.
REFINE_CONVERGED_M = 0.003
GRASP_CLEARANCE_M = 0.01
GRASP_HEIGHT_RETRIES_M = (0.0, 0.01, 0.02, 0.03)
# Tag-based height estimates are noisy by about 1 cm. If all cubes sit on a flat surface,
# set this to the cube CENTRE height in the planning frame (surface height + 0.025). This is
# the best way to get soft, exact placing. None = estimate the table height from the tags.
CUBE_CENTRE_Z_M = None
# Straight-line moves try these speed scalings in order (1.0 = full speed); slower ones are
# only used when the planner rejects a path as too fast.
LINEAR_SPEED_SCALES = (0.8, 0.5, 0.25, 0.1)
COMFORTABLE_RADIUS_M = 0.50
LIFT_HEIGHT_M = 0.12
CARRY_CLEARANCE_M = 0.05   # carry height above the placing height

# Gentle placing: lower normally to PLACE_SLOW_ZONE_M above the final height, then creep the
# last bit slowly, stop PLACE_CONTACT_GAP_M above the surface, wait, then open the gripper.
PLACE_SLOW_ZONE_M = 0.02
PLACE_SLOW_SPEEDS = (0.2, 0.1, 0.05)
PLACE_CONTACT_GAP_M = 0.002
RELEASE_SETTLE_S = 0.4
SETTLE_S = 0.25


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


def arm_target(arm, joints):
    """Build a JointTarget containing only the arm's planning-group joints."""
    group_joints = arm.model.group().joints
    names = tuple(name for name in group_joints if name in joints)
    missing = [name for name in group_joints if name not in joints]
    if missing:
        raise ValueError(f"Target is missing arm joints: {missing}")
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
    """Joint move to a named position (only arm joints: the gripper keeps its state)."""
    check_joint_limits(arm, joints)
    print(f"Moving to {label} position...")
    arm.move_joints(arm_target(arm, joints), speed_scaling=JOINT_SPEED)
    print(f"Reached {label} position")


def set_finger_gap(arm, gap_m):
    """Open the fingers to a total gap (metres), enforcing the 10 cm rule."""
    gripper = arm.ee
    if gripper is None:
        print("No gripper found; skipping fingers")
        return

    # RULE: never exceed the maximum gap.
    if gap_m > MAX_FINGER_GAP_M:
        print(f"Requested gap {gap_m * 100:.1f} cm exceeds the {MAX_FINGER_GAP_M * 100:.0f} cm limit; clamping")
        gap_m = MAX_FINGER_GAP_M
    gap_m = max(gap_m, 0.0)

    # The fingers are mirrored (left = +x, right = -x), so each finger moves half the gap.
    per_finger = gap_m / 2.0

    # Also respect the gripper's own hardware range from the robot model.
    info = arm.model.end_effector_for(arm.model.group().name).gripper
    lo, hi = sorted((info.min_position, info.max_position))
    if hi > lo and not (lo <= per_finger <= hi):
        clamped = min(max(per_finger, lo), hi)
        print(
            f"Per-finger {per_finger:.4f} m is outside gripper range [{lo:.4f}, {hi:.4f}]; "
            f"using {clamped:.4f} m (gap {clamped * 200:.1f} cm)"
        )
        per_finger = clamped

    result = gripper.goal(per_finger)
    print(f"Finger gap set to ~{per_finger * 200:.1f} cm (reached_goal={result.reached_goal})")

    # Verify the measured gap from joint states, if the fingers are reported there.
    current = arm.joints().position_by_name
    if "finger_left" in current and "finger_right" in current:
        measured = abs(current["finger_left"] - current["finger_right"])
        if measured > MAX_FINGER_GAP_M + 0.002:
            gripper.goal(MAX_FINGER_GAP_M / 2.0)
            raise RuntimeError(f"Finger gap {measured * 100:.1f} cm exceeded the 10 cm rule; corrected")


def hover_poses(origin):
    """Hover poses over the workspace at SCAN_HOVER_Z_M, using `origin`'s tool orientation."""
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
    """Labelled scan poses: an arc around the base first, then a grid around `origin`."""
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


def tool_orientation(azimuth, cube_yaw, pitch, turn):
    """Tool orientation for a top grasp.

    `pitch` = 180 deg is straight down; smaller values tilt the tool outward, away from
    the base (azimuth = direction from the base to the cube). The tool x axis is lined up
    with a cube edge; `turn` picks one of the four 90-degree equivalents.
    """
    k = azimuth - cube_yaw - turn * math.pi / 2
    return (
        Quaternion.from_rpy(0.0, 0.0, azimuth)
        * Quaternion.from_rpy(0.0, pitch, 0.0)
        * Quaternion.from_rpy(0.0, 0.0, k)
    )


def orientation_candidates(position, cube_yaw, reference):
    """Orientations to try, most vertical first; the two turns closest to `reference` for each tilt."""
    azimuth = math.atan2(position.y, position.x)
    result = []
    for pitch_deg in PITCH_CANDIDATES_DEG:
        pitch = math.radians(pitch_deg)
        options = [tool_orientation(azimuth, cube_yaw, pitch, turn) for turn in range(4)]
        options.sort(key=lambda q: -abs(quat_dot(q, reference)))
        result.extend((pitch_deg, q) for q in options[:2])
    return result


def stack_spot(side, z=0.0):
    sign = 1.0 if side == "left" else -1.0
    a = sign * math.radians(STACK_ANGLE_DEG)
    return Point(x=STACK_RADIUS_M * math.cos(a), y=STACK_RADIUS_M * math.sin(a), z=z)


def other_side(side):
    return "right" if side == "left" else "left"


with Client() as client:
    robot = client.robot()
    arm = robot.kinematics
    gripper = arm.ee
    if gripper is None:
        raise RuntimeError("This example requires a gripper end effector")

    rgb_topic = robot.core.topic(RGB_TOPIC, "sensor_msgs/msg/Image")
    info_topic = robot.core.topic(CAMERA_INFO_TOPIC, "sensor_msgs/msg/CameraInfo")
    intrinsics = camera_info_to_intrinsics(robot.core.get_latest(info_topic))
    detector = robot.vision.tags(TAG_FAMILY, size_m=MARKER_SIZE_M)

    output_topic = robot.core.topic(OUTPUT_TOPIC, "sensor_msgs/msg/Image")
    publisher = robot.core.publisher(output_topic)
    annotate_stop = threading.Event()

    def annotate_loop():
        """Publish the camera image with detected tags (outline, id, camera-frame distance) drawn on it."""
        view_detector = robot.vision.tags(TAG_FAMILY, size_m=MARKER_SIZE_M)
        while not annotate_stop.is_set():
            try:
                rgb = robot.core.get_image(rgb_topic)
                canvas = np.ascontiguousarray(rgb.array).copy()
                frame = view_detector.detect(rgb, intrinsics=intrinsics)
                for tag in frame.tags:
                    pts = np.array([(c.x, c.y) for c in tag.corners], np.int32)
                    cv2.polylines(canvas, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
                    cx, cy = pts.mean(axis=0).astype(int)
                    cv2.circle(canvas, (int(cx), int(cy)), 3, (0, 0, 255), -1)
                    cv2.putText(canvas, f"id {tag.id}", (int(pts[:, 0].min()), max(int(pts[:, 1].min()) - 6, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(canvas, f"{len(frame.tags)} tag(s)", (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                publisher.publish(Image(array=canvas, encoding=rgb.encoding, header=rgb.header))
            except Exception as exc:
                print(f"Annotation frame failed: {exc}")
                time.sleep(1.0)
            time.sleep(1.0 / ANNOTATE_MAX_FPS)

    threading.Thread(target=annotate_loop, daemon=True).start()
    print(f"Publishing annotated frames to {OUTPUT_TOPIC} - add it as a camera panel.")

    # cube name -> {"centre": Point, "yaw": float} in the planning frame (latest estimate)
    known = {}
    frozen = set()  # cubes whose estimate must not change (the one in the gripper)
    holding = {"v": False, "name": None}
    # Centre height of a box lying on the table; set after the boxes are found and then
    # refined from every observation of a box on the table.
    state = {"table_z": CUBE_CENTRE_Z_M}
    table_samples = []

    # ------------------------------------------------------------------
    # Box heights / levels
    # ------------------------------------------------------------------
    def box_centre_z(level):
        return state["table_z"] + BOX_SIZE_M * level

    def level_of(name):
        if state["table_z"] is None:
            return 0
        return max(0, int(round((known[name]["centre"].z - state["table_z"]) / BOX_SIZE_M)))

    def xy_dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def is_clear(name):
        """True if no other box sits on top of `name`."""
        c = known[name]["centre"]
        lvl = level_of(name)
        return not any(
            n != name and level_of(n) > lvl and xy_dist(known[n]["centre"], c) < 1.2 * BOX_SIZE_M
            for n in known
        )

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------
    def tag_centre(tag):
        return tag.to_frame(robot.frames, arm.planning_frame).apply(Point(z=-CUBE_HALF_SIZE_M))

    def cube_of(tag, c):
        """Which box a tag is on: the nearest known box (close in xy AND height), a new box
        while not all boxes are known yet, else None."""
        best, best_d = None, SAME_CUBE_DISTANCE_M
        for n, est in known.items():
            k = est["centre"]
            if abs(c.z - k.z) > SAME_CUBE_DZ_M:
                continue  # different level: e.g. the box below/above in a stack
            d = math.hypot(c.x - k.x, c.y - k.y)
            if d < best_d:
                best, best_d = n, d
        if best is not None:
            return best
        if len(known) < NUM_BOXES:
            name = f"box_{len(known) + 1}"
            known[name] = {"centre": c, "yaw": 0.0}  # placeholder, replaced by update_estimate
            print(f"New box found: {name} (tag {tag.id})")
            return name
        return None

    def update_estimate(name, tags):
        """Estimate the cube centre and yaw from ALL tags currently seen on it."""
        centres = []
        yaw_samples = []
        for t in tags:
            tp = t.to_frame(robot.frames, arm.planning_frame)
            centres.append(tp.apply(Point(z=-CUBE_HALF_SIZE_M)))
            origin = tp.apply(Point())
            for axis in (Point(x=1.0), Point(y=1.0), Point(z=1.0)):
                p = tp.apply(axis)
                dx, dy = p.x - origin.x, p.y - origin.y
                if math.hypot(dx, dy) > 0.7:  # this tag axis lies roughly horizontal
                    yaw_samples.append(math.atan2(dy, dx))
        n = len(centres)
        z_raw = sum(c.z for c in centres) / n
        if CUBE_CENTRE_Z_M is not None:
            lvl = max(0, int(round((z_raw - CUBE_CENTRE_Z_M) / BOX_SIZE_M)))
            z = CUBE_CENTRE_Z_M + BOX_SIZE_M * lvl
        else:
            z = z_raw
            tz = state["table_z"]
            if tz is not None and abs(z_raw - tz) < CUBE_HALF_SIZE_M:
                # a box lying on the table: use it to refine the table height
                table_samples.append(z_raw)
                del table_samples[:-50]
                state["table_z"] = statistics.median(table_samples)
        centre = Point(x=sum(c.x for c in centres) / n, y=sum(c.y for c in centres) / n, z=z)
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

    def detect_by_cube():
        """Return {cube_name: [tags]} and update the position estimate of every visible cube."""
        frame = detector.detect(robot.core.get_image(rgb_topic), intrinsics=intrinsics)
        if frame.tags:
            print("Tag IDs in view:", sorted(t.id for t in frame.tags))
        ee = current_pose().position if holding["v"] else None
        result = {}
        unmatched = []
        for tag in frame.tags:
            if tag.transform is None:
                continue
            c = tag_centre(tag)
            if ee is not None and math.dist((c.x, c.y, c.z), (ee.x, ee.y, ee.z)) < HELD_EXCLUSION_M:
                continue  # that is the cube in the gripper
            name = cube_of(tag, c)
            if name is None:
                unmatched.append((tag, c))
            else:
                result.setdefault(name, []).append(tag)

        # Tags that match no known box although all boxes are known: a box has moved
        # (bumped or fallen). Give them to the nearest box that is not in view where we expect it.
        groups = []
        for tag, c in unmatched:
            for g in groups:
                if xy_dist(g["c"], c) < SAME_CUBE_DISTANCE_M and abs(g["c"].z - c.z) < SAME_CUBE_DZ_M:
                    g["tags"].append(tag)
                    break
            else:
                groups.append({"c": c, "tags": [tag]})
        for g in groups:
            c = g["c"]
            candidates = [
                n for n in known
                if n not in frozen and n not in result
                and xy_dist(known[n]["centre"], c) < REASSIGN_MAX_M
            ]
            if not candidates:
                print(f"Ignoring tags {[t.id for t in g['tags']]}: they match no tracked box")
                continue
            n = min(candidates, key=lambda m: xy_dist(known[m]["centre"], c))
            print(f"{n} seems to have moved; updating its position")
            result[n] = g["tags"]

        for name, tags in result.items():
            if name not in frozen:
                update_estimate(name, tags)
        return result

    def scan_until(done, origin, avoid_obstacles, timeout=None):
        """Move the camera around `origin` until `done()` is True. Returns True on success."""
        deadline = None if timeout is None else time.monotonic() + timeout
        o = origin.position
        print(f"Scan origin: ({o.x:.3f}, {o.y:.3f}, {o.z:.3f}), radius from base {math.hypot(o.x, o.y):.3f} m")
        for label, pose in hover_poses(origin) + scan_poses(origin):
            if deadline is not None and time.monotonic() >= deadline:
                print(f"Search timed out after {timeout:.0f} s")
                return False
            moved = False
            error = None
            t0 = time.monotonic()
            print(f"Moving to {label} ...")
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
            time.sleep(SCAN_SETTLE_S)
            detect_by_cube()
            if done():
                return True
        return False

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------
    def current_pose():
        return robot.frames.lookup(arm.planning_frame, arm.end_effector_link).as_pose()

    def move_over(position, cube_yaw, avoid_obstacles, shift=False):
        """Move the tool to `position` with the most vertical orientation the arm can reach.

        With shift=True the target is moved by GRASP_SHIFT_TOOL_M along the tool x/y axes
        (used for grasping/placing, not for the camera view). Returns the pose reached.
        """
        last_error = None
        for pitch_deg, orientation in orientation_candidates(position, cube_yaw, current_pose().orientation):
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
            f"radius {math.hypot(position.x, position.y):.3f} m from the base. Last error: {last_error}"
        )

    def move_linear_safe(pose, scales=LINEAR_SPEED_SCALES):
        """Straight-line move, trying the given speed scalings in order."""
        last_error = None
        for scale in scales:
            try:
                return arm.move_linear(pose, speed_scaling=scale)
            except TypeError:
                print("move_linear has no speed_scaling argument in this SDK version")
                return arm.move_linear(pose)
            except Exception as exc:
                last_error = exc
                print(f"Straight-line move failed at speed scaling {scale}: {exc}")
        raise last_error

    def move_linear_or_normal(pose, scales=LINEAR_SPEED_SCALES, what="move"):
        try:
            move_linear_safe(pose, scales)
        except Exception as exc:
            print(f"Straight-line {what} failed ({exc}); using a normal move")
            arm.move_pose(pose, avoid_obstacles=False)

    home_pose = None  # end-effector pose at DETECTION_JOINTS, filled in after the first move

    def open_gripper():
        set_finger_gap(arm, OPEN_FINGER_GAP_M)  # never wider than the 10 cm rule

    def go_home(avoid_obstacles):
        """Move to the detection position (fallback: its Cartesian pose); return the pose reached."""
        errors = []
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
                    print("At home (Cartesian pose)")
                    return home_pose
                except Exception as exc:
                    errors.append(f"home pose: {exc}")
        raise RuntimeError(
            "Could not reach the home position (" + "; ".join(errors) + "). "
            "The arm may be in a bad state: reset the simulation and retry."
        )

    def go_initial():
        """Return to the initial position with the recorded finger gap (end of the program)."""
        if holding["v"]:
            # Do not squeeze the fingers shut on a held box; just bring the arm back.
            print(f"WARNING: {holding['name']} is still in the gripper; not changing the finger gap")
        move_to(arm, INITIAL_JOINTS, "initial")
        if not holding["v"]:
            set_finger_gap(arm, INITIAL_FINGER_GAP_M)

    # 1. Make sure we start at the initial position.
    print("Checking initial position...")
    if is_at(arm, INITIAL_JOINTS):
        print("Already at the initial position")
    else:
        move_to(arm, INITIAL_JOINTS, "initial")
    set_finger_gap(arm, INITIAL_FINGER_GAP_M)

    # 2. Go to the detection position with a 10 cm finger gap.
    home_pose = go_home(avoid_obstacles=True)
    set_finger_gap(arm, DETECTION_FINGER_GAP_M)
    hp = home_pose.position
    print(f"Detection pose: ({hp.x:.3f}, {hp.y:.3f}, {hp.z:.3f}), orientation {home_pose.orientation.array()}")

    # ------------------------------------------------------------------
    # Pick / place helpers
    # ------------------------------------------------------------------
    def refine(name, iterations=REFINE_ITERATIONS, avoid_obstacles=True):
        """Look at box `name` from above to improve its estimate. Returns True if it was seen."""
        seen = False
        prev = None
        for i in range(iterations):
            est = known[name]
            c = est["centre"]
            if prev is not None and xy_dist(c, prev) < REFINE_CONVERGED_M:
                break  # the estimate has settled: another look would only cost time
            prev = c
            view = Point(x=c.x, y=c.y, z=c.z + CUBE_HALF_SIZE_M + VIEW_HEIGHT_M)
            try:
                move_over(view, est["yaw"], avoid_obstacles=avoid_obstacles)
            except RuntimeError as exc:
                print(f"Could not get a closer view of {name}: {exc}")
                break
            time.sleep(SCAN_SETTLE_S)
            if name not in detect_by_cube():
                print(f"{name} not visible from the closer view; using the previous estimate")
                break
            seen = True
        return seen

    def pick_box(name):
        """Look closer at box `name`, approach it from above, grasp it and lift it.

        Returns (pick_pose, lift_pose).
        """
        refine(name, avoid_obstacles=True)

        est = known[name]
        c = est["centre"]
        level = level_of(name)
        cz = box_centre_z(level)  # snap to the exact level height: consistent grasp depth
        grasp_z = cz + CUBE_HALF_SIZE_M - GRASP_CLEARANCE_M
        print(
            f"Final estimate for {name}: ({c.x:.3f}, {c.y:.3f}), level {level}, centre z {cz:.3f} "
            f"(measured {c.z:.3f}), yaw {math.degrees(est['yaw']):+.0f} deg"
        )
        radius = math.hypot(c.x, c.y)
        if radius > COMFORTABLE_RADIUS_M:
            print(f"WARNING: {name} is {radius:.2f} m from the base, near the edge of the arm's reach.")

        approach_pose = move_over(
            Point(x=c.x, y=c.y, z=grasp_z + APPROACH_HEIGHT_M), est["yaw"], avoid_obstacles=True, shift=True
        )
        time.sleep(SETTLE_S)
        if STOP_AFTER_APPROACH:
            raise SystemExit("Stopped above the box (STOP_AFTER_APPROACH). Check that the box is between the fingers.")
        gx, gy = approach_pose.position.x, approach_pose.position.y

        pick_pose = None
        for extra in GRASP_HEIGHT_RETRIES_M:
            candidate = Pose(position=Point(x=gx, y=gy, z=grasp_z + extra), orientation=approach_pose.orientation)
            try:
                move_linear_safe(candidate)
                pick_pose = candidate
                break
            except Exception as exc:
                print(f"Descent to grasp height z={grasp_z + extra:.3f} m failed ({exc}); retrying higher")
                try:
                    arm.move_pose(approach_pose, avoid_obstacles=False)
                except Exception as back_exc:
                    print(f"Could not return to the approach pose: {back_exc}")
                time.sleep(SETTLE_S)
        if pick_pose is None:
            candidate = Pose(position=Point(x=gx, y=gy, z=grasp_z + 0.01), orientation=approach_pose.orientation)
            try:
                arm.move_pose(candidate, avoid_obstacles=False)
                pick_pose = candidate
            except Exception as exc:
                print(f"Normal descent failed too: {exc}")
        if pick_pose is None:
            raise RuntimeError(
                f"Could not descend to {name}. Check the table height and set CUBE_CENTRE_Z_M "
                f"(current table estimate: box centre z = {state['table_z']:.3f} m)"
            )
        time.sleep(SETTLE_S)
        gripper.close()
        time.sleep(SETTLE_S)
        frozen.add(name)
        holding["v"] = True
        holding["name"] = name

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
            raise RuntimeError(f"Could not lift {name}: it is probably at the edge of the arm's reach")
        time.sleep(SETTLE_S)
        return pick_pose, lift_pose

    def place_box(x, y, yaw, level, lift_z, label):
        """Gently put the carried box down at (x, y) on the given level (0 = table).

        The box is lowered normally to PLACE_SLOW_ZONE_M above its final height, then slowly
        until it is PLACE_CONTACT_GAP_M above the surface, held still, and only then released.
        """
        place_ee_z = (
            box_centre_z(level) + CUBE_HALF_SIZE_M - GRASP_CLEARANCE_M + PLACE_CONTACT_GAP_M
        )
        slow_z = place_ee_z + PLACE_SLOW_ZONE_M
        carry_z = max(lift_z, place_ee_z + CARRY_CLEARANCE_M)
        above = move_over(Point(x=x, y=y, z=carry_z), yaw, avoid_obstacles=False, shift=True)
        time.sleep(SETTLE_S)
        ax, ay, q = above.position.x, above.position.y, above.orientation
        pre_pose = Pose(position=Point(x=ax, y=ay, z=slow_z), orientation=q)
        place_pose = Pose(position=Point(x=ax, y=ay, z=place_ee_z), orientation=q)

        if carry_z - slow_z > 0.001:
            move_linear_or_normal(pre_pose, what="lowering")
        move_linear_or_normal(place_pose, scales=PLACE_SLOW_SPEEDS, what="slow lowering")
        time.sleep(SETTLE_S)          # let the arm come fully to rest before letting go
        open_gripper()
        holding["v"] = False
        holding["name"] = None
        time.sleep(RELEASE_SETTLE_S)  # let the box settle before the fingers move away

        try:  # leave slowly so the fingers do not drag the box
            move_linear_or_normal(pre_pose, scales=PLACE_SLOW_SPEEDS, what="slow retreat")
            move_linear_or_normal(above, what="retreat")
        except Exception as exc:
            print(f"Retreat failed: {exc}")
        print(f"Placed the box ({label}) at ({x:.3f}, {y:.3f}), level {level}")
        return above

    def set_placed(name, x, y, level, yaw):
        """Record where a box was just put down; vision refines it afterwards."""
        frozen.discard(name)
        known[name] = {"centre": Point(x=x, y=y, z=box_centre_z(level)), "yaw": yaw}

    # ------------------------------------------------------------------
    # Stack logic
    # ------------------------------------------------------------------
    def stack_members(spot):
        """Boxes forming a stack at `spot`, bottom first."""
        members = []
        ref = spot
        for level in range(NUM_BOXES):
            found = [
                n for n in known
                if n not in members and level_of(n) == level
                and xy_dist(known[n]["centre"], ref) <= STACK_MATCH_RADIUS_M
            ]
            if not found:
                break
            n = min(found, key=lambda m: xy_dist(known[m]["centre"], ref))
            members.append(n)
            ref = known[n]["centre"]
        return members

    def choose_mover(spot, stack, level):
        """The next box to put on the stack: a free box, a box in the way first, top boxes first."""
        candidates = [n for n in known if n not in stack and is_clear(n)]
        if not candidates:
            raise RuntimeError("No free box to move onto the stack")

        def key(n):
            d = xy_dist(known[n]["centre"], spot)
            blocking = level == 0 and d < STACK_BLOCK_RADIUS_M
            return (not blocking, -level_of(n), d)

        return min(candidates, key=key)

    def build_level(spot, level, side, redo=None):
        """Make sure there is a box at `level` of the stack at `spot`.

        Returns None if it was already there, else (mover, offset from the box below or None).
        """
        stack = stack_members(spot)
        if redo is None and len(stack) > level:
            print(f"Level {level} on the {side} already holds {stack[level]}")
            return None
        stack = stack[:level]
        if len(stack) < level:
            raise RuntimeError(f"The {side} stack is missing a box below level {level}")
        mover = redo or choose_mover(spot, stack, level)

        if level == 0:
            tx, ty, yaw = spot.x, spot.y, 0.0
            below = None
        else:
            below = stack[-1]
            refine(below, avoid_obstacles=False)  # precise position of the box we stack on
            b = known[below]
            tx, ty, yaw = b["centre"].x, b["centre"].y, b["yaw"]

        print(f"Moving {mover} to the {side} stack, level {level}")
        _, lift_pose = pick_box(mover)
        place_box(tx, ty, yaw, level, lift_pose.position.z, f"{side} stack")
        set_placed(mover, tx, ty, level, yaw)
        refine(mover, iterations=1, avoid_obstacles=False)  # check where it really ended up

        if below is None:
            return mover, None
        offset = xy_dist(known[mover]["centre"], known[below]["centre"])
        print(f"{mover} sits {offset * 1000:.1f} mm off-centre on {below}")
        if level_of(mover) != level:
            print(f"WARNING: {mover} is at level {level_of(mover)}, expected {level} (did it fall?)")
        return mover, offset

    def sort_to(side):
        spot = stack_spot(side, state["table_z"])
        print(f"Stacking all boxes on the {side} at ({spot.x:.2f}, {spot.y:.2f})")
        for level in range(NUM_BOXES):
            redo = None
            for attempt in range(1 + MAX_REPLACE_ATTEMPTS):
                result = build_level(spot, level, side, redo)
                if result is None:
                    break
                mover, offset = result
                if offset is None or offset <= STACK_TOLERANCE_M:
                    break
                if attempt < MAX_REPLACE_ATTEMPTS:
                    print(f"Off-centre by more than {STACK_TOLERANCE_M * 1000:.0f} mm; placing {mover} again")
                    redo = mover
                else:
                    print(f"WARNING: {mover} is still off-centre; continuing")
        stack = stack_members(spot)
        if len(stack) < NUM_BOXES:
            raise RuntimeError(f"The {side} stack has {len(stack)} of {NUM_BOXES} boxes after sorting")
        print(f"All {NUM_BOXES} boxes are stacked on the {side}: {stack}")

    def random_free_spot(moving):
        keep_out = [k["centre"] for n, k in known.items() if n != moving]
        keep_out += [stack_spot("left"), stack_spot("right")]
        for _ in range(500):
            a = math.radians(random.uniform(-SCATTER_ANGLE_DEG, SCATTER_ANGLE_DEG))
            r = random.uniform(*SCATTER_RADII_M)
            p = Point(x=r * math.cos(a), y=r * math.sin(a), z=state["table_z"])
            if all(xy_dist(p, o) >= SCATTER_MIN_SPACING_M for o in keep_out):
                return p
        raise RuntimeError("No free random spot found; reduce SCATTER_MIN_SPACING_M or widen the area")

    def scatter():
        print("Putting the boxes down at random spots")
        for name in sorted(known, key=lambda n: -level_of(n)):  # top boxes first
            spot = random_free_spot(name)
            yaw = math.radians(random.uniform(-45.0, 45.0))
            _, lift_pose = pick_box(name)
            place_box(spot.x, spot.y, yaw, 0, lift_pose.position.z, "random spot")
            set_placed(name, spot.x, spot.y, 0, yaw)

    def recover():
        """After an error: put down a held box, go home, and re-check every box."""
        name = holding["name"]
        if holding["v"] and name is not None:
            print(f"Still holding {name}: setting it down at a free spot")
            try:
                spot = random_free_spot(name)
                place_box(spot.x, spot.y, 0.0, 0, current_pose().position.z, "set down after an error")
                set_placed(name, spot.x, spot.y, 0, 0.0)
            except Exception as exc:
                raise RuntimeError(f"Could not set the held box down ({exc}); reset the simulation") from exc
        go_home(avoid_obstacles=False)
        for n in list(known):
            try:
                refine(n, iterations=1, avoid_obstacles=False)
            except Exception as exc:
                print(f"Could not re-check {n}: {exc}")
        go_home(avoid_obstacles=False)

    # ------------------------------------------------------------------
    # 1. Find all boxes (they start wherever they were put).
    # ------------------------------------------------------------------
    print(f"Searching for {NUM_BOXES} boxes...")
    time.sleep(SCAN_SETTLE_S)
    detect_by_cube()
    if len(known) < NUM_BOXES:
        scan_until(lambda: len(known) >= NUM_BOXES, home_pose, avoid_obstacles=True,
                   timeout=INITIAL_SEARCH_TIMEOUT_S)
    if len(known) < NUM_BOXES:
        raise RuntimeError(f"Found only {len(known)} of {NUM_BOXES} boxes; widen the scan range")
    if state["table_z"] is None:
        zmin = min(k["centre"].z for k in known.values())
        table_samples.extend(k["centre"].z for k in known.values() if k["centre"].z - zmin < CUBE_HALF_SIZE_M)
        state["table_z"] = statistics.median(table_samples)
    print(f"Table-level box centre height: {state['table_z']:.3f} m")
    home_pose = go_home(avoid_obstacles=True)

    # ------------------------------------------------------------------
    # 2. Endless loop: stack everything on one side, then on the other side, ...
    # ------------------------------------------------------------------
    side = START_SIDE.lower()
    if side not in ("left", "right"):
        raise ValueError("START_SIDE must be 'left' or 'right'")
    cycle = 0
    failures = 0
    try:
        while MAX_CYCLES is None or cycle < MAX_CYCLES:
            cycle += 1
            print(f"\n========== Cycle {cycle}: sorting to the {side.upper()} ==========")
            try:
                if SCATTER_BEFORE_EACH_SORT and cycle > 1:
                    scatter()
                sort_to(side)
                home_pose = go_home(avoid_obstacles=True)
                failures = 0
                side = other_side(side)  # next cycle goes to the other side
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                failures += 1
                print(f"Cycle {cycle} failed ({failures}/{MAX_CONSECUTIVE_FAILURES}): {exc}")
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    raise
                recover()  # then retry the same side
            time.sleep(PAUSE_BETWEEN_CYCLES_S)
        print(f"Finished {cycle} cycles")
    except KeyboardInterrupt:
        print("\nStopped by the user")
    finally:
        annotate_stop.set()
        # 3. Whatever happened (finished, Ctrl+C, too many failures): back to the initial position.
        try:
            go_home(avoid_obstacles=False)
            go_initial()
            print("Done: back at the initial position")
        except Exception as exc:
            print(f"Could not return to the initial position: {exc}")
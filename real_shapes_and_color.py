"""Detect ArUco-marked cubes on the wrist camera and publish annotated frames."""

import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from olo import Client, OloError, OloTimeoutError, OloUnavailable
from olo.core.images import Image, compressed_message_to_frame
from olo.kinematics import JointTarget
from olo.spatial import Point
from olo.vision import camera_info_to_intrinsics, project

# Camera topics differ between the simulation and the real robot.
USE_REAL_ROBOT = True       # False = simulation
if USE_REAL_ROBOT:
    RGB_TOPIC = "camera/color/image_raw/compressed"
    CAMERA_INFO_TOPIC = "camera/color/camera_info"   # assumed; check your topic list
    OUTPUT_TOPIC = "camera/annotated"
else:
    RGB_TOPIC = "wrist_rgbd/image/compressed"
    CAMERA_INFO_TOPIC = "wrist_rgbd/camera_info"
    OUTPUT_TOPIC = "wrist_rgbd/annotated"
MAX_FPS = 10
NO_FRAMES_WARNING_S = 5.0   # warn if the camera topic stays silent this long

# The cubes carry one DICT_4X4_50 marker per face (IDs 0-5: front, right,
# back, left, top, bottom). Several cubes share the same IDs, so markers are
# grouped into cubes by how close together they are in the image.
TAG_FAMILY = "aruco_4x4_50"
MARKER_SIZE_M = 0.036       # black outer edge of each marker
CUBE_SIZE_M = 0.050         # cube edge; the white margin fills the rest of the face
FACE_BY_ID = {0: "front", 1: "right", 2: "back", 3: "left", 4: "top", 5: "bottom"}
SAME_CUBE_FACTOR = 1.6      # markers closer than this many marker-widths are one cube
SMOOTHING = 0.4             # 0..1; lower = steadier outline, higher = follows motion faster
MAX_MATCH_DISTANCE_PX = 40  # a cube that jumps further than this between frames is treated as new
DRAW_WIREFRAME = False      # True also draws all 12 cube edges inside the outline

# The 8 cube corners in a face marker's frame. The marker lies on the z = 0
# face with +z pointing out of the cube, so the cube extends back to z = -size.
_H = CUBE_SIZE_M / 2
CUBE_CORNERS = [
    (x, y, z)
    for z in (0.0, -CUBE_SIZE_M)
    for x, y in ((-_H, _H), (_H, _H), (_H, -_H), (-_H, -_H))
]
CUBE_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),   # observed face
              (4, 5), (5, 6), (6, 7), (7, 4),   # opposite face
              (0, 4), (1, 5), (2, 6), (3, 7)]   # sides


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


@dataclass
class Cube:
    tags: list               # every marker seen on this cube
    face: object             # the marker the cube pose is taken from
    outline_px: np.ndarray   # 8x2 projected cube corners (4x2 face corners without intrinsics)
    center_px: np.ndarray    # image centre of the cube
    center_m: Point | None   # 3D cube centre (needs camera_info), in `frame`
    frame: str = ""


def marker_area(tag):
    corners = np.array([(c.x, c.y) for c in tag.corners], np.float32)
    return cv2.contourArea(corners)


def pick_face(tags):
    """Choose the marker that gives the most reliable cube pose.

    A marker seen face-on is large and its pose is stable; a side face seen at
    a grazing angle is thin and its pose flips around. Prefer the biggest
    marker and break ties with the lower reprojection error.
    """
    posed = [tag for tag in tags if tag.transform is not None] or tags
    return max(posed, key=lambda tag: (marker_area(tag), -tag.reprojection_error_px))


def face_outline(tag):
    """Grow the marker's corners out to the cube face (marker + white margin)."""
    corners = np.array([(c.x, c.y) for c in tag.corners], np.float32)
    centre = corners.mean(axis=0)
    return centre + (corners - centre) * (CUBE_SIZE_M / MARKER_SIZE_M)


def project_cube(face, intrinsics):
    """Project the full 3D cube, posed from one marker, into the image."""
    pixels = []
    for x, y, z in CUBE_CORNERS:
        point = face.transform.apply(Point(x=x, y=y, z=z))  # camera optical frame
        if point.z <= 0:
            return None  # behind the camera; cannot draw
        pixel = project(point, intrinsics)
        pixels.append((pixel.x, pixel.y))
    return np.array(pixels, np.float32)


def group_into_cubes(tags):
    """Cluster markers whose centres are within SAME_CUBE_FACTOR marker-widths."""
    groups = []
    for tag in tags:
        size = max(tag.box.width, tag.box.height)
        for group in groups:
            if any(
                np.hypot(tag.center.x - other.center.x, tag.center.y - other.center.y)
                < SAME_CUBE_FACTOR * max(size, other.box.width, other.box.height)
                for other in group
            ):
                group.append(tag)
                break
        else:
            groups.append([tag])
    return groups


def cube_centre(face, frames, base_frame):
    """The cube centre is half a cube behind the chosen face, along the marker's -z."""
    if face.transform is None:
        return None, ""
    behind = Point(z=-CUBE_SIZE_M / 2)
    try:
        return face.to_frame(frames, base_frame, timeout=0.2).apply(behind), base_frame
    except OloError:
        # No TF at the image stamp: fall back to the camera optical frame.
        return face.transform.apply(behind), face.transform.frame_id or "camera"


def detect_cubes(robot, image, intrinsics, base_frame):
    frame = robot.vision.detect_tags(
        image,
        TAG_FAMILY,
        intrinsics=intrinsics,
        size_m=MARKER_SIZE_M if intrinsics is not None else None,
    )
    tags = [tag for tag in frame.tags if tag.id in FACE_BY_ID]
    cubes = []
    for group in group_into_cubes(tags):
        face = pick_face(group)
        outline = None
        if intrinsics is not None and face.transform is not None:
            outline = project_cube(face, intrinsics)
        if outline is None:
            outline = face_outline(face)
        centre_m, centre_frame = cube_centre(face, robot.frames, base_frame)
        cubes.append(Cube(
            tags=group,
            face=face,
            outline_px=outline,
            center_px=outline.mean(axis=0),
            center_m=centre_m,
            frame=centre_frame,
        ))
    return cubes


class CubeSmoother:
    """Match cubes to the previous frame by position and low-pass their values.

    This removes the frame-to-frame jitter of the marker pose so the drawn
    outline and the reported position stay steady while the camera is still.
    """

    def __init__(self, alpha=SMOOTHING, max_jump_px=MAX_MATCH_DISTANCE_PX):
        self.alpha = alpha
        self.max_jump_px = max_jump_px
        self.previous = []

    def update(self, cubes):
        unmatched = list(self.previous)
        for cube in cubes:
            candidates = [old for old in unmatched if old.outline_px.shape == cube.outline_px.shape]
            if not candidates:
                continue
            match = min(candidates, key=lambda old: np.linalg.norm(old.center_px - cube.center_px))
            if np.linalg.norm(match.center_px - cube.center_px) > self.max_jump_px:
                continue
            unmatched.remove(match)
            a = self.alpha
            cube.outline_px = a * cube.outline_px + (1 - a) * match.outline_px
            cube.center_px = cube.outline_px.mean(axis=0)
            if cube.center_m is not None and match.center_m is not None and cube.frame == match.frame:
                cube.center_m = Point(
                    x=a * cube.center_m.x + (1 - a) * match.center_m.x,
                    y=a * cube.center_m.y + (1 - a) * match.center_m.y,
                    z=a * cube.center_m.z + (1 - a) * match.center_m.z,
                )
        self.previous = cubes
        return cubes


def annotate(rgb: Image, cubes) -> Image:
    canvas = rgb.array.copy()
    for n, cube in enumerate(cubes, start=1):
        pts = np.round(cube.outline_px).astype(np.int32)
        if DRAW_WIREFRAME and len(pts) == 8:
            for i, j in CUBE_EDGES:
                cv2.line(canvas, tuple(map(int, pts[i])), tuple(map(int, pts[j])),
                         (0, 255, 0), 1, cv2.LINE_AA)
        # The convex hull of the projected corners is the cube's silhouette.
        cv2.polylines(canvas, [cv2.convexHull(pts)], True, (0, 255, 0), 2, cv2.LINE_AA)
        marker = np.array([(c.x, c.y) for c in cube.face.corners], np.int32)
        cv2.polylines(canvas, [marker], True, (255, 160, 0), 1, cv2.LINE_AA)
        cx, cy = np.round(cube.center_px).astype(int)
        cv2.circle(canvas, (int(cx), int(cy)), 3, (0, 0, 255), -1)

        label = f"cube {n}"
        if cube.center_m is not None:
            p = cube.center_m
            label += f" ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) m"
        top = int(pts[:, 1].min())
        left = int(pts[:, 0].min())
        cv2.putText(canvas, label, (left, max(top - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    # Status line so an empty frame is clearly "annotation running, nothing found".
    cv2.putText(canvas, f"detection: {len(cubes)} cube(s)", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return Image(array=canvas, encoding="bgr8", header=rgb.header)


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

            # Camera intrinsics turn 2D markers into 3D cube poses. Without them
            # the cubes are still detected and outlined, just without a position.
            try:
                intrinsics = camera_info_to_intrinsics(robot.core.get_latest(info_topic, timeout=3.0))
            except OloError as exc:
                intrinsics = None
                print(f"No camera_info ({exc}); detecting cubes without 3D positions.")

            frames = queue.Queue(maxsize=1)
            errors = queue.Queue(maxsize=1)
            threading.Thread(target=receive_latest,
                             args=(robot, rgb_topic, frames, errors),
                             daemon=True).start()

            smoother = CubeSmoother()
            print(f"Publishing annotated frames to {OUTPUT_TOPIC} - add it as a camera panel.")
            next_frame = time.monotonic()
            last_count = -1
            while True:
                try:
                    message = frames.get(timeout=0.1)
                except queue.Empty:
                    if not errors.empty():
                        raise errors.get()
                    continue

                now = time.monotonic()
                if now < next_frame:
                    continue
                next_frame = now + 1 / MAX_FPS

                rgb = compressed_message_to_frame(message)
                cubes = smoother.update(detect_cubes(robot, rgb, intrinsics, base_frame))
                if len(cubes) != last_count:
                    print(f"{len(cubes)} cube(s)")
                    for n, cube in enumerate(cubes, start=1):
                        seen = ", ".join(FACE_BY_ID[tag.id] for tag in cube.tags)
                        where = ""
                        if cube.center_m is not None:
                            p = cube.center_m
                            where = f" at ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) m in {cube.frame}"
                        print(f"  cube {n}: faces [{seen}], pose from {FACE_BY_ID[cube.face.id]}{where}")
                    last_count = len(cubes)
                publisher.publish(annotate(rgb, cubes))
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

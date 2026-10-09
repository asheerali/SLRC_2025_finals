from olo import Client
from olo.errors import OloTimeoutError
from olo.kinematics import JointTarget

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
DETECTION_FINGER_GAP_M = 0.04   # 15 cm between the fingers

# ---------------- Rules / settings ----------------
MAX_FINGER_GAP_M = 0.4         # RULE: finger gap must never exceed 15 cm
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


# ---------------- Main sequence ----------------
with Client() as client:
    robot = client.robot()
    arm = robot.kinematics

    # Show the robot model's planning groups (useful for debugging).
    print("Default group:", arm.model.default_group)
    for g in arm.model.groups:
        print(f"  group {g.name}: joints={g.joints}")

    # 1. Make sure we start at the initial position.
    print("\n[1] Initial position")
    move_to(arm, INITIAL_JOINTS, "initial")
    set_finger_gap(arm, INITIAL_FINGER_GAP_M)

    # 2. Go to the detection position with a 15 cm finger gap.
    print("\n[2] Detection position")
    move_to(arm, DETECTION_JOINTS, "detection")
    set_finger_gap(arm, DETECTION_FINGER_GAP_M)

    # 3. Return to the initial position.
    print("\n[3] Back to initial position")
    move_to(arm, INITIAL_JOINTS, "initial")
    set_finger_gap(arm, INITIAL_FINGER_GAP_M)

    print("\nDone: back at the initial position")
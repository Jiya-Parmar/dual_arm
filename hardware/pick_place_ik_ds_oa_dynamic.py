#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import time
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped

# -----------------------------
# Obstacle parameters
# -----------------------------
OBSTACLE_RADIUS = 0.30

# -----------------------------
# Make RC-DS root importable
# -----------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.modulation_class import ModulationAvoider
from utils.dls_velocity_commander import DLSVelocityCommander
from utils.gripper_commands.franka_gripper import FrankaGripperController

# -----------------------------
# Modulation Avoider (center-free)
# -----------------------------
avoider = ModulationAvoider(
    gamma=lambda x_tilde: np.linalg.norm(x_tilde) / (OBSTACLE_RADIUS + 1e-9),
    grad_gamma=lambda x_tilde: (
        x_tilde / (np.linalg.norm(x_tilde) + 1e-9)
    ) / (OBSTACLE_RADIUS + 1e-9),
    rho=1.0,
    safety_eta=1.0,
    tail_effect=False,
)

# -----------------------------
# Gripper
# -----------------------------
franka_gripper = FrankaGripperController()
franka_gripper.open_gripper(width=0.08)
time.sleep(1.0)

# =====================================================
# Pose listeners
# =====================================================
class InitialPoseListener(Node):
    """Latch-first pose subscriber"""

    def __init__(self, topic: str, node_name: str):
        super().__init__(node_name)
        self.pose = None
        self.create_subscription(PoseStamped, topic, self.cb, 10)

    def cb(self, msg: PoseStamped):
        if self.pose is not None:
            return
        self.pose = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=float,
        )
        self.get_logger().info(
            f"Latched pose {self.get_name()}: {self.pose}"
        )


class ObstacleTracker(Node):
    """Continuously tracks obstacle pose"""

    def __init__(self, topic: str):
        super().__init__("obstacle_tracker")
        self.center = None
        self.create_subscription(PoseStamped, topic, self.cb, 10)

    def cb(self, msg: PoseStamped):
        self.center = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=float,
        )


def wait_for_pose(topic: str, node_name: str):
    node = InitialPoseListener(topic, node_name)
    while rclpy.ok() and node.pose is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    pose = node.pose
    node.destroy_node()
    return pose


# =====================================================
# Dynamical System (DS + dynamic obstacle modulation)
# =====================================================
def pick_place_ds(current_pose: dict) -> np.ndarray:
    global robotB, obstacle_tracker

    x = current_pose["position"]

    target_kdl = current_pose.get("target_kdl", None)
    if target_kdl is None:
        return np.zeros(6)

    x_goal = np.array(
        [target_kdl.x(), target_kdl.y(), target_kdl.z()],
        dtype=float,
    )

    # 1) Nominal DS
    v_nom = x_goal - x
    n = np.linalg.norm(v_nom)
    if n > robotB.max_cartesian_vel:
        v_nom = (v_nom / n) * robotB.max_cartesian_vel

    # 2) Obstacle modulation (only if obstacle seen)
    if obstacle_tracker.center is not None:
        x_tilde = x - obstacle_tracker.center
        v_mod = avoider.modulate(x=x, x_tilde=x_tilde, f=v_nom)
    else:
        v_mod = v_nom

    # 3) Orientation ignored
    w = np.zeros(3)

    return np.concatenate([v_mod, w])


# =====================================================
# MAIN
# =====================================================
def main(args=None):
    global robotB, obstacle_tracker

    rclpy.init(args=args)

    # -----------------------------
    # Get object poses
    # -----------------------------
    pick_pose = wait_for_pose(
        "/pose/franka/blue_cuboid", "pick_pose_listener"
    )
    place_pose = wait_for_pose(
        "/pose/franka/white_mug", "place_pose_listener"
    )

    # -----------------------------
    # Cartesian targets
    # -----------------------------
    PREGRASP_Z = 0.30
    GRASP_Z = 0.18
    LIFT_Z = 0.30
    PREPLACE_Z = 0.35
    PLACE_Z = 0.30

    pregrasp_pos = [pick_pose[0], pick_pose[1], pick_pose[2] + PREGRASP_Z]
    grasp_pos    = [pick_pose[0], pick_pose[1], pick_pose[2] + GRASP_Z]
    lift_pos     = [pick_pose[0], pick_pose[1], pick_pose[2] + LIFT_Z]
    preplace_pos = [place_pose[0], place_pose[1], place_pose[2] + PREPLACE_Z]
    place_pos    = [place_pose[0], place_pose[1], place_pose[2] + PLACE_Z]

    target_quat = [0.945, -0.324, 0.005, 0.027]

    # -----------------------------
    # Controller
    # -----------------------------
    robotB = DLSVelocityCommander(
        robot_id="robotB",
        base_link="fr3_link0",
        tip_link="fr3_link8",
        joint_names=[
            "fr3_joint1", "fr3_joint2", "fr3_joint3",
            "fr3_joint4", "fr3_joint5", "fr3_joint6",
            "fr3_joint7",
        ],
        target_pos=pregrasp_pos,
        target_quat=target_quat,
        joint_state_topic="/NS_1/franka/joint_states",
        velocity_command_topic="/NS_1/joint_velocity_controller/commands",
        robot_description_topic="/NS_1/robot_description",
        ee_pose_topic=None,
        ee_pose_is_stamped=False,
        max_cartesian_vel=0.2,
        max_angular_vel=0.3,
        dt=0.01,
        damping=0.1,
        custom_ds=pick_place_ds,
    )

    # -----------------------------
    # Obstacle tracker
    # -----------------------------
    obstacle_tracker = ObstacleTracker("/pose/franka/green_bottle")

    executor = MultiThreadedExecutor()
    executor.add_node(robotB)
    executor.add_node(obstacle_tracker)

    # -----------------------------
    # Helper
    # -----------------------------
    def move_and_wait(pos, name, timeout=6.0):
        robotB.get_logger().info(f"Moving to {name}")
        robotB.set_target(pos, target_quat)
        robotB.reset_goal_reached()
        start = time.time()
        while rclpy.ok() and not robotB.goal_reached():
            if time.time() - start > timeout:
                break
            executor.spin_once(timeout_sec=0.01)

    try:
        move_and_wait(pregrasp_pos, "PREGRASP")
        move_and_wait(grasp_pos, "GRASP")
        robotB.publish_zero_velocity()
        time.sleep(0.5)
        franka_gripper.close_gripper(width=0.04, force=20.0)

        move_and_wait(lift_pos, "LIFT")
        move_and_wait(preplace_pos, "PREPLACE")
        move_and_wait(place_pos, "PLACE")
        robotB.publish_zero_velocity()
        time.sleep(0.5)
        franka_gripper.open_gripper(width=0.08)

    finally:
        robotB.publish_zero_velocity()
        robotB.destroy_node()
        obstacle_tracker.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

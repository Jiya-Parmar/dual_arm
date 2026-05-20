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


# Make RC-DS root importable
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from utils.dls_velocity_commander import DLSVelocityCommander
from utils.gripper_commands.franka_gripper import FrankaGripperController
franka_gripper = FrankaGripperController()
franka_gripper.open_gripper(width=0.08)
time.sleep(1.0)
# -------------------------------------------------------
# One-shot pose listener
# -------------------------------------------------------
class InitialPoseListener(Node):
    def __init__(self, topic: str, node_name: str):
        super().__init__(node_name)
        self.pose = None

        self.create_subscription(
            PoseStamped,
            topic,
            self.pose_cb,
            10,
        )

    def pose_cb(self, msg: PoseStamped):
        if self.pose is not None:
            return

        self.pose = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ]

        self.get_logger().info(
            f"Latched pose from {self.get_name()}: "
            f"x={self.pose[0]:.4f}, y={self.pose[1]:.4f}, z={self.pose[2]:.4f}"
        )



def wait_for_pose(topic: str, node_name: str, timeout_sec: float, fallback_pose: list):
    node = InitialPoseListener(topic, node_name)
    start_time = time.time()

    while rclpy.ok() and node.pose is None:
        rclpy.spin_once(node, timeout_sec=0.1)

        # Check if the timeout has been exceeded
        if time.time() - start_time > timeout_sec:
            node.get_logger().warning(f"Timeout reached while waiting for pose on {topic}. Using fallback pose.")
            node.destroy_node()
            return fallback_pose

    pose = node.pose
    node.destroy_node()
    return pose


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    hardcoded_pick_pose = [0.28005232670635866, -0.36769667229137015, 0.08113820145774693] #[0.4921261145566418, 0.13996550307107655, 0.07466620414045444]  # Example hardcoded pick position for crate avoidance--> [0.28005232670635866, -0.36769667229137015, 0.08113820145774693]
    hardcoded_place_pose = [0.6431500139648929, 0.0010244106627500826, 0.12667222940941536] #[0.5584192431661987, -0.4388288685107432, 0.17439347134994837]  # Example hardcoded place position  for crate avoidance --> [0.6431500139648929, 0.0010244106627500826, 0.12667222940941536]
    # --------------------------------
    # Get PICK pose
    # --------------------------------
    pick_pose = wait_for_pose(
        "/pose/franka/blue_cuboid",
        "pick_pose_listener",
        timeout_sec=2.0,  # Timeout after 5 seconds
        fallback_pose=hardcoded_pick_pose,
    )

    # --------------------------------
    # Get PLACE pose
    # --------------------------------
    place_pose = wait_for_pose(
        "/pose/franka/white_mug",
        "place_pose_listener",
        timeout_sec=2.0,  # Timeout after 5 seconds
        fallback_pose=hardcoded_place_pose,
    )

    # --------------------------------
    # Build Cartesian targets
    # --------------------------------
    PREGRASP_Z = 0.30
    GRASP_Z = 0.10
    LIFT_Z = 0.30
    PREPLACE_Z = 0.50
    PLACE_Z = 0.15

    pregrasp_pos = [pick_pose[0], pick_pose[1], pick_pose[2] + PREGRASP_Z]
    grasp_pos    = [pick_pose[0], pick_pose[1], pick_pose[2] + GRASP_Z]
    lift_pos     = [pick_pose[0], pick_pose[1], pick_pose[2] + LIFT_Z]
    preplace_pos = [place_pose[0], place_pose[1], place_pose[2] + PREPLACE_Z]
    place_pos    = [place_pose[0], place_pose[1], place_pose[2] + PLACE_Z]
    
    target_quat = [0.945, -0.324, 0.005, 0.027]

    # --------------------------------
    # Create controller ONCE
    # --------------------------------
    robotB = DLSVelocityCommander(
        robot_id="robotB",
        base_link="fr3_link0",
        tip_link="fr3_link8",
        joint_names=[
            "fr3_joint1",
            "fr3_joint2",
            "fr3_joint3",
            "fr3_joint4",
            "fr3_joint5",
            "fr3_joint6",
            "fr3_joint7",
        ],
        target_pos=pregrasp_pos,   # initial dummy, immediately overridden
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
    )

    executor = MultiThreadedExecutor()
    executor.add_node(robotB)

    # --------------------------------
    # Helper: wait for convergence
    # --------------------------------
    def move_and_wait(pos, name, timeout=6.0):
        robotB.get_logger().info(f"Moving to {name}")
        robotB.set_target(pos, target_quat)
        robotB.reset_goal_reached()
        start =time.time()
        while rclpy.ok() and not robotB.goal_reached():
            if time.time() - start > timeout:
                robotB.get_logger().warning(f"Timeout reached while moving to {name}")
                break
            executor.spin_once(timeout_sec=0.01)

        robotB.get_logger().info(f"{name} reached")

    try:
        # -----------------------------
        # PICK SEQUENCE
        # -----------------------------
        
        move_and_wait(pregrasp_pos, "PREGRASP")
       
        move_and_wait(grasp_pos,    "GRASP")
        robotB.publish_zero_velocity()
        time.sleep(0.5)  # wait a bit at grasp pose
        franka_gripper.close_gripper(width=0.04, force=20.0)
        
        move_and_wait(lift_pos,     "LIFT")
        move_and_wait(preplace_pos,  "PRE-PLACE")
        move_and_wait(place_pos,    "PLACE")
        robotB.publish_zero_velocity()
        time.sleep(0.5)  # wait a bit at place pose
        franka_gripper.open_gripper(width=0.08)
        

        

    except KeyboardInterrupt:
        pass
    finally:
        robotB.publish_zero_velocity()
        robotB.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

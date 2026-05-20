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
from utils.gripper_commands.heal_dynamixel_gripper import GripperController
heal_gripper = GripperController()
heal_gripper.open_gripper()
# franka_gripper = FrankaGripperController()
# franka_gripper.open_gripper(width=0.08)
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

def pick_place_ds(current_pose: dict) -> np.ndarray:
    """
    current_pose keys (already set by DLS):
      - position : np.ndarray (3,)
      - orientation_xyzw : np.ndarray (4,)
    """

    x = current_pose["position"]          # EE position

    target_kdl = current_pose.get("target_kdl", None)
    if target_kdl is None:
        return np.zeros(6)         # KDL Vector
    
    x_goal = np.array(
        [target_kdl.x(), target_kdl.y(), target_kdl.z()],
        dtype=float,
    )

    # -----------------------------
    # 1) Nominal DS (goal attraction)
    # -----------------------------
    k_att = 1.0
    v_nom = k_att * (x_goal - x)

    # velocity limiting (important)
    n = np.linalg.norm(v_nom)
    if n > robotA.max_cartesian_vel:
        v_nom = (v_nom / n) * robotA.max_cartesian_vel


    # -----------------------------
    # 3) Orientation (simple stabilizer)
    # -----------------------------
    w = np.zeros(3)

    return np.concatenate([v_nom, w], axis=0)

#Static pose subscriber. Subscrivees to a topic and latches the first received pose.
def wait_for_pose(
    topic: str,
    node_name: str,
    timeout_sec: float,
    fallback_pose: list,
):
    node = InitialPoseListener(topic, node_name)
    start_time = time.time()

    while rclpy.ok() and node.pose is None:
        rclpy.spin_once(node, timeout_sec=0.1)

        if time.time() - start_time > timeout_sec:
            node.get_logger().warning(
                f"Timeout while waiting for pose on {topic}. Using fallback pose."
            )
            node.destroy_node()
            return fallback_pose

    pose = node.pose
    node.destroy_node()
    return pose

#Dynamic pose subscrber. Continuously updates the obstacle center.
# class ObstacleTracker(Node):
#     def __init__(self, topic: str):
#         super().__init__("obstacle_tracker")
#         self.center = None

#         self.create_subscription(
#             PoseStamped,
#             topic,
#             self.cb,
#             10,
#         )

#     def cb(self, msg: PoseStamped):
#         self.center = np.array(
#             [
#                 msg.pose.position.x,
#                 msg.pose.position.y,
#                 msg.pose.position.z,
#             ],
#             dtype=float,
#         )



# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main(args=None):
    hardcoded_pick_pose = [-0.003731244294287492, 0.6595048174402348, 0.3846573267124301]
    hardcoded_place_pose = [-0.0008232779826864515, 0.3678360083468431, 0.5538878347626814]
    # hardcoded_obstacle_pose = [0.02120461240644155, 0.4961942550283565, 0.25]
    global OBSTACLE_CENTER, avoider, robotA
    rclpy.init(args=args)

    # --------------------------------
    # Get PICK pose
    # --------------------------------
    pick_pose = wait_for_pose(
        "/pose/franka/blue_cuboid",
        "pick_pose_listener",
        timeout_sec=2.0,
        fallback_pose=hardcoded_pick_pose
    )

    # --------------------------------
    # Get PLACE pose
    # --------------------------------
    place_pose = wait_for_pose(
        "/pose/franka/white_mug",
        "place_pose_listener",
        timeout_sec=2.0,
        fallback_pose=hardcoded_place_pose
    )


    # --------------------------------
    # Build Cartesian targets
    # --------------------------------
    PREGRASP_Z = 0.18
    GRASP_Z = 0.0
    LIFT_Z = 0.20
    PREPLACE_Z = 0.18
    PLACE_Z = 0.0

    pregrasp_pos = [pick_pose[0], pick_pose[1], pick_pose[2] + PREGRASP_Z]
    grasp_pos    = [pick_pose[0], pick_pose[1], pick_pose[2] + GRASP_Z]
    lift_pos     = [pick_pose[0], pick_pose[1], pick_pose[2] + LIFT_Z]
    preplace_pos = [place_pose[0], place_pose[1], place_pose[2] + PREPLACE_Z]
    place_pos    = [place_pose[0], place_pose[1], place_pose[2] + PLACE_Z]
    
    target_quat = [0.945, -0.324, 0.005, 0.027]

    # --------------------------------
    # Create controller ONCE
    # --------------------------------
    
    robotA = DLSVelocityCommander(
        robot_id="robotA",
        base_link="base_link",
        tip_link="end-effector",
        joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        target_pos=[-0.098538959315282,0.5107166230209237, 0.24 + 0.035492071848395534],
        target_quat=[0.00, 0.00, 0.00, 0.999],
        joint_state_topic="/joint_states",
        velocity_command_topic="/velocity_controller/commands",
        robot_description_topic="/robot_description",
        ee_pose_topic=None,
        ee_pose_is_stamped=False,
        max_cartesian_vel=0.05,
        max_angular_vel=0.15,
        dt=0.01,
        damping=0.1,
        
        custom_ds=pick_place_ds,
    )
    # robotB = DLSVelocityCommander(
    #     robot_id="robotB",
    #     base_link="fr3_link0",
    #     tip_link="fr3_link8",
    #     joint_names=[
    #         "fr3_joint1",
    #         "fr3_joint2",
    #         "fr3_joint3",
    #         "fr3_joint4",
    #         "fr3_joint5",
    #         "fr3_joint6",
    #         "fr3_joint7",
    #     ],
    #     target_pos=pregrasp_pos,   # initial dummy, immediately overridden
    #     target_quat=target_quat,
    #     joint_state_topic="/NS_1/franka/joint_states",
    #     velocity_command_topic="/NS_1/joint_velocity_controller/commands",
    #     robot_description_topic="/NS_1/robot_description",
    #     ee_pose_topic=None,
    #     ee_pose_is_stamped=False,
    #     max_cartesian_vel=0.2,
    #     max_angular_vel=0.3,
    #     dt=0.01,
    #     damping=0.1,

    #     custom_ds=pick_place_ds,
    # )

    executor = MultiThreadedExecutor()
    executor.add_node(robotA)

    # --------------------------------
    # Helper: wait for convergence
    # --------------------------------
    def move_and_wait(pos, name, timeout=15.0):
        robotA.get_logger().info(f"Moving to {name}")
        robotA.set_target(pos, target_quat)
        robotA.reset_goal_reached()
        start =time.time()
        while rclpy.ok() and not robotA.goal_reached():
            if time.time() - start > timeout:
                robotA.get_logger().warning(f"Timeout reached while moving to {name}")
                break
            executor.spin_once(timeout_sec=0.01)

        robotA.get_logger().info(f"{name} reached")

    try:
        # -----------------------------
        # PICK SEQUENCE
        # -----------------------------
        
        # move_and_wait(pregrasp_pos, "PREGRASP")
       
        move_and_wait(grasp_pos,    "GRASP")
        robotA.publish_zero_velocity()
        time.sleep(0.5)  # wait a bit at grasp pose
        heal_gripper.close_gripper()
        # franka_gripper.close_gripper(width=0.02, force=30.0)
        
        # move_and_wait(lift_pos,     "LIFT")
        # move_and_wait(preplace_pos,  "PRE-PLACE")
        move_and_wait(place_pos,    "PLACE")
        robotA.publish_zero_velocity()
        time.sleep(0.5)  # wait a bit at place pose
        heal_gripper.open_gripper()
        # franka_gripper.open_gripper(width=0.08)
        

        

    except KeyboardInterrupt:
        pass
    finally:
        robotA.publish_zero_velocity()
        robotA.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

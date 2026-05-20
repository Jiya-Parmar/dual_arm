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

# OBSTACLE_CENTER = None
# OBSTACLE_RADIUS = 0.30
OBSTACLES = []   # list of dicts: {"center": np.array, "radius": float}
avoiders  = []

# Make RC-DS root importable
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from utils.modulation_class_v2 import ModulationAvoider

# avoider = ModulationAvoider(
#     gamma=lambda x: ModulationAvoider.gamma_sphere(
#         x, OBSTACLE_CENTER, OBSTACLE_RADIUS
#     ),
#     grad_gamma=lambda x: ModulationAvoider.grad_gamma_sphere(
#         x, OBSTACLE_CENTER, OBSTACLE_RADIUS
#     ),
#     rho=1.0,
#     safety_eta=1.0,
#     tail_effect=False,
# )

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
    # 2) Obstacle avoidance
    # -----------------------------
    v_mod = v_nom.copy()

    for avoider_i, obs in zip(avoiders, OBSTACLES):
        x_tilde = x - obs["center"]
        v_mod = avoider_i.modulate(
            x=x,
            x_tilde=x_tilde,    # ← obstacle-centered frame
            f=v_mod,
        )

    # -----------------------------
    # 3) Orientation (simple stabilizer)
    # -----------------------------
    w = np.zeros(3)

    return np.concatenate([v_mod, w], axis=0)

#Static pose subscriber. Subscrivees to a topic and latches the first received pose.
def wait_for_pose(topic, node_name, timeout_sec, fallback_pose):
    node = InitialPoseListener(topic, node_name)
    start_time = time.time()
    while rclpy.ok() and node.pose is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - start_time > timeout_sec:
            node.get_logger().warning(
                f"Timeout on {topic}. Using fallback."
            )
            node.destroy_node()
            return fallback_pose
    pose = node.pose
    node.destroy_node()
    return pose

# # Dynamic pose subscrber. Continuously updates the obstacle center.
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
    global OBSTACLES, robotA, avoiders
    rclpy.init(args=args)
    
    hardcoded_pick_pose = [0.008797492121682354, 0.6095668595754376, 0.3936943182815076]
    hardcoded_place_pose = [-0.4872101827077948, 0.4801915609348676, 0.3081947072858158]
# Heal Crate pick 1: [0.00797492121682354, 0.6695668595754376, 0.4136943182815076] | [ 0.19915900509377835, 0.0021127871733583026, 0.1804603558860462, 0.9632057343962909]
# Heal Place 1 : [-0.4872101827077948, 0.4801915609348676, 0.2281947072858158] | [0.013347910409805284, 0.44561474959715075, 0.2669506629506341, 0.8543925747449924]
# Heal crate side long : [0.025870643513917654, 0.5079258169172239, 0.19]
# Heal crate side small : [-0.26357436383966476, 0.6193718313193132, 0.19]

    fallback_long = [0.025870643513917654, 0.48079258169172239, 0.20]
    fallback_small = [0.025870643513917654-0.32, 0.48079258169172239+0.16, 0.19]
    
    # fallback_long = [0.4530114264519804, -0.022508355751175053, 0.20]
    # fallback_small = [0.6284382569122192, -0.3103458570094927, 0.20]
    # fallback_green  = [0.022408546698144082, 0.4835511939745351, 0.15]
    # fallback_orange = [0.09725623906492883, 0.5681313010009765, 0.06]
    # fallback_black  = [0.0077173576216621395, 0.3087350965950095, 0.095]

    # --------------------------------
    # Get PICK pose
    # --------------------------------
    # pick_pose = wait_for_pose(
    #     "/pose/franka/blue_cuboid",
    #     "pick_pose_listener",
    #     2.0,
    #     hardcoded_pick_pose,
    # )

    # # --------------------------------
    # # Get PLACE pose
    # # --------------------------------
    # place_pose = wait_for_pose(
    #     "/pose/franka/white_mug",
    #     "place_pose_listener",
    #     2.0,
    #     hardcoded_place_pose,
    # )

    # --------------------------------
    # Get OBSTACLE pose
    # --------------------------------
    
    OBSTACLES = [
        {
            "center": np.array(
                wait_for_pose(
                    "/pose/franka/green_bottle",
                    "obstacle_listener_1",
                    2.0,
                    fallback_long,
                ),
                dtype=float,
            ),
            "A": np.diag(1.0 / (np.array([0.32, 0.08, 0.30]) ** 2)),
        },
        {
            "center": np.array(
                wait_for_pose(
                    "/pose/franka/orange_hex_prism",
                    "obstacle_listener_2",
                    2.0,
                    fallback_small,
                ),
                dtype=float,
            ),
            "A": np.diag(1.0 / (np.array([0.08, 0.21, 0.30]) ** 2)),
        },
        # {
        #     "center": np.array(
        #         wait_for_pose(
        #             "/pose/franka/black_mug",
        #             "obstacle_listener_3",
        #             2.0,
        #             fallback_black,
        #         ),
        #         dtype=float,
        #     ),
        #     "A": np.diag(1.0 / (np.array([0.10, 0.10, 0.14]) ** 2)),
        # },
    ]
    
    avoiders = []
    for obs in OBSTACLES:
        avoiders.append(ModulationAvoider(
            gamma=lambda x, A=obs["A"]:
                ModulationAvoider.gamma_ellipsoid(x, np.zeros(3), A),
            grad_gamma=lambda x, A=obs["A"]:
                ModulationAvoider.grad_gamma_ellipsoid(x, np.zeros(3), A),
            rho=1.0,
            safety_eta=0.8,
            tail_effect=False,
        ))
    # --------------------------------
    # Build Cartesian targets
    # --------------------------------
    # PREGRASP_Z = 0.30
    # GRASP_Z = 0.0
    # LIFT_Z = 0.30
    # PREPLACE_Z = 0.35
    # PLACE_Z = 0.0

    pregrasp_pos = np.array([-0.0008232779826864515, 0.55095048174402348, 0.4846573267124301])
    grasp_pos    = np.array([-0.0008232779826864515, 0.55095048174402348, 0.36046573267124301])
    lift_pos     = np.array([-0.0008232779826864515, 0.55095048174402348, 0.54046573267124301])
    preplace_pos = np.array([-0.55, 0.5595048174402348, 0.45046573267124301])
    place_pos    = np.array([-0.55, 0.5595048174402348, 0.32046573267124301])
    
    
    target_quat = [0.00, 0.00, 0.00, 0.999]
    
    # pregrasp_pos = np.array([0.6430737308963683, -0.035183055429412836, 0.3063069872346974193])
    # grasp_pos    = np.array([0.6430737308963683, -0.035183055429412836, 0.28069872346974193])
    # lift_pos     = np.array([0.6430737308963683, -0.035183055429412836, 0.46069872346974193])
    # preplace_pos = np.array([0.3471876233208688, -0.4353593103380543, 0.45023858392689695])
    # place_pos    = np.array([0.3471876233208688, -0.4353593103380543, 0.28223858392689695])
    

# np.array([0.6430737308963683, -0.035183055429412836, 0.56069872346974193])
# np.array([0.6430737308963683, -0.035183055429412836, 0.26069872346974193])
# np.array([0.6430737308963683, -0.035183055429412836, 0.46069872346974193]),
# np.array([0.3471876233208688, -0.4353593103380543, 0.4623858392689695]),
# np.array([0.3471876233208688, -0.4353593103380543, 0.28223858392689695]),
    # --------------------------------
    # Create controller ONCE
    # --------------------------------
    # [0.00, 0.00, 0.00, 0.999]
    robotA = DLSVelocityCommander(
        robot_id="robotA",
        base_link="base_link",
        tip_link="end-effector",
        joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        target_pos=[-0.098538959315282,0.5107166230209237, 0.24 + 0.035492071848395534],
        target_quat=[0.006976676648530432, 0.0024512094056837355, 0.7879966608156741, 0.6156351030429786],
        joint_state_topic="/joint_states",
        velocity_command_topic="/velocity_controller/commands",
        robot_description_topic="/robot_description",
        ee_pose_topic=None,
        ee_pose_is_stamped=False,
        max_cartesian_vel=0.075,
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
    #     max_cartesian_vel=0.15,
    #     max_angular_vel=0.2,
    #     dt=0.01,
    #     damping=0.1,

    #     custom_ds=pick_place_ds,
    # )

    executor = MultiThreadedExecutor()
    executor.add_node(robotA)

    # --------------------------------
    # Helper: wait for convergence
    # --------------------------------
    def move_and_wait(pos, name, timeout=10.0, pos_tol=None):
        robotA.get_logger().info(f"Moving to {name}")
        robotA.set_target(pos, target_quat)
        robotA.reset_goal_reached()
        start =time.time()
        while rclpy.ok() and not robotA.goal_reached():
            if time.time() - start > timeout:
                robotA.get_logger().warning(f"Timeout reached while moving to {name}")
                break
            executor.spin_once(timeout_sec=0.01)
            
            # Position convergence check
            if pos_tol is not None:
                if robotA.current_pose is not None:   # ← this guard must be here
                    current_pos = robotA.current_pose["position"]
                    dist = np.linalg.norm(current_pos - np.array(pos))
                    if dist < pos_tol:
                        robotA.get_logger().info(f"{name} converged (dist={dist:.4f}m)")
                        break
            else:
                if robotA.goal_reached():
                    break

        robotA.get_logger().info(f"{name} reached")

    try:
        # -----------------------------
        # PICK SEQUENCE
        # -----------------------------
        
        move_and_wait(pregrasp_pos, "PREGRASP",timeout=30, pos_tol= 0.02)
       
        move_and_wait(grasp_pos,    "GRASP",timeout=20, pos_tol= 0.02)
        robotA.publish_zero_velocity()
        time.sleep(0.5)  # wait a bit at grasp pose
        # franka_gripper.close_gripper(width=0.04, force=20.0)
        heal_gripper.close_gripper()
        
        move_and_wait(lift_pos,     "LIFT", timeout=20, pos_tol=0.02)
        move_and_wait(preplace_pos,  "PRE-PLACE", timeout=20, pos_tol=0.04)
        move_and_wait(place_pos,    "PLACE", timeout=20, pos_tol=0.02)
        robotA.publish_zero_velocity()
        time.sleep(0.5)  # wait a bit at place pose
        # franka_gripper.open_gripper(width=0.08)
        heal_gripper.open_gripper()
        

        

    except KeyboardInterrupt:
        pass
    finally:
        robotA.publish_zero_velocity()
        robotA.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

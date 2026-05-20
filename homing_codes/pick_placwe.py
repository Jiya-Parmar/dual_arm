#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import numpy as np
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.quintic_planner import QuinticJointSpacePlanner
from utils.gripper_commands.franka_gripper import FrankaGripperController


class FrankaPickPlace(Node):

    def __init__(self):
        super().__init__("franka_pick_place_joint_velocity")

        self.joint_names = [
            "fr3_joint1","fr3_joint2","fr3_joint3",
            "fr3_joint4","fr3_joint5","fr3_joint6","fr3_joint7",
        ]

        # -------- POINTS (EDIT ONLY THESE IF NEEDED) --------
        self.Q1 = np.array([
  -0.21470452845096588,
   0.3397340178489685,
   0.011245395988225937,
  -2.011094570159912,
  -0.006312032230198383,
   2.3389716148376465,
   0.5835469961166382
        ])

        self.Q2 = np.array([
          -0.22894807159900665,
           0.5467138886451721,
           0.02211945131421089,
          -2.005707025527954,
           0.0049243806861341,
           2.5372860431671143,
           0.613952100276947
        ])

        self.Q3 = np.array([
           -0.6563398241996765,
            0.6065337061882019,
            0.1385745257139206,
           -1.7337499856948853,
           -0.15716783702373505,
            2.3202037811279297,
            0.44443365931510925
        ])

        # -------------- MOTION PARAMS --------------
        self.total_time = 10.0
        self.rate_hz = 200.0
        self.dt = 1.0 / self.rate_hz
        self.vmax = 0.6

        self._latest_js = None
        self._name_to_idx = {}
        self.planner = None
        self.t0 = None

        # NEW SEQUENCE STAGES
        self.stage = "MOVE_P1"
        self.gripper_closed = False

        # -------------- TOPICS (unchanged) --------------
        self.create_subscription(
            JointState,
            "/NS_1/franka/joint_states",
            self.joint_state_cb,
            qos_profile_sensor_data,
        )

        self.cmd_pub = self.create_publisher(
            Float64MultiArray,
            "/NS_1/joint_velocity_controller/commands",
            1,
        )

        self.timer = self.create_timer(self.dt, self.control_loop)
        self.gripper = FrankaGripperController()

        self.get_logger().info("Sequence node started.")

    # -----------------------------------------------------

    def joint_state_cb(self, msg):
        self._latest_js = msg
        self._name_to_idx = {n:i for i,n in enumerate(msg.name)}

    def extract_q(self):
        q = np.zeros(7)
        for i,n in enumerate(self.joint_names):
            q[i] = self._latest_js.position[self._name_to_idx[n]]
        return q

    def publish_zero(self):
        m = Float64MultiArray()
        m.data = [0.0]*7
        self.cmd_pub.publish(m)

    def start_motion(self, q_goal):
        self.planner = QuinticJointSpacePlanner(
            q_start=self.extract_q(),
            q_goal=q_goal,
            duration=self.total_time,
        )
        self.t0 = self.get_clock().now().nanoseconds * 1e-9

    # -----------------------------------------------------

    def control_loop(self):
        if self._latest_js is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        # -------- MOVE TO P1 --------
        if self.stage == "MOVE_P1":
            if self.planner is None:
                self.get_logger().info("Moving to Point-1")
                self.start_motion(self.Q1)
                return

            t = now - self.t0
            if t >= self.total_time:
                self.publish_zero()
                self.planner = None
                self.stage = "MOVE_P2"
                return

            qd = np.clip(self.planner.evaluate(t)[1], -self.vmax, self.vmax)
            self.cmd_pub.publish(Float64MultiArray(data=qd.tolist()))
            return

        # -------- MOVE TO P2 --------
        if self.stage == "MOVE_P2":
            if self.planner is None:
                self.get_logger().info("Moving to Point-2")
                self.start_motion(self.Q2)
                return

            t = now - self.t0
            if t >= self.total_time:
                self.publish_zero()
                self.planner = None
                self.stage = "GRIPPER_CLOSE"
                return

            qd = np.clip(self.planner.evaluate(t)[1], -self.vmax, self.vmax)
            self.cmd_pub.publish(Float64MultiArray(data=qd.tolist()))
            return

        # -------- CLOSE GRIPPER --------
        if self.stage == "GRIPPER_CLOSE":
            if not self.gripper_closed:
                self.get_logger().info("Closing gripper")
                threading.Thread(
                    target=self.gripper.close_gripper,
                    kwargs={"width":0.04,"force":30.0},
                    daemon=True,
                ).start()
                self.gripper_closed = True
                return

            self.stage = "RETURN_P1"
            return

        # -------- RETURN TO P1 --------
        if self.stage == "RETURN_P1":
            if self.planner is None:
                self.get_logger().info("Returning to Point-1")
                self.start_motion(self.Q1)
                return

            t = now - self.t0
            if t >= self.total_time:
                self.publish_zero()
                self.planner = None
                self.stage = "MOVE_P3"
                return

            qd = np.clip(self.planner.evaluate(t)[1], -self.vmax, self.vmax)
            self.cmd_pub.publish(Float64MultiArray(data=qd.tolist()))
            return

        # -------- MOVE TO P3 --------
        if self.stage == "MOVE_P3":
            if self.planner is None:
                self.get_logger().info("Moving to Point-3")
                self.start_motion(self.Q3)
                return

            t = now - self.t0
            if t >= self.total_time:
                self.publish_zero()
                self.planner = None
                self.stage = "GRIPPER_OPEN"
                return

            qd = np.clip(self.planner.evaluate(t)[1], -self.vmax, self.vmax)
            self.cmd_pub.publish(Float64MultiArray(data=qd.tolist()))
            return

        # -------- OPEN GRIPPER & EXIT --------
        if self.stage == "GRIPPER_OPEN":
            self.get_logger().info("Opening gripper")
            threading.Thread(
                target=self.gripper.open_gripper,
                daemon=True,
            ).start()

            self.publish_zero()
            self.get_logger().info("Sequence complete.")
            rclpy.shutdown()
            return


def main(args=None):
    rclpy.init(args=args)
    node = FrankaPickPlace()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.publish_zero()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

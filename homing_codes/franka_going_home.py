#!/usr/bin/env python3
"""
franka_going_home.py

Use QuinticJointSpacePlanner to generate a smooth joint-space trajectory
for the Franka FR3 from its current configuration to a "home" pose.

- Subscribes:  /NS_1/franka/joint_states   (sensor_msgs/JointState)
- Publishes:   /NS_1/joint_velocity_controller/commands (Float64MultiArray)
- Uses:        utils.quintic_planner.QuinticJointSpacePlanner
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Dict, List, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# ---------------------------------------------------------------------------
# Make sure we can import from RC-DS/utils even when running this script directly
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.quintic_planner import QuinticJointSpacePlanner  # type: ignore


class FrankaGoHome(Node):
    """
    Simple node:
      1. Waits for one JointState from /NS_1/franka/joint_states
      2. Builds a QuinticJointSpacePlanner from q_start -> q_home
      3. On each timer step, evaluates q(t), qdot(t) and publishes qdot
      4. Stops (publishes zero) once t >= T or very close to goal
    """

    def __init__(
        self,
        *,
        joint_state_topic: str = "/NS_1/franka/joint_states",
        command_topic: str = "/NS_1/joint_velocity_controller/commands",
        joint_names: Optional[List[str]] = None,
        total_time: float = 15.0,
        rate_hz: float = 200.0,
    ):
        super().__init__("franka_go_home")

        # Franka joint name order we will use for planning/commands
        if joint_names is None:
            joint_names = [
                "fr3_joint1",
                "fr3_joint2",
                "fr3_joint3",
                "fr3_joint4",
                "fr3_joint5",
                "fr3_joint6",
                "fr3_joint7",
            ]

        self.joint_state_topic = joint_state_topic
        self.command_topic = command_topic
        self.joint_names = joint_names
        self.n_joints = len(self.joint_names)

        # Home configuration (same as your controllers.yaml)
        self.q_home = np.array(
            [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
            dtype=float,
        )

        # Planner and timing
        self.total_time = float(total_time)
        self.dt = 1.0 / float(rate_hz)
        self.planner: Optional[QuinticJointSpacePlanner] = None
        self.t0: Optional[float] = None

        # State
        self._latest_js: Optional[JointState] = None
        self._name_to_idx: Dict[str, int] = {}
        self._q_start: Optional[np.ndarray] = None
        self._done = False

        # QoS matching Franka joint_states (SensorDataQoS)
        self.create_subscription(
            JointState,
            self.joint_state_topic,
            self.joint_state_callback,
            qos_profile_sensor_data,
        )

        self.cmd_pub = self.create_publisher(Float64MultiArray, self.command_topic, 1)

        # Control loop timer
        self._timer = self.create_timer(self.dt, self._on_timer)

        self.get_logger().info(
            "FrankaGoHome node started.\n"
            f"  joint_state_topic = {self.joint_state_topic}\n"
            f"  command_topic     = {self.command_topic}\n"
            f"  joint_names       = {self.joint_names}\n"
            f"  total_time        = {self.total_time} s\n"
            f"  rate              = {rate_hz} Hz"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def joint_state_callback(self, msg: JointState):
        """Store the latest JointState and name->index map."""
        self._latest_js = msg
        self._name_to_idx = {n: i for i, n in enumerate(msg.name)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_q_in_order(self, msg: JointState) -> Optional[np.ndarray]:
        """Return joint vector q in self.joint_names order, or None if missing."""
        q = np.zeros(self.n_joints, dtype=float)
        missing = []
        for i, name in enumerate(self.joint_names):
            idx = self._name_to_idx.get(name, None)
            if idx is None:
                missing.append(name)
                q[i] = 0.0
            else:
                q[i] = float(msg.position[idx])

        if missing:
            self.get_logger().warn(
                f"Missing joints in JointState: {missing}. "
                "Using 0.0 for those entries."
            )

        return q

    def _maybe_init_planner(self, now_sec: float):
        """
        Initialize the quintic planner once we have a valid starting JointState.
        """
        if self.planner is not None or self._latest_js is None:
            return

        # Extract q_start in the desired order
        q_start = self._extract_q_in_order(self._latest_js)
        if q_start is None:
            return

        self._q_start = q_start.copy()
        self.planner = QuinticJointSpacePlanner(
            q_start=self._q_start,
            q_goal=self.q_home,
            duration=self.total_time,  # ✅ match planner signature
        )
        self.t0 = now_sec

        self.get_logger().info(
            "Initialized QuinticJointSpacePlanner.\n"
            f"  q_start = {self._q_start}\n"
            f"  q_goal  = {self.q_home}\n"
            f"  T       = {self.total_time}"
        )

    def _publish_zero(self):
        msg = Float64MultiArray()
        msg.data = [0.0] * self.n_joints
        self.cmd_pub.publish(msg)

    # ------------------------------------------------------------------
    # Timer: main control loop
    # ------------------------------------------------------------------

    def _on_timer(self):
        if self._done:
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        # Ensure we have planner initialized
        self._maybe_init_planner(now)
        if self.planner is None or self.t0 is None:
            # Still waiting for first JointState
            return

        t = now - self.t0

        if t >= self.total_time:
            # End of planned motion: send zero once and finish
            self._publish_zero()
            self._done = True
            self.get_logger().info(
                f"Reached end of trajectory (t={t:.3f} >= T={self.total_time:.3f}). "
                "Published zero velocities and stopping."
            )
            return

        # Evaluate planner (q, qd, qdd) at time t
        q, qd, qdd = self.planner.evaluate(t)

        # Optional: simple guard on maximum velocity
        vmax = 1.0  # rad/s (you can tune this)
        qd_clamped = np.clip(qd, -vmax, vmax)

        msg = Float64MultiArray()
        msg.data = qd_clamped.tolist()
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FrankaGoHome(
        joint_state_topic="/NS_1/franka/joint_states",
        command_topic="/NS_1/joint_velocity_controller/commands",
        total_time=15.0,      # you set this in your run
        rate_hz=200.0,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # On shutdown, send zero velocities once more
        try:
            node._publish_zero()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
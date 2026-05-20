#!/usr/bin/env python3
# dual_dls_velocity_commander_ros2.py
from __future__ import annotations

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    qos_profile_sensor_data,
)

from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromUrdfModel
import PyKDL as kdl


# =========================
# Quaternion / SO(3) utils
# =========================

def normalize_quaternion(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.array(q_xyzw, dtype=float).reshape(4,)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n


def quaternion_multiply(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1_xyzw
    x2, y2, z2, w2 = q2_xyzw
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w], dtype=float)


def quaternion_log_error(q_target_xyzw: np.ndarray, q_current_xyzw: np.ndarray) -> np.ndarray:
    """
    Log-map orientation error in R^3 such that omega = angle * axis.
    Both quaternions are assumed [x, y, z, w].
    """
    q1 = normalize_quaternion(q_target_xyzw)
    q2 = normalize_quaternion(q_current_xyzw)

    # inverse of q2
    q2_inv = np.array([-q2[0], -q2[1], -q2[2], q2[3]], dtype=float)
    q_err = quaternion_multiply(q1, q2_inv)
    q_err = normalize_quaternion(q_err)

    # shortest-arc convention
    if q_err[3] < 0.0:
        q_err *= -1.0

    w = float(np.clip(q_err[3], -1.0, 1.0))
    angle = 2.0 * math.acos(w)

    if angle < 1e-6:
        return np.zeros(3, dtype=float)

    s = math.sin(angle / 2.0)
    axis = q_err[:3] / max(s, 1e-12)
    return angle * axis


# =========================
# DLS Velocity Commander
# =========================

class DLSVelocityCommander(Node):
    """
    Generic DLS Cartesian velocity controller for a single robot.

    Added in this version:
      - set_target(pos, quat) so DS/bridge can update the effective goal
      - debounced goal detection:
          goal_reached = pose_converged AND ||dq|| small for N consecutive ticks
    """

    def __init__(
        self,
        *,
        robot_id: str,
        base_link: str,
        tip_link: str,
        joint_names: list[str],
        target_pos: list[float],
        target_quat: list[float],
        joint_state_topic: str,
        velocity_command_topic: str,
        robot_description_topic: str = "/robot_description",
        ee_pose_topic: str | None = None,
        ee_pose_is_stamped: bool = True,
        custom_ds=None,                    # fn(pose_dict)->twist[6]
        max_cartesian_vel: float = 0.05,
        max_angular_vel: float = 0.05,
        dt: float = 0.001,
        damping: float = 0.01,
        coordinated_sync_mode: bool = False,
        virtual_pose_fn=None,
        virtual_goal=None,
        
    ):
        super().__init__(f"dls_velocity_commander_{robot_id}")

        self.robot_id = robot_id
        self.base_link = base_link
        self.tip_link = tip_link

        self.dt = float(dt)
        self.damping = float(damping)

        self.max_cartesian_vel = float(max_cartesian_vel)
        self.max_angular_vel = float(max_angular_vel)
        self.custom_ds = custom_ds

        # --- pose thresholds for convergence ---
        self.pos_thresh = 0.003
        self.ori_thresh = 0.005

        # --- stopped / debounce settings (NEW) ---
        self.dq_stop_thresh = 0.01        # rad/s norm threshold (tune per robot)
        self.converge_hold_ticks = 10     # ticks required (dt=0.01 -> 0.1s)
        self._hold_counter = 0
        self._goal_reached = False

        # coordinated sync fields (kept but not used in this simple reaching demo)
        self.coordinated_sync_mode = bool(coordinated_sync_mode)
        self.get_virtual_pose_fn = virtual_pose_fn
        self._virtual_goal = virtual_goal
        self.virtual_pos_thresh = 0.002
        self.virtual_ori_thresh = 0.005

        self.sync_mode = False
        self.partner: DLSVelocityCommander | None = None

        self.q_dA = None
        self.q_dB = None
        self.r_A = None
        self.r_B = None
        
        self.k_pos = 1.0
        self.k_ori = 1.0  # try 3–12 for Franka


        # pub
        self.pub = self.create_publisher(Float64MultiArray, velocity_command_topic, 1)

        # joint naming
        self.command_joint_names = list(joint_names)  # order expected by velocity controller
        self.chain_joint_names: list[str] = []        # KDL chain order

        # joint state buffers
        self._last_joint_state_msg: JointState | None = None
        self._js_name_to_idx: dict[str, int] | None = None

        # optional EE pose feedback
        self.current_pose = None
        self.ee_pose_topic = ee_pose_topic
        if self.ee_pose_topic:
            msg_type = PoseStamped if ee_pose_is_stamped else Pose
            self.create_subscription(msg_type, self.ee_pose_callback, 10)

        # KDL deferred init
        self._kdl_ready = False
        self.chain = None
        self.n_joints = 0
        self.fk_solver = None
        self.jac_solver = None

        # QoS: joint_states is typically SensorDataQoS (BEST_EFFORT)
        self.create_subscription(
            JointState, joint_state_topic, self.joint_state_callback, qos_profile_sensor_data
        )

        # QoS: robot_description is often latched/transient_local
        robot_desc_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._robot_desc_sub = self.create_subscription(
            String, robot_description_topic, self._robot_description_cb, robot_desc_qos
        )

        # target frame (initial)
        self.target_pos_kdl = kdl.Vector(*target_pos)
        qx, qy, qz, qw = normalize_quaternion(np.array(target_quat, dtype=float))
        self.target_quat_xyzw = np.array([qx, qy, qz, qw], dtype=float)
        self.target_rot_kdl = kdl.Rotation.Quaternion(qx, qy, qz, qw)
        self.target_frame = kdl.Frame(self.target_rot_kdl, self.target_pos_kdl)

        # timers
        self._timer = self.create_timer(self.dt, self._on_timer)
        self._status_timer = self.create_timer(2.0, self._status_tick)

        self._zero_published = False
        self._last_dq_norm = None

        self.get_logger().info(
            f"[{self.robot_id}] Started. Waiting for URDF + joint_states...\n"
            f"  base_link={self.base_link}, tip_link={self.tip_link}\n"
            f"  joint_state_topic={joint_state_topic}\n"
            f"  cmd_topic={velocity_command_topic}\n"
            f"  robot_description_topic={robot_description_topic}"
        )

    # ------------------------
    # Debug status
    # ------------------------

    def _status_tick(self):
        kdl_s = "READY" if self._kdl_ready else "WAIT"
        js_s = "OK" if self._last_joint_state_msg is not None else "WAIT"
        self.get_logger().info(
            f"[{self.robot_id}] status: KDL={kdl_s}, joint_state={js_s}, "
            f"n_joints={self.n_joints}, last|dq|={self._last_dq_norm}, "
            f"goal_reached={self._goal_reached}, hold={self._hold_counter}/{self.converge_hold_ticks}"
        )

    # ------------------------
    # Target update (NEW)
    # ------------------------

    def set_target(self, pos_xyz: np.ndarray, quat_xyzw: np.ndarray):
        """
        Update the internal target used by has_converged() and classic mode.
        Bridge/DS should call this when you change DS goals.
        """
        pos_xyz = np.asarray(pos_xyz, dtype=float).reshape(3,)
        quat_xyzw = normalize_quaternion(np.asarray(quat_xyzw, dtype=float).reshape(4,))

        self.target_pos_kdl = kdl.Vector(*pos_xyz.tolist())
        self.target_quat_xyzw = quat_xyzw.copy()
        self.target_rot_kdl = kdl.Rotation.Quaternion(*self.target_quat_xyzw)
        self.target_frame = kdl.Frame(self.target_rot_kdl, self.target_pos_kdl)

        # reset convergence latch when target changes
        self._hold_counter = 0
        self._goal_reached = False

    def goal_reached(self) -> bool:
        return bool(self._goal_reached)

    def reset_goal_reached(self):
        self._hold_counter = 0
        self._goal_reached = False

    # ---------------------------------
    # URDF -> KDL once
    # ---------------------------------

    def _robot_description_cb(self, msg: String):
        if self._kdl_ready:
            return

        urdf_xml = msg.data
        if not urdf_xml or "<robot" not in urdf_xml:
            self.get_logger().error("Got robot_description but it doesn't look like URDF XML.")
            return

        try:
            robot = URDF.from_xml_string(urdf_xml)
            ok, tree = treeFromUrdfModel(robot)
            if not ok:
                self.get_logger().error("Failed to construct KDL tree from URDF.")
                return

            chain = tree.getChain(self.base_link, self.tip_link)
            n = int(chain.getNrOfJoints())
            if n <= 0:
                self.get_logger().error(
                    f"KDL chain has 0 joints. Check base_link='{self.base_link}' and tip_link='{self.tip_link}'."
                )
                return

            # Extract only movable joints in chain order
            names: list[str] = []
            for i in range(chain.getNrOfSegments()):
                seg = chain.getSegment(i)
                j = seg.getJoint()
                jt = j.getTypeName()  # "None", "Fixed", "RotZ", ...
                if jt not in ("None", "Fixed"):
                    names.append(j.getName())
                    if len(names) == n:
                        break

            if len(names) != n:
                self.get_logger().error(
                    f"Joint-name extraction mismatch: KDL reports {n} joints but extracted {len(names)} names: {names}"
                )
                return

            self.chain = chain
            self.n_joints = n
            self.chain_joint_names = names
            self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
            self.jac_solver = kdl.ChainJntToJacSolver(self.chain)

            # If user didn't provide explicit command_joint_names, publish in chain order
            if len(self.command_joint_names) == 0:
                self.command_joint_names = list(self.chain_joint_names)
            else:
                missing = [jn for jn in self.command_joint_names if jn not in set(self.chain_joint_names)]
                if missing:
                    self.get_logger().warn(
                        f"command_joint_names contains names not in KDL chain: {missing}. "
                        "Those entries will be commanded as 0.0."
                    )

            self._kdl_ready = True
            self.get_logger().info(
                f"[{self.robot_id}] KDL READY. n_joints={self.n_joints}\n"
                f"  chain_joint_names={self.chain_joint_names}\n"
                f"  command_joint_names={self.command_joint_names}"
            )

            # stop listening after success
            self.destroy_subscription(self._robot_desc_sub)

        except Exception as e:
            self.get_logger().error(f"Exception creating KDL from URDF: {e}")

    # ------------------------
    # Joint state & EE pose
    # ------------------------

    def joint_state_callback(self, msg: JointState):
        self._last_joint_state_msg = msg
        self._js_name_to_idx = {n: i for i, n in enumerate(msg.name)}

    def ee_pose_callback(self, msg):
        pose = msg.pose if isinstance(msg, PoseStamped) else msg
        pos = pose.position
        ori = pose.orientation
        quat = normalize_quaternion(np.array([ori.x, ori.y, ori.z, ori.w], dtype=float))
        self.current_pose = {
            "position": np.array([pos.x, pos.y, pos.z], dtype=float),
            "rotation_kdl": kdl.Rotation.Quaternion(*quat),
            "orientation_xyzw": quat,
        }

    # ------------------------
    # Helpers for joint mapping
    # ------------------------

    def _get_q_chain(self) -> np.ndarray | None:
        """
        Build joint vector q in KDL chain order from the latest JointState.
        """
        if (self._last_joint_state_msg is None) or (self._js_name_to_idx is None) or (self.n_joints <= 0):
            return None

        q_chain = np.zeros(self.n_joints, dtype=float)
        missing = False
        for i, jn in enumerate(self.chain_joint_names):
            idx = self._js_name_to_idx.get(jn, None)
            if idx is None:
                missing = True
                q_chain[i] = 0.0
            else:
                q_chain[i] = float(self._last_joint_state_msg.position[idx])

        if missing:
            self.get_logger().warn(
                f"[{self.robot_id}] Some chain_joint_names were missing in JointState. "
                "This will break Cartesian control. Check naming consistency!"
            )

        return q_chain

    def _chain_to_cmd_order(self, qdot_chain: np.ndarray) -> np.ndarray:
        """
        Map qdot from chain order -> command_joint_names order.
        """
        name_to_chain_idx = {n: i for i, n in enumerate(self.chain_joint_names)}
        qdot_cmd = np.zeros(len(self.command_joint_names), dtype=float)
        for k, jn in enumerate(self.command_joint_names):
            ci = name_to_chain_idx.get(jn, None)
            qdot_cmd[k] = float(qdot_chain[ci]) if ci is not None else 0.0
        return qdot_cmd

    # ------------------------
    # Stop
    # ------------------------

    def publish_zero_velocity(self):
        if self._zero_published or (len(self.command_joint_names) <= 0):
            return
        msg = Float64MultiArray()
        msg.data = [0.0] * len(self.command_joint_names)
        self.pub.publish(msg)
        self._zero_published = True
        self.get_logger().info(f"[{self.robot_id}] Published zero velocity.")

    # ------------------------
    # Convergence (pose only)
    # ------------------------

    def has_converged(self) -> bool:
        if self.current_pose is None:
            return False

        target_pos = np.array(
            [self.target_pos_kdl.x(), self.target_pos_kdl.y(), self.target_pos_kdl.z()],
            dtype=float,
        )
        pos_error = float(np.linalg.norm(self.current_pose["position"] - target_pos))
        ori_error = float(
            np.linalg.norm(
                quaternion_log_error(self.target_quat_xyzw, self.current_pose["orientation_xyzw"])
            )
        )
        return (pos_error < self.pos_thresh) and (ori_error < self.ori_thresh)

    # ------------------------
    # Core DLS IK
    # ------------------------

    def compute_dls_ik(self, q_current: kdl.JntArray) -> np.ndarray:
        if not self._kdl_ready:
            return np.zeros(self.n_joints, dtype=float)

        # FK (always needed to update current_pose)
        current_frame = kdl.Frame()
        self.fk_solver.JntToCart(q_current, current_frame)

        current_pos = np.array(
            [current_frame.p.x(), current_frame.p.y(), current_frame.p.z()],
            dtype=float,
        )
        current_quat_xyzw = normalize_quaternion(
            np.array(current_frame.M.GetQuaternion(), dtype=float)
        )

        # update current_pose so bridge/custom_ds can read it
        self.current_pose = {
            "position": current_pos,
            "rotation_kdl": current_frame.M,
            "orientation_xyzw": current_quat_xyzw,
        }
        
        self.current_pose["target_kdl"] = self.target_pos_kdl
        # ==========================================================
        # 1) DS MODE: if a custom_ds is attached, use its twist
        # ==========================================================
        if self.custom_ds is not None:
            try:
                twist = np.asarray(self.custom_ds(self.current_pose), dtype=float).reshape(-1)
            except Exception as e:
                self.get_logger().error(
                    f"[{self.robot_id}] custom_ds raised exception: {e}"
                )
                twist = np.zeros(6, dtype=float)

            if twist.size != 6:
                self.get_logger().error(
                    f"[{self.robot_id}] custom_ds must return 6 values, got shape {twist.shape}. "
                    "Falling back to zero twist."
                )
                twist = np.zeros(6, dtype=float)

            v = twist[:3]
            w = twist[3:]

            # clip DS velocities to the same limits
            n_lin = float(np.linalg.norm(v))
            if n_lin > self.max_cartesian_vel:
                v = (v / max(n_lin, 1e-12)) * self.max_cartesian_vel

            n_ang = float(np.linalg.norm(w))
            if n_ang > self.max_angular_vel:
                w = (w / max(n_ang, 1e-12)) * self.max_angular_vel

            e = np.concatenate([v, w], axis=0)  # (6,)

        # ==========================================================
        # 2) CLASSIC MODE: no DS, just error to target
        # ==========================================================
        else:
            # linear error
            linear_vec = self.target_frame.p - kdl.Vector(*current_pos)
            linear = np.array(
                [linear_vec.x(), linear_vec.y(), linear_vec.z()],
                dtype=float,
            )

            # hemisphere continuity w.r.t. target
            tgt = self.target_quat_xyzw
            cur = current_quat_xyzw
            if float(np.dot(tgt, cur)) < 0.0:
                tgt = -tgt
                self.target_quat_xyzw = tgt
                self.target_rot_kdl = kdl.Rotation.Quaternion(*tgt)
                self.target_frame = kdl.Frame(self.target_rot_kdl, self.target_pos_kdl)

            angular = quaternion_log_error(self.target_quat_xyzw, cur)

            # NEW: weight position vs orientation
            linear  = self.k_pos * linear
            angular = self.k_ori * angular

            # clip (after weighting)
            n_lin = float(np.linalg.norm(linear))
            if n_lin > self.max_cartesian_vel:
                linear = (linear / max(n_lin, 1e-12)) * self.max_cartesian_vel

            n_ang = float(np.linalg.norm(angular))
            if n_ang > self.max_angular_vel:
                angular = (angular / max(n_ang, 1e-12)) * self.max_angular_vel

            e = np.concatenate([linear, angular], axis=0)

        # ==========================================================
        # 3) Common DLS step
        # ==========================================================
        jac = kdl.Jacobian(self.n_joints)
        self.jac_solver.JntToJac(q_current, jac)
        J = np.array(
            [[jac[i, j] for j in range(self.n_joints)] for i in range(6)],
            dtype=float,
        )

        lam2 = self.damping ** 2
        A = J @ J.T + lam2 * np.eye(6)
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            self.get_logger().error(f"[{self.robot_id}] DLS inversion failed!")
            return np.zeros(self.n_joints, dtype=float)

        dq_chain = (J.T @ A_inv) @ e
        return dq_chain

    # ------------------------
    # Control loop
    # ------------------------

    def _on_timer(self):
        # wait until we have KDL + at least one JointState
        if (not self._kdl_ready) or (self._last_joint_state_msg is None):
            return

        q_chain = self._get_q_chain()
        if q_chain is None:
            return

        # Build KDL joint array
        q_kdl = kdl.JntArray(self.n_joints)
        for i in range(self.n_joints):
            q_kdl[i] = float(q_chain[i])

        dq_chain = self.compute_dls_ik(q_kdl)
        self._last_dq_norm = float(np.linalg.norm(dq_chain))

        # Map into controller command order
        dq_cmd = self._chain_to_cmd_order(dq_chain)

        msg = Float64MultiArray()
        msg.data = dq_cmd.tolist()
        self.pub.publish(msg)

        # -----------------------------
        # NEW: debounced goal detection
        # -----------------------------
        pose_ok = self.has_converged()
        stopped_ok = (self._last_dq_norm is not None) and (self._last_dq_norm < self.dq_stop_thresh)

        if pose_ok and stopped_ok:
            self._hold_counter += 1
        else:
            self._hold_counter = 0
            self._goal_reached = False

        if self._hold_counter >= self.converge_hold_ticks:
            self._goal_reached = True

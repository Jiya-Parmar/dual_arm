#!/usr/bin/env python3
"""
Quintic joint-space trajectory planner.

- Plan a smooth trajectory between two joint configurations
  with specified position, velocity, and acceleration at both ends.
- Supports multiple joints (vectorized with numpy).
- Typical usage:
    planner = QuinticJointSpacePlanner(
        q_start=[0, 0, 0, 0, 0, 0, 0],
        q_goal=[0.5, -0.3, 0.2, 0.0, 0.4, -0.1, 0.2],
        duration=3.0
    )
    q, dq, ddq = planner.evaluate(t)

Author: (you / IITGN)
"""

from __future__ import annotations
from typing import Sequence, Tuple

import numpy as np


class QuinticJointSpacePlanner:
    """
    Quintic joint-space trajectory between two states.

    For each joint j, the trajectory is:
        q_j(t) = a0 + a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5

    Boundary conditions at t = 0 and t = T:
        q(0)  = q_start
        dq(0) = v_start
        ddq(0) = a_start

        q(T)  = q_goal
        dq(T) = v_goal
        ddq(T) = a_goal
    """

    def __init__(
        self,
        q_start: Sequence[float],
        q_goal: Sequence[float],
        duration: float,
        v_start: Sequence[float] | None = None,
        v_goal: Sequence[float] | None = None,
        a_start: Sequence[float] | None = None,
        a_goal: Sequence[float] | None = None,
    ) -> None:
        """
        Initialize the planner.

        Args:
            q_start: Initial joint positions [n_joints]
            q_goal:  Final joint positions [n_joints]
            duration: Total motion time T > 0 (seconds)
            v_start: Initial joint velocities [n_joints], default = 0
            v_goal:  Final joint velocities [n_joints], default = 0
            a_start: Initial joint accelerations [n_joints], default = 0
            a_goal:  Final joint accelerations [n_joints], default = 0
        """
        self.T = float(duration)
        if self.T <= 0.0:
            raise ValueError("duration must be positive")

        self.q_start = np.array(q_start, dtype=float).ravel()
        self.q_goal = np.array(q_goal, dtype=float).ravel()

        if self.q_start.shape != self.q_goal.shape:
            raise ValueError("q_start and q_goal must have the same shape")

        self.n_joints = self.q_start.size

        # Default boundary conditions: zero vel/acc
        if v_start is None:
            self.v_start = np.zeros(self.n_joints)
        else:
            self.v_start = np.array(v_start, dtype=float).ravel()

        if v_goal is None:
            self.v_goal = np.zeros(self.n_joints)
        else:
            self.v_goal = np.array(v_goal, dtype=float).ravel()

        if a_start is None:
            self.a_start = np.zeros(self.n_joints)
        else:
            self.a_start = np.array(a_start, dtype=float).ravel()

        if a_goal is None:
            self.a_goal = np.zeros(self.n_joints)
        else:
            self.a_goal = np.array(a_goal, dtype=float).ravel()

        # Sanity checks
        for arr_name, arr in [
            ("v_start", self.v_start),
            ("v_goal", self.v_goal),
            ("a_start", self.a_start),
            ("a_goal", self.a_goal),
        ]:
            if arr.shape != (self.n_joints,):
                raise ValueError(f"{arr_name} must be shape ({self.n_joints},), got {arr.shape}")

        # Coefficients matrix: shape (n_joints, 6)
        self.coeffs = np.zeros((self.n_joints, 6), dtype=float)

        # Precompute coefficients
        self._compute_coefficients()

    # ------------------------------------------------------------------
    # Internal: compute polynomial coefficients for all joints
    # ------------------------------------------------------------------
    def _compute_coefficients(self) -> None:
        T = self.T

        # Precompute the 3x3 system for a3, a4, a5 (per standard quintic derivation)
        T2 = T * T
        T3 = T2 * T
        T4 = T2 * T2
        T5 = T3 * T2

        # Matrix for unknowns [a3, a4, a5]^T
        M = np.array(
            [
                [T3,   T4,    T5],
                [3*T2, 4*T3,  5*T4],
                [6*T,  12*T2, 20*T3],
            ],
            dtype=float,
        )

        Minv = np.linalg.inv(M)

        for j in range(self.n_joints):
            q0 = self.q_start[j]
            qf = self.q_goal[j]
            v0 = self.v_start[j]
            vf = self.v_goal[j]
            acc0 = self.a_start[j]
            accf = self.a_goal[j]

            # First three coefficients from t = 0 boundary conditions
            a0 = q0
            a1 = v0
            a2 = acc0 / 2.0

            # Right-hand side for [a3, a4, a5]
            b1 = qf - (a0 + a1 * T + a2 * T2)
            b2 = vf - (a1 + 2.0 * a2 * T)
            b3 = accf - (2.0 * a2)

            b = np.array([b1, b2, b3], dtype=float)

            a3, a4, a5 = Minv @ b

            self.coeffs[j, :] = [a0, a1, a2, a3, a4, a5]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(self, t: float, clamp: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate q(t), dq(t), ddq(t) at time t.

        Args:
            t: Time (seconds)
            clamp: If True, clamp t into [0, T]. If False, use t as-is.

        Returns:
            q:   Joint positions [n_joints]
            dq:  Joint velocities [n_joints]
            ddq: Joint accelerations [n_joints]
        """
        if clamp:
            if t <= 0.0:
                t = 0.0
            elif t >= self.T:
                t = self.T

        t1 = t
        t2 = t1 * t1
        t3 = t2 * t1
        t4 = t2 * t2
        t5 = t3 * t2

        # Powers stacked for vectorized dot product per joint
        #   q(t)   = a0 + a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5
        #   dq(t)  = a1 + 2 a2 t + 3 a3 t^2 + 4 a4 t^3 + 5 a5 t^4
        #   ddq(t) = 2 a2 + 6 a3 t + 12 a4 t^2 + 20 a5 t^3
        a = self.coeffs  # (n_joints, 6)

        q = (
            a[:, 0]
            + a[:, 1] * t1
            + a[:, 2] * t2
            + a[:, 3] * t3
            + a[:, 4] * t4
            + a[:, 5] * t5
        )

        dq = (
            a[:, 1]
            + 2.0 * a[:, 2] * t1
            + 3.0 * a[:, 3] * t2
            + 4.0 * a[:, 4] * t3
            + 5.0 * a[:, 5] * t4
        )

        ddq = (
            2.0 * a[:, 2]
            + 6.0 * a[:, 3] * t1
            + 12.0 * a[:, 4] * t2
            + 20.0 * a[:, 5] * t3
        )

        return q, dq, ddq

    def duration(self) -> float:
        """Return total trajectory duration T."""
        return self.T

    def is_finished(self, t: float) -> bool:
        """Convenience: True if t >= T."""
        return t >= self.T


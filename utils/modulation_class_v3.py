"""
modulation_math.py
==================
Pure mathematics for DS-based obstacle avoidance (Khansari-Billard 2012).

No simulation loop, no hardware loop, no timing, no state integration.
Call from your own simulation runner or hardware runner.

Two classes
-----------
ModulationAvoider
    Stateless.  All methods are pure functions of their inputs.
    Call M(), E(), D(), modulate() at every control tick.

SaddlePointEscaper
    Stateful across ticks (tracks whether an escape is in progress).
    Call update() at every control tick.
    Returns (escape_velocity, is_escaping).
    - escape_velocity is zeros when no escape is needed.
    - is_escaping tells the caller to skip its normal DS command.

Caller responsibilities
-----------------------
Simulation runner:
    x = x + dt * (escape_vel if is_escaping else modulated_vel)

Hardware runner:
    command(escape_vel if is_escaping else modulated_vel)
    apply your own low-pass filtering, velocity limits, safety stops, etc.

See bottom of file for minimal usage examples.
"""

from __future__ import annotations
from typing import Callable, Optional
import numpy as np

_EPS = 1e-9


# ============================================================
# ModulationAvoider — stateless math
# ============================================================

class ModulationAvoider:
    """
    Computes the dynamic modulation matrix M(ξ̃) and the modulated
    velocity ξ̇ = M(ξ̃) f(ξ) for a single convex obstacle.

    All public methods are stateless — they depend only on their
    arguments.  No integration, no timing, no side effects.

    Parameters
    ----------
    gamma : callable (x_tilde) -> float
        Obstacle function.  γ < 1 inside, γ = 1 on boundary, γ > 1 outside.
        x_tilde = x - obstacle_center.
    grad_gamma : callable (x_tilde) -> np.ndarray
        Gradient of gamma w.r.t. x_tilde.  Must return shape (d,).
    rho : float
        Reactivity parameter ρ > 0.  Larger → deflects earlier / stronger.
    safety_eta : float
        Safety factor η ∈ (0, 1].  Inflates the obstacle.
        η = 1 means no inflation.
        The effective boundary seen by the robot is γ = 1/η.
    tail_effect : bool
        If True (default), remove the tail-effect correction:
        λ₁ is set to 1 when the robot moves away from the obstacle,
        so the trajectory is not bent after passing it.
    lambda_min : float
        Floor on λ₁ (normal eigenvalue).
    lambda_max_factor : float
        λ_τ is capped at 1 + lambda_max_factor.
    """

    def __init__(
        self,
        gamma       : Callable[[np.ndarray], float],
        grad_gamma  : Callable[[np.ndarray], np.ndarray],
        rho         : float = 1.0,
        safety_eta  : float = 1.0,
        tail_effect : bool  = True,
        lambda_min  : float = -10.0,
        lambda_max_factor : float = 100.0,
    ):
        assert rho > 0,             "rho must be > 0"
        assert 0 < safety_eta <= 1, "safety_eta must be in (0, 1]"

        self.gamma             = gamma
        self.grad_gamma        = grad_gamma
        self.rho               = float(rho)
        self.safety_eta        = float(safety_eta)
        self.tail_effect       = bool(tail_effect)
        self.lambda_min        = float(lambda_min)
        self.lambda_max_factor = float(lambda_max_factor)

    # ----------------------------------------------------------
    # Primary call: modulated velocity
    # ----------------------------------------------------------

    def modulate(
        self,
        x       : np.ndarray,
        x_tilde : np.ndarray,
        f       : np.ndarray,
    ) -> np.ndarray:
        """
        Compute modulated velocity  ξ̇ = M(ξ̃) f(ξ)
        and enforce impenetrability at the safety-inflated boundary.

        Parameters
        ----------
        x       : robot position in world frame, shape (d,)
        x_tilde : x - obstacle_center, shape (d,)
        f       : original DS velocity at x, shape (d,)

        Returns
        -------
        dx_dt : modulated velocity, shape (d,)
        """
        x       = np.asarray(x,       float)
        x_tilde = np.asarray(x_tilde, float)
        f       = np.asarray(f,       float)

        dx_dt = self.M(x_tilde, f=f).dot(f)

        # --- Impenetrability clamp ---
        # If inside the inflated boundary, remove any inward normal component.
        x_tilde_safe      = x_tilde * self.safety_eta
        n                 = self._get_normal(x_tilde_safe)
        n_unit            = n / (np.linalg.norm(n) + _EPS)
        gamma_val         = self.gamma(x_tilde)
        effective_boundary = 1.0 / self.safety_eta   # equivalent: γ(x̃_safe) = 1

        if gamma_val <= effective_boundary:
            normal_vel = float(n_unit.dot(dx_dt))
            if normal_vel < 0:
                dx_dt = dx_dt - normal_vel * n_unit

        return dx_dt

    # ----------------------------------------------------------
    # Modulation matrix  M = E D E⁻¹
    # ----------------------------------------------------------

    def M(
        self,
        x_tilde : np.ndarray,
        f       : Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute the d×d modulation matrix M(ξ̃) = E(ξ̃) D(ξ̃) E(ξ̃)⁻¹.

        Paper Eq. (15).

        The tail-effect correction (Eq. 23) requires the *modulated*
        velocity ξ̇ = M·f for its sign check — not f itself.
        We use a two-pass approach: first build M without the tail-effect
        correction to get a preliminary ξ̇, then recompute D with the
        correct sign check.

        Parameters
        ----------
        x_tilde : x - obstacle_center, shape (d,)
        f       : original DS velocity.  If None, tail-effect is skipped.
        """
        x_tilde      = np.asarray(x_tilde, float)
        x_tilde_safe = x_tilde * self.safety_eta
        E            = self.E(x_tilde_safe)

        try:
            E_inv = np.linalg.inv(E)
        except np.linalg.LinAlgError:
            E_inv = np.linalg.pinv(E)

        if f is None or not self.tail_effect:
            # No tail-effect correction needed
            D = self.D(x_tilde, modulated_vel=None)
            return E.dot(D).dot(E_inv)

        # Pass 1: M without tail-effect → preliminary modulated velocity
        D_no_tail  = self.D(x_tilde, modulated_vel=None)
        M_prelim   = E.dot(D_no_tail).dot(E_inv)
        f_mod_prelim = M_prelim.dot(np.asarray(f, float))

        # Pass 2: D with correct sign check on modulated velocity (Eq. 23)
        D = self.D(x_tilde, modulated_vel=f_mod_prelim)
        return E.dot(D).dot(E_inv)

    # ----------------------------------------------------------
    # Basis matrix  E = [n | e¹ | … | eᵈ⁻¹]
    # ----------------------------------------------------------

    def E(self, x_tilde: np.ndarray) -> np.ndarray:
        """
        Build the d×d basis matrix whose first column is the
        unit normal and remaining columns span the tangent hyperplane.

        Paper Eq. (16).

        Parameters
        ----------
        x_tilde : position in *safety-inflated* obstacle frame, shape (d,)
        """
        n      = self._get_normal(x_tilde)
        d      = n.shape[0]
        n_unit = n / (np.linalg.norm(n) + _EPS)

        tangents = []
        for i in range(d):
            ei   = np.zeros(d); ei[i] = 1.0
            vi   = ei - np.dot(ei, n_unit) * n_unit
            for t in tangents:
                vi = vi - np.dot(vi, t) * t
            norm_vi = np.linalg.norm(vi)
            if norm_vi > 1e-8:
                tangents.append(vi / norm_vi)
            if len(tangents) >= d - 1:
                break

        # Fallback: QR decomposition
        if len(tangents) < d - 1:
            Q, _ = np.linalg.qr(np.random.randn(d, d))
            tangents = [
                Q[:, i] for i in range(d)
                if abs(np.dot(Q[:, i], n_unit)) < 1 - 1e-3
            ][:d - 1]

        return np.column_stack([n_unit] + tangents[:d - 1])

    # ----------------------------------------------------------
    # Eigenvalue matrix  D = diag(λ₁, λ_τ, …, λ_τ)
    # ----------------------------------------------------------

    def D(
        self,
        x_tilde       : np.ndarray,
        modulated_vel : Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build the diagonal eigenvalue matrix D(ξ̃).

        Paper Eq. (18) with reactivity ρ (Eq. 22) and optional
        tail-effect correction (Eq. 23).

        Parameters
        ----------
        x_tilde       : x - obstacle_center (UN-inflated), shape (d,)
        modulated_vel : the *modulated* velocity ξ̇ = M·f, used for the
                        tail-effect sign check (Eq. 23).
                        Pass None to skip the tail-effect correction.

        Notes
        -----
        The tail-effect check must use the modulated velocity ξ̇,
        NOT the original DS velocity f.  Callers (including M()) are
        responsible for passing the right vector here.
        """
        x_tilde_safe = x_tilde * self.safety_eta
        gamma_val    = abs(float(self.gamma(x_tilde_safe))) + _EPS
        d            = x_tilde.shape[0]

        denom   = gamma_val ** (1.0 / self.rho)
        lam_n   = 1.0 - 1.0 / denom   # λ₁ — normal eigenvalue
        lam_tau = 1.0 + 1.0 / denom   # λ_τ — tangent eigenvalues

        # --- Tail-effect correction (Eq. 23) ---
        # λ₁ = 1 when the robot is moving AWAY from the obstacle
        # i.e. n(ξ̃)ᵀ ξ̇ ≥ 0
        # Must use the modulated velocity ξ̇, not the raw DS f.
        if self.tail_effect and modulated_vel is not None:
            n      = self._get_normal(x_tilde_safe)
            n_unit = n / (np.linalg.norm(n) + _EPS)
            if float(np.dot(n_unit, modulated_vel)) >= 0:
                lam_n = 1.0

        lam_n   = max(lam_n,   self.lambda_min)
        lam_tau = min(lam_tau, 1.0 + self.lambda_max_factor)

        return np.diag([lam_n] + [lam_tau] * (d - 1))

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _get_normal(self, x_tilde: np.ndarray) -> np.ndarray:
        """Return grad_gamma evaluated at x_tilde.  Shape (d,)."""
        n = np.asarray(self.grad_gamma(x_tilde), float)
        if n.ndim != 1 or n.shape[0] != x_tilde.shape[0]:
            raise ValueError("grad_gamma must return a 1D vector of the same dimension as x_tilde")
        return n

    # ----------------------------------------------------------
    # Static gamma / grad_gamma factories
    # ----------------------------------------------------------

    @staticmethod
    def gamma_sphere(x: np.ndarray, center: np.ndarray, radius: float) -> float:
        return float(np.linalg.norm(x - center) / (radius + _EPS))

    @staticmethod
    def grad_gamma_sphere(x: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
        v = x - center
        return (v / (np.linalg.norm(v) + _EPS)) / (radius + _EPS)

    @staticmethod
    def gamma_ellipsoid(x: np.ndarray, center: np.ndarray, A: np.ndarray) -> float:
        v = x - center
        return float(np.sqrt(np.dot(v, A.dot(v)) + _EPS))

    @staticmethod
    def grad_gamma_ellipsoid(x: np.ndarray, center: np.ndarray, A: np.ndarray) -> np.ndarray:
        v = x - center
        return A.dot(v) / (np.sqrt(np.dot(v, A.dot(v)) + _EPS) + _EPS)

    @staticmethod
    def gamma_superellipsoid(
        x: np.ndarray, center: np.ndarray, radii: np.ndarray, p: float
    ) -> float:
        v = (np.asarray(x) - np.asarray(center)) / np.asarray(radii)
        return float((np.sum(np.abs(v) ** p) + _EPS) ** (1.0 / p))

    @staticmethod
    def grad_gamma_superellipsoid(
        x: np.ndarray, center: np.ndarray, radii: np.ndarray, p: float
    ) -> np.ndarray:
        x      = np.asarray(x);      center = np.asarray(center)
        radii  = np.asarray(radii)
        v      = (x - center) / radii
        abs_v  = np.abs(v)
        S      = np.sum(abs_v ** p) + _EPS
        coeff  = S ** (1.0 / p - 1.0)
        return coeff * ((abs_v ** (p - 2)) * v / radii)


# ============================================================
# SaddlePointEscaper — stateful across ticks, math only
# ============================================================

class SaddlePointEscaper:
    """
    Implements Algorithm 1 from Khansari-Billard 2012 for handling
    saddle points and local minima on the obstacle boundary.

    Stateful: tracks whether an escape manoeuvre is in progress.

    Call update() at every control tick.  It returns:
        escape_velocity : np.ndarray  — zeros when no escape needed
        is_escaping     : bool        — True while escape is active

    The caller decides how to use the output:
        Simulation:  x_next = x + dt * (escape_vel if is_escaping else v_mod)
        Hardware:    command(escape_vel if is_escaping else v_mod)

    Hardware-specific concerns (filtering, velocity limits, safety stops)
    are intentionally left to the caller.  This class only decides:
        1. Are we stuck?  (stuck_ticks_required consecutive ticks below thresh)
        2. Which direction to escape?  (Algorithm 1, tangent cost selection)
        3. What velocity to command?   (displacement / time, clipped to max_escape_vel)
        4. Is it safe to hand back control?  (gamma + velocity check)

    Parameters
    ----------
    max_escape_vel : float
        Maximum speed during escape (m/s or rad/s).  Tune per robot.
    stuck_vel_thresh : float
        Modulated velocity magnitude below which the robot is considered stuck.
    stuck_ticks_required : int
        How many consecutive ticks below thresh before escape triggers.
        Use 1 for simulation (deterministic), 10–30 for hardware (noisy).
    vel_filter_alpha : float
        IIR smoothing on ||v_mod|| for the stuck check.
        1.0 = no filtering (simulation), 0.05–0.2 for hardware.
        alpha = 1 disables filtering entirely (instantaneous check).
    cooldown_ticks : int
        Ticks to wait after an escape before allowing re-trigger.
        0 is fine for simulation.  50–100 recommended for hardware.
    resume_gamma_margin : float
        Resume normal DS when gamma > (1/safety_eta) * resume_gamma_margin.
        1.05 means 5% outside the inflated boundary.
    """

    def __init__(
        self,
        max_escape_vel        : float = 0.05,
        stuck_vel_thresh      : float = 0.01,
        stuck_ticks_required  : int   = 1,
        vel_filter_alpha      : float = 1.0,
        cooldown_ticks        : int   = 0,
        resume_gamma_margin   : float = 1.5,
    ):
        assert 0 < vel_filter_alpha <= 1.0, "vel_filter_alpha must be in (0, 1]"

        self.max_escape_vel       = float(max_escape_vel)
        self.stuck_vel_thresh     = float(stuck_vel_thresh)
        self.stuck_ticks_required = int(stuck_ticks_required)
        self.vel_filter_alpha     = float(vel_filter_alpha)
        self.cooldown_ticks_total = int(cooldown_ticks)
        self.resume_gamma_margin  = float(resume_gamma_margin)

        self._reset_state()

    # ----------------------------------------------------------
    # Public interface
    # ----------------------------------------------------------

    @property
    def is_escaping(self) -> bool:
        return self._escaping

    def reset(self):
        """Hard reset — call when starting a new motion or after an error."""
        self._reset_state()

    def update(
        self,
        x          : np.ndarray,
        obs_center : np.ndarray,
        avoider    : ModulationAvoider,
        f_nom      : Callable[[np.ndarray], np.ndarray],
        goal       : np.ndarray,
        dt         : float,
        alpha      : float = 0.005,
        trigger_zone : float = 1.1,
        max_iters  : int   = 500,
    ) -> tuple[np.ndarray, bool]:
        """
        Compute escape velocity for this tick.

        Parameters
        ----------
        x          : current robot position, shape (d,)
        obs_center : obstacle center in world frame, shape (d,)
        avoider    : ModulationAvoider instance configured for this obstacle
        f_nom      : callable, returns original DS velocity at a position
        goal       : goal position, shape (d,)  (used for tangent selection)
        dt         : control timestep [s]
        alpha      : step size along tangent in Algorithm 1
        trigger_zone : gamma multiplier above effective boundary to start
                       checking for stuck condition (paper: exactly 1.0,
                       practical: 1.05–1.2 to account for numeric drift)
        max_iters  : maximum iterations in Algorithm 1 escape loop

        Returns
        -------
        escape_velocity : np.ndarray, shape (d,)
            Non-zero only while is_escaping is True.
            Caller should command this instead of normal DS velocity.
        is_escaping : bool
            True while an escape manoeuvre is in progress.
        """
        x          = np.asarray(x,          float)
        obs_center = np.asarray(obs_center, float)
        goal       = np.asarray(goal,       float)

        # --- Cooldown: wait after previous escape before re-triggering ---
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return np.zeros_like(x), False

        # --- Already escaping: sustain velocity until ticks are done ---
        if self._escaping:
            self._ticks_remaining -= 1

            if self._ticks_remaining > 0:
                return self._v_escape.copy(), True

            # Ticks finished — check whether it is safe to hand back control.
            # We use the current real position + live gamma + modulated velocity.
            # This is robust to obstacle motion and avoids fragile forward sim.
            x_tilde_now        = x - obs_center
            gamma_now          = avoider.gamma(x_tilde_now)
            effective_boundary = 1.0 / avoider.safety_eta
            f_now              = np.asarray(f_nom(x), float)
            v_mod_now          = avoider.M(x_tilde_now, f=f_now).dot(f_now)

            outside   = gamma_now > effective_boundary * self.resume_gamma_margin
            has_vel   = np.linalg.norm(v_mod_now) > self.stuck_vel_thresh

            if outside and has_vel:
                # Safe — hand back control
                self._escaping           = False
                self._v_escape           = np.zeros_like(x)
                self._ticks_remaining    = 0
                self._stuck_counter      = 0
                self._vel_mag_filtered   = 0.0
                self._cooldown_remaining = self.cooldown_ticks_total
                return np.zeros_like(x), False
            else:
                # Not safe yet — extend escape in same direction
                self._ticks_remaining = max(10, self.stuck_ticks_required)
                return self._v_escape.copy(), True

        # --- Not escaping: check trigger conditions ---

        x_tilde            = x - obs_center
        x_tilde_safe       = x_tilde * avoider.safety_eta
        effective_boundary = 1.0 / avoider.safety_eta
        gamma_val          = avoider.gamma(x_tilde)

        # Condition 1 (paper): Γ(ξ̃) = 1 — robot is near the boundary
        # trigger_zone > 1 because robot asymptotically approaches γ = 1/η
        if gamma_val > effective_boundary * trigger_zone:
            self._stuck_counter    = 0
            self._vel_mag_filtered = 0.0
            return np.zeros_like(x), False

        # Condition 2 (paper): ξ̇ = 0 — robot is stuck
        f_here = np.asarray(f_nom(x), float)
        v_mod  = avoider.M(x_tilde, f=f_here).dot(f_here)
        v_mag  = float(np.linalg.norm(v_mod))

        # IIR filter — when vel_filter_alpha = 1.0 this is instantaneous
        self._vel_mag_filtered = (
            self.vel_filter_alpha * v_mag
            + (1.0 - self.vel_filter_alpha) * self._vel_mag_filtered
        )

        if self._vel_mag_filtered >= self.stuck_vel_thresh:
            self._stuck_counter = 0
            return np.zeros_like(x), False

        self._stuck_counter += 1
        if self._stuck_counter < self.stuck_ticks_required:
            return np.zeros_like(x), False

        # Both conditions confirmed — run Algorithm 1 internally
        self._stuck_counter    = 0
        self._vel_mag_filtered = 0.0
        
        n_at_boundary  = avoider._get_normal(x_tilde_safe)
        pos_scale       = float(np.linalg.norm(x_tilde_safe))
        grad_scale      = 1.0 / (np.linalg.norm(n_at_boundary) + _EPS)
        obstacle_scale  = max(pos_scale, grad_scale)
        max_escape_dist = max(0.30, 4.0 * obstacle_scale)
        
        escape_vel, n_ticks = self._run_algorithm1(
            x, x_tilde, x_tilde_safe, obs_center,
            avoider, f_nom, goal,
            alpha, dt, effective_boundary, max_iters,
            max_escape_dist=max_escape_dist,
        )

        self._escaping        = True
        self._v_escape        = escape_vel
        self._ticks_remaining = n_ticks

        return self._v_escape.copy(), True

    # ----------------------------------------------------------
    # Algorithm 1 — internal, runs on a position copy
    # ----------------------------------------------------------

    def _run_algorithm1(
        self,
        x            : np.ndarray,
        x_tilde      : np.ndarray,
        x_tilde_safe : np.ndarray,
        obs_center   : np.ndarray,
        avoider      : ModulationAvoider,
        f_nom        : Callable,
        goal         : np.ndarray,
        alpha        : float,
        dt           : float,
        effective_boundary : float,
        max_iters    : int,
        max_escape_dist : float = 0.10,
    ) -> tuple[np.ndarray, int]:
        """
        Run Algorithm 1 on a copy of the position.
        Returns (escape_velocity, ticks_needed).

        Paper Algorithm 1 (exact):
            while true:
                ξᵗ⁺¹ ← ξᵗ + α · eⁱ · δt
                ξ̇ᵗ⁺¹ ← M(ξ̃ᵗ⁺¹) f(ξᵗ⁺¹)
                exit if (eⁱ)ᵀ ξ̇ᵗ⁺¹ > 0  or  n(ξ̃)ᵀ ξ̇ᵗ⁺¹ > 0
        """

        MAX_ESCAPE_DIST = max_escape_dist
        
           

        # Pick best tangent direction (handles both saddle and local-min cases)
        E        = avoider.E(x_tilde_safe)
        tangents = [E[:, i] for i in range(1, E.shape[1])]
        e_i      = self._pick_tangent(
            tangents, x, obs_center, avoider,
            goal, alpha, dt, effective_boundary,
        )

        x_sim       = x.copy()
        x_start     = x.copy()
        steps_taken = 0

        for _ in range(max_iters):
            # Paper line 5: ξᵗ⁺¹ ← ξᵗ + α · eⁱ · δt
            x_sim       = x_sim + alpha * e_i * dt
            steps_taken += 1

            if np.linalg.norm(x_sim - x_start) > MAX_ESCAPE_DIST:
                break

            # Paper line 6: ξ̇ᵗ⁺¹ = M(ξ̃ᵗ⁺¹) f(ξᵗ⁺¹)
            x_tilde_sim      = x_sim - obs_center
            x_tilde_safe_sim = x_tilde_sim * avoider.safety_eta
            f_sim            = np.asarray(f_nom(x_sim), float)
            v_new            = avoider.M(x_tilde_sim, f=f_sim).dot(f_sim)

            n_sim  = avoider._get_normal(x_tilde_safe_sim)
            n_unit = n_sim / (np.linalg.norm(n_sim) + _EPS)

            # Paper lines 7-8: exit conditions
            tangent_exit = float(np.dot(e_i, v_new))   > 0.0
            normal_exit  = float(np.dot(n_unit, v_new)) > 0.0

            if tangent_exit or normal_exit:
                break

        # Escape velocity = total displacement / total time
        displacement = x_sim - x_start
        time_needed  = max(steps_taken * dt, _EPS)
        v_escape     = displacement / time_needed

        # Clip to max_escape_vel, adjust tick count so robot travels full distance
        v_norm = float(np.linalg.norm(v_escape))
        if v_norm > self.max_escape_vel:
            v_escape    = v_escape / v_norm * self.max_escape_vel
            dist_needed = float(np.linalg.norm(displacement))
            steps_taken = max(1, int(dist_needed / (self.max_escape_vel * dt)) + 1)

        return v_escape, steps_taken

    # ----------------------------------------------------------
    # Tangent selection — cost = steps_to_escape + dist_to_goal
    # ----------------------------------------------------------

    def _pick_tangent(
        self,
        tangents           : list[np.ndarray],
        x_start            : np.ndarray,
        obs_center         : np.ndarray,
        avoider            : ModulationAvoider,
        goal               : np.ndarray,
        alpha              : float,
        dt                 : float,
        effective_boundary : float,
        max_steps          : int = 300,
    ) -> np.ndarray:
        """
        Pick the tangent direction (±eⁱ) with lowest cost:
            cost = steps_to_leave_boundary + remaining_dist_to_goal / (alpha * dt)

        Works correctly for both saddle points (all ~1 step, picks goal-aligned)
        and local minima (picks shortest path to nearest saddle point toward goal).
        """
        def cost(tangent: np.ndarray) -> float:
            x_test = x_start.copy()

            # Push slightly inside the boundary so escape distance is meaningful
            # (robot may be numerically just outside γ = 1/η)
            x_tilde_s = x_start - obs_center
            # n_in      = avoider._get_normal(x_tilde_s * avoider.safety_eta)
            # n_in_unit = n_in / (np.linalg.norm(n_in) + _EPS)
            x_test = x_start.copy()   # scale-consistent push

            for i in range(max_steps):
                x_test = x_test + alpha * tangent * dt
                xt     = x_test - obs_center
                if avoider.gamma(xt) > effective_boundary:
                    remaining = float(np.linalg.norm(goal - x_test))
                    return i + remaining / (alpha * dt)

            # Never escaped — very high cost
            return max_steps + float(np.linalg.norm(goal - x_test)) / (alpha * dt)

        best_cost = np.inf
        best_dir  = tangents[0]

        for t in tangents:
            for direction in [t, -t]:
                c = cost(direction)
                if c < best_cost:
                    best_cost = c
                    best_dir  = direction

        return best_dir

    # ----------------------------------------------------------
    # Internal state management
    # ----------------------------------------------------------

    def _reset_state(self):
        self._escaping           = False
        self._v_escape           = np.zeros(3)
        self._ticks_remaining    = 0
        self._stuck_counter      = 0
        self._vel_mag_filtered   = 0.0
        self._cooldown_remaining = 0


# ============================================================
# Usage examples
# ============================================================

"""
--- SHARED SETUP (same in both runners) ---

import numpy as np
from modulation_math import ModulationAvoider, SaddlePointEscaper

obs_center = np.array([0.0, 0.0, 0.0])
obs_radius = 0.3

avoider = ModulationAvoider(
    gamma      = lambda xt: ModulationAvoider.gamma_sphere(xt, np.zeros(3), obs_radius),
    grad_gamma = lambda xt: ModulationAvoider.grad_gamma_sphere(xt, np.zeros(3), obs_radius),
    rho        = 1.0,
    safety_eta = 0.9,
    tail_effect = True,
)

def f_nom(x):
    goal = np.array([1.0, 0.0, 0.0])
    return -2.0 * (x - goal)


--- SIMULATION RUNNER ---

escaper = SaddlePointEscaper(
    max_escape_vel       = 0.05,
    stuck_vel_thresh     = 0.01,
    stuck_ticks_required = 1,       # single tick is fine — no noise
    vel_filter_alpha     = 1.0,     # no filtering needed
    cooldown_ticks       = 0,
    resume_gamma_margin  = 1.05,
)

x  = np.array([-1.0, 0.0, 0.0])
dt = 0.01

for _ in range(5000):
    x_tilde   = x - obs_center
    f         = f_nom(x)
    v_mod     = avoider.modulate(x, x_tilde, f)

    escape_vel, is_escaping = escaper.update(
        x, obs_center, avoider, f_nom,
        goal=np.array([1.0, 0.0, 0.0]),
        dt=dt,
    )

    vel = escape_vel if is_escaping else v_mod
    x   = x + dt * vel


--- HARDWARE RUNNER ---

escaper = SaddlePointEscaper(
    max_escape_vel       = 0.03,    # conservative for real robot
    stuck_vel_thresh     = 0.008,
    stuck_ticks_required = 20,      # ~40ms at 500Hz before triggering
    vel_filter_alpha     = 0.1,     # smooth over ~10 ticks
    cooldown_ticks       = 100,     # ~200ms cooldown at 500Hz
    resume_gamma_margin  = 1.08,
)

# In your hardware control loop (called at fixed dt):
def control_tick(x_current, dt):
    x_tilde = x_current - obs_center
    f       = f_nom(x_current)

    # Apply your own low-pass filter to x_current before passing in
    # Apply your own safety checks after getting vel out

    v_mod = avoider.modulate(x_current, x_tilde, f)

    escape_vel, is_escaping = escaper.update(
        x_current, obs_center, avoider, f_nom,
        goal=np.array([1.0, 0.0, 0.0]),
        dt=dt,
    )

    vel = escape_vel if is_escaping else v_mod

    # Your hardware layer handles: joint limit checks, torque limits,
    # collision stops, watchdogs, etc.
    return vel
"""
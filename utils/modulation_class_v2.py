from typing import Callable, Optional
import numpy as np

_EPS = 1e-9


class ModulationAvoider:

    def __init__(
        self,
        gamma: Callable[[np.ndarray], float],
        grad_gamma: Callable[[np.ndarray], np.ndarray],
        dim: Optional[int] = None,
        rho: float = 1.0,
        safety_eta: float = 1.0,
        tail_effect: bool = True,
        lambda_min: float = -10.0,
        lambda_max_factor: float = 100.0,
    ):
        assert rho > 0, "rho must be > 0"
        assert 0 < safety_eta <= 1.0, "safety_eta must be in (0,1]"

        self.gamma = gamma
        self.grad_gamma = grad_gamma
        self.rho = float(rho)
        self.safety_eta = float(safety_eta)
        self.tail_effect = bool(tail_effect)
        self.lambda_min = lambda_min
        self.lambda_max_factor = lambda_max_factor
        self.dim = dim

    # ---------------------- Public API ----------------------

    def modulate(self, x: np.ndarray, x_tilde: np.ndarray, f: np.ndarray) -> np.ndarray:
        """Compute modulated velocity dx_dt = M(x_tilde) f(x)."""

        x = np.asarray(x, float)
        x_tilde = np.asarray(x_tilde, float)
        f = np.asarray(f, float)

        d = x.shape[0]
        if self.dim is None:
            self.dim = d
        assert d == self.dim, "x dimension mismatch"

        # Safety-inflated modulation anchor
        # x_tilde_safe = x_tilde / self.safety_eta
        M = self.M(x_tilde, f=f)
        dx_dt = M.dot(f)

        # Enforce impenetrability
        x_tilde_safe = x_tilde * self.safety_eta
        n = self._get_normal(x_tilde_safe)
        n_unit = n / (np.linalg.norm(n) + _EPS)

        gamma_val = self.gamma(x_tilde)
        effective_boundary = 1/self.safety_eta

        if gamma_val <= effective_boundary:
            normal_vel = float(n_unit.dot(dx_dt))
            if normal_vel < 0:
                dx_dt = dx_dt - normal_vel * n_unit

        return dx_dt

    def M(self, x_tilde: np.ndarray, f: Optional[np.ndarray] = None) -> np.ndarray:
        x_tilde = np.asarray(x_tilde, float)
        x_tilde_safe = x_tilde * self.safety_eta
        E = self.E(x_tilde_safe)
        D = self.D(x_tilde, f=f)

        try:
            E_inv = np.linalg.inv(E)
        except np.linalg.LinAlgError:
            E_inv = np.linalg.pinv(E)

        return E.dot(D).dot(E_inv)

    def E(self, x_tilde: np.ndarray) -> np.ndarray:
        n = self._get_normal(x_tilde)
        d = n.shape[0]
        n_unit = n / (np.linalg.norm(n) + _EPS)

        tangents = []
        for i in range(d):
            ei = np.zeros(d)
            ei[i] = 1.0
            vi = ei - np.dot(ei, n_unit) * n_unit

            for t in tangents:
                vi = vi - np.dot(vi, t) * t

            norm_vi = np.linalg.norm(vi)
            if norm_vi > 1e-8:
                tangents.append(vi / norm_vi)
            if len(tangents) >= d - 1:
                break

        if len(tangents) < d - 1:
            rand_mat = np.random.randn(d, d)
            Q, _ = np.linalg.qr(rand_mat)
            tangents = [Q[:, i] for i in range(d) if abs(np.dot(Q[:, i], n_unit)) < 1 - 1e-3]
            tangents = tangents[: max(0, d - 1)]

        E = np.column_stack([n_unit] + tangents[: d - 1])
        return E

    def D(self, x_tilde: np.ndarray, f: Optional[np.ndarray] = None) -> np.ndarray:
        # gamma_raw = abs(float(self.gamma(x_tilde))) + _EPS
        
        #Inflate obstacle boundary
        x_tilde_safe = x_tilde * self.safety_eta
        gamma_val = abs(float(self.gamma(x_tilde_safe))) + _EPS
        d = x_tilde.shape[0]

        denom = gamma_val ** (1.0 / self.rho)
        lam_n = 1.0 - 1.0 / denom
        lam_tau = 1.0 + 1.0 / denom

        # Tail-effect
        if self.tail_effect and f is not None:
            x_tilde_safe = x_tilde * self.safety_eta
            n = self._get_normal(x_tilde_safe)
            n_unit = n / (np.linalg.norm(n) + _EPS)
            if float(np.dot(n_unit, f)) >= 0:
                lam_n = 1.0

        lam_n = max(lam_n, self.lambda_min)
        lam_tau = min(lam_tau, 1.0 + self.lambda_max_factor)

        return np.diag([lam_n] + [lam_tau] * (d - 1))

    # ---------------------- Helpers ----------------------

    def _get_normal(self, x_tilde: np.ndarray) -> np.ndarray:
        n = np.asarray(self.grad_gamma(x_tilde), float)
        if n.ndim != 1 or n.shape[0] != x_tilde.shape[0]:
            raise ValueError("grad_gamma must return a 1D vector of same dimension")
        return n

    # ---------------------- Static Utility Methods ----------------------

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
    def gamma_superellipsoid(x, center, radii, p):
        x = np.asarray(x)
        center = np.asarray(center)
        radii = np.asarray(radii)

        v = (x - center) / radii
        S = np.sum(np.abs(v) ** p)

        return (S + 1e-9) ** (1.0 / p)

    @staticmethod
    def grad_gamma_superellipsoid(x, center, radii, p):
        x = np.asarray(x)
        center = np.asarray(center)
        radii = np.asarray(radii)

        v = (x - center) / radii
        abs_v = np.abs(v)

        S = np.sum(abs_v ** p) + 1e-9
        coeff = S ** (1.0 / p - 1.0)

        term = (abs_v ** (p - 2)) * v
        grad = coeff * (term / radii)

        return grad
    

# # ============================================================
# # Saddle Point Escaper — Algorithm 1, Khansari-Billard 2012
# # ============================================================

# class SaddlePointEscaper:
#     """
#     Implements Algorithm 1 from Khansari-Billard paper exactly.

#     Paper trigger conditions:
#         Γ(ξ̃) = 1  AND  ||ξ̇|| = 0

#     Paper loop:
#         ξt+1  ←  ξt + α·eⁱ·δt
#         ξ̇t+1  ←  M(ξ̃t+1)·f(ξt+1)
#         exit if (eⁱ)ᵀ·ξ̇t+1 > 0  OR  n(ξ̃)ᵀ·ξ̇t+1 > 0

#     Hardware/simulation compatible:
#         - Runs Algorithm 1 internally on a position copy
#         - Returns equivalent escape velocity sustained over
#           exactly enough ticks to physically move robot to
#           the escaped position
#     """

#     def __init__(self):
#         self._escaping         = False
#         self._v_escape         = np.zeros(3)
#         self._ticks_remaining  = 0

#     @property
#     def is_escaping(self) -> bool:
#         return self._escaping

#     def reset(self):
#         self._escaping         = False
#         self._v_escape         = np.zeros(3)
#         self._ticks_remaining  = 0

#     # ----------------------------------------------------------
#     # Main entry — call every control tick
#     # ----------------------------------------------------------

#     def update(
#         self,
#         x                : np.ndarray,
#         obs_center       : np.ndarray,
#         avoider          : ModulationAvoider,
#         f_nom            : Callable[[np.ndarray], np.ndarray],
#         goal             : np.ndarray,
#         dt               : float,
#         alpha            : float = 0.005,
#         stuck_vel_thresh : float = 0.01,
#         max_escape_vel   : float = 0.05,
#         trigger_zone     : float = 1.1,
#         max_iters        : int   = 500,
#     ) -> np.ndarray:
#         """
#         Returns escape velocity when triggered, zeros otherwise.

#         Simulation : x = x + dt * v_escape   (integrate directly)
#         Hardware   : command v_escape as cartesian velocity
#         Both       : skip normal DS while is_escaping == True
#         """

#         # ── Already mid-escape: sustain velocity until ticks done ──
#         if self._escaping:
#             self._ticks_remaining -= 1
            
#             if self._ticks_remaining <= 0:
#                 # Simulate 100 steps of normal DS from current position
#                 # Only hand back control if robot stays clear of obstacle
#                 x_sim_check  = x.copy()
#                 will_reenter = False
                
#                 for _ in range(100):
#                     xt_check = x_sim_check - obs_center
#                     f_check  = np.asarray(f_nom(x_sim_check), float)
                    
#                     # Clamp f_check magnitude
#                     f_norm = np.linalg.norm(f_check)
#                     if f_norm > 0.05:
#                         f_check = f_check / f_norm * 0.05
                        
#                     v_check      = avoider.M(xt_check, f=f_check).dot(f_check)
#                     x_sim_check  = x_sim_check + dt * v_check
                    
#                     if avoider.gamma(x_sim_check - obs_center) < 1.005:
#                         will_reenter = True
#                         break
                    
#                 if will_reenter:
#                     # Not safe — keep escaping in same direction 50 more ticks
#                     self._ticks_remaining = 50
#                 else:
#                     # Safe — hand back to normal DS
#                     self._escaping        = False
#                     self._v_escape        = np.zeros_like(x)
#                     self._ticks_remaining = 0
            
#             return self._v_escape.copy()

#         # ── Paper trigger condition 1: near boundary? ──
#         x          = np.asarray(x, float)
#         obs_center = np.asarray(obs_center, float)
#         goal       = np.asarray(goal, float)

#         x_tilde            = x - obs_center
#         x_tilde_safe       = x_tilde * avoider.safety_eta
#         effective_boundary = 1.0 / avoider.safety_eta
#         gamma_val          = avoider.gamma(x_tilde)

#         # trigger_zone > 1.0 because robot asymptotically approaches
#         # gamma=1.0 and never exactly reaches it numerically
#         if gamma_val > effective_boundary * trigger_zone:
#             return np.zeros_like(x)

#         # ── Paper trigger condition 2: velocity near zero? ──
#         f_here = np.asarray(f_nom(x), float)
#         v_mod  = avoider.M(x_tilde, f=f_here).dot(f_here)

#         if np.linalg.norm(v_mod) >= stuck_vel_thresh:
#             return np.zeros_like(x)

#         # ── Both conditions met — run Algorithm 1 ──

#         # Step 1: choose tangent eⁱ
#         # Paper: any tangent works for saddle point (1 step exits)
#         # For local minimum: must walk toward nearest saddle point
#         # We pick using escape_cost which measures:
#         #   (steps to escape boundary) + (remaining dist to goal)
#         # This correctly handles both cases
#         E        = avoider.E(x_tilde_safe)
#         tangents = [E[:, i] for i in range(1, E.shape[1])]

#         # Initialize x_sim before escape_cost (needed inside function)
#         x_sim       = x.copy()
#         x_start     = x.copy()
#         steps_taken = 0

#         e_i = self._pick_tangent(
#             tangents, x_sim, obs_center, avoider,
#             goal, alpha, effective_boundary,
#             dt,
#         )

#         MAX_ESCAPE_DIST = 0.35
#         # Step 2: Algorithm 1 loop — move along tangent until exit
#         for _ in range(max_iters):

#             # Paper Eq: ξt+1 ← ξt + α·eⁱ·δt
#             x_sim       = x_sim + alpha * e_i 
#             steps_taken += 1

#             if np.linalg.norm(x_sim - x_start) > MAX_ESCAPE_DIST:
#                 break

#             # Paper Eq: ξ̇t+1 = M(ξ̃t+1)·f(ξt+1)
#             x_tilde_sim      = x_sim - obs_center
#             x_tilde_safe_sim = x_tilde_sim * avoider.safety_eta
#             f_sim            = np.asarray(f_nom(x_sim), float)
#             v_new            = avoider.M(x_tilde_sim, f=f_sim).dot(f_sim)

#             n_sim   = avoider._get_normal(x_tilde_safe_sim)
#             n_unit  = n_sim / (np.linalg.norm(n_sim) + _EPS)

#             # Paper exit conditions (exact):
#             # (eⁱ)ᵀ·ξ̇t+1 > 0  OR  n(ξ̃)ᵀ·ξ̇t+1 > 0
#             tangent_exit = float(np.dot(e_i, v_new))   > 0.0
#             normal_exit  = float(np.dot(n_unit, v_new)) > 0.0

#             if tangent_exit or normal_exit:
#                 break   # escaped — saddle point broken

#         # ── Compute escape velocity = displacement / time ──
#         total_displacement = x_sim - x_start
#         time_needed        = steps_taken * dt
#         v_escape           = total_displacement / (time_needed + _EPS)

#         # Clip to hardware safe speed
#         # Recompute steps so robot still physically travels full displacement
#         v_norm = np.linalg.norm(v_escape)
#         if v_norm > max_escape_vel:
#             v_escape    = v_escape / v_norm * max_escape_vel
#             dist_needed = np.linalg.norm(total_displacement)
#             steps_taken = max(1, int(dist_needed / (max_escape_vel * dt)) + 1)

#         # ── Store state — persists across ticks ──
#         self._escaping        = True
#         self._v_escape        = v_escape
#         self._ticks_remaining = steps_taken

#         return self._v_escape.copy()

#     # ----------------------------------------------------------
#     # Tangent selection
#     # ----------------------------------------------------------

#     def _pick_tangent(
#         self,
#         tangents,
#         x_sim_start,
#         obs_center,
#         avoider,
#         goal,
#         alpha,
#         effective_boundary,
#         dt,
#         max_steps: int = 300,
#     ) -> np.ndarray:
#         """
#         Pick tangent direction with lowest cost:
#             cost = steps_to_escape_boundary + remaining_dist_to_goal / alpha

#         Tries both +t and -t for each tangent basis vector.
#         Correctly handles:
#           - Saddle point: all directions have cost ~1, picks goal-aligned one
#           - Local minimum: picks direction toward nearest saddle point
#             which is also the direction with shortest escape path to goal
#         """

#         def cost(tangent):
#             x_test = x_sim_start.copy()

#             # Push slightly inside boundary so escape distance is meaningful
#             # (robot may already be just outside gamma=1.0)
#             x_tilde_s = x_sim_start - obs_center
#             n_in      = avoider._get_normal(x_tilde_s * avoider.safety_eta)
#             n_in_unit = n_in / (np.linalg.norm(n_in) + _EPS)
#             x_test    = x_test - 0.03 * n_in_unit

#             for i in range(max_steps):
#                 x_test = x_test + alpha * tangent 
#                 xt     = x_test - obs_center
#                 if avoider.gamma(xt) > effective_boundary:
#                     remaining = np.linalg.norm(goal - x_test)
#                     return i + remaining / alpha
#             # Never escaped
#             return max_steps + np.linalg.norm(goal - x_test) / alpha

#         best_cost = np.inf
#         best_dir  = tangents[0]

#         for t in tangents:
#             for direction in [t, -t]:
#                 c = cost(direction)
#                 if c < best_cost:
#                     best_cost = c
#                     best_dir  = direction

#         return best_dir


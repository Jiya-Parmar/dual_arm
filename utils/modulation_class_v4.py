
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

_EPS = 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
#  Single-Obstacle Avoider
# ═══════════════════════════════════════════════════════════════════════════════

class ModulationAvoider:
 

    def __init__(
        self,
        gamma: Callable[[np.ndarray], float],
        grad_gamma: Callable[[np.ndarray], np.ndarray],
        dim: Optional[int] = None,
        rho: float = 1.0,
        safety_eta: float = 1.0,
        tail_effect: bool = True,
        lambda_min: float = 0.0,
        lambda_max_factor: float = 100.0,
        # ── NEW [Huber2019 Eq.5] ─────────────────────────────────────────────
        # A reference point ξ^r *inside* the obstacle.  When supplied the
        # first column of E becomes the "reference direction"
        #   r(ξ) = (ξ - ξ^r) / ‖ξ - ξ^r‖
        # instead of the gradient normal.  This is what makes the method work
        # on star-shaped / concave obstacles (Huber2019 Sec. III-B & III-C).
        # Leave as None to keep the original KB2012 gradient-normal behaviour.
        reference_point: Optional[np.ndarray] = None,
    ):
        assert rho > 0,              "rho must be > 0"
        assert 0 < safety_eta <= 1.0, "safety_eta must be in (0, 1]"

        self.gamma            = gamma
        self.grad_gamma       = grad_gamma
        self.rho              = float(rho)
        self.safety_eta       = float(safety_eta)
        self.tail_effect      = bool(tail_effect)
        self.lambda_min       = float(lambda_min)
        self.lambda_max_factor= float(lambda_max_factor)
        self.dim              = dim
        # [Huber2019 Sec. III-B] ξ^r – reference point inside the obstacle
        self.reference_point  = (
            np.asarray(reference_point, float) if reference_point is not None else None
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────────

    def modulate(
        self,
        x: np.ndarray,
        x_tilde: np.ndarray,
        f: np.ndarray,
        # ── NEW [Huber2019 Eq.17] ────────────────────────────────────────────
        # Linear velocity ẋ_{L,o} and angular velocity ẋ_{R,o} of the obstacle.
        # Pass None (default) for a static obstacle → original KB2012 behaviour.
        obstacle_linear_vel:  Optional[np.ndarray] = None,
        obstacle_angular_vel: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        
        x       = np.asarray(x,       float)
        x_tilde = np.asarray(x_tilde, float)
        f       = np.asarray(f,       float)

        d = x.shape[0]
        if self.dim is None:
            self.dim = d
        assert d == self.dim, "x dimension mismatch"

        M = self.M(x_tilde, f=f)

        # ── [Huber2019 Eq.17] Moving-obstacle branch ─────────────────────────
        if obstacle_linear_vel is not None or obstacle_angular_vel is not None:
            xi_dot = self._obstacle_velocity(
                x_tilde, obstacle_linear_vel, obstacle_angular_vel
            )
            # Modulate in obstacle frame, then add back obstacle velocity
            dx_dt = M.dot(f - xi_dot) + xi_dot
        else:
            # ── [KB2012 Eq.19] Static obstacle ───────────────────────────────
            dx_dt = M.dot(f)

        # Hard impenetrability clamp (safety fallback)
        x_tilde_safe = x_tilde * self.safety_eta
        n_unit = self._unit_normal(x_tilde_safe)
        gamma_val        = self.gamma(x_tilde)
        effective_bnd    = 1.0 / self.safety_eta
        if gamma_val <= effective_bnd:
            normal_vel = float(n_unit.dot(dx_dt))
            if normal_vel < 0:
                dx_dt = dx_dt - normal_vel * n_unit

        return dx_dt

    # ── NEW [KB2012 Algorithm 1] ───────────────────────────────────────────────
    def saddle_point_escape(
        self,
        x: np.ndarray,
        x_tilde: np.ndarray,
        f_func: Callable[[np.ndarray], np.ndarray],
        alpha: float = 0.01,
        max_iters: int = 2000,
    ) -> np.ndarray:

       
        x       = np.asarray(x,       float)
        x_tilde = np.asarray(x_tilde, float)

        gamma_val = float(self.gamma(x_tilde))
        dx_dt     = self.modulate(x, x_tilde, f_func(x))

        # [KB2012 Alg.1 Line 1] Check preconditions
        on_boundary = abs(gamma_val - 1.0) < 0.05
        stuck       = np.linalg.norm(dx_dt) < 1e-2
        if not (on_boundary and stuck):
            return x                                # not stuck → nothing to do

        # Get basis matrix; column 0 = normal/reference, column 1 = first tangent
        x_tilde_safe = x_tilde * self.safety_eta
        E     = self.E(x_tilde_safe)
        n_col = E[:, 0]                             # normal or reference direction
        ei    = E[:, 1]                             # [KB2012 Alg.1 Line 2] e¹

        x_curr      = x.copy()
        x_tilde_cur = x_tilde.copy()

        for _ in range(max_iters):
            # [KB2012 Alg.1 Line 5]  ξ_{t+1} ← ξ_t + α e¹
            x_curr      = x_curr      + alpha * ei
            x_tilde_cur = x_tilde_cur + alpha * ei  # approximate rigid shift

            f_curr    = f_func(x_curr)
            dx_dt_cur = self.modulate(x_curr, x_tilde_cur, f_curr)

            # Refresh basis at new position
            E_cur  = self.E(x_tilde_cur * self.safety_eta)
            n_col  = E_cur[:, 0]
            ei     = E_cur[:, 1]

            # [KB2012 Alg.1 Lines 7-8] Exit conditions
            if float(ei.dot(dx_dt_cur)) > 0 or float(n_col.dot(dx_dt_cur)) > 0:
                return x_curr

        return x_curr   # best position if max_iters exhausted

    # ──────────────────────────────────────────────────────────────────────────
    #  M, E, D  (core matrix pipeline)
    # ──────────────────────────────────────────────────────────────────────────

    def M(self, x_tilde: np.ndarray, f: Optional[np.ndarray] = None) -> np.ndarray:
       
        x_tilde      = np.asarray(x_tilde, float)
        x_tilde_safe = x_tilde * self.safety_eta
        E = self.E(x_tilde_safe)
        D = self.D(x_tilde, f=f)
        try:
            E_inv = np.linalg.inv(E)
        except np.linalg.LinAlgError:
            E_inv = np.linalg.pinv(E)
        return E.dot(D).dot(E_inv)

    def E(self, x_tilde: np.ndarray) -> np.ndarray:
        
        x_tilde = np.asarray(x_tilde, float)
        d       = x_tilde.shape[0]

        if self.reference_point is not None:
            # ── [Huber2019 Eq.5]  Reference-direction mode ───────────────────
            ref_vec  = x_tilde - self.reference_point
            norm_ref = np.linalg.norm(ref_vec)
            first_col = ref_vec / norm_ref if norm_ref > _EPS else np.eye(d)[:, 0]

            # Tangents are orthonormal to ∇Γ  (Huber2019 Sec. III-A)
            grad   = self._get_normal(x_tilde)
            n_unit = grad / (np.linalg.norm(grad) + _EPS)
            tangents = _gram_schmidt_tangents(n_unit, d)

        else:
            # ── [KB2012 Eqs.12-14]  Gradient-normal mode (original) ──────────
            n       = self._get_normal(x_tilde)
            n_unit  = n / (np.linalg.norm(n) + _EPS)
            first_col = n_unit
            tangents  = _gram_schmidt_tangents(n_unit, d)

        E = np.column_stack([first_col] + tangents[: d - 1])
        return E

    def D(self, x_tilde: np.ndarray, f: Optional[np.ndarray] = None) -> np.ndarray:
        
        x_tilde_safe = x_tilde * self.safety_eta
        gamma_val    = abs(float(self.gamma(x_tilde_safe))) + _EPS
        d            = x_tilde.shape[0]

        denom   = gamma_val ** (1.0 / self.rho)
        lam_n   = 1.0 - 1.0 / denom           # [KB2012 Eq.18] λ₁
        lam_tau = 1.0 + 1.0 / denom           # [KB2012 Eq.18] λᵢ, i ≥ 2

        # [KB2012 Eq.23]  Tail-effect: disable normal compression when moving away
        if self.tail_effect and f is not None:
            n_unit = self._unit_normal(x_tilde_safe)
            if float(np.dot(n_unit, f)) >= 0:
                lam_n = 1.0

        lam_n   = max(lam_n,   self.lambda_min)
        lam_tau = min(lam_tau, 1.0 + self.lambda_max_factor)

        return np.diag([lam_n] + [lam_tau] * (d - 1))

    # ──────────────────────────────────────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _unit_normal(self, x_tilde: np.ndarray) -> np.ndarray:
        n = self._get_normal(x_tilde)
        return n / (np.linalg.norm(n) + _EPS)

    def _get_normal(self, x_tilde: np.ndarray) -> np.ndarray:
        n = np.asarray(self.grad_gamma(x_tilde), float)
        if n.ndim != 1 or n.shape[0] != x_tilde.shape[0]:
            raise ValueError("grad_gamma must return a 1-D vector of same dimension as x")
        return n

    # ── NEW [Huber2019 Eq.17] ─────────────────────────────────────────────────
    @staticmethod
    def _obstacle_velocity(
        x_tilde: np.ndarray,
        v_linear:  Optional[np.ndarray],
        v_angular: Optional[np.ndarray],
    ) -> np.ndarray:
        
        d      = x_tilde.shape[0]
        xi_dot = np.zeros(d)
        if v_linear is not None:
            xi_dot = xi_dot + np.asarray(v_linear, float)
        if v_angular is not None and d == 3:
            xi_dot = xi_dot + np.cross(np.asarray(v_angular, float), x_tilde)
        return xi_dot

    # ──────────────────────────────────────────────────────────────────────────
    #  Static Gamma / GradGamma utilities  (unchanged + new star-shaped)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def gamma_sphere(x, center, radius) -> float:
        return float(np.linalg.norm(np.asarray(x) - np.asarray(center)) / (radius + _EPS))

    @staticmethod
    def grad_gamma_sphere(x, center, radius) -> np.ndarray:
        v = np.asarray(x) - np.asarray(center)
        return (v / (np.linalg.norm(v) + _EPS)) / (radius + _EPS)

    @staticmethod
    def gamma_ellipsoid(x, center, A) -> float:
        v = np.asarray(x) - np.asarray(center)
        return float(np.sqrt(np.dot(v, A.dot(v)) + _EPS))

    @staticmethod
    def grad_gamma_ellipsoid(x, center, A) -> np.ndarray:
        v = np.asarray(x) - np.asarray(center)
        return A.dot(v) / (np.sqrt(np.dot(v, A.dot(v)) + _EPS) + _EPS)

    @staticmethod
    def gamma_superellipsoid(x, center, radii, p) -> float:
        v = (np.asarray(x) - np.asarray(center)) / np.asarray(radii)
        return float((np.sum(np.abs(v) ** p) + _EPS) ** (1.0 / p))

    @staticmethod
    def grad_gamma_superellipsoid(x, center, radii, p) -> np.ndarray:
        x, center, radii = map(np.asarray, (x, center, radii))
        v     = (x - center) / radii
        S     = np.sum(np.abs(v) ** p) + _EPS
        coeff = S ** (1.0 / p - 1.0)
        return coeff * ((np.abs(v) ** (p - 2)) * v / radii)

    # ── NEW [Huber2019 Eq.3] ──────────────────────────────────────────────────
    @staticmethod
    def gamma_star_shaped(
        x: np.ndarray,
        reference_point: np.ndarray,
        radius_func: Callable[[np.ndarray], float],
    ) -> float:
        
        x   = np.asarray(x,               float)
        ref = np.asarray(reference_point, float)
        dist = np.linalg.norm(x - ref) + _EPS
        R    = float(radius_func(x)) + _EPS
        return dist / R


# ═══════════════════════════════════════════════════════════════════════════════
#  Multi-Obstacle Avoider
# ═══════════════════════════════════════════════════════════════════════════════

class MultiObstacleAvoider:
    

    def __init__(
        self,
        avoiders: List[ModulationAvoider],
        obstacle_centers: List[np.ndarray],
        method: str = "product",       # "product" | "interpolate"
    ):
        
        assert len(avoiders) == len(obstacle_centers), \
            "Must supply one center per avoider"
        self.avoiders         = avoiders
        self.obstacle_centers = [np.asarray(c, float) for c in obstacle_centers]
        self.method           = method

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────────

    def modulate(
        self,
        x:  np.ndarray,
        f:  np.ndarray,
        obstacle_linear_vels:  Optional[List[Optional[np.ndarray]]] = None,
        obstacle_angular_vels: Optional[List[Optional[np.ndarray]]] = None,
    ) -> np.ndarray:
        
        x = np.asarray(x, float)
        f = np.asarray(f, float)
        K = len(self.avoiders)

        x_tildes = [x - c for c in self.obstacle_centers]
        lin_vels  = obstacle_linear_vels  or [None] * K
        ang_vels  = obstacle_angular_vels or [None] * K

        if self.method == "product":
            return self._modulate_product(x, x_tildes, f, lin_vels, ang_vels)
        elif self.method == "interpolate":
            return self._modulate_interpolate(x, x_tildes, f, lin_vels, ang_vels)
        else:
            raise ValueError(f"Unknown method '{self.method}'. Use 'product' or 'interpolate'.")

    # ── NEW [KB2012 Algorithm 1] multi-obstacle version ───────────────────────
    def saddle_point_escape(
        self,
        x: np.ndarray,
        f_func: Callable[[np.ndarray], np.ndarray],
        alpha: float = 0.01,
        max_iters: int = 200,
    ) -> np.ndarray:
        
        x  = np.asarray(x, float)
        f  = f_func(x)
        dx = self.modulate(x, f)

        if np.linalg.norm(dx) > 1e-4:
            return x                        # not stuck

        x_tildes = [x - c for c in self.obstacle_centers]
        gammas   = [abs(float(av.gamma(xt)))
                    for av, xt in zip(self.avoiders, x_tildes)]
        # Identify nearest boundary
        k = int(np.argmin([abs(g - 1.0) for g in gammas]))

        av   = self.avoiders[k]
        xt   = x_tildes[k]
        if abs(gammas[k] - 1.0) > 1e-3:
            return x                        # not actually on a boundary

        E    = av.E(xt * av.safety_eta)
        ei   = E[:, 1]                      # first tangent vector
        n_c  = E[:, 0]

        x_curr = x.copy()
        for _ in range(max_iters):
            x_curr  = x_curr + alpha * ei
            f_curr  = f_func(x_curr)
            dx_curr = self.modulate(x_curr, f_curr)

            # Refresh tangent for the tracked obstacle
            xt_curr = x_curr - self.obstacle_centers[k]
            E_curr  = av.E(xt_curr * av.safety_eta)
            ei      = E_curr[:, 1]
            n_c     = E_curr[:, 0]

            # [KB2012 Alg.1 Lines 7-8]
            if float(ei.dot(dx_curr)) > 0 or float(n_c.dot(dx_curr)) > 0:
                return x_curr

        return x_curr

    # ──────────────────────────────────────────────────────────────────────────
    #  KB2012 Product approach
    # ──────────────────────────────────────────────────────────────────────────

    def _modulate_product(self, x, x_tildes, f, lin_vels, ang_vels):
        
        K      = len(self.avoiders)
        gammas = [abs(float(av.gamma(xt))) + _EPS
                  for av, xt in zip(self.avoiders, x_tildes)]

        omegas = _compute_omega_weights(gammas)   # [KB2012 Eq.25]

        M_bar = np.eye(x.shape[0])
        for k, (av, xt) in enumerate(zip(self.avoiders, x_tildes)):
            Mk    = _M_with_omega(av, xt, f, omegas[k])   # [KB2012 Eq.24+26]
            M_bar = M_bar.dot(Mk)                          # [KB2012 Eq.27]

        return M_bar.dot(f)   # [KB2012 Eq.19]

    # ──────────────────────────────────────────────────────────────────────────
    #  Huber2019 Directional-Interpolation approach
    # ──────────────────────────────────────────────────────────────────────────

    def _modulate_interpolate(self, x, x_tildes, f, lin_vels, ang_vels):
        
        K      = len(self.avoiders)
        d      = x.shape[0]
        gammas = [abs(float(av.gamma(xt))) + _EPS
                  for av, xt in zip(self.avoiders, x_tildes)]

        # [Huber2019 Eq.12] Weights
        weights = _compute_huber_weights(gammas)

        # Per-obstacle modulated velocities
        dx_dots = [
            av.modulate(x, xt, f, lv, av_)
            for av, xt, lv, av_ in zip(self.avoiders, x_tildes, lin_vels, ang_vels)
        ]

        # [Huber2019 Eq.13]  Weighted magnitude
        mags          = np.array([np.linalg.norm(v) for v in dx_dots])
        mean_magnitude = float(np.dot(weights, mags))

        # [Huber2019 Eqs.14-16]  Direction via κ-space
        nf = f / (np.linalg.norm(f) + _EPS)   # unit vector along original DS

        if d == 2:
            # 2-D: κ = signed angle from f to ẋᵒ  (Huber2019 Sec. IV-A)
            kappas = []
            for dxo in dx_dots:
                n_dxo = dxo / (np.linalg.norm(dxo) + _EPS)
                cos_a = float(np.clip(np.dot(nf, n_dxo), -1.0, 1.0))
                sin_a = float(nf[0] * n_dxo[1] - nf[1] * n_dxo[0])
                kappas.append(np.arctan2(sin_a, cos_a))
            kappa_bar = float(np.dot(weights, kappas))   # [Huber2019 Eq.14]
            # [Huber2019 Eq.15]  Reconstruct direction in original space
            c, s = np.cos(kappa_bar), np.sin(kappa_bar)
            R_mat = np.array([[c, -s], [s, c]])
            mean_dir = R_mat.dot(nf)
        else:
            # d > 2: approximate as weighted mean of unit velocity vectors.
            # Full κ-space generalization [Huber2019 Eqs.14-15] requires an
            # SO(d) rotation; the weighted unit-vector mean is equivalent for
            # small deflection angles and is numerically stable.
            mean_dir_raw = sum(
                w * v / (np.linalg.norm(v) + _EPS)
                for w, v in zip(weights, dx_dots)
            )
            norm_raw = np.linalg.norm(mean_dir_raw)
            mean_dir = mean_dir_raw / (norm_raw + _EPS)

        return mean_magnitude * mean_dir   # [Huber2019 Eq.16]


# ═══════════════════════════════════════════════════════════════════════════════
#  Module-level helpers  (shared by both classes)
# ═══════════════════════════════════════════════════════════════════════════════

def _gram_schmidt_tangents(normal: np.ndarray, d: int) -> List[np.ndarray]:
    
    tangents: List[np.ndarray] = []
    for i in range(d):
        ei = np.zeros(d); ei[i] = 1.0
        vi = ei - np.dot(ei, normal) * normal
        for t in tangents:
            vi = vi - np.dot(vi, t) * t
        norm_vi = np.linalg.norm(vi)
        if norm_vi > 1e-8:
            tangents.append(vi / norm_vi)
        if len(tangents) >= d - 1:
            break
    # Fallback via QR if Gram-Schmidt degenerates
    if len(tangents) < d - 1:
        Q, _ = np.linalg.qr(np.random.randn(d, d))
        extra = [Q[:, i] for i in range(d)
                 if abs(np.dot(Q[:, i], normal)) < 1 - 1e-3]
        tangents = (tangents + extra)[: d - 1]
    return tangents[: d - 1]


def _compute_omega_weights(gammas: List[float]) -> List[float]:
    
    K = len(gammas)
    if K == 1:
        return [1.0]

    omegas = []
    for k in range(K):
        w = 1.0
        for i in range(K):
            if i == k:
                continue
            num   = max(gammas[i] - 1.0, 0.0)
            denom = max(gammas[k] - 1.0, 0.0) + max(gammas[i] - 1.0, 0.0) + _EPS
            w    *= num / denom
        omegas.append(float(np.clip(w, 0.0, 1.0)))
    return omegas


def _compute_huber_weights(gammas: List[float]) -> np.ndarray:
    
    K = len(gammas)
    if K == 1:
        return np.array([1.0])

    prods = []
    for k in range(K):
        p = 1.0
        for i in range(K):
            if i != k:
                p *= max(gammas[i] - 1.0, 0.0) + _EPS
        prods.append(p)

    total = sum(prods) + _EPS
    return np.array(prods) / total


def _M_with_omega(
    av: ModulationAvoider,
    x_tilde: np.ndarray,
    f: np.ndarray,
    omega: float,
) -> np.ndarray:
    
    x_tilde_safe = x_tilde * av.safety_eta
    E = av.E(x_tilde_safe)
    d = x_tilde.shape[0]

    gamma_val = abs(float(av.gamma(x_tilde_safe))) + _EPS
    denom     = gamma_val ** (1.0 / av.rho)

    lam_n   = 1.0 - omega / denom
    lam_tau = 1.0 + omega / denom

    # [KB2012 Eq.23] Tail-effect
    if av.tail_effect:
        n_unit = av._unit_normal(x_tilde_safe)
        if float(n_unit.dot(f)) >= 0:
            lam_n = 1.0

    lam_n   = max(lam_n,   av.lambda_min)
    lam_tau = min(lam_tau, 1.0 + av.lambda_max_factor)

    D = np.diag([lam_n] + [lam_tau] * (d - 1))
    try:
        E_inv = np.linalg.inv(E)
    except np.linalg.LinAlgError:
        E_inv = np.linalg.pinv(E)
    return E.dot(D).dot(E_inv)
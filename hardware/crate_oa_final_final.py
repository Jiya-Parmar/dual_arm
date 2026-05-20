from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from utils.modulation_class_v4 import ModulationAvoider, MultiObstacleAvoider
# ─────────────────────────────────────────────────────────────────────────────

_EPS = 1e-9

# ============================================================
# DEMO PARAMETERS  (unchanged)
# ============================================================

START = np.array([-0.0008232779826864515, 0.3678360083468431, 0.5538878347626814])

GOALS = [
    np.array([-0.0008232779826864515, 0.6595048174402348, 0.6046573267124301]),
    np.array([-0.0008232779826864515, 0.6595048174402348, 0.46046573267124301]),
    np.array([-0.55, 0.6595048174402348, 0.136046573267124301]),
    np.array([-0.0008232779826864515, 0.3678360083468431,  0.5538878347626814]),
]

OBSTACLES = [
    {
        "center": np.array([0.025870643513917654, 0.45079258169172239, 0.20]),
        "radii":  np.array([0.32, 0.05, 0.40])
    },
    {
        "center": np.array([0.025870643513917654 - 0.32,
                            0.45079258169172239  + 0.21,
                            0.19]),
        "radii":  np.array([0.05, 0.20, 0.40])
    },
]

# ============================================================
# Build per-obstacle ModulationAvoider instances  (unchanged)
# ============================================================

avoiders = []
for obs in OBSTACLES:
    A = np.diag(1.0 / (obs["radii"] ** 2))

    avoider = ModulationAvoider(
        gamma      = lambda x, A=A: ModulationAvoider.gamma_ellipsoid(x, np.zeros(3), A),
        grad_gamma = lambda x, A=A: ModulationAvoider.grad_gamma_ellipsoid(x, np.zeros(3), A),
        rho        = 3.0,
        safety_eta = 0.8,
        tail_effect= False,
        # reference_point is left as None → KB2012 gradient-normal E matrix.
        # To use the Huber2019 star-shaped E matrix instead, pass:
        #   reference_point = obs["center"]   (but ensure it's relative to x_tilde)
    )
    avoiders.append(avoider)


multi_avoider = MultiObstacleAvoider(
    avoiders         = avoiders,
    obstacle_centers = [obs["center"] for obs in OBSTACLES],
    method           = "product",    # KB2012 Eqs.24-27
)
# ─────────────────────────────────────────────────────────────────────────────

# ============================================================
# Sanity Check  (unchanged logic, cleaner loop)
# ============================================================

for obs in OBSTACLES:
    center = obs["center"]
    radii  = obs["radii"]
    A      = np.diag(1.0 / (radii ** 2))
    print(f"Obstacle: center={center}, radii={radii}")
    print(f"  gamma(START) = "
          f"{ModulationAvoider.gamma_ellipsoid(START - center, np.zeros(3), A):.3f}")
    for i, g in enumerate(GOALS):
        gv = ModulationAvoider.gamma_ellipsoid(g - center, np.zeros(3), A)
        print(f"  gamma(GOAL {i+1}) = {gv:.3f}  "
              f"({'outside' if gv > 1 else 'INSIDE'})")

# ============================================================
# Simulation — loop over goal sequence
# ============================================================

dt    = 0.01
T     = 100.0
steps = int(T / dt)

all_traj           = []
all_gamma_min      = []
all_vel_norms      = []
segment_boundaries = [0]

x = START.copy()

for goal_idx, GOAL in enumerate(GOALS):
    print(f"\n--- Segment {goal_idx+1}: heading to GOAL {goal_idx+1} = {GOAL} ---")

    def f_nom(pos, goal=GOAL):
        return 1.0 * (goal - pos)

    for step in range(steps):

        v_nom = f_nom(x)

        v_mod = multi_avoider.modulate(x, v_nom)

        x = multi_avoider.saddle_point_escape(x, f_nom, alpha=0.005)
        # ─────────────────────────────────────────────────────────────────────

        # Gamma values for logging  (same as original, but via list comp)
        gamma_vals_current = [
            av.gamma(x - obs["center"])
            for av, obs in zip(avoiders, OBSTACLES)
        ]

        all_traj.append(x.copy())
        all_gamma_min.append(min(gamma_vals_current))
        all_vel_norms.append(np.linalg.norm(v_mod))

        x = x + dt * v_mod

        if np.linalg.norm(x - GOAL) < 1e-3:
            print(f"  Goal {goal_idx+1} reached at step {step}")
            break

    segment_boundaries.append(len(all_traj))

traj = np.array(all_traj)
print(f"\nTotal trajectory points: {len(traj)}")

# ============================================================
# 3D Plot  (unchanged)
# ============================================================

fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection='3d')

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for i in range(len(GOALS)):
    s   = segment_boundaries[i]
    e   = segment_boundaries[i + 1]
    seg = traj[s:e]
    if len(seg) > 0:
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                color=colors[i % len(colors)],
                label=f"Segment {i+1} → Goal {i+1}")

ax.scatter(*START, s=100, c='black', zorder=5, label="Start")
for i, g in enumerate(GOALS):
    ax.scatter(*g, s=100, marker="*", zorder=5, label=f"Goal {i+1}")

for obs in OBSTACLES:
    center = obs["center"]
    radii  = obs["radii"]
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x_e = radii[0] * np.outer(np.cos(u), np.sin(v))
    y_e = radii[1] * np.outer(np.sin(u), np.sin(v))
    z_e = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x_e + center[0], y_e + center[1], z_e + center[2], alpha=0.2)

ax.set_title("Multi-Obstacle Modulated Trajectory (Goal Sequence)")
ax.legend(fontsize=7)
plt.tight_layout()
plt.show()

# ============================================================
# Gamma Plot  (unchanged)
# ============================================================

plt.figure()
plt.plot(all_gamma_min)
plt.axhline(1.0, linestyle='--', color='red', label='Obstacle boundary')
for b in segment_boundaries[1:-1]:
    plt.axvline(b, linestyle=':', color='gray', label='Goal switch')
plt.title("Minimum Gamma Along Trajectory")
plt.xlabel("Time step")
plt.ylabel("Gamma")
plt.legend()
plt.show()

# ============================================================
# Velocity Norm Plot  (unchanged)
# ============================================================

plt.figure()
plt.plot(all_vel_norms)
for b in segment_boundaries[1:-1]:
    plt.axvline(b, linestyle=':', color='gray')
plt.title("Modulated Velocity Norm")
plt.xlabel("Time step")
plt.ylabel("||v||")
plt.show()
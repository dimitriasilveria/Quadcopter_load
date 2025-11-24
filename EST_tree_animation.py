"""
EST_tree_animation.py

Stand-alone script that animates the growing EST tree for your quadcopter-with-load planner.

Usage:
 - Put this file in the same folder as the file that defines your EST class (the code you pasted).
 - Rename your EST file to one of: est.py, est_impl.py, est_module.py or adjust the IMPORT_NAMES list below.
 - Run: python EST_tree_animation.py

What it does:
 - Tries to import your EST class and the quad_w_load_dyn model.
 - Launches est.search(...) in a background thread so the GUI animation can update while the tree grows.
 - Animates the current vertices (V), edges (E), obstacles (by calling map.display), start and goal.

Note: this script expects your EST instance to expose: V (list of 3-tuples), E (dict mapping 3-tuple->numpy array of states),
map (with display(ax) and is_free), start and goal attributes and path when found. If your filenames or symbols differ,
adjust IMPORT_NAMES or variable names below.
"""

import importlib
import sys
import threading
import time
import traceback
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Try to import EST from likely filenames; adjust this list if needed
IMPORT_NAMES = ["est", "est_impl", "est_module", "EST", "est_script"]
EST_CLASS_NAME = "EST"
QUAD_CLASS_NAMES = ["quad_w_load_dyn", "quad_w_load_dyn"]

est_module = None
EST = None
quad_module = None

for name in IMPORT_NAMES:
    try:
        est_module = importlib.import_module(name)
        if hasattr(est_module, EST_CLASS_NAME):
            EST = getattr(est_module, EST_CLASS_NAME)
            break
    except Exception:
        continue

# If we didn't import, try to exec probable local files
if EST is None:
    tried = False
    for fname in ("est.py", "est_impl.py", "est_module.py", "est_script.py"):
        try:
            with open(fname, "r") as f:
                code = f.read()
            loc = {}
            exec(code, loc)
            if EST_CLASS_NAME in loc:
                EST = loc[EST_CLASS_NAME]
                tried = True
                break
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Error while executing {fname}: {e}")
            traceback.print_exc()
            continue

if EST is None:
    raise ImportError(
        "Could not find an EST class in common module names.\n"
        "Place your EST implementation in est.py (or add its module name to IMPORT_NAMES),\n"
        "or rename accordingly."
    )

# Import quad class from whatever module EST expects; try to find quad_w_load_dyn in the same module or globally
quad_cls = None
if est_module is not None and hasattr(est_module, "quad_w_load_dyn"):
    quad_cls = getattr(est_module, "quad_w_load_dyn")
else:
    # try to import from quad_w_load_dyn.py if present
    try:
        qm = importlib.import_module("quad_w_load_dyn")
        if hasattr(qm, "quad_w_load_dyn"):
            quad_cls = getattr(qm, "quad_w_load_dyn")
    except Exception:
        quad_cls = None

# If quad class is still None, we'll instantiate using fallback import — user should have quad_w_load_dyn available
if quad_cls is None:
    try:
        from quad_w_load_dyn import quad_w_load_dyn as quad_cls
    except Exception:
        # we'll allow the user to pass their own quad manually in the script; for now we'll error clearly
        raise ImportError("Could not import quad_w_load_dyn. Make sure quad_w_load_dyn.py exists in the same folder.")

# ---------- Build an EST instance ----------
# Adjust initial conditions here if you want different start/goal
quad = quad_cls()
start_state = np.zeros((quad.n_states + 3, 1))
start_state[0:quad.n_states] = quad.x.copy()
# push start to a convenient point if desired
start_point = (10, 10, 10)
goal = (30, 30, 30)

est = EST(start_point, start_state, goal, quad)

# Animation parameters
FPS = 20
INTERVAL_MS = int(1000 / FPS)
ITERATIONS_IN_BACKGROUND = 10000  # max iterations the background search will attempt

# Start the search in a background thread so the GUI remains responsive
search_thread = threading.Thread(target=lambda: est.search(max_iterations=ITERATIONS_IN_BACKGROUND), daemon=True)
search_thread.start()

# Create figure and axes
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((1,1,1))

# Draw static elements once (map obstacles). We'll re-draw them each frame because map.display may rely on axes state.
# But we keep a copy of the map for reference.

start_scatter = None
goal_scatter = None
vertices_scatter = None
edge_lines = []
text_iteration = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

# limits - try to infer from map size if available
try:
    size_x, size_y, size_z = est.map.size
    ax.set_xlim(0, size_x)
    ax.set_ylim(0, size_y)
    ax.set_zlim(0, size_z)
except Exception:
    # fallback
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)

# helper to draw the current tree
def draw_tree(ax, est):
    # Clear dynamic artists (but keep the axes limits and labels)
    ax.cla()
    ax.set_box_aspect((1,1,1))
    try:
        size_x, size_y, size_z = est.map.size
        ax.set_xlim(0, size_x)
        ax.set_ylim(0, size_y)
        ax.set_zlim(0, size_z)
    except Exception:
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_zlim(0, 100)

    # re-draw obstacles using your map.display method if available
    try:
        est.map.display(ax)
    except Exception:
        # if map.display isn't available, just continue
        pass

    # draw edges
    for key, segment in getattr(est, 'E', {}).items():
        try:
            seg = np.array(segment)
            if seg.size == 0:
                continue
            xs = seg[:, 0]
            ys = seg[:, 1]
            zs = seg[:, 2]
            ax.plot(xs, ys, zs, linewidth=0.7)
        except Exception:
            continue

    # draw vertices
    try:
        V = np.array(est.V)
        if V.size != 0:
            ax.scatter(V[:, 0], V[:, 1], V[:, 2], s=8)
    except Exception:
        pass

    # draw start and goal
    try:
        s = est.start
        g = est.goal
        ax.scatter([s[0]], [s[1]], [s[2]], s=60, marker='o', label='start', depthshade=True)
        ax.scatter([g[0]], [g[1]], [g[2]], s=80, marker='*', label='goal', depthshade=True)
    except Exception:
        pass

    ax.legend(loc='upper left')

# Animation update function
frame_counter = {"i": 0}

def update(frame):
    frame_counter['i'] += 1
    draw_tree(ax, est)
    text_iteration.set_text(f"Iterations: {frame_counter['i']}")

    # If a path is found, highlight it and stop animation
    if getattr(est, 'path', None) and len(est.path) > 0:
        for segment in est.path:
            seg = np.array(segment)
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], linewidth=2.5, color='red', label='found path')
        # Stop the animation
        ani.event_source.stop()

ani = animation.FuncAnimation(fig, update, interval=INTERVAL_MS)

print("Animation started. Close the window to stop.")
plt.show()

# Wait for background thread to finish (optional)
if search_thread.is_alive():
    print("Search thread still running; the GUI was closed. You may want to let the search finish or rerun.")
else:
    print("Search completed.")


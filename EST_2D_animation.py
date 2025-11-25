"""
est_animation.py

Animation script that visualizes an EST search (search tree + map) as it runs.

Usage: put this script in the same folder as your EST implementation (the code you
posted). It imports EST from that module. Run:

    python est_animation.py

If your EST class is in a different file name change the import at the top.

This script does not modify your EST class; instead it performs one-step
``est_step`` iterations that mimic the body of EST.search and yields updates to the
animation. The animation draws:
 - the Map (uses est.map.display(ax) if available)
 - all vertices so far (scatter)
 - tree edges (lines for each extension stored in est.E_points)
 - the latest extension in a brighter color
 - the goal (green circle)

If your Map class or EST class differs slightly, adapt the simple function
`est_step` accordingly.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from icecream import ic
import sys

# Adjust this import if your EST class is defined in another file/module.
# The user's EST class (from the message) should be importable as `EST`.
try:
    from est_2D import EST  # try a custom module name first
except Exception:
    # fallback: try importing EST from the current namespace (if user copied it)
    try:
        from __main__ import EST
    except Exception:
        # As a last resort, try to import from a file named `est.py` or the file
        # where the user pasted the EST class. You can change this line to match
        # your environment.
        try:
            from est import EST
        except Exception:
            print("Could not import EST automatically. Make sure EST is importable (est.py or est_module.py).")
            # We won't exit here so the user can still read the file and adapt it.


# One-step EST executor. It mirrors the body of EST.search() from the user's code
# but yields information needed for animation instead of storing everything at once.

def est_step(est):
    """Perform a single EST iteration on `est` instance.

    Returns a dict with keys:
      - 'added' : True/False (whether a new vertex was added)
      - 'parent': the sampled vertex (tuple)
      - 'child' : new vertex (tuple) if added else None
      - 'path_points': list of 2D tuples along the extension (may be empty on collision)
      - 'collision': True/False
      - 'reached_goal': True/False
      - 'iter_info': optional debug info
    """
    info = {
        'added': False,
        'parent': None,
        'child': None,
        'path_points': [],
        'collision': False,
        'reached_goal': False,
        'iter_info': None
    }

    try:
        x_rand = est.sample()
    except Exception as e:
        info['iter_info'] = f"sampling error: {e}"
        return info

    info['parent'] = x_rand

    # draw from actuator sampler
    try:
        tau, f = est.sample_actuation()
    except Exception as e:
        info['iter_info'] = f"actuation sample error: {e}"
        return info

    # determine x0 (state) for steering
    if x_rand == est.start:
        x0 = est.start_state
    else:
        # If E_states doesn't contain the parent (e.g. first iterations), fallback
        if x_rand not in est.E_states:
            # If we cannot find previous state, use start_state as fallback
            x0 = est.start_state
        else:
            x0 = est.E_states[x_rand][:, -1].reshape((est.quad.n_states, 1))

    # compute extension (trajectory of states)
    try:
        X_new = est.steer(x0, tau, f)
    except Exception as e:
        info['iter_info'] = f"steer error: {e}"
        return info

    # construct 2D points from the trajectory
    x_new = tuple(X_new[0:2, -1])
    path_points = []
    collision = False
    for point in X_new.T:
        pt = tuple(point[0:2])
        path_points.append(pt)
        if hasattr(est, 'map') and est.map is not None:
            try:
                if not est.map.is_free(pt):
                    collision = True
                    break
            except Exception:
                # if map.is_free is not available or errors, skip collision checking
                pass

    info['path_points'] = path_points
    info['collision'] = collision

    if collision:
        return info

    # accept the extension: store E_points and E_states similar to the user's code
    try:
        est.E_points[x_new] = path_points
        est.E_states[x_new] = X_new
    except Exception:
        # ensure dictionaries exist
        if not hasattr(est, 'E_points'):
            est.E_points = {}
        if not hasattr(est, 'E_states'):
            est.E_states = {}
        est.E_points[x_new] = path_points
        est.E_states[x_new] = X_new

    # update proximity stats
    try:
        est.update_proximity(x_new)
    except Exception as e:
        # update_proximity may require V to be numeric etc. If it fails, try a
        # mild fallback: just append vertex and set uniform weights.
        try:
            est.V.append(x_new)
            est.w[x_new] = len(est.V)  # crude
            est.p = {v: 1.0 / len(est.V) for v in est.V}
        except Exception:
            pass

    info['added'] = True
    info['child'] = x_new

    # check goal
    try:
        if est.check_goal_reached(x_new):
            info['reached_goal'] = True
    except Exception:
        pass

    return info


# Animation wrapper
class ESTAnimator:
    def __init__(self, est, max_iters=5000):
        self.est = est
        self.max_iters = max_iters
        self.iter = 0
        self.history_last_added = None

        # figure
        self.fig, self.ax = plt.subplots()
        self.sc_vertices = None
        self.lines = []
        self.latest_line = None

        # Plot goal
        goal = getattr(est, 'goal', None)
        if goal is not None:
            self.ax.scatter(goal[0], goal[1], s=100, marker='*', label='goal', zorder=4)

        # attempt to plot map (if map.display exists)
        if hasattr(est, 'map') and est.map is not None:
            try:
                est.map.display(self.ax)
            except Exception:
                # fallback: if map has a grid or obstacle list, try to draw them
                if hasattr(est.map, 'obstacles'):
                    for obs in est.map.obstacles:
                        xs, ys = zip(*obs)
                        self.ax.fill(xs, ys, alpha=0.5)

        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title('EST search tree (live)')
        self.ax.legend()

    def init_plot(self):
        V = np.array(list(self.est.V)) if len(self.est.V) > 0 else np.zeros((0, 2))
        if V.shape[0] > 0:
            self.sc_vertices = self.ax.scatter(V[:, 0], V[:, 1], s=10, label='vertices', zorder=2)
        else:
            self.sc_vertices = self.ax.scatter([], [], s=10, label='vertices', zorder=2)
        return [self.sc_vertices]

    def update(self, frame):
        # perform one EST step
        if self.iter >= self.max_iters:
            return []
        info = est_step(self.est)
        self.iter += 1

        # update vertex scatter
        try:
            V = np.array(list(self.est.V)) if len(self.est.V) > 0 else np.zeros((0, 2))
            if V.shape[0] > 0:
                self.sc_vertices.set_offsets(V)
        except Exception:
            pass

        # clear existing lines and redraw all edges (simple approach)
        # remove old lines
        for ln in self.lines:
            try:
                ln.remove()
            except Exception:
                pass
        self.lines = []

        # draw tree edges from est.E_points
        try:
            for child, pts in getattr(self.est, 'E_points', {}).items():
                pts_arr = np.array(pts)
                # single polyline per extension
                ln, = self.ax.plot(pts_arr[:, 0], pts_arr[:, 1], linewidth=0.8, alpha=0.7)
                self.lines.append(ln)
        except Exception:
            pass

        # draw latest extension brighter
        if info['added'] and info['path_points']:
            try:
                pts_arr = np.array(info['path_points'])
                if self.latest_line is not None:
                    try:
                        self.latest_line.remove()
                    except Exception:
                        pass
                self.latest_line, = self.ax.plot(pts_arr[:, 0], pts_arr[:, 1], linewidth=2.0, alpha=1.0, zorder=5)
            except Exception:
                pass

        # goal check: if reached, annotate and stop animation
        if info['reached_goal']:
            self.ax.set_title(f"Goal reached at iter {self.iter}!")
            print('Goal reached; stopping animation')
            self.anim.event_source.stop()

        return [self.sc_vertices] + self.lines + ([self.latest_line] if self.latest_line is not None else [])

    def run(self, interval=20):
        self.anim = animation.FuncAnimation(self.fig, self.update, init_func=self.init_plot,
                                            frames=self.max_iters, interval=interval, blit=False)
        plt.show()


if __name__ == '__main__':
    # Example usage: create a default quad and EST if available in user's environment.
    try:
        from quad_w_load_dyn_2D import quad_w_load_dyn
        quad = quad_w_load_dyn()
        start_state = np.zeros((quad.n_states, 1))
        start_state[0:quad.n_states] = quad.x.copy()
        start_state[0:2] = np.array([[25],[50]])
        start_point = (25,50)
        goal = (75,50)
        seed = np.random.randint(0, 10000)
        est = EST(start_point, start_state, goal, quad, seed=seed)
    except Exception as e:
        print('Could not auto-create EST (import error). Please create an EST instance yourself and call ESTAnimator(est).run()')
        raise

    animator = ESTAnimator(est, max_iters=2000)
    animator.run(interval=30)

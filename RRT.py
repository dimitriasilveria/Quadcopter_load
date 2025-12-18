import numpy as np
import random
from Maps2d import Map
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from controller_quad import controller, closed_loop_dynamics_point
from quad_w_load_dyn_2D import quad_w_load_dyn as quad_dyn
from numpy.typing import NDArray
from icecream import ic
import os
import yaml
class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True
class RRT:
    def __init__(self, start, goal, quad, obstacles,l=30, epsilon=0.01, step=2.5, goal_tolerance=0.5, file_name="rrt_path.yaml"):
        self.start = start
        self.goal = goal
        self.epsilon = epsilon
        self.step = step
        self.goal_tolerance = goal_tolerance
        self.map_height = 10
        self.map_width = 10
        self.map = Map(self.map_width, self.map_height,step)
        self.file_name = file_name
        self.info_dict = {'f':[], 'tau':[]}
        if obstacles == 5:
            obstacle_gap = 3.0
            self.map.obstacles_five(obstacle_gap)
        elif obstacles == 1:
            self.map.obstacles_one(4)
            file_dir = "rrt_obstacle_1"
            os.makedirs(file_dir, exist_ok=True)
        

        # if map_type == 1:
        #     self.map.obstacles_one(l)
        # elif map_type == 2:
        #     self.map.obstacles_two()
        # elif map_type == 3:
        #     self.map.obstacles_three()
        # elif map_type == 4:
        #     self.map.obstacles_four()

        self.path_length = 0
        self.path = []
        self.V = [self.array_to_tuple(self.start)]  # List of vertices
        self.quad = quad
        self.quad.x[0:2] = np.array(self.start[0:2]).reshape((2,1))  # initial load position
        self.quad.x[2:4] = np.array(self.start[2:4]).reshape((2,1))  # initial load velocity
        self.E_states = {}  # Edges in the tree
        self.E_states[self.array_to_tuple(self.start)] = self.quad.x.flatten()
        self.states_path = []
        self.start_state = self.quad.x.flatten().copy()
        self.E_efforts = {}
        self.E = {}  # Edges in the tree
        # self.E[self.start] = None  # start has no parent
        self.dt = self.quad.dt
        self.min_vel = -5.0
        self.max_vel = 5.0
        self.K = np.diag([1.0, 1.0, 4, 4])  # weighting matrix for cost function

    def sample(self) -> NDArray:
        p = random.random()
        if p < self.epsilon:
            return np.hstack((self.goal, np.array([0.0, 0.0]))).reshape((6,1))  # position, velocity, acceleration
        else:
            x = random.uniform(0, self.map.width)
            y = random.uniform(0, self.map.height)
            vx = random.uniform(self.min_vel, self.max_vel)
            vy = random.uniform(self.min_vel, self.max_vel)
            return np.array([x, y, vx, vy, 0.0, 0.0]).reshape((6,1)) # position, velocity, acceleration
                
    def nearest(self, q_rand: NDArray) -> NDArray:
        min_dist = np.inf
        q_near = None
        for v in self.V:
            v_array = np.array(v).reshape((4,1))
            dist = (q_rand[0:4] - v_array).T @ self.K @ (q_rand[0:4] - v_array)
            if dist < min_dist:
                min_dist = dist
                q_near = v
        return q_near
    
    def steer(self, l_near: NDArray, l_rand: NDArray) -> NDArray:
        #l_near and l_rand contain positions, velocities, and accelerations
        tf = 1
        t = 0.0
        
        x = l_near.flatten()
        self.quad.x = x.reshape((8,1))
        # self.quad.x[2:4] = x[2:4].reshape((2,1))  # initial load velocity
        Pos_vel = [l_near[0:4]] #ensuring that parent info is the first point
        #simulate
        Commands_control = []
        latest_control = [None, None]
        while t < tf:
            n_points = 10  # number of intermediate states per dt
            t_eval = np.linspace(t, t + self.dt, n_points)
            t_span = (t, t + self.dt)
            sol = solve_ivp(
                fun=lambda tt, xx: closed_loop_dynamics_point(
                    tt, xx, self.quad, controller, l_rand, latest_control
                ),
                t_span=t_span,
                y0=x,
                method="RK45",
                t_eval=t_eval,      # like MATLAB output grid
                rtol=1e-6,
                atol=1e-8
            )
                # Update state
            x = sol.y.T[-1]
            x[4] = np.clip(x[4], -np.pi/4, np.pi/4)  # keep angles within -pi/4 to pi/4
            x[5] = np.clip(x[5], -np.pi/1.5, np.pi/1.5)  # limit angular velocities
            x[6] = np.clip(x[6], -np.pi/2, np.pi/2)
            x[7] = np.clip(x[7], -np.pi/1.5, np.pi/1.5)
            t = sol.t[-1]
            x_l = x[0:2].reshape((2,1))
            for state in sol.y.T:
                x_l = state[0:2].reshape((2,1))
                quad_pos = x_l + self.quad.l * np.array([[-np.sin(state[4])],[np.cos(state[4])]])
                if self.map.is_free((state[0], state[1]), quad_pos.flatten(), self.quad.L, state[6]) == False:
                    return None, None, None
            Pos_vel.append(x[0:4])  # store load position and velocity
            Commands_control.append(latest_control)

        Quad_states = x  # store quad states
        return Pos_vel, Quad_states, Commands_control

    def search(self, num_iter=1e5, seed=None):
        if seed is not None:
            random.seed(seed)

        for i in range(int(num_iter)):
            l_rand = self.sample()
            l_nearest = self.nearest(l_rand)
            nearest_states = self.E_states[l_nearest]
            L_new, Q_new, Commands_new = self.steer(nearest_states, l_rand)

            if L_new is None:
                continue

            l_new = L_new[-1]
            l_new_point = self.array_to_tuple(l_new)

            if l_new_point not in self.V:
                self.V.append(l_new_point)

            self.E[l_new_point] = [L_new]
            self.E_states[l_new_point] = Q_new
            self.E_efforts[l_new_point] = Commands_new

            # 🔥 Animate tree expansion
            # self.animate_tree()

            # Goal check
            if np.linalg.norm(np.array(l_new[0:2]) - np.array(self.goal[0:2])) < self.goal_tolerance:
                # print("Goal reached!")
                return self.reconstruct_path(l_new), i

        print("Goal not reached within max iterations.")
        return None, i


    def array_to_tuple(self, arr):
        return tuple(float(x) for x in arr.flatten())

    def reconstruct_path(self, q_new):
        current = self.array_to_tuple(q_new)
        # self.states_path = [self.E_states[current]]
        commands = []
        while current != self.array_to_tuple(self.start):
            path_points = self.E[current][0]
            commands += self.E_efforts[current]
            self.states_path = [self.E_states[current].copy().tolist()] + self.states_path

            self.path = path_points[:-1] + self.path  # prepend to path            
            current = self.array_to_tuple(path_points[0])
        self.path = [self.start] + self.path

        self.states_path = [self.start_state] + self.states_path
        P = np.vstack(self.path)
        diffs = np.diff(P[:,0:2], axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        path_length = np.sum(segment_lengths)
        self.info_dict['path_length'] = str(path_length)
        self.info_dict['num_iterations'] = len(self.V)
        self.info_dict['commands'] = commands
        self.plot_result(fname='figures/rrt_scenario_1')
        # with open(self.file_name, 'w') as file:
        #     yaml.dump(self.info_dict, file, Dumper=NoAliasDumper)
        # self.path.reverse()  # reverse to get from start to goal

        return self.path

    def plot_result(self, show=True, save=True, fname="rrt_result.png"):
        fig, ax = plt.subplots(figsize=(7, 7))

        # --- Plot obstacles ---
        for obs in self.map.obstacles:
            (ox1, oy1), (ox2, oy2) = obs
            ax.fill([ox1, ox2, ox2, ox1],
                    [oy1, oy1, oy2, oy2],
                    color="gray", alpha=0.5)

        # --- Plot sampled nodes ---
        V = np.array(self.V)
        ax.scatter(V[:, 0], V[:, 1],
                s=10, c="lightblue", label="Sampled nodes")



        # --- Plot path if it exists ---
        if self.path:
            path = np.array(self.path)

            # Load path
            # ax.plot(path[:, 0], path[:, 1],
            #         c="blue", linewidth=2, label="Load path")

            # --- Quadcopter path ---
            quad_path = []

            for x in self.states_path:
                self.quad.x = np.array(x).reshape((8,1)).copy()
                q = self.quad.quad_position().flatten()
                quad_path.append(q)

            quad_path = np.array(quad_path)

            # Insert NaNs between discontinuous segments
            quad_x = quad_path[:, 0].astype(float)
            quad_y = quad_path[:, 1].astype(float)

            # Break line wherever jump is too large
            jump_thresh = 0.5   # meters (tune if needed)

            dx = np.diff(quad_x)
            dy = np.diff(quad_y)
            dist = np.sqrt(dx**2 + dy**2)

            breaks = np.where(dist > jump_thresh)[0] + 1

            quad_x = np.insert(quad_x, breaks, np.nan)
            quad_y = np.insert(quad_y, breaks, np.nan)

            ax.plot(quad_x, quad_y,
                    linestyle="--",
                    linewidth=2,
                    color="orange",
                    label="Quad path")

            # --- Draw final quad + load ---
            if self.states_path:

                # Initial configuration
                self.draw_quad_glyph(
                    ax,
                    self.states_path[0],
                    color="#bbbbbb",   # light gray
                    zorder=8
                )

                # Final configuration
                self.draw_quad_glyph(
                    ax,
                    self.states_path[-1],
                    color="#7b4ab2",   # purple
                    zorder=9
                )
                
        # --- Start & goal ---
        ax.scatter(*self.start[0:2], s=120, c="green", marker="o", label="Start")
        ax.scatter(*self.goal[0:2], s=120, c="red", marker="*", label="Goal")

        ax.set_aspect("equal")
        ax.set_xlim(0, self.map.width)
        ax.set_ylim(0, self.map.height)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend()
        ax.grid(True)

        if save:
            plt.savefig(fname, dpi=300)
        if show:
            plt.show()
        plt.close()

    def draw_quad_glyph(self, ax, x_state, color="#7b4ab2", zorder=6):
        """
        Draw a stylized quad + load configuration (T-shape)
        """
        x_state = np.array(x_state).reshape((8,1))
        self.quad.x = x_state.copy()

        # Positions
        x_l = x_state[0:2].flatten()
        x_q = self.quad.quad_position().flatten()

        theta = float(x_state[4, 0])
        L = self.quad.L

        # Horizontal arm direction
        arm_dir = np.array([np.cos(theta), np.sin(theta)])

        # Motor positions (LEFT and RIGHT)
        m_left  = x_q - L * arm_dir
        m_right = x_q + L * arm_dir

        # ---- Arm ----
        ax.plot([m_left[0], m_right[0]],
                [m_left[1], m_right[1]],
                linewidth=5,
                color=color,
                zorder=zorder)

        # ---- Quad body ----
        ax.scatter(x_q[0], x_q[1],
                s=70, marker="s",
                color=color,
                zorder=zorder + 1)

        # ---- Motors (BOTH) ----
        ax.scatter([m_left[0], m_right[0]],
                [m_left[1], m_right[1]],
                s=40,
                color="#f4a6c1",
                zorder=zorder + 2)

        # ---- Cable ----
        ax.plot([x_q[0], x_l[0]],
                [x_q[1], x_l[1]],
                linewidth=3,
                color="black",
                zorder=zorder - 1)

        # ---- Load ----
        ax.scatter(x_l[0], x_l[1],
                s=50,
                color="black",
                zorder=zorder)

    def animate_tree(self, interval=0.001):
        """
        Animate the RRT tree expansion.
        Call this after the search() loop or inside search() every iteration.
        """

        # Create a figure only once
        if not hasattr(self, "_fig"):
            self._fig, self._ax = plt.subplots()
            self.map.display(self._ax)
            self._ax.scatter(self.start[0], self.start[1], c='green', s=80, label="start")
            self._ax.scatter(self.goal[0], self.goal[1], c='orange', s=80, label="goal")
            self._ax.set_title("RRT Tree Expansion")
            self._ax.set_xlim(0, self.map.width)
            self._ax.set_ylim(0, self.map.height)
            self._ax.legend()

            # store artists so we don’t redraw everything
            self._edge_lines = []

        # Draw new edges
        for child, trajectories in self.E.items():
            if hasattr(self, "_drawn") and child in self._drawn:
                continue  # already drawn

            L_new = trajectories[0]       # full segment: [parent, ..., child]
            parent = L_new[0]
            child_xy = L_new[-1]

            # Extract x,y only
            px, py = parent[0], parent[1]
            cx, cy = child_xy[0], child_xy[1]

            # Draw edge
            line, = self._ax.plot([px, cx], [py, cy], '-', color="blue", linewidth=0.7)
            self._edge_lines.append(line)

            # Mark as drawn
            if not hasattr(self, "_drawn"):
                self._drawn = set()
            self._drawn.add(child)

        plt.pause(interval)
    def plot_path(self, path, fig_name="rrt_path.pdf"):
        fig, ax = plt.subplots()
        ax = self.map.display(ax)
        xs, ys, vx, vy = zip(*self.V)
        ax.scatter(xs, ys, c='blue', s=5)
        if path:
            path_xs, path_ys, path_vx, path_vy = zip(*path)
            ax.plot(path_xs, path_ys, c='red', linewidth=2)
        plt.scatter([self.start[0]], [self.start[1]], c='green', s=50, label='Start')
        plt.scatter([self.goal[0]], [self.goal[1]], c='orange', s=50, label='Goal')
        # for child, (parent, _) in self.E.items():
        #     plt.plot([child[0], parent[0]], [child[1], parent[1]], c='gray', linewidth=0.5)
        plt.legend()
        plt.savefig(fig_name)
        plt.show()

if __name__ == "__main__":
    folder_name = "RRT_paths_2D_obstacle_1"
    os.makedirs(folder_name, exist_ok=True)
    for i in range(0,1):
        print(i)
        quad = quad_dyn()
        start = np.array([5.2, 1.50, 0.0, 0.0])
        goal = np.array([3, 2.0, 0.0, 0.0])
        rrt = RRT(start=start, goal=goal, obstacles=1, quad=quad, file_name=f"{folder_name}/rrt_path_seed_{i}.yaml")
        path, iterations = rrt.search(seed=i)
        # print(path)
        # rrt.plot_path(path)
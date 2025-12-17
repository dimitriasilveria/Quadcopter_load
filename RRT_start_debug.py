import numpy as np
import random
from Maps2d import Map
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from controller_quad import controller, closed_loop_dynamics_point
from quad_w_load_dyn_2D import quad_w_load_dyn as quad_dyn
from numpy.typing import NDArray
from icecream import ic
from scipy.spatial import KDTree
import os
import yaml
import time
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
        self.info_dict = {}
        self.states_file_step = "states_aux.csv"
        self.states_file_step_node = "states_best_parent_aux.csv"
        if obstacles == 5:
            obstacle_gap = 3.0
            self.map.obstacles_five(obstacle_gap)
        self.goal_found = False
        self.best_goal_node = None
        self.best_goal_cost = np.inf
        self.check_list = [199, 249, 499]
        # self.check_list = [49, 249, 400]
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
        self.V = [self.start]  # List of vertices
        self.quad = quad
        self.quad.x[0:2] = np.array(self.start[0:2]).reshape((2,1))  # initial load position
        self.quad.x[2:4] = np.array(self.start[2:4]).reshape((2,1))  # initial load velocity
        self.E_states = {}  # Edges in the tree
        self.E_states[self.start] = self.quad.x.flatten()
        self.E = {}  # Edges in the tree
        self.E_commands = {} #dictionary to save the commands
        # self.E[self.start] = None  # start has no parent
        self.dt = self.quad.dt
        self.min_vel = -5.0
        self.max_vel = 5.0
        self.K = np.diag([1.0, 1.0, 4, 4])  # weighting matrix for cost function
        self.cost_0 = None
        self.cost_1 = None
        self.cost_2 = None
        self.path_length_0 = None
        self.path_length_1 = None
        self.path_length_2 = None

    def sample(self) -> tuple:
        p = random.random()
        if p < self.epsilon:
            return self.goal  # position, velocity, acceleration
        else:
            x = random.uniform(0, self.map.width)
            y = random.uniform(0, self.map.height)
            vx = random.uniform(self.min_vel, self.max_vel)
            vy = random.uniform(self.min_vel, self.max_vel)
            return (x, y, vx, vy) # position, velocity, acceleration
                
    def nearest(self, q_rand: tuple) -> tuple:
        min_dist = np.inf
        q_near = None
        for v in self.V:
            v_array = np.array(v).reshape((4,1))
            q_rand_array = np.array(q_rand).reshape((4,1))
            dist = (q_rand_array[0:4] - v_array).T @ self.K @ (q_rand_array[0:4] - v_array)
            if dist < min_dist:
                min_dist = dist
                q_near = v
        return q_near
    
    def cost_to_come(self, q):
        cost = 0
        current = q
        while current != self.start:
            if current in self.E:
                path = self.E[current][0]
                # print(f"Current node: {current}, Parent: {self.E[current][0]}")
                P = np.vstack(path)
                diffs = np.diff(P[:,:], axis=0)
                for i in range(diffs.shape[0]):
                    cost += diffs[i].reshape((4,1)).T @ self.K @ diffs[i]
                current = self.array_to_tuple(path[0])
            else:
                print(f"node {current} not in the tree")
                break
        return float(cost)
    
    def rewire(self, q_new, neighbors):

        cost_new = self.cost_to_come(q_new)
        for neighbor in neighbors:
            if neighbor == self.start:
                continue  # Don't rewire the start node
            diff = (np.array(q_new) - np.array(neighbor)).reshape((4,1))
            cost_new_neigh = diff.T @ self.K @ diff
            tentative_cost = cost_new + float(cost_new_neigh)
            current_cost = self.cost_to_come(neighbor)
            if  tentative_cost < current_cost:
                # Rewire
                nearest_states = self.E_states[q_new]
                L_new, States_new, Commands_new = self.steer_toward_node(nearest_states, np.hstack((np.array(neighbor), np.array([0.0, 0.0]))).reshape((6,1)))
                if L_new is not None:
                    # l_new = L_new[-1]
                    # l_new_point = self.array_to_tuple(l_new)
                    # neighbor_tuple = self.array_to_tuple(neighbor)
                    self.E[neighbor] = [L_new]
                    States_new[0:4] = np.asarray(q_new)
                    self.E_states[neighbor] = States_new
                    ic(q_new, neighbor)
                    self.E_commands[neighbor] = Commands_new
    
    def neighborhood(self, q_new):
        k = int(np.ceil(np.e*1.5*np.log(len(self.V))))
        if k < 1:
            k = 1
        elif k > len(self.V):
            k = len(self.V)
        tree = KDTree(self.V)
        _, idxs = tree.query(q_new, k=k)
        if len(idxs.shape) == 0:
            idxs = [idxs]
        neighbors = [self.V[i] for i in idxs]
        return neighbors

    def best_parent(self, q_new, neighbors):
        best_cost = np.inf
        best_parent = None
        L_best = None
        States_best = None
        Commands_best = None
        q_new_array = np.array(q_new)
        for neighbor in neighbors:
            if not np.array_equal(neighbor, self.goal):
                cost_q = self.cost_to_come(neighbor)
                diff = (q_new_array - np.array(neighbor)).reshape((4,1))
                cost = diff.T @ self.K @ diff
                total_cost = cost_q + cost
                if total_cost < best_cost:
                    states = self.E_states[neighbor]
                    L_new, States_new, Commands_new = self.steer_toward_node(states, np.hstack((q_new_array, np.array([0.0, 0.0]))).reshape((6,1)))
                    if L_new is not None:
                        best_cost = total_cost
                        best_parent = neighbor
                        L_best = L_new
                        States_best = States_new
                        Commands_best = Commands_new
        return best_parent, L_best, States_best, Commands_best

    def steer_toward_node(self, l_near: NDArray, l_rand: NDArray) -> NDArray:
        #l_near and l_rand contain positions, velocities, and accelerations
        tf = 2
        t = 0.0
        x = l_near.flatten()
        self.quad.x = x.reshape((8,1))
        # self.quad.x[2:4] = x[2:4].reshape((2,1))  # initial load velocity
        Pos_vel = [l_near[0:4]] #ensuring that parent info is the first point
        latest_control = [None, None]
        Commands_control = []
        while (np.linalg.norm(x[0:2]-l_rand[0:2].flatten()) > 0.001):
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
            x[6] = np.clip(x[6], -np.pi/2, np.pi/2)
            x[5] = np.clip(x[5], -np.pi, np.pi)  # limit angular velocities
            x[7] = np.clip(x[7], -np.pi, np.pi)
            t = sol.t[-1]
            if (t > tf):
                # print("Exceeded time limit in steer_toward_node")
                return None, None, None
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

    # def float_to_int(self, v):
    #     multiplier = 1e5
    #     return (int(v[0]*10000),int(v[1]*10000),int(v[2]*10000),int(v[3]*10000))

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
            x[6] = np.clip(x[6], -np.pi/2, np.pi/2)
            x[5] = np.clip(x[5], -np.pi, np.pi)  # limit angular velocities
            x[7] = np.clip(x[7], -np.pi, np.pi)
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

    def search(self, num_iter=500, seed=None):
        if seed is not None:
            random.seed(seed)
            self.seed = seed
        begin = time.time()
        for i in range(int(num_iter)):
            ic(i)
            if i in self.check_list:
                self.get_stats(i)
            l_rand = self.sample()
            l_nearest = self.nearest(l_rand)
            nearest_states = self.E_states[l_nearest]
            L_new, Q_new, Commands_new = self.steer(nearest_states, np.hstack((np.array(l_rand), np.array([0.0, 0.0]))).reshape((6,1)))
            if L_new is None:
                continue
            l_new = L_new[-1]
            l_new_point = self.array_to_tuple(l_new)
            neighbors = self.neighborhood(l_new_point)
            best_parent, L_best, Q_best, Commands_best = self.best_parent(l_new_point, neighbors)
            if L_best is not None:
                L_new = L_best
                Q_new = Q_best
                Commands_new = Commands_best
                l_new = L_new[-1]
                l_new_point = self.array_to_tuple(l_new)
            if l_new_point not in self.V:
                self.V.append(l_new_point)
                self.E[l_new_point] = [L_new]
                self.E_states[l_new_point] = Q_new
                self.E_commands[l_new_point] = Commands_new
            if l_new_point != self.goal:
                self.rewire(l_new_point, neighbors)

            # 🔥 Animate tree expansion
            # self.animate_tree()
            # Goal check

            if self.in_goal_region(l_new):
                # print(f"Goal reached iteration {i}!")
                self.goal_found = True
                self.info_dict[f"goal found with {i} iterations"] = {}
                cost = self.cost_to_come(l_new_point)
                if cost < self.best_goal_cost:
                    self.best_goal_cost = cost
                    self.best_goal_node = l_new_point
                # self.info_dict['num_iteretions_to_find_goal'] = i

        end = time.time()
        time_diff = end - begin

        # Get total seconds
        total_seconds = time_diff

        # Convert to minutes and seconds
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        self.info_dict[f'execution time'] = (minutes, seconds)
        if self.goal_found:
            num_iterations = i
            path, path_length, commands = self.reconstruct_path(self.best_goal_node)
            self.info_dict[f'{num_iterations} iterations:'] = {}
            self.info_dict[f'{num_iterations} iterations:']['path_lenght'] = float(path_length)
            self.info_dict[f'{num_iterations} iterations:']['commands'] = commands
            self.info_dict[f'{num_iterations} iterations:']['cost'] = self.best_goal_cost
            with open(self.file_name, 'w') as file:
                yaml.dump(self.info_dict, file, Dumper=NoAliasDumper)
            return path
        else:
            with open(self.file_name, 'w') as file:
                yaml.dump(self.info_dict, file, Dumper=NoAliasDumper)
            print("Goal not reached within max iterations.")
            return None
    def edge_cost(self, L):
        cost = 0.0
        for i in range(1, len(L)):
            diff = (L[i] - L[i-1]).reshape(4,1)
            cost += float(diff.T @ self.K @ diff)
        return cost
    
    def in_goal_region(self,l_new):
        return np.linalg.norm(l_new[0:2]-self.goal[0:2]) <= self.goal_tolerance

    def array_to_tuple(self, arr):
        return tuple(float(x) for x in arr.flatten())

    def reconstruct_path(self, q_new):
        path = []
        current = q_new
        # cost = self.cost_to_come(q_new)
        commands = []
        while current != self.start:
            if current not in self.E:
                print(f"Error: Node {current} not found in self.E. Cannot reconstruct path.")
                self.info_dict['error seed'] = self.seed
                self.info_dict['self.E'] = self.E
                return None, 0, None  # Stop and return an error state
            path_points = self.E[current][0]
            commands += self.E_commands[current]
            path = path_points[:-1] + path  # prepend to path
            current = self.array_to_tuple(path_points[0])
        path = [self.start] + path
        P = np.vstack(path)
        diffs = np.diff(P[:,0:2], axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        path_length = np.sum(segment_lengths)
        # num_iterations = len(self.V)
        commands = commands


        # self.path.reverse()  # reverse to get from start to goal
        return path, path_length, commands

    def get_stats(self, num_iterations):
        if num_iterations == self.check_list[0]:
            if self.goal_found:
                path, path_length, commands = self.reconstruct_path(self.best_goal_node)
                self.info_dict[f'{num_iterations} iterations:'] = {}
                self.info_dict[f'{num_iterations} iterations:']['path_lenght'] = float(path_length)
                self.info_dict[f'{num_iterations} iterations:']['commands'] = commands
                self.info_dict[f'{num_iterations} iterations:']['cost'] = self.best_goal_cost
                # self.E_0 = self.E.copy()
                # self.V_0 = self.V.copy()
        if num_iterations == self.check_list[1]:
            if self.goal_found:
                self.cost_1 = self.cost_to_come(self.goal)
                path, path_length, commands = self.reconstruct_path(self.best_goal_node)
                self.info_dict[f'{num_iterations} iterations:'] = {}
                self.info_dict[f'{num_iterations} iterations:']['path_lenght'] = float(path_length)
                self.info_dict[f'{num_iterations} iterations:']['commands'] = commands
                self.info_dict[f'{num_iterations} iterations:']['cost'] = self.best_goal_cost
                # self.E_1 = self.E.copy()
                # self.V_1 = self.V.copy()
        if num_iterations == self.check_list[2]:
            if self.goal_found:
                path, path_length, commands = self.reconstruct_path(self.best_goal_node)
                self.info_dict[f'{num_iterations} iterations:'] = {}
                self.info_dict[f'{num_iterations} iterations:']['path_lenght'] = float(path_length)
                self.info_dict[f'{num_iterations} iterations:']['commands'] = commands
                self.info_dict[f'{num_iterations} iterations:']['cost'] = self.best_goal_cost
                # self.E_2 = self.E.copy()
                # self.V_2 = self.V.copy() 

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
    folder_name = "RRT_star_paths_2D_obstacle_1"
    os.makedirs(folder_name, exist_ok=True)
    for i in range(97,50,-1):
        print(i)
        quad = quad_dyn()
        start = (5.2, 1.50, 0.0, 0.0)
        goal = (3, 2.0, 0.0, 0.0)
        rrt = RRT(start=start, goal=goal, obstacles=1, quad=quad, file_name=f"{folder_name}/rrt_path_seed_{i}.yaml")
        path = rrt.search(seed=i, num_iter=2000)
        # rrt.plot_path(path, fig_name=f"{folder_name}/rrt_path_seed_{i}.pdf")
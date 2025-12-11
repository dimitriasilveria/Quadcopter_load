import numpy as np
import random
from Maps2d import Map
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from controller_quad import controller, closed_loop_dynamics_point
from quad_w_load_dyn_2D import quad_w_load_dyn as quad_dyn

class RRT:
    def __init__(self, start, goal, quad, map_type,l=30, epsilon=0.01, step=2.5, goal_tolerance=1.0):
        self.start = start
        self.goal = goal
        self.epsilon = epsilon
        self.step = step
        self.goal_tolerance = goal_tolerance
        self.map_height = 10
        self.map_width = 10
        self.map = Map(self.map_width, self.map_height,step)
        if map_type == 1:
            self.map.obstacles_one(l)
        elif map_type == 2:
            self.map.obstacles_two()
        elif map_type == 3:
            self.map.obstacles_three()
        elif map_type == 4:
            self.map.obstacles_four()

        self.path_length = 0
        self.V = [self.start]  # List of vertices
        self.E = {}      # Dictionary of edges\
        self.quad = quad
        self.dt = self.quad.dt
        self.min_vel = -5.0
        self.max_vel = 5.0

    def sample(self):
        p = random.random()
        if p < self.epsilon:
            return self.goal
        else:
            x = random.uniform(0, self.map.width)
            y = random.uniform(0, self.map.height)
            vx = random.uniform(self.min_vel, self.max_vel)
            vy = random.uniform(self.min_vel, self.max_vel)
            return (x, y, vx, vy, 0.0, 0.0)  # position, velocity, acceleration

                
    def nearest(self, q_rand):
        min_dist = np.inf
        q_near = None
        for v in self.V:
            dist = np.linalg.norm(np.array(q_rand) - np.array(v))
            if dist < min_dist:
                min_dist = dist
                q_near = v
        return q_near
    
    def steer(self, l_near, l_rand):
        #l_near and l_rand contain positions, velocities, and accelerations
        tf = 1
        t = 0.0
        l_near = np.array(l_near)
        x = np.array(l_rand)
        self.quad.x[0:2] = l_near[0:2].reshape((2,1))  # initial load position
        self.quad.x[2:4] = l_near[2:4].reshape((2,1))  # initial load velocity
        x = l_near.copy()
        Pos = []
        Vel = []
        Quad_pos = []
        #simulate
        while t < tf:
            t_span = (t, t + self.dt)
            sol = solve_ivp(
                fun=lambda tt, xx: closed_loop_dynamics_point(
                    tt, xx, self.quad, controller, l_rand
                ),
                t_span=t_span,
                y0=x,
                method="RK45",
                t_eval=t_span,      # like MATLAB output grid
                rtol=1e-6,
                atol=1e-8
            )
                # Update state
            x = sol.y.T[-1]
            t = sol.t[-1]
            Pos.append(x[0:2])  # store load position only
            Vel.append(x[2:4])  # store load velocity only
            Quad_pos.append(self.quad.quad_position().flatten())
        return Pos, Vel, Quad_pos

    def search(self, num_iter = 1e5, seed=None):
        if seed is not None:
            random.seed(seed)

        i = 0
        for i in range(int(num_iter)):
            l_rand = self.sample()
            l_nearest = self.nearest(l_rand)
            L_new, V_new, Q_new = self.steer(l_nearest, l_rand) #get load position and velocity, and quad position trajectories
            if self.map.is_valid(q_nearest, q_new):
                if q_new not in self.V:
                    self.V.append(q_new)
                self.E[q_new] = [q_nearest,np.linalg.norm(np.array(q_new) - np.array(q_nearest))]
            i += 1
            if  q_new == self.goal:
                print("Goal reached!")
                return self.reconstruct_path(q_new), i
            
        print("Goal not reached within max iterations.")
        return None, i
    
    def reconstruct_path(self, q_new):
        path = [self.goal]
        current = q_new
        self.path_length = np.linalg.norm(np.array(self.goal) - np.array(q_new))
        while current != self.start:
            path.append(current)
            if current in self.E:
                self.path_length += self.E[current][1]  # Add edge length to path length
                current = self.E[current][0]  # Move to the parent node
            else:
                break
        path.append(self.start)
        path.reverse()
        return path
    
    def plot_path(self, path, fig_name="rrt_path.pdf"):
        fig, ax = plt.subplots()
        ax = self.map.display(ax)
        xs, ys = zip(*self.V)
        ax.scatter(xs, ys, c='blue', s=5)
        if path:
            path_xs, path_ys = zip(*path)
            ax.plot(path_xs, path_ys, c='red', linewidth=2)
        plt.scatter([self.start[0]], [self.start[1]], c='green', s=50, label='Start')
        plt.scatter([self.goal[0]], [self.goal[1]], c='orange', s=50, label='Goal')
        for child, (parent, _) in self.E.items():
            plt.plot([child[0], parent[0]], [child[1], parent[1]], c='gray', linewidth=0.5)
        plt.legend()
        plt.savefig(fig_name)
        # plt.show()

if __name__ == "__main__":
    quad = quad_dyn()
    rrt = RRT(start=(25, 50), goal=(75, 50), map_type=1, quad=quad)
    path, iterations = rrt.search()
    rrt.plot_path(path)
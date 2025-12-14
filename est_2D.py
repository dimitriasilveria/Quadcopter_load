import numpy as np
from quad_w_load_dyn_2D import quad_w_load_dyn
from icecream import ic
from Maps2d import Map
from scipy.spatial import cKDTree 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
import os

class EST():
    def __init__(self, start_point, start_state, goal, quad, obstacles=5, seed=None):
        self.seed = seed
        self.start = start_point
        self.start_state = start_state
        self.goal = goal
        self.quad = quad
        self.map = Map(10,10,5)
        if obstacles == 5:
            obstacle_gap = 3.0
            self.map.obstacles_five(obstacle_gap)
            file_dir = "est_obstacle_5"
            os.makedirs(file_dir, exist_ok=True)
            self.file_name = f"{file_dir}/est_seed_{seed}_obstacle_{obstacle_gap}.yaml" if seed is not None else "est.yaml"
        elif obstacles == 1:
            self.map.obstacles_one(4)
            file_dir = "est_obstacle_1"
            os.makedirs(file_dir, exist_ok=True)
            self.file_name = f"{file_dir}/est_seed_{seed}_obstacle_1.yaml" if seed is not None else "est.yaml"
        # self.map.obstacles_one(3)
        # self.map.obstacles_two()
        
        self.path = []
        self.states_path = [start_state]
        self.V = [start_point]
        self.E_points = {}
        self.tau_path = []
        self.f_path = []
        self.E_tau = {start_point: [0]}
        self.E_f = {start_point: [9.81*(self.quad.mq + self.quad.ml)]}
        self.E_states = {start_point: start_state}
        self.w = {start_point: 1.0}
        self.w_prime = {start_point: 1.0}
        self.delta = 2.0
        self.goal_tol = 0.5
        self.p = {start_point: 1.0}
        min_thrust, min_tau = self.quad.calc_min_torque_thrust()
        max_thrust, max_tau = self.quad.calc_max_torque_thrust()
        self.min_u = np.vstack((min_tau, min_thrust))
        self.max_u = np.vstack((max_tau, max_thrust))
        self.gif_folder = "gifs"

    def steer(self,x0, tau, f):
        N = 30
        points = np.zeros((self.quad.n_states, N))
        quad_pos = np.zeros((2, N)) #saving quadcopter position to check for collisions
        quad_pos[:,0] = self.quad.quad_position().flatten()
        points[:,0] = x0.flatten()
        commands = np.zeros((2, N-1))
        x = x0
        for i in range(1, N):
            if i != 1:
                tau = 0
            commands[0,i-1] = tau
            commands[1,i-1] = f
            x = self.quad.runge_kutta_step(x, f, tau)
            points[:,i] = x.flatten()
            quad_pos[:,i] = self.quad.quad_position().flatten()
        return points, quad_pos, commands
        
    def sample_actuation(self, x_rand):
        tau = np.random.uniform(self.min_u[0]/2, self.max_u[0]/2)
        f = np.random.uniform(self.min_u[1], self.max_u[1])

        return tau[0], f[0]
    
    def update_proximity(self, x_new,):
        tree = cKDTree(self.V)
        indices = tree.query_ball_point(x_new, r=self.delta, return_sorted=True)
        n = len(indices)
        self.w[x_new] = n
        self.V.append(x_new)
        for index in indices:
            neighbor = self.V[index]
            self.w[neighbor] += 100  # increase weight of neighbors
        max_w = max(self.w.values())
        for vertex in self.V:
            self.w_prime[vertex] = max_w - self.w[vertex] + 1

        total_w_prime = sum(self.w_prime.values())
        for vertex in self.V:
            self.p[vertex] = self.w_prime[vertex] / total_w_prime

    def sample(self):
        sampled_index = np.random.choice(len(self.V), p=list(self.p.values()))
        sampled_vertex = self.V[sampled_index]
        return sampled_vertex
    
    def search(self, max_iterations=1000):
        np.random.seed(self.seed)
        for it in range(max_iterations):
            x_rand = self.sample()
            tau, f = self.sample_actuation(x_rand)

            if x_rand == self.start:
                x0 = self.start_state
            else:
                x0 = self.E_states[x_rand][:,-1].reshape((self.quad.n_states,1))

            X_new, Q_new, Commands = self.steer(x0, tau, f)
            x_new = (float(X_new[0,-1]), float(X_new[1,-1]))

            x_new_points = []
            is_free = True
            for (point, q_pos) in zip(X_new.T, Q_new.T):
                point_tuple = (float(point[0]), float(point[1]))
                quad_pos = (float(q_pos[0]), float(q_pos[1]))
                x_new_points.append(point_tuple)
                if not self.map.is_free(point_tuple, quad_pos, self.quad.L):
                    is_free = False
                    break

            if is_free and x_new not in self.V:
                self.E_points[x_new] = x_new_points
                self.E_states[x_new] = X_new
                self.E_tau[x_new] = Commands[0,:].tolist()
                self.E_f[x_new] = Commands[1,:].tolist()

                # Parent = first point
                parent = x_new_points[0]

                self.update_proximity(x_new)

                # Yield for animation
                # yield ("extend", parent, x_new)

                if self.check_goal_reached(x_new):
                    self.path = self.reconstruct_path(x_new)
                    self.save_info()
                    # yield ("goal", None, None)
                    return
        if (it == max_iterations - 1) and not self.path:
            self.save_info()
            print("Max iterations reached, no path found.")

    def save_info(self):
        #function to save the control efforts along the path, the path length (in meters) and number of iteration in yaml file
        if self.path:
      
            path_length = 0.0
            for i in range(len(self.path)-1):
                p1 = np.array(self.path[i])
                p2 = np.array(self.path[i+1])
                path_length += np.linalg.norm(p2 - p1)
            info = {
                'tau_list': self.tau_path,
                'f_list': self.f_path,
                'path_length': str(path_length),
                'iterations': len(self.V)
            }
        else:
            info = {
                'path_length': 'np.inf',
                'iterations': len(self.V)
                
            }
        
        with open(self.file_name, 'w') as file:
            yaml.dump(info, file)

    def reconstruct_path(self, x_goal):
        x_current = x_goal
        while x_current != self.start:
            path_points = self.E_points[x_current]
            self.path = path_points[:-1] + self.path
            self.states_path = self.E_states[x_current][:, :-1] + self.states_path
            self.tau_path = self.E_tau[x_current][:-1] + self.tau_path
            self.f_path = self.E_f[x_current][:-1] + self.f_path
            x_current = path_points[0]
        self.path.reverse()
        return self.path
    
    def check_goal_reached(self, x):
        return np.linalg.norm(np.array(x) - np.array(self.goal)) < self.goal_tol


if __name__ == "__main__":
    # def search(obstacles, seed):

    for i in range(100):
        print(i)
        seed = i
        quad = quad_w_load_dyn()
        start_state = np.zeros((quad.n_states,1))
        start_state[0:quad.n_states] = quad.x.copy()
        start_point = (5.2,1.5)
        start_state[0:2] = np.array([[start_point[0]],[start_point[1]]])
        goal = (3,2.0)
        # goal = (3,1.0)
        est = EST(start_point, start_state, goal, quad, obstacles=1, seed=seed)
        est.search(100000)
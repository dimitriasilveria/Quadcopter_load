import numpy as np
from quad_w_load_dyn import quad_w_load_dyn
from icecream import ic
from Maps import Map
from scipy.spatial import cKDTree 
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class EST():
    def __init__(self, start_point, start_state, goal, quad):
        self.start = start_point
        self.start_state = start_state
        self.goal = goal
        self.quad = quad
        self.map = Map(40,40,40)
        # self.map.obstacles_one(30)
        self.path = []
        self.V = [start_point]
        self.E_points = {}
        self.E_states = {}
        self.w = {start_point: 1.0}
        self.w_prime = {start_point: 1.0}
        self.delta = 10.0
        self.goal_tol = 2.0
        self.p = {start_point: 1.0}
        _, min_tau = self.quad.calc_min_torque_thrust()
        max_thrust, max_tau = self.quad.calc_max_torque_thrust()
        min_thrust = (self.quad.ml + self.quad.mq) * self.quad.g
        self.min_u = np.vstack((min_tau, min_thrust))
        self.max_u = np.vstack((max_tau, max_thrust))

    def steer(self,x0, tau, f):
        N = 50
        points = np.zeros((N, self.quad.n_states+3))
        points[0,] = x0.flatten()
        x = x0
        for i in range(1, N):
            x, Rot = self.quad.runge_kutta_step(x[0:self.quad.n_states], f, tau)
            orientation = R.from_matrix(Rot).as_euler('xyz').reshape((3,1))
            points[i,0:self.quad.n_states] = x.flatten()
            points[i,self.quad.n_states:self.quad.n_states+3] = orientation.flatten()
        return points
        
    def sample_actuation(self):
        tau = np.random.uniform(self.min_u[0:3], self.max_u[0:3])
        f = np.random.uniform(self.min_u[3], self.max_u[3])
        return tau.reshape((3,1)), f
    
    def update_proximity(self, x_new):
        tree = cKDTree(self.V)
        indices = tree.query_ball_point(x_new, r=self.delta, return_sorted=True)
        n = len(indices)
        self.w[x_new] = n
        self.V.append(x_new)
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
    
    def search(self, max_iterations=1000, plot_every=50, path_pause=0.05):
        # --- setup interactive plot ---
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot obstacles, start, goal
        self.map.display(ax)
        ax.scatter(*self.start, color='green', s=100, label='Start')
        ax.scatter(*self.goal, color='red', s=100, label='Goal')

        # Set axis limits based on map size
        ax.set_xlim(0, self.map.width)
        ax.set_ylim(0, self.map.height)
        ax.set_zlim(0, self.map.depth)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=30, azim=45)  # better 3D perspective
        plt.ion()
        plt.show()

        tree_lines = []

        for it in range(max_iterations):
            x_rand = self.sample()
            tau, f = self.sample_actuation()
            if x_rand == self.start:
                x0 = self.start_state
            else:
                x0 = self.E_states[x_rand][-1,:].reshape((self.quad.n_states+3,1))

            X_new = self.steer(x0, tau, f)  # path from x_rand to x_new
            x_new = tuple(X_new[-1,0:3])
            x_new_points = []

            for point in X_new:
                point_tuple = tuple(point[0:3])
                x_new_points.append(point_tuple)
                if not self.map.is_free(point_tuple):
                    break
            else:
                # no collision, add path
                self.E_points[x_new] = x_new_points
                self.E_states[x_new] = X_new
                self.update_proximity(x_new)

                # --- plot tree segment ---
                if it % plot_every == 0 or self.check_goal_reached(x_new):
                    segment = np.array(x_new_points)
                    line, = ax.plot(segment[:,0], segment[:,1], segment[:,2],
                                    color='blue', alpha=0.7)
                    tree_lines.append(line)
                    plt.pause(0.001)

                # check goal
                if self.check_goal_reached(x_new):
                    print("Goal reached!")
                    self.path = self.reconstruct_path(x_new)
                    break

        # --- highlight final path ---
        if self.path:
            print("Animating final path...")
            for segment in self.path:
                segment = np.array(segment)
                ax.plot(segment[:,0], segment[:,1], segment[:,2],
                        color='red', linewidth=3)
                plt.pause(path_pause)

        plt.ioff()
        plt.show()




    def reconstruct_path(self, x_goal):
        x_current = x_goal
        while x_current in self.E:
            X_segment = self.E[x_current]
            self.path.append(list(X_segment))
            x_current = X_segment[0,:].reshape((self.quad.n_states,1))
        return self.path
    
    def check_goal_reached(self, x):
        return np.linalg.norm(np.array(x) - np.array(self.goal)) < self.goal_tol

if __name__ == "__main__":
    quad = quad_w_load_dyn()
    start_state = np.zeros((quad.n_states+3,1))
    start_state[0:quad.n_states] = quad.x.copy()
    orientation = R.from_matrix(quad.R).as_euler('xyz').reshape((3,1))
    start_state[quad.n_states:quad.n_states+3] = orientation
    start_state[0:3] = np.array([[10],[10],[10]])
    start_point = (10,10,10)
    goal = (30,30,30)
    est = EST(start_point, start_state, goal, quad)
    est.search(100000)
    ic(len(est.path))
    #plot the path
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    est.map.display(ax)
    for segment in est.path:
        ax.plot(segment[:,0], segment[:,1], segment[:,2], color='b')
    plt.show()








    


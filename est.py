import numpy as np
from quad_w_load_dyn import quad_w_load_dyn
from icecream import ic
from Maps import Map
from scipy.spatial import cKDTree 
from scipy.spatial.transform import Rotation as R


class EST():
    def __init__(self, start, goal, quad):
        self.start = start
        self.goal = goal
        self.quad = quad
        self.map = Map(100,100,100)
        self.path = []
        self.V = [start]
        self.E = {}
        self.w = {}
        self.w_prime = {}
        self.delta = 10.0
        self.goal_tol = 2.0
        self.p = []
        _, min_tau = self.quad.calc_min_torque_thrust()
        max_thrust, max_tau = self.quad.calc_max_torque_thrust()
        min_thrust = (self.quad.ml + self.quad.mq) * self.quad.g
        self.min_u = np.vstack((min_tau, min_thrust))
        self.max_u = np.vstack((max_tau, max_thrust))

    def steer(self,x0, tau, f):
        N = 10
        points = np.zeros((N, self.quad.n_states+3))
        points[0,:] = x0.flatten()
        x = x0
        for i in range(1, N):
            x, Rot = self.quad.runge_kutta_step(x, f, tau)
            orientation = R.from_matrix(Rot).as_euler('xyz').reshape((3,1))
            points[i,0:self.quad.n_states] = x.flatten()
            points[i,self.quad.n_states:self.quad.n_states+3] = orientation.flatten()
        return points
        
    def sample_actuation(self):
        tau = np.random.uniform(self.min_u[0:3], self.max_u[0:3])
        f = np.random.uniform(self.min_u[3], self.max_u[3])
        return tau.reshape((3,1)), f
    
    def update_proximity(self, x_new):
        tree = cKDTree(np.array(self.V).squeeze().T)
        dists, indices = tree.query_ball_point(x_new.flatten(), r=self.delta, return_sorted=True)
        n = len(dists)
        self.w[x_new] = n
        self.V.append(x_new)
        max_w = max(self.w.values())
        for vertex in self.V:
            self.w_prime[vertex] = max_w - self.w[vertex] + 1

        total_w_prime = sum(self.w_prime.values())
        for vertex in self.V:
            self.p[vertex] = self.w_prime[vertex] / total_w_prime

    def sample(self):
        sampled_vertex = np.random.choice(self.V, p=self.p)
        return sampled_vertex
    
    def search(self, max_iterations=1000):
        for it in range(max_iterations):
            ic(it)
            x_rand = self.sample()
            tau, f = self.sample_actuation()
            x0 = self.E[x_rand][ -1,:].reshape((self.quad.n_states,1))
            X_new = self.steer(x0, tau, f) # path from x_rand to x_new
            x_new = X_new[ -1,0:3].reshape((3,1))
            if not self.map.collision_check(x_rand, X_new):
                self.E[x_new] = X_new
                self.update_proximity(x_new)
                if np.linalg.norm(x_new - self.goal) < self.goal_tol:
                    ic("Goal reached!")
                    self.path = self.reconstruct_path(x_new)
                    break

    def reconstruct_path(self, x_goal):
        path = []
        x_current = x_goal
        while x_current in self.E:
            X_segment = self.E[x_current]
            path.append(list(X_segment))
            x_current = X_segment[0,:].reshape((self.quad.n_states,1))
        return path
    

if __name__ == "__main__":
    quad = quad_w_load_dyn()
    start = quad.x
    start[0:3] = np.array([[10],[10],[10]])
    goal = np.array([[50],[50],[50]])
    est = EST(start, goal, quad)
    est.search(5000)
    ic(len(est.path))







    


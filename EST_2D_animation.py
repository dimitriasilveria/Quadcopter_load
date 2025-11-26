import numpy as np
from quad_w_load_dyn_2D import quad_w_load_dyn
from icecream import ic
from Maps2d import Map
from scipy.spatial import cKDTree
import matplotlib 
matplotlib.use("Agg")   # no GUI, no figure windows will appear
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Option 1: import the class directly
from matplotlib.animation import FuncAnimation

import imageio

class EST():
    def __init__(self, start_point, start_state, goal, quad, seed=None):
        self.seed = seed
        self.start = start_point
        self.start_state = start_state
        self.goal = goal
        self.quad = quad
        self.map = Map(10,10,10)
        self.map.obstacles_five(2)
        # self.map.obstacles_one(3)
        # self.map.obstacles_two()
        self.path = []
        self.states_path = [start_state]
        self.V = [start_point]
        self.E_points = {}
        self.E_commands = {start_point: (0, 9.81*(self.quad.mq + self.quad.ml))}
        self.E_states = {start_point: start_state}
        self.w = {start_point: 1.0}
        self.w_prime = {start_point: 1.0}
        self.delta = 2.0
        self.goal_tol = 1.5
        self.p = {start_point: 1.0}
        min_thrust, min_tau = self.quad.calc_min_torque_thrust()
        max_thrust, max_tau = self.quad.calc_max_torque_thrust()
        self.min_u = np.vstack((min_tau, min_thrust))
        self.max_u = np.vstack((max_tau, max_thrust))
        self.gif_folder = "gifs"

    def steer(self,x0, tau, f):
        N = 30
        points = np.zeros((self.quad.n_states, N))
        points[:,0] = x0.flatten()
        x = x0
        for i in range(1, N):
            if i != 1:
                tau = 0
            x = self.quad.runge_kutta_step(x, f, tau)
            points[:,i] = x.flatten()
        return points
        
    def sample_actuation(self, x_rand):
        # (tau, f) = self.E_commands[x_rand]
        # percent = 0.7
        # if tau == 0:
        #     tau = np.random.uniform(self.min_u[0]/2, self.max_u[0]/2)
        # else:
        #     tau = np.random.uniform(tau*(1-percent), tau*(1+percent))
        #     tau = np.clip(tau, self.min_u[0], self.max_u[0])
        # if f == 0:
        #     f = np.random.uniform(self.min_u[1], self.max_u[1])
        # else:
        #     f = np.random.uniform(f*(1-percent), f*(1+percent))
        #     f = np.clip(f, self.min_u[1], self.max_u[1])
        tau = np.random.uniform(self.min_u[0]/2, self.max_u[0]/2)
        f = np.random.uniform(self.min_u[1], self.max_u[1])

        return tau, f
    
    def update_proximity(self, x_new, it):
        tree = cKDTree(self.V)
        indices = tree.query_ball_point(x_new, r=self.delta, return_sorted=True)
        n = len(indices)
        self.w[x_new] = n
        self.V.append(x_new)
        for index in indices:
            neighbor = self.V[index]
            self.w[neighbor] += 1*it  # increase weight of neighbors
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

            X_new = self.steer(x0, tau, f)
            x_new = (float(X_new[0,-1]), float(X_new[1,-1]))

            x_new_points = []
            is_free = True
            for point in X_new.T:
                point_tuple = (float(point[0]), float(point[1]))
                x_new_points.append(point_tuple)
                if not self.map.is_free(point_tuple):
                    is_free = False
                    break

            if is_free and x_new not in self.V:
                self.E_points[x_new] = x_new_points
                self.E_states[x_new] = X_new
                # self.E_commands[x_new] = (tau, f)

                # Parent = first point
                parent = x_new_points[0]

                self.update_proximity(x_new, it)

                # Yield for animation
                yield ("extend", parent, x_new)

                if self.check_goal_reached(x_new):
                    self.path = self.reconstruct_path(x_new)
                    yield ("goal", None, None)
                    return


    def reconstruct_path(self, x_goal):
        x_current = x_goal
        while x_current != self.start:
            path_points = self.E_points[x_current]
            self.path = path_points[:-1] + self.path
            self.states_path = [self.E_states[x_current][:, :-1]] + self.states_path
            x_current = path_points[0]
        self.path.reverse()
        return self.path
    
    def check_goal_reached(self, x):
        return np.linalg.norm(np.array(x) - np.array(self.goal)) < self.goal_tol

def search_animate():
    quad = quad_w_load_dyn()
    start_state = np.zeros((quad.n_states,1))
    start_state[0:quad.n_states] = quad.x.copy()
    start_point = (5.0,2.0)
    start_state[0:2] = np.array([[start_point[0]],[start_point[1]]])
    goal = (5.0,8.0)
    est = EST(start_point, start_state, goal, quad)

    fig, ax = plt.subplots()
    est.map.display(ax)
    ax.scatter([start_point[0]], [start_point[1]], c="green", s=80, zorder=5)
    ax.scatter([goal[0]], [goal[1]], c="red", s=80, zorder=5)
    plt.title("EST Search (live)")
    plt.tight_layout()

    plt.ion()
    # plt.show()

    frames = []
    capture_every = 50
    frame_count = 0

    # --- run search ---
    for event in est.search(max_iterations=1000000):
        if event[0] == "extend":
            parent, child = event[1], event[2]
            pts = est.E_points[child]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, color="orange", linewidth=1, zorder=1)

        elif event[0] == "goal":
            # final path in red
            px = [p[0] for p in est.path]
            py = [p[1] for p in est.path]
            ax.plot(px, py, color="red", linewidth=3, zorder=10)
            fig.canvas.draw()
            plt.pause(3)  # small pause to show final path
            # capture frame for GIF
            frame_count += 1
            w, h = fig.canvas.get_width_height()
            img = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8').reshape((h, w, 4))
            img = img[:, :, [1, 2, 3]].copy()  # ARGB -> RGB
            frames.append(img)
            # --- CLOSE THE FIGURE IMMEDIATELY ---
            plt.close(fig)  # <- figure disappears immediately
            plt.ioff()
            # --- SAVE GIF AFTER CLOSING FIGURE ---
            if len(frames) > 0:
                gif_name = f"{est.gif_folder}/est_run_mod_comp.gif"
                fps = 20
                imageio.mimsave(gif_name, frames, fps=fps)
                print(f"Saved GIF: {gif_name}  (frames={len(frames)})")

            break  # <- stop the loop immediately

        # capture frame for GIF
        frame_count += 1
        if frame_count % capture_every == 0:
            w, h = fig.canvas.get_width_height()
            img = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8').reshape((h, w, 4))
            img = img[:, :, [1, 2, 3]].copy()  # ARGB -> RGB
            frames.append(img)

        # update figure live
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.0005)


    # code continues immediately here
    print("Animation finished, figure closed, code continues!")

    Xfull = np.hstack(est.states_path)
    N = Xfull.shape[1]
    X_load_quad = np.zeros((4, N))
    X_load_quad[0:2, :] = Xfull[0:2, :]
    for i in range(N):
        x_l = Xfull[0:2, i].reshape((2,1))
        x_q = x_l + quad.l * np.array([[-np.sin(Xfull[4, i])],[np.cos(Xfull[4, i])]])
        X_load_quad[2:4, i] = x_q.flatten()

    # --- animation setup (2D yz plane) ---
    fig, ax = plt.subplots()
    est.map.display(ax)
    ax.scatter([start_point[0]], [start_point[1]], c="green", s=80, zorder=5)
    ax.scatter([goal[0]], [goal[1]], c="red", s=80, zorder=5)
    plt.title("EST Search (live)")
    plt.tight_layout()

    plt.ioff()

    all_y = np.hstack([X_load_quad[0, :], X_load_quad[2, :]])
    all_z = np.hstack([X_load_quad[1, :], X_load_quad[3, :]])
    # trails and markers
    load_trail, = ax.plot([], [], lw=1, label='Load Trajectory')
    quad_trail, = ax.plot([], [], lw=1, label='Quad Trajectory')
    pend_line, = ax.plot([], [], lw=2)
    load_point, = ax.plot([], [], marker='o', markersize=6)
    quad_point, = ax.plot([], [], marker='s', markersize=8)

    # oriented quad: arm line and two motor markers (created once)
    arm_line, = ax.plot([], [], lw=3, solid_capstyle='round', label='Quad arm')
    motor1_point, = ax.plot([], [], marker='o', ms=6)  # +L motor
    motor2_point, = ax.plot([], [], marker='o', ms=6)  # -L motor

    ax.legend(loc='upper right')

    # If you prefer to keep a list of quad-related artists, define it BEFORE update()
    quad_artists = [arm_line, motor1_point, motor2_point]  # optional, but fine to have

    def init():
        load_trail.set_data([], [])
        quad_trail.set_data([], [])
        pend_line.set_data([], [])
        load_point.set_data([], [])
        quad_point.set_data([], [])
        arm_line.set_data([], [])
        motor1_point.set_data([], [])
        motor2_point.set_data([], [])
        return load_trail, quad_trail, pend_line, load_point, quad_point, arm_line, motor1_point, motor2_point

    def update(i):
        # load position (scalars -> pass as lists)
        y_l, z_l = float(X_load_quad[0, i]), float(X_load_quad[1, i])
        # quad position
        y_q, z_q = float(X_load_quad[2, i]), float(X_load_quad[3, i])

        # pendulum line (quad -> load)
        pend_line.set_data([y_q, y_l], [z_q, z_l])

        # trails
        load_trail.set_data(X_load_quad[0, :i+1], X_load_quad[1, :i+1])
        quad_trail.set_data(X_load_quad[2, :i+1], X_load_quad[3, :i+1])

        # points (single-point must be sequences)
        load_point.set_data([y_l], [z_l])
        quad_point.set_data([y_q], [z_q])

        # --- oriented quad arm & motors using phi_q from Xfull ---
        phi_q = float(Xfull[6, i])   # quad angle stored at index 6 in your state
        u_y = np.cos(phi_q)
        u_z = np.sin(phi_q)

        L = quad.L  # distance from center to motor along the body axis
        m1_y = y_q +  L * u_y
        m1_z = z_q +  L * u_z
        m2_y = y_q -  L * u_y
        m2_z = z_q -  L * u_z

        arm_line.set_data([m1_y, m2_y], [m1_z, m2_z])
        motor1_point.set_data([m1_y], [m1_z])
        motor2_point.set_data([m2_y], [m2_z])
        # plt.pause(0.0001)

        # return ALL artists that changed
        return load_trail, quad_trail, pend_line, load_point, quad_point, arm_line, motor1_point, motor2_point

    anim = FuncAnimation(fig, update, frames=N, init_func=init, blit=True, interval=quad.dt*100)
    anim.save(f"{est.gif_folder}/quad_pendulum_est_mod_comp.gif", writer="pillow", fps=30)
    return
    # In a script use plt.show(); in Jupyter use HTML(anim.to_jshtml())
    # plt.show()
if __name__ == "__main__":
    for i in range(100):
        search_animate()
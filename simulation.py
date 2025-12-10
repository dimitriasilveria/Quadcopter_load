from scipy.integrate import solve_ivp
import numpy as np
from controller import controller, trajectory, closed_loop_dynamics
from quad_w_load_dyn_2D import quad_w_load_dyn as quad_dyn
import matplotlib.pyplot as plt 

quad = quad_dyn()

dt = 0.01
plotEvery = 5
tf = 50.0

t = 0.0
des0 = trajectory(0.0)
quad.x[0:2] = des0[0:2].reshape((2,1))  # initial load position
quad.x[2:4] = des0[2:4].reshape((2,1))  # initial load velocity
x = quad.x.flatten()
Pos = [x[0:2]]  # store initial load position
desPos = [des0[0:2]]  # store initial desired load position

while t < tf:
    # print(t)
    t_span = (t, t + plotEvery*dt)
    t_eval = np.linspace(t_span[0], t_span[1], plotEvery + 1)

    sol = solve_ivp(
        fun=lambda tt, xx: closed_loop_dynamics(
            tt, xx, quad, controller, trajectory
        ),
        t_span=t_span,
        y0=x,
        method="RK45",
        t_eval=t_eval,      # like MATLAB output grid
        rtol=1e-6,
        atol=1e-8
    )

    tStep = sol.t
    qStep = sol.y.T

    # Update state
    x = qStep[-1]
    t = tStep[-1]
    desPos.append(trajectory(t)[0:2])  # store desired load position only
    Pos.append(qStep[:, 0:2])  # store load position only

#plot results

Pos = np.vstack(Pos)
desPos = np.vstack(desPos)
plt.plot(Pos[:,0], Pos[:,1], label='Load Path')
plt.plot(desPos[:,0], desPos[:,1], '--', label='Desired Path')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Quadcopter Load Position Over Time')
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()
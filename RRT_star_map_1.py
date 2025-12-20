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
from RRT_start_debug import RRT
class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

if __name__ == "__main__":
    folder_name = "RRT_star_paths_2D_obstacle_1"
    os.makedirs(folder_name, exist_ok=True)
    for i in range(20,0,-1):
        print(i)
        quad = quad_dyn()
        start = (5.2, 1.50, 0.0, 0.0)
        goal = (3, 2.0, 0.0, 0.0)
        rrt = RRT(start=start, goal=goal, obstacles=1, quad=quad, file_name=f"{folder_name}/rrt_path_seed_{i}.yaml")
        path = rrt.search(seed=i, num_iter=2000)
        # rrt.plot_path(path, fig_name=f"{folder_name}/rrt_path_seed_{i}.pdf")
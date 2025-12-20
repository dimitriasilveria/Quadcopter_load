# import glob
# import yaml

# #est algorithm
# algorithm = 'est'
# est_folder = "/home/dimitria/PhD_codes/quadcopter_load/est_obstacle_1/*.yaml"
# est_files = glob.glob(est_folder)

# success_list = []

# for file in est_files:
#     with open(file, 'r') as f:
#         data = yaml.safe_load(f)

#     if 'tau_list' in data:
#         print(file)
#         seed = file[-18:-16]
#         success_list.append(seed)
    
# for i in range(len(success_list)):
#     if success_list[i][0] == '_':
#         success_list[i] = success_list[i][1]
#     success_list[i] = int(success_list[i])

# print(len(success_list), success_list)

import numpy as np
import glob
#I have one npz file per run
#each variable 'states' constitute a list of arrays with states of each iteration
#
def open_files(method='RRT', obstacle=1):

    folder = f"/home/dimitria/PhD_codes/quadcopter_load/states/{method}_obstacle_{obstacle}_states/*.npz"
    states_files = glob.glob(folder)
    load_angles_per_run = []
    load_angular_vel_per_run = []
    for file in states_files:
        data = np.load(file)
        states = data['states']
        if len(states) > 0:

            print(states)


#obstacle 5

obstacle = 5

#est
method = 'EST'
open_files(method,obstacle)

#rrt
# method = 'RRT'
# open_files(method,obstacle)


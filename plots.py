import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import glob
from icecream import ic
import yaml
import numpy as np
import statistics
import os
import re

import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True
mpl.rcParams["font.size"] = 14
# plotting rrt results
figures_dir = 'figures'
os.makedirs(figures_dir, exist_ok=True)

algorithm = 'rrt'
rrt_folder = '/home/dimitria/PhD_codes/quadcopter_load/RRT_paths_2D_obstacle_5/*.yaml'
rrt_files = glob.glob(rrt_folder)

path_length_list = []
num_success = 0
tau_list = []
f_list = []
num_iteratios_list = []
costs_list = []
for file in rrt_files:
    with open(file, 'r') as f:
        data = yaml.safe_load(f)
    if 'path_length' in data:
        path_length_list.append(float(data['path_length']))
        num_success += 1
        for command in data['commands']:
            tau_list.append(command[0])
            f_list.append(command[1])
            num_iteratios_list.append(data['num_iterations'])
        if 'path_cost' in data:
            costs_list.append(float(data['path_cost']))

min_tau = min(tau_list)
max_tau = max(tau_list)
ic(f"percentage of success:", num_success/len(rrt_files))
median_path_length = statistics.median(path_length_list)
ic(f"median path length of algorithm {algorithm}: {median_path_length}")
median_cost = statistics.median(costs_list)
ic(f"median of number of iterations to reach the goal: {median_cost}")
median_iterations = statistics.median(num_iteratios_list)
ic(f"median of number of iterations to reach the goal: {median_iterations}")

#plotting torque
plt.hist(tau_list, bins=10)
# plt.title("Histogram of $\tau$")
plt.ylabel("Frequency")
plt.xlabel(r"$\tau$")
plt.savefig(f"{figures_dir}/{algorithm}_tau_histogram.pdf", bbox_inches="tight")
plt.close()
#plotting force
plt.hist(f_list, bins=10)
# plt.title("Histogram of $\tau$")
plt.ylabel("Frequency")
plt.xlabel(r"$f$")
plt.savefig(f"{figures_dir}/{algorithm}_force_histogram.pdf", bbox_inches="tight")
plt.close()

#rrt star
algorithm = 'rrt_star'
rrt_star_folder = '/home/dimitria/PhD_codes/quadcopter_load/RRT_star_paths_2D_obstacle_5/*.yaml'
rrt_star_files = glob.glob(rrt_star_folder)

path_length_list = []
num_success = 0
tau_list = []
f_list = []
num_iteratios_list = []
costs_list = []
for file in rrt_star_files:
    with open(file, 'r') as f:
        data = yaml.safe_load(f)
    if 'path_lenght' in data['1999 iterations:']:
        path_length_list.append(float(data['1999 iterations:']['path_lenght']))
        num_success += 1
        for command in data['1999 iterations:']['commands']:
            tau_list.append(command[0])
            f_list.append(command[1])
            iterations = []

            for key in data.keys():
                m = re.search(r"goal found with (\d+) iterations", key)
                if m:
                    iterations.append(int(m.group(1)))
            num_iteratios_list.append(min(iterations))

    if 'cost' in data['1999 iterations:']:
        costs_list.append(float(data['1999 iterations:']['cost']))

ic(f"percentage of success:", num_success/len(rrt_star_files))
median_path_length = statistics.median(path_length_list)
ic(f"median path length of algorithm {algorithm}: {median_path_length}")
median_cost = statistics.median(costs_list)
ic(f"median of number of iterations to reach the goal: {median_cost}")
median_iterations = statistics.median(num_iteratios_list)
ic(f"median of number of iterations to reach the goal: {median_iterations}")

#plotting torque
plt.hist(tau_list, bins=10)
# plt.title("Histogram of $\tau$")
plt.ylabel("Frequency")
plt.xlabel(r"$\tau$")
plt.savefig(f"{figures_dir}/{algorithm}_tau_histogram.pdf", bbox_inches="tight")
plt.close()
#plotting force
plt.hist(f_list, bins=10)
# plt.title("Histogram of $\tau$")
plt.ylabel("Frequency")
plt.xlabel(r"$f$")
plt.savefig(f"{figures_dir}/{algorithm}_force_histogram.pdf", bbox_inches="tight")
plt.close()
#est algorithm
algorithm = 'est'
est_folder = "/home/dimitria/PhD_codes/quadcopter_load/info_obstacle_5_2/*.yaml"
est_files = glob.glob(est_folder)
tau_list = []
f_list = []
path_length_list = []
num_iteratios_list = []
num_success = 0

for file in est_files:
    with open(file, 'r') as f:
        data = yaml.safe_load(f)
    if data['path_length'] != 'np.inf':
        path_length_list.append(float(data['path_length']))
        tau_list += data['tau_list']
        f_list += data['f_list']
        num_iteratios_list.append(data['iterations'])
        num_success +=1

ic(f"percentage of success:", num_success/len(est_files))
median_path_length = statistics.median(path_length_list)
ic(f"median path length of algorithm {algorithm}: {median_path_length}")
median_iterations = statistics.median(num_iteratios_list)
ic(f"median of number of iterations to reach the goal: {median_iterations}")
#plotting histogram of tau

tau_list_zero = [float(x) for x in tau_list if float(x) != 0]
f_list_zero   = [float(x) for x in f_list if float(x) != 0]
#plotting torque
plt.hist(tau_list_zero, bins=10)
# plt.title("Histogram of $\tau$")
plt.ylabel("Frequency")
plt.xlabel(r"$\tau$")
plt.savefig(f"{figures_dir}/{algorithm}_tau_histogram.pdf", bbox_inches="tight")
plt.xlim(min_tau, max_tau)
plt.close()

#plotting force
plt.hist(f_list_zero, bins=10)
# plt.title("Histogram of $\tau$")
plt.ylabel("Frequency")
plt.xlabel(r"$f$")
plt.savefig(f"{figures_dir}/{algorithm}_force_histogram.pdf", bbox_inches="tight")
plt.close()
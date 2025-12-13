import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import glob
from icecream import ic
import yaml
import numpy as np
import statistics
import os

# import matplotlib.pyplot as plt
# plt.rcParams["text.usetex"] = True
#plotting est results
figures_dir = 'figures'
os.makedirs(figures_dir, exist_ok=True)
algorithm = 'est'
est_folder = "/home/dimitria/PhD_codes/quadcopter_load/info_obstacle_5_2/*.yaml"
est_files = glob.glob(est_folder)
tau_list = []
f_list = []
path_length_list = []
num_success = 0

for file in est_files:
    with open(file, 'r') as f:
        data = yaml.safe_load(f)
    if data['path_length'] != 'np.inf':
        path_length_list.append(float(data['path_length']))
        tau_list += data['tau_list']
        f_list += data['f_list']
        num_success +=1

ic(f"percentage of success:", num_success/len(est_files))
median_path_length = statistics.median(path_length_list)
ic(f"median path length of algorithm {algorithm}: {median_path_length}")
#plotting histogram of tau

tau_list_zero = [float(x) for x in tau_list if float(x) != 0]
f_list_zero   = [float(x) for x in f_list if float(x) != 0]
#plotting torque
plt.hist(tau_list_zero, bins=5)
# plt.title("Histogram of $\tau$")
plt.ylabel("Frequency")
plt.xlabel(r"$\tau$")
plt.savefig(f"{figures_dir}/{algorithm}_tau_histogram.pdf", bbox_inches="tight")
plt.close()
#plotting force
plt.hist(f_list_zero, bins=5)
# plt.title("Histogram of $\tau$")
plt.ylabel("Frequency")
plt.xlabel(r"$f$")
plt.savefig(f"{figures_dir}/{algorithm}_force_histogram.pdf", bbox_inches="tight")
plt.close()
algorithm = 'rrt'
rrt_folder = '/home/dimitria/PhD_codes/quadcopter_load/RRT_paths_2D_obstacle_5/*.yaml'
rrt_files = glob.glob(rrt_folder)

path_length_list = []
num_success = 0
tau_list = []
f_list = []
for file in rrt_files:
    with open(file, 'r') as f:
        data = yaml.safe_load(f)
    if 'path_length' in data:
        path_length_list.append(float(data['path_length']))
        num_success += 1
        for command in data['commands']:
            tau_list.append(command[0])
            f_list.append(command[1])

ic(f"percentage of success:", num_success/len(est_files))
median_path_length = statistics.median(path_length_list)
ic(f"median path length of algorithm {algorithm}: {median_path_length}")

#plotting torque
plt.hist(tau_list, bins=5)
# plt.title("Histogram of $\tau$")
plt.ylabel("Frequency")
plt.xlabel(r"$\tau$")
plt.savefig(f"{figures_dir}/{algorithm}_tau_histogram.pdf", bbox_inches="tight")
plt.close()
#plotting force
plt.hist(f_list, bins=5)
# plt.title("Histogram of $\tau$")
plt.ylabel("Frequency")
plt.xlabel(r"$f$")
plt.savefig(f"{figures_dir}/{algorithm}_force_histogram.pdf", bbox_inches="tight")
import numpy as np
import matplotlib.pyplot as plt

class Map:
    def __init__(self, width, height, step=2.5):
        self.width = width
        self.height = height
        self.step = step
        self.obstacles = []

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)

    def obstacles_one(self,l):
        self.add_obstacle([(45, l), (55, self.height-l)])

    def obstacles_two(self):
        self.add_obstacle([(10, 35), (15, 49)])
        self.add_obstacle([(15, 35), (35, 40)])
        self.add_obstacle([(35, 35), (40, 65)])
        self.add_obstacle([(10, 51), (15, 65)])
        self.add_obstacle([(15, 60), (35, 65)])

    def obstacles_three(self):
        self.add_obstacle([(60, 35), (65, 65)])
        self.add_obstacle([(65, 60), (85, 65)])
        self.add_obstacle([(85, 51), (90, 65)])
        self.add_obstacle([(60, 35), (85, 40)])
        self.add_obstacle([(85, 35), (90, 49)])

    def obstacles_four(self):
        self.obstacles_two()
        self.obstacles_three()

    #example [(x1, y1), (x2, y2)]
    #where (x1, y1) is bottom-left and (x2, y2) is top-right
    def is_free(self, point):
        if point[0] < 0 or point[0] > self.width or point[1] < 0 or point[1] > self.height:
            return False
        for obs in self.obstacles:
            if point[0] >= obs[0][0] and point[1] >= obs[0][1] and point[0] <= obs[1][0] and point[1] <= obs[1][1]:
                return False
        return True

    def is_valid(self, q_nearest, q_new):
        if not self.is_free(q_new):
            return False
        # Check the line segment between q_nearest and q_new for collisions
        num_checks = int(np.ceil(np.linalg.norm(np.array(q_new) - np.array(q_nearest)))*20)
        for i in range(1, num_checks + 1):
            t = i / num_checks
            intermediate_point = (q_nearest[0] + t * (q_new[0] - q_nearest[0]),
                                  q_nearest[1] + t * (q_new[1] - q_nearest[1]))
            if not self.is_free(intermediate_point):
                return False
        return True

    def display(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        for obs in self.obstacles:
            rect = plt.Rectangle(obs[0], obs[1][0]-obs[0][0], obs[1][1]-obs[0][1], color='gray')
            ax.add_patch(rect)
        return ax
        # plt.show()

if __name__ == "__main__":
    m = Map(100, 100)
    m.obstacles_one(30)
    m.display()

    m2 = Map(100, 100)
    m2.obstacles_two()
    m2.display()

    m3 = Map(100, 100)
    m3.obstacles_three()
    m3.display()
    plt.show()

    m4 = Map(100, 100)
    m4.obstacles_four()
    m4.display()
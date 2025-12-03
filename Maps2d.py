import numpy as np
import matplotlib.pyplot as plt

class Map:
    def __init__(self, width, height, step=2.5):
        self.width = width
        self.height = height
        self.step = step
        self.obstacles = []
        self.obs_buffer = 0.25 # buffer around obstacles

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)

    def obstacles_one(self,l):
        self.add_obstacle([(4.5, 0), (4.8, self.height-l)])
        self.add_obstacle([(6.0, 0), (6.3, self.height-l)])

    def obstacles_two(self):
        self.add_obstacle([(1.0, 3.5), (1.5, 4.9)])
        self.add_obstacle([(1.5, 3.5), (3.5, 4.0)])
        self.add_obstacle([(3.5, 3.5), (4.0, 6.5)])
        self.add_obstacle([(1.0, 5.7), (1.5, 6.5)])
        self.add_obstacle([(1.5, 6.0), (3.5, 6.5)])

    def obstacles_three(self):
        self.add_obstacle([(6.0, 3.5), (6.5, 6.5)])
        self.add_obstacle([(6.5, 6.0), (8.5, 6.5)])
        self.add_obstacle([(8.5, 5.1), (9.0, 6.5)])
        self.add_obstacle([(6.0, 3.5), (8.5, 4.0)])
        self.add_obstacle([(8.5, 3.5), (9.0, 4.9)])

    def obstacles_four(self):
        self.obstacles_two()
        self.obstacles_three()

    def obstacles_five(self, l):
        self.add_obstacle([(0, self.height/2 - l/2), (self.width/2, self.height/2 - l/2 - 0.01*self.height)])
        self.add_obstacle([(self.width/2, self.height/2 + l/2 + 0.01*self.height), (self.width, self.height/2 + l/2)])

    #example [(x1, y1), (x2, y2)]
    #where (x1, y1) is bottom-left and (x2, y2) is top-right

    def is_free(self, point):
        x, y = point
        # check bounds first
        if x < 0 or x > self.width or y < 0 or y > self.height:
            return False
        # check against each obstacle expanded by obs_buffer
        for obs in self.obstacles:
            x1, y1 = obs[0]
            x2, y2 = obs[1]
            if x >= (x1 - self.obs_buffer) and y >= (y1 - self.obs_buffer) and x <= (x2 + self.obs_buffer) and y <= (y2 + self.obs_buffer):
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
    m = Map(10, 10)
    m.obstacles_one(3)
    m.display()

    # m2 = Map(10, 10)
    # m2.obstacles_two()
    # m2.display()

    # m3 = Map(10, 10)
    # m3.obstacles_three()
    # m3.display()
    # plt.show()

    # m4 = Map(10, 10)
    # m4.obstacles_four()
    # m4.display()
    m = Map(10, 10)
    m.obstacles_five(2)
    m.display()
    plt.show()
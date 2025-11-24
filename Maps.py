import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

class Map:
    def __init__(self, width, height, depth, step=2.5):
        self.width = width
        self.height = height
        self.depth = depth
        self.step = step
        self.obstacles = []
        self.obstacle_buffer = 1.0  # buffer around obstacles


    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)

    def obstacles_one(self,l):
        self.add_obstacle([(45, l,0), (55, self.height-l,0), (45, l,20), (55, self.height-l,20)])

    # def obstacles_two(self):
    #     self.add_obstacle([(10, 35), (15, 49)])
    #     self.add_obstacle([(15, 35), (35, 40)])
    #     self.add_obstacle([(35, 35), (40, 65)])
    #     self.add_obstacle([(10, 51), (15, 65)])
    #     self.add_obstacle([(15, 60), (35, 65)])

    # def obstacles_three(self):
    #     self.add_obstacle([(60, 35), (65, 65)])
    #     self.add_obstacle([(65, 60), (85, 65)])
    #     self.add_obstacle([(85, 51), (90, 65)])
    #     self.add_obstacle([(60, 35), (85, 40)])
    #     self.add_obstacle([(85, 35), (90, 49)])

    # def obstacles_four(self):
    #     self.obstacles_two()
    #     self.obstacles_three()

    #example [(x1, y1), (x2, y2)]
    #where (x1, y1) is bottom-left and (x2, y2) is top-right
    def is_free(self, point):
        x, y, z = point
        if x < 0 or x > self.width or y < 0 or y > self.height or z < 0 or z > self.depth:
            return False
        for obs in self.obstacles:
            xs = [p[0] for p in obs]
            ys = [p[1] for p in obs]
            if len(obs[0]) > 2:
                zs = [p[2] for p in obs]
            else:
                zs = [0, self.depth]

            xmin, xmax = min(xs) - self.obstacle_buffer, max(xs) + self.obstacle_buffer
            ymin, ymax = min(ys) - self.obstacle_buffer, max(ys) + self.obstacle_buffer
            zmin, zmax = min(zs) - self.obstacle_buffer, max(zs) + self.obstacle_buffer

            if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax:
                return False
        return True
    def collision_check(self, points):
        for point in points:
            if not self.is_free(point):
                return False
        return True

    def display(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_zlim(0, self.depth)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        for obs in self.obstacles:
            # obs can be 2D: [(x1,y1),(x2,y2)] or 3D points [(x,y,z),...]
            pts = list(obs)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if len(pts[0]) > 2:
                zs = [p[2] for p in pts]
            else:
                # extrude 2D obstacle through full depth
                zs = [0, self.depth]

            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            zmin, zmax = min(zs), max(zs)

            # 8 vertices of the cuboid
            v = [
                (xmin, ymin, zmin),
                (xmax, ymin, zmin),
                (xmax, ymax, zmin),
                (xmin, ymax, zmin),
                (xmin, ymin, zmax),
                (xmax, ymin, zmax),
                (xmax, ymax, zmax),
                (xmin, ymax, zmax),
            ]

            # 6 faces (each as list of 4 vertices)
            faces = [
                [v[0], v[1], v[2], v[3]],
                [v[4], v[5], v[6], v[7]],
                [v[0], v[1], v[5], v[4]],
                [v[2], v[3], v[7], v[6]],
                [v[1], v[2], v[6], v[5]],
                [v[4], v[7], v[3], v[0]],
            ]

            pc = Poly3DCollection(faces, facecolors='gray', edgecolors='k', linewidths=0.5, alpha=0.5)
            ax.add_collection3d(pc)

        return ax

if __name__ == "__main__":
    m = Map(40, 40,40)
    # m.obstacles_one(30)
    m.display()
    plt.show()

    # m2 = Map(100, 100)
    # m2.obstacles_two()
    # m2.display()

    # m3 = Map(100, 100)
    # m3.obstacles_three()
    # m3.display()

    # m4 = Map(100, 100)
    # m4.obstacles_four()
    # m4.display()
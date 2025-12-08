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

    def is_free(self, load, quad_pos, quad_length):
        """
        Check if load, quadcopter, cable, and motors are collision-free
        """
        x, y = load
        qx, qy = quad_pos
        
        # 1. Check load position
        if not self._is_point_free(x, y):
            return False
        
        # 2. Check quadcopter center position
        if not self._is_point_free(qx, qy):
            return False
        
        # 3. Check cable (line between load and quad)
        if not self._is_line_free(load, quad_pos):
            return False
        
        # 4. Check quadcopter motors (assuming motors are at quad_length from center)
        # For a planar quadcopter, check all 4 motor positions (or 2 if truly 1D)
        motor_offsets = [
            (quad_length, 0),   # right motor
            (-quad_length, 0)
        ]
        
        for dx, dy in motor_offsets:
            mx, my = qx + dx, qy + dy
            if not self._is_point_free(mx, my):
                return False
        
        # 5. Optional: Check quadcopter body (treat as circle or rectangle)
        # If quad has physical body, check intermediate points
        # body_radius = quad_length * 0.5  # adjust as needed
        # if not self._is_circle_free(qx, qy, body_radius):
        #     return False
        
        return True

    def _is_point_free(self, x, y):
        """Check if a single point is within bounds and obstacle-free"""
        # Check bounds
        if x < 0 or x > self.width or y < 0 or y > self.height:
            return False
        
        # Check obstacles
        for obs in self.obstacles:
            (ox1, oy1), (ox2, oy2) = obs
            if ox1 <= x <= ox2 and oy1 <= y <= ox2:
                return False
        
        return True

    def _is_line_free(self, p1, p2):
        """Check if line segment between two points is collision-free"""
        x1, y1 = p1
        x2, y2 = p2
        
        # Calculate number of checks based on distance
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        n_checks = max(int(np.ceil(distance * 10)), 10)  # at least 10 checks
        
        for i in range(n_checks + 1):
            t = i / n_checks
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            if not self._is_point_free(x, y):
                return False
        
        return True

    def _is_circle_free(self, cx, cy, radius):
        """Check if circle is collision-free (sample points on perimeter)"""
        n_samples = max(int(2 * np.pi * radius * 2), 8)
        
        for i in range(n_samples):
            angle = 2 * np.pi * i / n_samples
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            
            if not self._is_point_free(x, y):
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
    # m = Map(10, 10)
    # m.obstacles_one(2)
    # m.display()

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
    m.obstacles_five(3.)
    m.display()
    plt.show()
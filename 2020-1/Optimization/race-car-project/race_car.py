import numpy as np
import matplotlib.pyplot as plt

class racetrack:
    def __init__(self,segments):
        self.segments = segments
    
    def render_pwl(self, increments, init_ang=0):
        """
        find the points along the track for plotting
        """
        points = [(0.,0.)]
        ang = init_ang
        for seg in segments:
            # find the parametric formula of the segment
            # given a curve that vertex at (0,0) with curvature k:
            # focus is at (0,k/4)
            pass


class parabola_segment:
    def __init__(self, a, max_x):
        self.a = a # curvature
        self.shift = np.array([0,0],dtype=float)
        self.theta = 0.
        self.m = max_x

    def copy(self):
        ps = parabola_segment(self.a, self.m)
        ps.shift = self.shift.copy()
        ps.theta = self.theta
        return ps

    def set_shift(self,shift_vec):
        self.shift = np.array(shift_vec,dtype=float)
        return self

    def add_shift(self, shift_vec):
        self.shift += shift_vec
        return self

    def set_rotation(self,rad):
        self.theta = rad % (2*np.pi)
        return self
    
    def add_rotation(self,rad):
        self.theta = (self.theta+rad) % (2*np.pi)
        return self

    def get_parametric(self):
        """
        returns the polynomials of t for x and y
        """
        a, m, theta = self.a, self.m, self.theta
        c, s = np.cos, np.sin
        x_poly = np.array([self.shift[0], m*c(theta), -a*m**2*s(theta)])
        y_poly = np.array([self.shift[1], m*s(theta), a*m**2*c(theta)])
        return x_poly, y_poly

    def get_points(self,n_points):
        """
        returns n points on the parabola
        """
        t = np.linspace(0., 1., n_points)
        t2 = t * t
        D = np.vstack([np.ones(n_points), t, t2])
        x_poly, y_poly = self.get_parametric()
        x = np.dot(x_poly[np.newaxis,:],D)
        y = np.dot(y_poly[np.newaxis,:],D)
        return np.vstack([x,y])



if __name__ == "__main__":
    seg = parabola_segment(0.2,5)
    last_seg = seg
    segs = [last_seg]
    for i in range(6):
        new_seg = last_seg.copy().add_shift([2.5,0.5]).add_rotation(0.2)
        segs.append(new_seg)
        last_seg = new_seg

    for seg in segs:
        xy = seg.get_points(20)
        plt.plot(xy[0,:],xy[1,:],'.-')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
 

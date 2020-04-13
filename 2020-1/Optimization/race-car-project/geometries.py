import numpy as np
import matplotlib.pyplot as plt

rotation_mat = lambda dir: np.array([[np.cos(dir), -np.sin(dir)],[np.sin(dir), np.cos(dir)]])

turtulehead= [
        [(0.004,100)]*2,
        [(-0.025,50)]*2,
        [(-0.01,50)]*2,
        [(0,200)]
    ]
turtulehead = sum(turtulehead,[])

turtulehead2 = [(a*1.3, b) for (a,b) in turtulehead]

lionhead = [
        [(0.06,15)],
        [(-0.06,15)],
        [(-0.25,4)]*2,
        [(0.18,8)],
        [(-0.4,4)],
        [(0,12)]
    ]
lionhead = sum(lionhead,[])

class piecewise_path:
    def __init__(self,segments):
        # assume the first segment goes right from (0,0) 
        # with initial direction along x axis
        # assumes the segments were already stitched
        self.segments = segments 
        self.n_segments = len(segments)
   
    def perturbation_to_curvature_matrix(self):
        ret1 = np.zeros((self.n_segments,self.n_segments))
        ret2 = np.zeros((self.n_segments,self.n_segments))
        a = self.get_segment_curvatures()
        m = self.get_segment_lengths()
        for i in range(self.n_segments):
            pert = np.zeros(self.n_segments)
            pert[i] = 1.
            p = self.get_path_by_perturbations(pert)
            ret1[:,i] = p.get_segment_curvatures()-a
            ret2[:,i] = p.get_segment_lengths()-m
        return ret1, ret2


    def get_path_by_perturbations(self,pert):
        """
        return a new piecewise path with endpoints perturbed by delta_b
        calculate a new m to preserve joints on line perpindicular to original curvature
        """
        seg_start_dir = 0.
        prev_endpoint = np.zeros(2)
        segments = []        
        for seg,b in zip(self.segments,pert):
            curr_endpoint = seg.get_points(2)[:,1]
            theta = np.arctan(2*seg.a*seg.m)
            new_endpoint = curr_endpoint+np.array([-b*np.sin(seg_start_dir+theta),b*np.cos(seg_start_dir+theta)])

            new_segment = parabola_segment(0,0).from_direction_and_endpoints(seg_start_dir,prev_endpoint,new_endpoint)
            segments.append(new_segment)

            seg_start_dir += np.arctan(2*new_segment.a*new_segment.m)[0]
            prev_endpoint = new_endpoint
        return piecewise_path(segments)

    def get_segment_curvatures(self):
        return np.array([seg.a for seg in self.segments]).squeeze()

    def get_segment_lengths(self):
        return np.array([seg.m for seg in self.segments]).squeeze()
        
    def copy(self):
        return piecewise_path([seg.copy() for seg in self.segments])


class parabola_segment:
    def __init__(self, a, max_x):
        self.a = a # curvature
        self.shift = np.array([0,0],dtype=float)
        self.theta = 0.
        self.m = max_x

    def __str__(self):
        return str(self.__dict__)

    def copy(self):
        ps = parabola_segment(self.a, self.m)
        ps.shift = self.shift.copy()
        ps.theta = self.theta
        return ps

    def set_shift(self,shift_vec):
        "The (x,y) of the starting point"
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

    def from_endpoint(self,endpoint_vec):
        self.shift = np.array([0.,0.])
        self.theta = 0
        self.a = endpoint_vec[1]/endpoint_vec[0]**2
        self.m = endpoint_vec[0]
        return self

    def from_direction_and_endpoints(self,dir,start_point,end_point):
        """
        set parabola parameters based on direction (theta) and endpoints
        """
        rel_point = end_point-start_point
        rel_point_angle = np.arctan2(rel_point[1],rel_point[0])
        if np.abs(rel_point_angle-dir)> np.pi/2:
            raise ValueError("The direction must be within 90 degrees of the line connecting the points")
        rel_point = rotation_mat(-dir).dot(rel_point[:,np.newaxis])
        self.from_endpoint(rel_point)
        self.set_rotation(dir).set_shift(start_point)
        return self

    def set_curve_length(self, s):
        "set the max_x so that total curve length roughly equals s"
        tol = 1e-4
        q = 20
        eta = 4
        self.m = s
        last_sign = 1
        while True:
            pts = self.get_points(q+1)
            dp = np.diff(pts)
            new_len = np.sum(np.sqrt(np.sum(dp*dp,0)))
            df = np.abs(new_len-s)
            # print("{:.5f}, {}".format(df, last_sign))
            if abs(df) < tol:
                break
            if new_len < s:
                self.m += self.m/eta
                if last_sign <0: 
                    eta *= 2
                last_sign = 1
            else:
                self.m -= self.m/eta
                if last_sign > 0: 
                    eta *=2
                last_sign = -1
        return self
            
    def set_curve_length_old(self, s):
        "set the max_x so that total curve length roughly equals s"
        tol = 1e-4
        q = 20
        self.m = s
        while True:
            pts = self.get_points(q+1)
            dp = np.diff(pts)
            new_len = np.cumsum(np.sqrt(np.sum(dp*dp,0)))
            if np.abs(new_len[-1]-s) < tol:
                break
            if new_len[-1] < s:
                # shouldn't get here
                self.m *= 2
            else: 
                best_ind = np.nonzero(new_len>s)[0][0]
                if best_ind +1 == q:
                    q *=2
                else:
                    self.m = self.m * (best_ind+1)/q
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

    def stitch_at_end(self,other):
        "stitch this parabola at the end of the other"
        other_endpoints = other.get_points(2)
        other_x_poly, other_y_poly = other.get_parametric()
        deriv0 = np.array([0.,1.,2.]) # evaluates from the polynomial its derivative at t=1
        deriv_x = deriv0.dot(other_x_poly)
        deriv_y = deriv0.dot(other_y_poly)
        theta = np.arctan2(deriv_y, deriv_x)
        self.set_shift(other_endpoints[:,1].squeeze())
        self.set_rotation(theta)
        return self

def compose_track(segments=turtulehead,plot=False):
    if type(segments[0])!=parabola_segment:
        segments = [parabola_segment(*k) for k in segments]
    last_seg = segments[0]
    for seg in segments[1:]:
        seg.stitch_at_end(last_seg)
        last_seg=seg
    if plot:
        plot_segments(segments,20)
    return piecewise_path(segments)

def test_fan():
    "draw a fan of identical, but rotated and shifted, parabolas"
    seg = parabola_segment(0.2,5)
    last_seg = seg
    segs = [last_seg]
    for i in range(6):
        new_seg = last_seg.copy().add_shift([2.5,0.5]).add_rotation(0.2)
        segs.append(new_seg)
        last_seg = new_seg
    plot_segments(segs,20)

def plot_segments(segs, n_points, show =True, color=''):
    if color == '':
        color = '-'
    for seg in segs:
        xy = seg.get_points(n_points)
        plt.plot(xy[0,:],xy[1,:],color)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend([str(i) for i in range(len(segs))])
    if show:
        plt.show()


def test_stitch():
    "draw parabolas stitched together"
    k = np.linspace(0.1, 1, 5)
    last_seg = parabola_segment(0,5)
    segs = [last_seg]
    for crv in k[:3]:
        new_seg = parabola_segment(crv,5)# .stitch_at_end(last_seg)
        segs.append(new_seg)
        last_seg = new_seg
    plot_segments(segs,20)

def test_curve_length():
    k = np.linspace(0, 1, 8)
    last_seg = parabola_segment(k[0],5).set_curve_length(3)
    segs = [last_seg]
    for crv in k[1:]:
        new_seg = parabola_segment(crv,5).set_curve_length(1).stitch_at_end(last_seg)
        segs.append(new_seg)
        last_seg = new_seg
    plot_segments(segs,20)

def test_from_endpoint_vec():
    p = [1,10]
    r = parabola_segment(0,0).from_endpoint(p)
    plot_segments([r],10)

def test_from_endpoints():
    sp = np.array([0,0])
    ep = np.array([5,5])
    r = parabola_segment(0,0).from_direction_and_endpoints(np.pi,sp,ep)
    plot_segments([r],10)


def test_perturbation_to_curvature_matrix():
    track = compose_track()
    n_segments = track.n_segments
    pert = (-np.ones(n_segments))**np.arange(n_segments) # alternating +1/-1    
    mat = track.perturbation_to_curvature_matrix()
    a = track.get_segment_curvatures()
    k = np.dot(mat,pert[:,np.newaxis]).squeeze() + a
    m = track.get_segment_lengths()
    dm = -pert * 2 * a * m / np.sqrt((2*a*m)**2+1)
    new_segments = [parabola_segment(k[i],m[i]+dm[i]) for i in range(n_segments)]
    path = compose_track(new_segments)

    startpoints = lambda pw: np.vstack([seg.get_points(2)[:,1] for seg in pw.segments])

    path_startpoints = startpoints(path)
    track_startpoints = startpoints(track)

    plot_segments(track.segments, 20, show=False, color='b-')
    plot_segments(track.segments, 2, show=False, color='b.')
    
    plot_segments(path.segments, 20, show=False, color='r-')
    plot_segments(path.segments, 2, show=False, color='r.')

    plt.show()

def test_get_path_by_perturbations():
    track = compose_track()
    n_segments = track.n_segments
    pert = (-np.ones(n_segments))**np.arange(n_segments) # alternating +1/-1    
    #pert = np.ones(n_segments)
    #pert[3] = -1.
    path = track.get_path_by_perturbations(pert)
    
    startpoints = lambda pw: np.vstack([seg.get_points(2)[:,1] for seg in pw.segments])

    path_startpoints = startpoints(path)
    track_startpoints = startpoints(track)

    plot_segments(track.segments, 20, show=False, color='b-')
    plot_segments(track.segments, 2, show=False, color='b.')
    
    plot_segments(path.segments, 20, show=False, color='r-')
    plot_segments(path.segments, 2, show=False, color='r.')

    plt.show()

def test_perturbation_to_curvature_matrix2():
    track = compose_track()
    n_segments = track.n_segments
    k_mat, m_mat = track.perturbation_to_curvature_matrix2()
    pert = (-np.ones(n_segments))**np.arange(n_segments) # alternating +1/-1    
    
    new_ks = np.dot(k_mat,pert[:,np.newaxis]).squeeze()
    new_ks += track.get_segment_curvatures()
    new_ms = np.dot(m_mat,pert[:,np.newaxis]).squeeze()
    new_ms += track.get_segment_lengths()

    path = track.get_path_by_perturbations(pert)
    1


if __name__ == "__main__":
#    plot_segments([r],10)
#    test_from_endpoints()
#    test_get_path_by_perturbations()
#    test_perturbation_to_curvature_matrix2()
    compose_track(plot=True)
"""
parabola_curve_length
p = (mx, a.mx^2 ) # endpoint
pointing vector: (1,2a.mx)
normally pointing vector = (-2a.mx,1)/sqrt(4a^2.mx^2+1)
new x coord = -b*2a.mx / sqrt(4a^2.mx^2+1)
"""

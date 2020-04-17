import numpy as np
import geometries as gm

def minimum(*args):
    m = np.inf
    for v in args:
        m = np.minimum(v,m)
    return m

def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())

class problem:
    def __init__(self,track_segments):
        self.track_segments = track_segments.copy()
        self.segment_lengths = self.track_segments.get_segment_lengths()
        self.N = self.track_segments.n_segments
        self.gr = None # represents coefficient of friction
        self.gs = None # represents bonus to friction due to acceleration
        self.top_speed = None
        self.initial_speed = None
        self.max_accel = None # represents the top possible acceleration
        self.max_decel = None # represent the top possible deceleration
        self.nominal_speed = None

    def set_mu(self, mu, gs):
        """
        mu*M*g = M*v^2/R
        mu*g = k/q^2
        q^2*gr - k = 0 ==> gr = k/q^2
        gr = mu*g # where g is the constant of gravity
        """
        mu_vector = mu
        if type(mu) != np.ndarray:
            mu_vector = np.array(mu_vector) 
        if mu_vector.size !=1 and (
            mu_vector.squeeze().ndim != 1 or 
            mu_vector.size != self.n_segments
            ):
            raise ValueError("wrong size or shape of input")
        self.gr = mu_vector*10 
        self.gs = gs
        return self

    def set_top_speed(self,speed=80):
        """
        in m/s. Multiply by 3.6 to get the car's km/h top speed
        """
        self.top_speed = speed
        return self

    def set_acc_and_brake_factors(self, acc=8, brake=6):
        """
        in m/(s*s): assumes accelaration/decelarations are constants
        """
        self.max_accel = acc
        self.max_decel = brake        
        return self

    def set_road_width(self, half_width=4):
        self.road_width = half_width

    def set_nominal_speed(self):
        """
        This is the speed that the car travels when following the centerline exactly.
        It is set at the N track endpoints.
        The 0th point is set to a speed of top speed/2.
        """
        a = np.abs(self.track_segments.get_segment_curvatures())
        N, gr, gs = self.N, self.gr, self.gs
        x = self.segment_lengths
        top_centrip_prev_segment = self.gr/a
        top_centrip_next_segment = np.zeros(N)
        top_centrip_next_segment[:N-1] = self.gr*gs/a[1:]
        top_centrip_next_segment[N-1] = np.inf
        
        nominal_speed = minimum(top_centrip_next_segment, top_centrip_prev_segment, self.top_speed**2)
        top_pos_speed_change = 2* x * self.max_accel
        top_neg_speed_change = 2* x * self.max_decel
        count = 0
        while True:
            count +=1
            prev_point_speed = np.roll(nominal_speed,1)
            prev_point_speed[0] = np.inf
            next_point_speed = np.roll(nominal_speed,-1)
            next_point_speed[-1] = np.inf
            if np.all(nominal_speed <= prev_point_speed + top_pos_speed_change) and \
                np.all(nominal_speed <= next_point_speed + top_neg_speed_change):
                break
           
            new_nominal_speed = minimum(nominal_speed, 
                prev_point_speed + top_pos_speed_change,
                next_point_speed + top_neg_speed_change
                )
            nominal_speed = new_nominal_speed
            if count>N:
              raise NotImplementedError("could not solve for nominal speed")
            
        initial_speed_sq= min(nominal_speed[0]+top_neg_speed_change[0]*0.7,self.top_speed**2*0.7)
        self.nominal_speed = np.sqrt(nominal_speed)
        self.initial_speed = np.sqrt(initial_speed_sq)
            
        return self

    def get_problem_dimension(self):
        return 6*self.N-1
    
    def get_problem_matrices(self,sigmas_d,sigmas_u,boundary_d=None,boundary_u=None, goal_only=False):
        """
        goal:
        quadratic coefficient:
        2*s[i]/(4*m[i])+s[i+1](4*m[i+1])
        coupling coefficient:
        s[i]/(2*m[i]) 
        linear coefficient:
        -s[i+1]

        accelaration constraint: 2N constraints (for accelaration and decelaration) 
        m[i]*u[i] - m[i-1]*u[i-1] <= 2*max_accel/(m[i]+m[i-1]) + m[i-1]-m[i]
        
        radial speed constraint: 2N constraints (for segment before and after)
        m[i]^2*(1+2u[i]) - mu*gs/a[i] + mu*gs/a[i]^2 * delta_a <=0

        boundary constraints: replace with penalty on deviations d[i]^2
 
        total parameters: 2N problem parameters + 3N constraints
        """
        tol = 1e-5
        N, dim = self.N, self.get_problem_dimension()
        H = np.zeros((dim, dim))
        F = np.zeros(dim)

        s = self.segment_lengths
        m = self.nominal_speed
        
        # filling in goal
        beta = m[:-1]/(m[:-1]+m[1:])
        tau = 2*s[1:]/(m[:-1]+m[1:])

        tau0 = 2*s[0]/(self.initial_speed+m[0])
        beta0 = m[0]/(self.initial_speed+m[1])  # actually, 1-beta0
        H[0,0] += tau0*beta0**2
        F[0] -= tau0*beta0
        for i in range(N-1):
            H[i,i] += tau[i]*beta[i]**2 
            H[i+1, i+1] += tau[i]*(1-beta[i])**2
            H[i+1,i] += tau[i]*beta[i]*(1-beta[i])
            F[i] -= tau[i]*beta[i]
            F[i+1] -= tau[i]*(1-beta[i])
        
        if goal_only:
            H = symmetrize(H)            
            return H, F

        # filling in the speed and deviation regularization
        for i in range(N):
            H[i,i] = sigmas_u[i]
            H[N+i,N+i] = sigmas_d[i]
        

        # radial end of segment: from 2N to 3N-1
        gr, gs = self.gr, self.gs
        delta_k, _ = self.track_segments.perturbation_to_curvature_matrix()
        a = self.track_segments.get_segment_curvatures()
        r = 1/a
        delta_k[r<0,:] = -delta_k[r<0,:]
        r[r<0] = -r[r<0]
        B = 2*N
        for i in range(N):
            if np.abs(a[i])>tol:
                H[B+i, i] += 1
                H[B+i, N:2*N] += 0.5*r[i]*delta_k[i,:]

                F[B + i] +=  1- gr*gs*r[i]/m[i]**2

        B += N
        # radial start of segment: from 3N to 4N-2
        for i in range(N-1):
            if np.abs(a[i+1]) > tol:
                H[B+i, i] += 1
                H[B+i, N:2*N] += 0.5*r[i+1]*delta_k[i+1,:]

                F[B+i] += 1 - gr*r[i+1]/m[i]**2

        B += N-1
        # deviations
        W = self.road_width
        if boundary_d is not None:
            for i in range(N):
                H[B+i,N+i] += boundary_d[i] # either left or right side constraint
                F[B+i] = -W * np.abs(boundary_d[i])
        B += N
        if boundary_u is not None:
            for i in range(N):
                H[B+i,i] += boundary_u[i] 
                F[B+i] = -0.5 * np.abs(boundary_u[i])


        H = symmetrize(H)            
        return H,F


if __name__ == "__main__":
    prob = problem(gm.compose_track())
    prob.set_top_speed().set_acc_and_brake_factors().set_mu(0.7,1.2)
    prob.set_nominal_speed()
    H, F = prob.get_problem_matrices(np.ones(8))
    print(prob.nominal_speed)
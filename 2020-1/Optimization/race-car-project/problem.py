import numpy as np
import geometries as gm

def minimum(*args):
    m = np.inf
    for v in args:
        m = np.minimum(v,m)
    return m
    
class problem:
    def __init__(self,track_segments):
        self.track_segments = track_segments.copy()
        self.segment_lengths = self.track_segments.get_segment_lengths()
        self.speed_weights = np.convolve(self.segment_lengths,np.array([0.5, 0.5]))
        self.N = self.track_segments.n_segments
        self.gr = None # represents coefficient of friction
        self.gs = None # represents bonus to friction due to acceleration
        self.top_speed = None
        self.initial_speed = None
        self.acc_factor = None # represents the top possible acceleration
        self.brake_factor = None # represent the top possible deceleration
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

    def set_acc_and_brake_factors(self, acc=0.12, brake=0.1):
        """
        in m/(s*m): The speed change in m/s per meter of track
        assuming going from 50 m/s to 100 m/s takes at least 5 seconds
        the car travels 75*5 = 375 meters
        so 50/375 = 0.133
        we set a constant of 0.12 for acc and 0.1 for brake
        for slow speed: assuming going from 5 m/s to 15 m/s takes 3 seconds
        a change of +10m/s on 30 meters is 0.333
        """
        if self.top_speed is None:
            raise ValueError("first set top speed")
        self.acc_factor = acc
        self.brake_factor = brake        
        return self

    def set_nominal_speed(self):
        """
        This is the speed that the car travels when following the centerline exactly.
        It is set at the N track endpoints.
        The 0th point is set to a speed of top speed/2.
        """
        a = np.abs(self.track_segments.get_segment_curvatures())
        N, gr, gs = self.N, self.gr, self.gs
        x = self.segment_lengths
        top_centrip_prev_segment = np.sqrt(self.gr/a)
        top_centrip_next_segment = np.zeros(N)
        top_centrip_next_segment[:N-1] = np.sqrt(self.gr*gs/a[1:])
        top_centrip_next_segment[N-1] = np.inf
        
        nominal_speed = minimum(top_centrip_next_segment, top_centrip_prev_segment, self.top_speed)
        top_pos_speed_change = x * self.acc_factor
        top_neg_speed_change = x* self.brake_factor
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
            self.initial_speed = min(nominal_speed[0]+top_neg_speed_change[0]/2,self.top_speed/2)
            if count>N:
              raise NotImplementedError("could not solve for nominal speed")
            
        self.nominal_speed = nominal_speed
        return self

    def get_problem_matrices(self):
        """
        goal: x[i]/m(1-(u[i-1]+u[i])/2+(u[i-1]+u[i])^2/4) =
        x[i]/(4m) (4 - 2u[i-1]-2u[i] +u[i-1]^2 + 2u[i-1]u[i] + u[i]^2)
        accelaration constraint: u[i]-u[i-1] < acc_factor*x
        """

class old_problem:
    """
    The problem includes N segments which were already stitched (for C1-continuity).
    These N segments have N+1 endpoints (not a loop), which I will refer to as joints.
    The race begins with the car at point 0 (or joint 0), on the track's midline,
    driving at half its max speed.
    Its instantneous orientation just as it crosses the race's start line
    is tangent to the track's midline.
    The decision variables are:
    * b1-bN the "pertrubations", or deviations of the car from the center of the track. 
    The car's turning curvatures are "implicit variables" that are fully determined by the
    N+1 b's (with b0=0)
    * v1-vN the speeds at which the car crosses the path's joints. Instead of using the v's
    directly, our optimization is based on qi = 1/vi
    * The average co-speed in a segment i is (q[i-1]+q[i])/2
    so we need a vector x' = [0.5*x[1], 0.5*(x[1]+x[2]),..., 0.5*(x[n-1]+x[n]), 0.5*x[n]] =
    [0.5*x[1],0.5*x[1:n]]+[0.5*x[1:n],0.5*x[n]
    * The goal is to minimize dot(x', q), which is the time it takes the car to traverse the path
    """
    def __init__(self, track_segments):
        self.track_segments = track_segments.copy()
        self.segment_lengths = self.track_segments.get_segment_lengths()
        self.speed_weights = np.convolve(self.segment_lengths,np.array([0.5, 0.5]))
        self.N = self.track_segments.n_segments
        self.gr = None # represents coefficient of friction
        self.gs = None # represents bonus to friction due to acceleration
        self.top_speed = None
        self.initial_speed = None
        self.acc_factor = None # represents the top possible acceleration
        self.brake_factor = None # represent the top possible deceleration
    


    def set_mu(self, mu_vector= 0.2, gs=1.1):
        """
        mu*M*g = M*v^2/R
        mu*g = k/q^2
        q^2*gr - k = 0 ==> gr = k/q^2
        gr = mu*g # where g is the constant of gravity
        """
        if type(mu_vector) != np.ndarray:
            mu_vector = np.array(mu_vector) 
        if mu_vector.size !=1 and (
            mu_vector.squeeze().ndim != 1 or 
            mu_vector.size != self.n_segments
            ):
            raise ValueError("wrong size or shape of input")
        self.gr = mu_vector*10 
        self.gs = gs
        return self

    # def set_grip_acc_factor(self, grip_acc_factor):
    #     self.grip_acc_factor = grip_acc_factor
    #     return self

    def set_top_speed(self, speed=80):
        """
        in m/s. Multiply by 3.6 to get the car's km/h top speed
        """
        self.top_speed = speed
        self.initial_speed = self.top_speed/2
        return self

    def set_acc_and_brake_factors(self, acc=0.5, br=0.7):
        """
        in m/s^2. Uses a speed constant (top speed) to scale dimensions
        """
        if self.top_speed is None:
            raise ValueError("first set top speed")
        self.acc_factor = 2*acc/self.top_speed
        self.brake_factor = 2*br/self.top_speed
        return self

    def get_problem_matrices(self, sigmas_centrip, sigmas_acc, sigmas_br, sigmas_width, sigmas_top_speed):
        """
        * The problem matrices are the result of adding each constraint
        weighted by its sigma
        * There are N speeds and N pertrubations, since the 0th joint 
        always has perturbation 0 and speed max_speed/2
        * The constant speed does not enter the constraint calculation
        However its coupling with q1 does enter
        * There are penalties for trying to exceed the centripetal force:
        when q_bar[i]^2*gr[i] - k[i] < 0 we have exceeded the allowed centripetal force which should
        lead to a proportional penalty.
        q_bar[i]^2=(c5*q[i-1]+(1-c5)*q[i])^2
        """

        ts = self.track_segments
        # each row represents a single curvature
        # since we want the absolute curvature |k| in the constraint,
        # we multiply each row by the sign of the curvature a
        
        N = self.N
        H = np.zeros((N*2,N*2))
        F = np.zeros(N*2)

        # centripetal constraints
        # k[j]-q[j]^2*gr < 0
        # k[j]-q[j+1]^2*gr*gs < 0
        k_mat, _ = ts.perturbation_to_curvature_matrix()
        a = ts.get_segment_curvatures()
        k_mat[a<0,:] = -k_mat[a<0,:]

        gr, gs, sc, ss = self.gr, self.gs, sigmas_centrip[:N], sigmas_centrip[N:]

        for i in range(N):
            H[i,i] -= sc[i]*gr
            F[N:] += sc[i]*k_mat[i,:] 
        for i in range(N):
            H[i+1,i+1] -= ss[i]*gr*gs
            F[N:] += sc[i]*k_mat[i,:] 

        # accelaration constraints:
        # q[i]-q[i+1]-q[i+1]q[i]*e*x[i+1] < 0 
        sa, sb, e, r, x = sigmas_acc, sigmas_br, self.acc_factor, self.brake_factor, self.segment_lengths
        F[0] -= sa[0]*e*x[0]/self.initial_speed
        for i in range(N-1):
            H[i+1,i] -= 0.5*sa[i+1]*e*x[i+1] 
            H[i,i+1] -= 0.5*sa[i+1]*e*x[i+1]
            F[i] += sa[i]
            F[i+1] -= sa[i]

        # braking constraints
        for i in range(N-1):
            H[i+1,i] -= 0.5*sb[i+1]*r*x[i+1]
            H[i,i+1] -= 0.5*sb[i+1]*r*x[i+1]
            F[i] -= sb[i+1]
            F[i+1] += sb[i+1]
        
        # # penalize high speeds
        for i in range(N):
            H[i,i] -= sigmas_top_speed[i]
        # problem unconstrained goal
        F[:N] += x
        
        # rough boundaries
        for i in range(N):
            H[N+i,N+i] += sigmas_width[i]
        return H, F

    def check_constrains_fulfilled(self,x_star):
        ts, N = self.track_segments, self.N
        a = ts.get_segment_curvatures()
        q = x_star[:N]
        b = x_star[N:]
        k_mat,_ = ts.perturbation_to_curvature_matrix()

        k =  np.dot(k_mat,b[:, np.newaxis]).squeeze()
        k = np.abs(k + a)
        gr, gs = self.gr, self.gs

        segment_start_centrip = k[1:]-q[:-1]**2*gr
        segment_end_centrip = k-q**2*gr*gs
        
        left_boundaries = b - 4 # will be negative as long as b < 4
        right_boundaries = -b - 4 # will be negative as long as b > -4
        1



    def test_penalty_mat(self, pertrubations, cospeeds):
        ts = self.track_segments
        mat = ts.perturbation_to_curvature_matrix()
        pertrubations=pertrubations.reshape(pertrubations.size,1)
        delta_k = mat.dot(pertrubations)
        a = ts.get_segment_curvatures()[:,np.newaxis]
        k = a + delta_k
        c5, N, gr = self.c5, self.N, self.grip_factor
        gr = self.grip_factor
        qbar_column = np.convolve(cospeeds,np.array([c5,1-c5]))[1:] # the first one is dumped
        qbar_column = qbar_column.reshape(-1,1)
        no_penalty = qbar_column**2*gr-np.abs(k) > 0.
        
        # reset rows where there's no current penality
        penalty_mat[no_penality,:]=0 
        penalty_mat = np.diag(k) - np.dot(qbar_mat,np.diag(gr))



"""
    def get_static_penalty_mat(self):
        c5, N, gr = self.c5, self.N, self.grip_factor
        penalty_q_mat1 = np.diag((1-c5)*np.ones(N)) +  np.diag(c5*np.ones(N-1),-1)
        penalty_q_mat2 = np.dot(penalty_q_mat1.T,penalty_q_mat1)
        penalty_q_mat2 = np.dot(np.diag(gr),penalty_q_mat2) # factor each row by corresponding coef 
        penalty_k_mat = self.track_segments.pertrubations_to_curvature_matrix()
        z = np.zeros(N,N)
        penalty_mat = np.concatenate([[-penalty_q_mat2,z],[z,penalty_k_mat]])
        return penalty_mat


    def get_step_penalty_mat(self, pertrubations, cospeeds):
        if self.static_penality is None:
            self.static_penalty = self.get_static_penalty_mat()

        penalty_mat = self.static_penalty.copy()

        state_vec = np.vstack([cospeeds[:,np.newaxis], pertrubations[:,np.newaxis]])
        penalty_values = np.dot(penalty_mat,state_vec)
        penalty_mat[penalty_values<0,:]=0
        return penalty_mat
"""

if __name__ == "__main__":
    prob = problem(gm.compose_track())
    prob.set_top_speed().set_acc_and_brake_factors().set_mu(0.7,1.2)
    prob.set_nominal_speed()
    print(prob.nominal_speed)
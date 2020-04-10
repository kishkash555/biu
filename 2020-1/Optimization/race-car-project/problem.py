import numpy as np
import geometries as gm

class problem:
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
        self.grip_factor = None
        self.top_speed = None
        self.initial_speed = None
        self.segment_lengths = self.track_segments.get_segment_lengths()
        self.speed_weights = np.convolve(self.segment_lengths,np.array([0.5, 0.5]))
        self.N = self.track_segments.n_segments
        self.c5 = 0.
        self.static_penalty = None

    def set_mu(self, mu_vector= 0.4):
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
        self.grip_factor = mu_vector*10 
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

    def get_problem_matrices(self, sigmas):
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
        k_mat = ts.perturbation_to_curvature_matrix()
        a = ts.get_segment_curvatures()#[:,np.newaxis]
        k_mat[a<0,:] = -k_mat[a<0,:]

        c5, N, gr = self.c5, self.N, self.grip_factor

        H = np.zeros((N*2,N*2))
        F = np.zeros(N*2)
        for i in range(N+1):
            if i == 0:
                q0 = 1/self.initial_speed
                H[0,0] = -sigmas[0] * gr[0] * (1-c5)**2
                F[0] = -sigmas[0] * gr[0] * 2 * q0 * c5 * (1-c5)
            else:
                H[i-1:i,i-1:i] = -gr[i]*sigmas[i]*np.array(
                    [[c5**2, c5*(1-c5)],
                    [c5*(1-c5), (1-c5)**2]]
                )
            F[N:] = F[N:] + sigmas[i]*k_mat[i,:]

        return H, F

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
        penalty_mat = np.diag(k) - np.dot(qbar_mat2,np.diag(gr))


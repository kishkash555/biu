import numpy as np
import geometries as gm

class problem:
    def __init__(self, track_segments):
        self.track_segments = track_segments
        self.grip_factor = None
        self.grip_acc_factor = None
        self.top_speed = None
        self.initial_speed = None

    def set_grip_factor(self,grip_factor):
        self.grip_factor = grip_factor
        return self

    def set_grip_acc_factor(self, grip_acc_factor):
        self.grip_acc_factor = grip_acc_factor
        return self

    def set_top_speed(self, speed):
        self.top_speed = speed
        self.initial_speed = top_speed/2
        return self

    def get_hessian(self, pertrubations, speeds):
        if pertrubations.ndim==1:
            x = pertrubations[:,np.newaxis]
        elif pertrubations.shape[1]>1 and pertrubations.shape[0]==1:
            x = pertrubations.T
        elif np.all(pertrubations.shape >1):
            raise ValueError("pertrubations should be 1D")

        ts = self.track_segments
        mat = ts.pertrubations_to_curvature_matrix()
        delta_k = mat.dot(pertrubations)
        a = ts.get_segment_curvatures()[:,np.newaxis]
        k = a + delta_k

### Kalman Filters in python
# Source(s) :=
# | Understand & Code a Kalman Filter [Part 2, Python] 
#        (https://www.youtube.com/watch?v=m5Bw1m8jJuY)
# | Kalman Filtering – A Practical Implementation Guide (with code!) 
#        (https://www.robotsforroboticists.com/kalman-filtering/) 
# | A practical approach to Kalman filter and how to implement it:w
#        (http://blog.tkjelectronics.dk/2012/09/a-practical-approach-to-kalman-filter-and-how-to-implement-it/)
# | Object Tracking: Simple Implementation of Kalman Filter in Python
#        (https://machinelearningspace.com/object-tracking-python/)
# | (*) An Elementary Introduction to Kalman Filtering 
#        (https://arxiv.org/pdf/1710.04055)
# | (**) Kalman Filter - VISUALLY EXPLAINED
#        (https://youtu.be/-DiZGpAh7T4)
# Author := mason (orig. code @ https://github.com/cbecker/kalman_python_cpp)

import numpy as np

class KF:
    def __init__(self,  X: np.matrix,
                        H: np.matrix,
                        A: np.matrix,
                        B: np.matrix,
                        Q: np.matrix,
                        dt: float,
                        std_a: float,
                        std_z: float):

        # (N)umber of state variables
        self.N = len(X)

        #NOTE these matrices vary for each problem statement, and should be provided
        self.X = X # State matrix/array
        self.H = H # Transformation Matrix
        self.A = A # State transition matrix (state)
        self.B = B # State transition matrix (action)
        self.Q = Q # Process noise covariance matrix
        self.P = np.eye(self.N) # Covariance matrix
 
        # hyperparams 
        self.dt = dt
        self.std_a = std_a # system disturbance variance
        self.std_z = std_z # sensor measurement variance

    def predict(self, u: np.ndarray) -> None:
        """Predict next state given current state self.X and action u"""
        A, B, X, P, Q = self.A, self.B, self.X, self.P, self.Q

        new_X = A @ X + B @ u # x = A x + B u
        new_P = A @ P @ A.T + Q # P = A P A^T + Q

        self.X = new_X
        self.P = new_P

    def update(self, z: np.ndarray) -> None:
        """Update KF given measurement z"""
        H, P, X, I = self.H, self.P, self.X, np.eye(self.N)

        # Kalman Gain K 
        S = H @ P @ H.T + self.std_z # S = H P H^T + std_z
        K = P @ H.T @ np.linalg.inv(S) # K = P H^T S^-1

        # innovation
        innovation = z - H @ X

        # apply Kalman Gain and innovation to update prediction
        new_X = X + K @ innovation # x = x + K innovation
        new_P = (I - K @ H) @ P # P = (I - K H) P 

        self.X = new_X
        self.P = new_P

    @property
    def state(self) -> np.array:
        return self.X
 
    @property
    def cov(self) -> np.array: 
        return self.P

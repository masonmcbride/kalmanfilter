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
# Author := mason (orig. code @ https://github.com/cbecker/kalman_python_cpp)

import numpy as np

# offsets of each variable in state vector
iX = 0
iV = 1
NUMVARS = 2 * iV


class KF:
    def __init__(self,  initial_x: float, 
                        initial_v: float, 
                        dt: float,
                        std_a: float,
                        std_z: float) -> None:

        # state variable, contains [position, velocity]
        self.X = np.zeros((NUMVARS, 1))

        # initialize state
        self.X[iX] = initial_x
        self.X[iV] = initial_v

        # initialize H matrix, this varies for each problem statement
        self.H = np.zeros((1, NUMVARS))
        self.H[0, iX] = 1
        # this is [1 0] bc only position is observed

        # Initialie covariance matrix 
        self.P = np.eye(NUMVARS)
        
        # hyperparams 
        self.dt = dt
        self.std_a = std_a # system disturbance variance
        self.std_z = std_z # sensor variance


    # BEGIN main funtionality

    def predict(self) -> None:
        # x = Ax [+ Bu]  -- this is missing an action nterm
        A = np.eye(NUMVARS)
        A[iX, iV] = self.dt
        new_X = A.dot(self.X)

        # P = A P A.T + G G.T * std_a
        G = np.zeros((NUMVARS, 1))
        G[iX], G[iV] = 0.5 * self.dt**2, self.dt
        new_P = A.dot(self.P).dot(A.T) + G.dot(G.T) * self.std_a

        self.X = new_X
        self.P = new_P

    def update(self, z: np.ndarray) -> None:
        # S = H P Ht + std_z
        # K = P Ht S^-1

        # define S to be the inverse part of Kalman Gain expr
        S = self.H.dot(self.P).dot(self.H.T) + self.std_z
        # calculate Kalman Gain K 
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))

        # innovation = z - H x
        # x = x + K innovation
        # P = (I - K H) * P 
        
        # calculate innovation
        innovation = z - self.H.dot(self.X)

        # apply Kalman Gain and innovation to update prediction
        new_X = self.X + K.dot(innovation)
        new_P = (np.eye(NUMVARS) - K.dot(self.H)).dot(self.P)

        self.X = new_X
        self.P = new_P

    # END main functionality

    @property
    def state(self) -> np.array:
        return self.X
    
    @property
    def cov(self) -> np.array: 
        return self.P

    @property
    def pos(self) -> float:
        return self.X[iX]

    @property 
    def vel(self) -> float:
        return self.X[iV]
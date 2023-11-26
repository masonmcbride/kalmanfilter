import pytest
from kf import KF
import numpy as np

DT = 0.1
#TODO test 1-D KF and 2-D KF problem statements

class TestKF(unittest.TestCase):
    def test_can_construct_with_x_and_v(self):
        x = 0.2
        v = 2.3

        kf = KF(initial_x=x, initial_v=v, accel_variance=1.2)
        self.assertAlmostEqual(kf.pos, x)
        self.assertAlmostEqual(kf.vel, v)

    def test_after_calling_predict_comma_mean_and_cov_are_right_shape(self):
        x = 0.2
        v = 2.3

        kf = KF(initial_x=x, initial_v=v, accel_variance=1.2)
        kf.predict(dt=DT)

        self.assertAlmostEqual(kf.mean.shape, (2,))
        self.assertAlmostEqual(kf.cov.shape, (2,2))

    def test_after_calling_predict_increases_state_uncertainty(self):
        x = 0.2
        v = 2.3

        kf = KF(initial_x=x, initial_v=v, accel_variance=1.2)

        for _ in range(10):
            det_before = np.linalg.det(kf.cov)
            kf.predict(dt=DT)
            det_after = np.linalg.det(kf.cov)

            self.assertGreater(det_after, det_before)
        

    def test_state_vector_is_size_double_iV_index(self):
        # assert(iV*2 == len(kf.x))
        pass





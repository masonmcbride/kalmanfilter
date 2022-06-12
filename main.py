from kf import KF 
import numpy as np
import matplotlib.pyplot as plt


def main():
    DT = 0.1

    # establish ground truth function := f(t) = t
    # parabola with roots 0, 1
    t = np.arange(0, 100, DT)
    f_t = 0.1*(t**2 - t)

    # initialize kalman filter 
    kf = KF(initial_x=0, initial_v=0.1, dt=DT, std_a=1.3, std_z=25)

    predictions = []
    measurements = np.array([x + np.random.normal(0, kf.std_z) for x in f_t])
    for z_k in measurements:

        kf.predict()
        predictions.append(kf.X) 
        kf.update(z_k)

    # turn into np.ndarray so I can apply H to it and get pos
    predictions = np.array(predictions)
    predicted_positions = kf.H.dot(predictions).flatten()

    # plot that shit
    fig = plt.figure()
    fig.suptitle('Example of Kalman filter for tracking a moving object in 1-D', fontsize=20)
    plt.plot(t, measurements, label='Measurements', color='b',linewidth=0.5)
    plt.plot(t, f_t, label='Ground Truth Function', color='y', linewidth=1.5)
    plt.plot(t, predicted_positions, label='Kalman Filter Prediction', color='r', linewidth=1.5)
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Position (m)', fontsize=20)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

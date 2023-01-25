import math
import matplotlib.pyplot as plt
import numpy as np
from tyai_mcl import TyaiMCL


def calc_dummy_likelihood(mcl, map_x, map_y):
    likelihood = np.zeros(mcl.nump)
    for i in range(mcl.nump):
        p = mcl.get_particle(i)
        diff_x = p[0] - map_x
        diff_y = p[1] - map_y
        likelihood[i] = np.exp(-(diff_x**2 + diff_y**2)/10)
    return likelihood


if __name__ == '__main__':
    mcl = TyaiMCL(1000)
    #mcl.set_pos_gaussian(-2, 0, -1, 1, 1, 1)
    mcl.set_pos_uniform(-9, 9, -6, 6, -math.pi, math.pi)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cnt = 1
    while True:
        mcl.predict_motion(0.1, 0, 0, std_x=0.01, std_y=0.0, std_th=0.05)

        if cnt % 10 == 0:
            likelihood = calc_dummy_likelihood(mcl, cnt * 0.1, 1)
            mcl.observation_update(likelihood)
        cnt += 1

        mcl.plot_particles(ax)
        plt.draw()
        plt.pause(0.1)
        ax.clear()

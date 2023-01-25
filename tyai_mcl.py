# TyaiMCL
#
# Copyright (C) 2009-2023 Kiyoshi Irie
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math


class TyaiMCL:
    def __init__(self, num_particles=100):
        self.p = np.zeros((num_particles, 4), dtype=np.float64) # x, y, th, weight
        self.init_weights()

    def init_weights(self):
        self.p[:, 3] = 1.0

    def get_pos(self):
        # calc mean orientation
        dx = 0
        dy = 0
        for i in range(self.nump):
            dx += math.cos(self.p[i, 2])
            dy += math.sin(self.p[i, 2])
        th = math.atan2(dy, dx)
        return [np.mean(self.p[:, 0]), np.mean(self.p[:, 1]), th]

    def get_particle(self, i):
        return self.p[i]

    @property
    def nump(self):
        return self.p.shape[0]

    def set_pos_gaussian(self, x, y, th, std_x, std_y, std_th):
        self.p[:, 0] = np.random.normal(loc=x,  scale=std_x,  size=self.nump)
        self.p[:, 1] = np.random.normal(loc=y,  scale=std_y,  size=self.nump)
        self.p[:, 2] = np.random.normal(loc=th, scale=std_th, size=self.nump)

    def set_pos_uniform(self, minx, maxx, miny, maxy, minth, maxth):
        self.p[:, 0] = np.random.uniform(low=minx, high=maxx, size=self.nump)
        self.p[:, 1] = np.random.uniform(low=miny, high=maxy, size=self.nump)
        self.p[:, 2] = np.random.uniform(low=minth, high=maxth, size=self.nump)

    def move_particle(self, i, dx, dy, dth):
        self.p[i, 0] += dx * math.cos(self.p[i, 2]) - dy * math.sin(self.p[i, 2])
        self.p[i, 1] += dx * math.sin(self.p[i, 2]) + dy * math.cos(self.p[i, 2])
        self.p[i, 2] += dth

    def predict_motion(self, dx, dy, dth, std_x, std_y, std_th):
        xrand = np.random.normal(scale=std_x, size=self.nump)
        yrand = np.random.normal(scale=std_y, size=self.nump)
        trand = np.random.normal(scale=std_th, size=self.nump)
        for i in range(self.nump):
            self.move_particle(i, dx + xrand[i], dy + yrand[i], dth + trand[i])
        self.p[:, 2] = np.unwrap(self.p[:, 2])

    def resample(self):
        new_p = np.zeros(self.p.shape, dtype=np.float64)
        self.normalize_weight()

        r = random.random()/self.nump
        c = self.p[0, 3]
        i = 1
        for m in range(self.nump):
            u = r + m/self.nump
            while u > c and i < self.nump:
                i += 1
                c += self.p[i-1, 3]
            new_p[m, :] = self.p[i-1, :]

        self.p = new_p
        self.init_weights()

    def normalize_weight(self):
        sum_w = sum(self.p[:, 3])
        self.p[:, 3] /= sum_w

    def plot_particles(self, ax):
        line_len = 0.4

        # show particles
        ax.axis('equal')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-7, 7)
        ax.scatter(self.p[:, 0], self.p[:, 1], facecolors='none', edgecolors='black')
        line_ends_x = self.p[:, 0] + line_len * np.cos(self.p[:, 2])
        line_ends_y = self.p[:, 1] + line_len * np.sin(self.p[:, 2])
        for i in range(self.nump):
            ax.plot([self.p[i, 0], line_ends_x[i]], [self.p[i, 1], line_ends_y[i]], color='black')

        # show gravity
        g = self.get_pos()
        ax.plot(g[0], g[1], marker="o", color='red')
        xe = g[0] + line_len * math.cos(g[2])
        ye = g[1] + line_len * math.sin(g[2])
        ax.plot([g[0], xe], [g[1], ye], color='red')

    def observation_update(self, likelihood_list):
        self.p[:, 3] *= likelihood_list
        self.resample()



import time
from threading import Thread

import numpy as np

import gym
from gym import spaces, logger

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use(['fivethirtyeight', 'seaborn-whitegrid', 'seaborn-ticks'])
from matplotlib import rcParams
from matplotlib import gridspec
import matplotlib.ticker as plticker

rcParams['axes.facecolor'] = 'FFFFFF'
rcParams['savefig.facecolor'] = 'FFFFFF'
rcParams['figure.facecolor'] = 'FFFFFF'
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

rcParams.update({'figure.autolayout': True})

class KerbalLanderSimpleEnvironment(gym.Env):

    def __init__(self):

        super(KerbalLanderSimpleEnvironment, self).__init__()

        self.thrust = 60000.
        self.vesselMass = 2355.

        self.fuelBurnRate = 17.70

        self.vel = []
        self.alt = []
        self.acc = []
        self.throt = []

        self.totalRewards = []

        self.reward_range = (-200, 1000)

        # Observations are:
        # Current throttle (float)
        # Altitude above surface (float)
        # Velocity vector (3 floats)

        self.maxVelocity = 1E3
        self.maxAltitude = 3.5E4

        self.lowObs = np.array([
            0., # Acc,
            -10., # Altitude,
            -self.maxVelocity # vz
        ])

        self.highObs = np.array([
            self.thrust / self.vesselMass, # Acc,
            self.maxAltitude, # Altitude,
            self.maxVelocity # vz
        ])

        self.observation_space = spaces.Box(self.lowObs, self.highObs, dtype = np.float32)

        self.lowAct = np.array([
            0., # throttle
        ])

        self.highAct = np.array([
            1., # throttle
        ])

        self.action_space = spaces.Box(self.lowAct, self.highAct, dtype = np.float32)

        self.init()

    @property
    def mass(self):
        return self.vesselMass + self.fuelMass

    def init(self):

        self.stepCounter = 0

        self.altitude = np.random.uniform(0, 20000)
        self.velocity = np.random.uniform(-200, 0)
        self.throttle = 0
        self.acceleration = self.g(self.altitude)

        self.fuelMass = 2041.

        self.episodeReward = 0

    def g(self, altitude):

        m = 9.7599066E20
        r = 200000
        G = 6.67430E-11

        return -G * m  / (r + altitude) ** 2

    def thrustAcc(self, throttle):

        maxAcc = self.thrust / self.mass

        return throttle * maxAcc

    def exploded(self):

        if self.altitude < 1.0 and self.velocity < -20:
            return True
        else:
            return False

    def terminate(self):

        termDown = self.altitude < 1.0
        termHigh = self.altitude > self.maxAltitude
        termFuel = self.fuelMass < 1E-4

        self.terminated = termDown or termHigh# or termFuel

        return self.terminated

    def reset(self):

        self.init()

        return self._nextObservation()

    def mapRange(self, inLow, inHigh, outLow, outHigh, val):
        # From https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range

        return outLow + ((outHigh - outLow) / (inHigh - inLow)) * (val - inLow)

    def forward(self):

        dt = 0.1 # seconds

        itr = 1

        for i in range(itr):

            # g acts in -ve direction

            newAcceleration = self.thrustAcc(self.throttle) + self.g(self.altitude)

            newAltitude = self.altitude + self.velocity * dt + 0.5 * self.acceleration * dt * dt
            newVelocity = self.velocity + 0.5 * (self.acceleration + newAcceleration) * dt

            self.altitude = newAltitude
            self.velocity = newVelocity
            self.acceleration = newAcceleration

            self.fuelMass = self.fuelMass - dt * self.fuelBurnRate * self.throttle

    def _nextObservation(self):

        thrustAcc = self.throttle * (self.thrust / self.mass)

        obs = np.array([
            self.mapRange(self.lowObs[0], self.highObs[0], -1.0, 1.0, thrustAcc),
            self.mapRange(self.lowObs[1], self.highObs[1], -1.0, 1.0, self.altitude),
            self.mapRange(self.lowObs[2], self.highObs[2], -1.0, 1.0, self.velocity),
        ])

        # Sometimes (1, 3), sometimes (,3) - not sure why
        return obs.flatten()

    def _takeAction(self, action):

        # Output actions are sigmoid + OU noise, so clip then scale
        # Clipping should be okay, assuming that variance of OU noise is small compared to action range

        throttle = np.clip(action, 0, 1)

        self.throttle = throttle[0] # Single element vector

    def calculateReward(self):

        reward = 0

        if not self.exploded() and self.altitude < 1.0: # landed
            print('LANDED!')
            reward += 2.

        if self.altitude < 1.0:
            reward += 1 * np.exp(-0.01 * np.abs(self.velocity))
            print('Hit @', self.velocity, reward)

        if self.fuelMass < 1E-4:
            print('No fuel')
            reward -= 1.

        if self.altitude > self.maxAltitude:
            print('Max alt')
            reward -= 1.

        return np.array(reward)

    def makeEpisodePlot(self):

        fig, axs = plt.subplots(2, 2, figsize=(26, 26))
        plt.subplots_adjust(wspace = 0.25)

        axs[0][0].plot(self.alt, linewidth = 2.0)
        axs[0][0].set_xlabel('Steps', fontsize = 32)
        axs[0][0].set_ylabel('Altitude', fontsize = 32)
        axs[0][0].tick_params(labelsize = 24)

        axs[0][1].plot(self.vel, linewidth = 2.0)
        axs[0][1].set_xlabel('Steps', fontsize = 32)
        axs[0][1].set_ylabel('Velocity', fontsize = 32)
        axs[0][1].tick_params(labelsize = 24)

        axs[1][0].plot(self.acc, linewidth = 2.0)
        axs[1][0].set_xlabel('Steps', fontsize = 32)
        axs[1][0].set_ylabel('Acceleration', fontsize = 32)
        axs[1][0].tick_params(labelsize = 24)

        axs[1][1].plot(self.throt, linewidth = 2.0)
        axs[1][1].set_xlabel('Steps', fontsize = 32)
        axs[1][1].set_ylabel('Throttle', fontsize = 32)
        axs[1][1].tick_params(labelsize = 24)

        plt.savefig('episode.pdf')
        plt.clf()

    def makeRewardPlot(self):

        plt.plot(self.totalRewards, lw = 0.25, alpha = 1.0)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig('rewards.pdf')
        plt.clf()

    def step(self, action):

        self.vel.append( self.velocity )
        self.alt.append( self.altitude )
        self.acc.append( self.acceleration )
        self.throt.append( action[0] )

        self.stepCounter += 1

        self._takeAction(action)

        self.forward()

        done = self.terminate()
        obs = self._nextObservation()
        reward = self.calculateReward()

        self.episodeReward += reward

        # print('Action', action)
        # print('Obs:', obs)
        # print('Reward:', reward)
        # print('Done:', done)

        if done:
            self.totalRewards.append( self.episodeReward )

            # self.makeEpisodePlot()

            if len(self.totalRewards) > 0 and len(self.totalRewards) % 1000 == 0:
                self.makeRewardPlot()

        return obs, reward, done, {}

if __name__ == '__main__':

    lander = KerbalLanderSimpleEnvironment()

    action = [0.0]

    alt = []
    vel = []
    acc = []

    for i in range(10000):
        obs, reward, done, _ = lander.step(action)

        if done:
            lander.reset()
            break

        alt.append( lander.altitude )
        vel.append( lander.velocity )
        acc.append( lander.acceleration )

    plt.plot(alt, linewidth = 1.0)
    plt.xlabel('steps')
    plt.ylabel('Altitude')
    plt.savefig('alt.pdf')
    plt.clf()

    plt.plot(vel, linewidth = 1.0)
    plt.xlabel('steps')
    plt.ylabel('Velocity')
    plt.savefig('vel.pdf')
    plt.clf()

    plt.plot(acc, linewidth = 1.0)
    plt.xlabel('steps')
    plt.ylabel('Acceleration')
    plt.savefig('acc.pdf')
    plt.clf()

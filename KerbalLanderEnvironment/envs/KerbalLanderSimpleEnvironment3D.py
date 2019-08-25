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

class KerbalLanderSimpleEnvironment3D(gym.Env):

    def __init__(self):

        super(KerbalLanderSimpleEnvironment3D, self).__init__()

        self.thrust = 60000.
        self.vesselMass = 2355.
        self.fuelMass = 2041.

        self.fuelBurnRate = 17.70

        self.vel = []
        self.alt = []
        self.acc = []
        self.throt = []

        self.totalRewards = []

        self.reward_range = (-2, 3)

        # Observations are:
        # Current throttle (float)
        # Altitude above surface (float)
        # Velocity vector (3 floats)

        self.maxVelocity = 1.5E3
        self.maxAltitude = 1E5 # training
        # self.maxAltitude = 1E5 # Testing

        self.lowObs = np.array([
            0., # FuelMass
            -10., # Altitude,
            -self.maxVelocity, # vx
            -self.maxVelocity, # vy
            -self.maxVelocity, # vz
        ])

        self.highObs = np.array([
            self.fuelMass, # FuelMass
            self.maxAltitude, # Altitude,
            self.maxVelocity, # vx
            self.maxVelocity, # vy
            self.maxVelocity, # vz
        ])

        self.observation_space = spaces.Box(self.lowObs, self.highObs, dtype = np.float32)

        # Pitch:
        # The pitch of the vessel relative to the horizon, in degrees. A value between -90 and +90.

        # Heading:
        #The heading of the vessel (its angle relative to north), in degrees. A value between 0 and 360.

        # For stable-baselines

        self.lowAct = np.array([
            -1.0, # throttle
            -1.0, # pitch
            -1.0, # heading
        ])

        self.highAct = np.array([
            1.0, # throttle
            1.0, # pitch
            1.0, # heading
        ])

        self.action_space = spaces.Box(self.lowAct, self.highAct, dtype = np.float32)

        self.init()

    @property
    def mass(self):
        return self.vesselMass + self.fuelMass

    def init(self):

        self.stepCounter = 0

        # Correlated random initial conditions?

        rnd = np.random.uniform(0, 1)

        self.altitude = rnd * 50000
        self.x = np.array([0, 0, self.altitude])

        vz = -1000 if np.random.uniform(0, 1) < 0.05 else - max(2 * rnd * 500, 50)
        vx = np.random.uniform(-50, 50)
        vy = np.random.uniform(-50, 50)

        self.velocity = np.array([vx, vy, vz])

        if np.random.uniform(0, 1) < 0.05:
            self.altitude = 20
            self.x = np.array([0, 0, self.altitude])

            self.velocity = np.array([0, 0, 0])

        # self.velocity = np.array([0, 0, 0])

        # self.altitude = 36000 # Test
        # self.velocity = -650 # Test

        self.throttle = 0.0

        self.acceleration = self.g(self.altitude) # -ve z direction

        self.fuelMass = 2041.

        self.episodeReward = 0

    def g(self, altitude):

        m = 9.7599066E20 # Mun mass
        r = 200000 # Mun radius
        G = 6.67430E-11 # Gravitational constant

        return np.array([0, 0, -G * m  / (r + altitude) ** 2])

    def thrustAcc(self, throttle):

        thrustX = np.cos(np.radians(self.pitch)) * np.sin(np.radians(self.heading))
        thrustY = np.cos(np.radians(self.pitch)) * np.cos(np.radians(self.heading))
        thrustZ = np.sin(np.radians(self.pitch))

        maxAcc = self.thrust / self.mass
        accMag = throttle * maxAcc
        accVec = np.array([thrustX, thrustY, thrustZ])
        acc = accMag * accVec

        return acc

    def exploded(self):

        # Might also want to restrict movement in x, y
        if self.altitude < 1.0 and (self.velocity[2] < -10 or np.sum(np.abs(self.velocity[:2])) > 10):
            return True
        else:
            return False

    def terminate(self):

        termDown = self.altitude < 1.0
        termHigh = self.altitude > self.maxAltitude
        termFuel = self.fuelMass < 1E-4

        self.terminated = termDown or termHigh or termFuel

        return self.terminated

    def reset(self):

        self.init()

        return self._nextObservation()

    def mapRange(self, inLow, inHigh, outLow, outHigh, val):
        # From https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range

        return outLow + ((outHigh - outLow) / (inHigh - inLow)) * (val - inLow)

    def forward(self):

        dt = 0.1 # seconds

        itr = 5 * 5

        for i in range(itr):

            # g acts in -ve z direction

            newAcceleration = self.thrustAcc(self.throttle) + self.g(self.altitude)

            newX = self.x + self.velocity * dt + 0.5 * self.acceleration * dt * dt
            newVelocity = self.velocity + 0.5 * (self.acceleration + newAcceleration) * dt

            self.x = newX
            self.altitude = self.x[2]
            self.velocity = newVelocity
            self.acceleration = newAcceleration

            self.fuelMass = self.fuelMass - dt * self.fuelBurnRate * self.throttle

    def _nextObservation(self):

        obs = np.array([
            self.mapRange(self.lowObs[0], self.highObs[0], -1.0, 1.0, self.fuelMass),
            self.mapRange(self.lowObs[1], self.highObs[1], -1.0, 1.0, self.altitude),
            self.mapRange(self.lowObs[2], self.highObs[2], -1.0, 1.0, self.velocity[0]),
            self.mapRange(self.lowObs[3], self.highObs[3], -1.0, 1.0, self.velocity[1]),
            self.mapRange(self.lowObs[4], self.highObs[4], -1.0, 1.0, self.velocity[2]),
        ])

        return obs.flatten()

    def _takeAction(self, action):

        # Output actions are sigmoid + OU noise, so clip then scale
        # Clipping should be okay, assuming that variance of OU noise is small compared to action range

        self.throttle = self.mapRange(self.lowAct[0], self.highAct[0], 0.0, 1.0, np.clip(action[0], -1, 1))
        self.pitch = self.mapRange(self.lowAct[1], self.highAct[1], -90., 90., np.clip(action[1], -1, 1))
        self.heading = self.mapRange(self.lowAct[2], self.highAct[2], 0., 360., np.clip(action[2], -1, 1))

    def calculateReward(self):

        reward = 0

        if not self.exploded() and self.altitude < 1.0: # landed
            print('LANDED!')
            reward += 2.

        if self.altitude < 1.0:
            reward += 1 * np.exp(-0.01 * np.abs(self.velocity[2]))
            print('Hit @', self.velocity[2], reward)

            reward += 1 * np.exp(-0.01 * np.sum(np.abs(self.velocity[:2])))
            print('Hit @', self.velocity[:2], reward)

            # New
            reward -= 0.1 * np.exp(-0.001 * np.abs(self.fuelMass))
            print('Fuel @', self.fuelMass, reward)

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

        # self.vel.append( self.velocity )
        # self.alt.append( self.altitude )
        # self.acc.append( self.acceleration )
        # self.throt.append( action[0] )

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

        self.stepCounter += 1

        return obs, reward, done, {}

if __name__ == '__main__':

    lander = KerbalLanderSimpleEnvironment3D()

    # action = [-1.0, 1.0, -1.0] # Nothing
    action = [0.0, 0.0, 0.0]

    throttle = lander.mapRange(lander.lowAct[0], lander.highAct[0], 0.0, 1.0, np.clip(action[0], -1, 1))
    pitch = lander.mapRange(lander.lowAct[1], lander.highAct[1], -90., 90., np.clip(action[1], -1, 1))
    heading = lander.mapRange(lander.lowAct[2], lander.highAct[2], 0., 360., np.clip(action[2], -1, 1))

    print(throttle, pitch, heading)

    x, y, z = [], [], []
    velX, velY, velZ = [], [], []
    accX, accY, accZ = [], [], []

    for i in range(10000):
        obs, reward, done, _ = lander.step(action)

        if done:
            lander.reset()
            break

        x.append( lander.x[0] )
        y.append( lander.x[1] )
        z.append( lander.x[2] )

        velX.append( lander.velocity[0] )
        velY.append( lander.velocity[1] )
        velZ.append( lander.velocity[2] )

        accX.append( lander.acceleration[0] )
        accY.append( lander.acceleration[1] )
        accZ.append( lander.acceleration[2] )

    plt.plot(x, linewidth = 1.0)
    plt.xlabel('steps')
    plt.ylabel('x')
    plt.savefig('x.pdf')
    plt.clf()

    plt.plot(y, linewidth = 1.0)
    plt.xlabel('steps')
    plt.ylabel('y')
    plt.savefig('y.pdf')
    plt.clf()

    plt.plot(z, linewidth = 1.0)
    plt.xlabel('steps')
    plt.ylabel('z')
    plt.savefig('z.pdf')
    plt.clf()

    plt.plot(velX, linewidth = 1.0)
    plt.xlabel('steps')
    plt.ylabel('vx')
    plt.savefig('vx.pdf')
    plt.clf()

    plt.plot(velY, linewidth = 1.0)
    plt.xlabel('steps')
    plt.ylabel('vy')
    plt.savefig('vy.pdf')
    plt.clf()

    plt.plot(velZ, linewidth = 1.0)
    plt.xlabel('steps')
    plt.ylabel('vz')
    plt.savefig('vz.pdf')
    plt.clf()

    plt.plot(accX, linewidth = 1.0)
    plt.xlabel('steps')
    plt.ylabel('ax')
    plt.savefig('ax.pdf')
    plt.clf()

    plt.plot(accY, linewidth = 1.0)
    plt.xlabel('steps')
    plt.ylabel('ay')
    plt.savefig('ay.pdf')
    plt.clf()

    plt.plot(accZ, linewidth = 1.0)
    plt.xlabel('steps')
    plt.ylabel('az')
    plt.savefig('az.pdf')
    plt.clf()

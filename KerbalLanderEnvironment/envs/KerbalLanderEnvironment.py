import time
from threading import Thread

import numpy as np

import gym
from gym import spaces, logger

import krpc

class KerbalLanderEnvironment(gym.Env):

    def __init__(self, serverAddress = '192.168.1.104', saveName = 'lander100k'):

        super(KerbalLanderEnvironment, self).__init__()

        self.serverAddress = serverAddress
        self.saveName = saveName

        self.conn = krpc.connect( name = 'LanderRL',
                                  address = serverAddress)

        self.loadSave(self.saveName)

        self.initKRPC(self.serverAddress)

        self.reward_range = (-100, 300)

        # Observations are:
        # Current throttle (float)
        # Altitude above surface (float)
        # Velocity vector (3 floats)

        # Could add position vector (3 floats)(?)

        self.lowObs = np.array([
            0., # Throttle,
            0., # Altitude,
            -np.finfo(np.float32).max, # vx,
            -np.finfo(np.float32).max, # vy,
            -np.finfo(np.float32).max # vz
        ])

        self.highObs = np.array([
            1., # Throttle,
            100000., # Altitude,
            np.finfo(np.float32).max, # vx,
            np.finfo(np.float32).max, # vy,
            np.finfo(np.float32).max # vz
        ])

        self.observation_space = spaces.Box(self.lowObs, self.highObs, dtype = np.float32)

        # Actions are:
        # Point in a heading at pitch (2 floats)
        # Set Thrust (float)

        self.lowAct = np.array([
            -90., # pitch
            0., # heading
            0., # throttle
        ])

        self.highAct = np.array([
            90., # pitch
            360., # heading
            1., # throttle
        ])

        self.action_space = spaces.Box(self.lowAct, self.highAct, dtype = np.float32)

    def loadSave(self, name):
        self.conn.space_center.load(name)

    def initKRPC(self, serverAddress):

        self.vessel = self.conn.space_center.active_vessel

        self.frame = self.vessel.orbit.body.reference_frame

        self.telemetry = self.vessel.flight(self.frame)
        self.control = self.vessel.control

        self.engine = self.vessel.parts.engines[0]
        self.thrust = self.engine.available_thrust

        self.autopilot = self.vessel.auto_pilot
        self.autopilot.reference_frame = self.frame

        self.propellant = self.engine.propellants[0]
        self.initFuel = self.propellant.current_amount
        self.currentFuel = self.initFuel
        self.hasFuel = self.engine.has_fuel

        # Let's be optimistic
        self.control.gear = True

    def terminate(self):

        # Between 3 and 1 -> on its side
        termAlt = self.telemetry.surface_altitude > 100000 or self.telemetry.surface_altitude < 2
        termFuel = not self.engine.has_fuel
        termLanded = self.vessel.situation == self.vessel.situation.landed

        # Also have a nSteps requirement in here?

        # print('Terminate:', termAlt, termFuel, termLanded, self.exploded())

        self.terminated = termAlt or termFuel or termLanded or self.exploded()

        return self.terminated

    def exploded(self):
        return self.telemetry.surface_altitude < 1.

    def totalVelocity(self):
        return np.sum(np.abs(self.telemetry.velocity))

    def reset(self):

        self.loadSave(self.saveName)
        self.initKRPC(self.serverAddress)

        self.stepCounter = 0
        self.terminated = False

        return self._nextObservation()

    def _nextObservation(self):

        velocity = self.telemetry.velocity

        obs = np.array([
            self.control.throttle,
            self.telemetry.surface_altitude,
            velocity[0],
            velocity[1],
            velocity[2],
        ])

        return obs

    def mapRange(self, inLow, inHigh, outLow, outHigh, val):
        # From https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range

        return outLow + ((outHigh - outLow) / (inHigh - inLow)) * (val - inLow)

    def scaleAction(self, action):
        # Inputs are in [0, 1], from sigmoid

        pitchIn, headingIn, throttleIn = action

        pitch = self.mapRange(0., 1., self.lowAct[0], self.highAct[0], pitchIn)
        heading = self.mapRange(0., 1., self.lowAct[1], self.highAct[1], headingIn)
        throttle = self.mapRange(0., 1., self.lowAct[2], self.highAct[2], throttleIn)

        return pitch, heading, throttle

    def _takeAction(self, action):

        # Output actions are sigmoid + OU noise, so clip then scale
        # Clipping should be okay, assuming that variance of OU noise is small compared to action range

        action = np.clip(action, 0, 1)

        pitch, heading, throttle = self.scaleAction(action)

        self.autopilot.target_pitch_and_heading(pitch.item(), heading.item())
        self.autopilot.engage()

        # Block whilst the vessel orients

        # Sometimes this enters an infinite loop if we're waiting whilst the vessel
        # has crashed, so just wait 10 seconds instead and hope everything is okay

        # self.autopilot.wait()

        print('ap wait')
        # thread = Thread(target = lambda : self.vessel.auto_pilot.wait())
        # thread.start()
        # thread.join(timeout = 5.0)

        time.sleep(10)

        print(pitch, heading, throttle)
        self.control.throttle = throttle.item()

    def calculateReward(self):

        reward = 0

        # Could require some experimentation
        # For now:

        # Give 1 unit per 1000 metres altitude descended past 100k

        reward += (1E5 - self.telemetry.surface_altitude) / 1E3

        # Give -100 units if we explode

        if self.exploded() : reward -= 100

        # Give 100 units for touching down safely

        touchedDown = False

        if not self.exploded() and self.totalVelocity() < 1:
            reward += 100
            touchedDown = True

        # Give 100 * fraction of fuel left units if touched down safely

        if touchedDown:
            frac = self.propellant.current_amount / self.initFuel
            reward += 100 * frac

        # Give -100 units for running out of fuel

        if self.terminated and self.propellant.current_amount < 1E-5:
            reward -= 100

        # Could give something based on velocity close to the surface?

        return reward

    def step(self, action):

        self.stepCounter += 1

        try:

            print('Taking action')
            self._takeAction(action)

            # Wait a second to see what happens
            time.sleep(1)

            print('Taking obs')
            obs = self._nextObservation()
            reward = self.calculateReward()
            done = self.terminate()

            self.obs = obs
            self.reward = reward

        except:
            print('Obs excpt')
            # Return the previous ones, and done

            obs = self.obs
            reward = self.reward
            done = True

        return obs, reward, done, {}

if __name__ == '__main__':

    import time

    lander = KerbalLanderEnvironment(saveName = 'lander100k')

    # print(lander.step([0, 0, 1]))

    while 1:
        print(lander.terminate())
        print(lander.exploded())
        time.sleep(0.5)

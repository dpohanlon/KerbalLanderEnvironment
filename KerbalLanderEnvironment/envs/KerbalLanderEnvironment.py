import time
from threading import Thread

import numpy as np

import gym
from gym import spaces, logger

import krpc

class KerbalLanderEnvironment(gym.Env):

    # def __init__(self, serverAddress = '192.168.1.104', saveName = 'lander100k'):
    # def __init__(self, serverAddress = '192.168.1.104', saveName = 'lander 38 fixed'):
    # def __init__(self, serverAddress = '176.206.35.46', saveName = 'lander100k'):
    def __init__(self, serverAddress = '176.206.35.46', saveName = 'lander 38 fixed'):

        super(KerbalLanderEnvironment, self).__init__()

        self.serverAddress = serverAddress
        self.saveName = saveName

        self.conn = krpc.connect( name = 'LanderRL',
                                  address = serverAddress)

        self.loadSave(self.saveName)

        self.initKRPC(self.serverAddress)

        self.reward_range = (-200, 1000)

        # Observations are:
        # Current throttle (float)
        # Altitude above surface (float)
        # Velocity vector (3 floats)

        # Could add position vector (3 floats)(?)

        maxVelocity = 1E3

        self.lowObs = np.array([
            0., # Throttle,
            0., # Altitude,
            # -maxVelocity, # vx,
            # -maxVelocity, # vy,
            -maxVelocity # vz
        ])

        self.highObs = np.array([
            1., # Throttle,
            50000., # Altitude,
            # maxVelocity, # vx,
            # maxVelocity, # vy,
            maxVelocity # vz
        ])

        self.observation_space = spaces.Box(self.lowObs, self.highObs, dtype = np.float32)

        # Actions are:
        # Point in a heading at pitch (2 floats)
        # Set Thrust (float)

        self.lowAct = np.array([
            # -90., # pitch
            # 0., # heading
            0., # throttle
        ])

        self.highAct = np.array([
            # 90., # pitch
            # 360., # heading
            1., # throttle
        ])

        self.action_space = spaces.Box(self.lowAct, self.highAct, dtype = np.float32)

    def loadSave(self, name):
        self.conn.space_center.load(name)

    def initKRPC(self, serverAddress):

        self.vessel = self.conn.space_center.active_vessel

        self.engine = self.vessel.parts.engines[0]

        self.propellant = self.engine.propellants[0]
        self.initFuel = self.propellant.current_amount
        self.currentFuel = self.initFuel
        self.hasFuel = self.engine.has_fuel

        self.vessel.control.speed_mode = self.vessel.control.speed_mode.surface

        self.bodyFrame = self.vessel.orbit.body.reference_frame
        self.frame = self.vessel.surface_velocity_reference_frame

        self.telemetry = self.vessel.flight(self.bodyFrame)
        self.control = self.vessel.control

        self.thrust = self.engine.available_thrust

        self.autopilot = self.vessel.auto_pilot
        self.autopilot.reference_frame = self.frame

        # Let's be optimistic
        self.control.gear = True

        # Simple mode
        self.autopilot.disengage()
        self.autopilot.sas = True
        self.autopilot.sas_mode = self.conn.space_center.SASMode.retrograde

        time.sleep(5)

    def terminate(self):

        # Between 3 and 1 -> on its side
        termAlt = self.telemetry.surface_altitude > 40000 or self.telemetry.surface_altitude < 2
        termFuel = not self.engine.has_fuel
        termLanded = self.landed()
        # termUp = self.telemetry.velocity[0] > 500
        termUp = self.telemetry.velocity[0] > 0 and self.telemetry.surface_altitude > 100

        # Also have a nSteps requirement in here?

        # print('Terminate:', termAlt, termFuel, termLanded, self.exploded())

        self.terminated = termAlt or termFuel or termLanded or self.exploded() or termUp

        return self.terminated

    def exploded(self):
        # return self.telemetry.surface_altitude < 1. or len(self.vessel.parts.engines) == 0
        return len(self.vessel.parts.engines) == 0

    def landed(self):
        return self.vessel.situation == self.vessel.situation.landed

    def totalVelocity(self):
        return np.sum(np.abs(self.telemetry.velocity))

    def reset(self):

        self.loadSave(self.saveName)

        time.sleep(5)

        self.initKRPC(self.serverAddress)

        self.stepCounter = 0
        self.terminated = False

        return self._nextObservation()

    def _nextObservation(self):

        velocity = self.telemetry.velocity

        # obs = np.array([
        #     self.mapRange(self.lowObs[0], self.highObs[0], -1.0, 1.0, self.control.throttle),
        #     self.mapRange(self.lowObs[1], self.highObs[1], -1.0, 1.0, self.telemetry.surface_altitude),
        #     self.mapRange(self.lowObs[2], self.highObs[2], -1.0, 1.0, velocity[0]),
        #     self.mapRange(self.lowObs[3], self.highObs[3], -1.0, 1.0, velocity[1]),
        #     self.mapRange(self.lowObs[4], self.highObs[4], -1.0, 1.0, velocity[2]),
        # ])

        obs = np.array([
            self.mapRange(self.lowObs[0], self.highObs[0], 0, 1.0, self.control.throttle),
            self.mapRange(self.lowObs[1], self.highObs[1], -1.0, 1.0, self.telemetry.surface_altitude),
            self.mapRange(self.lowObs[2], self.highObs[2], -1.0, 1.0, self.telemetry.velocity[0]),
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

        throttle = np.clip(action, 0, 1)

        # pitch, heading, throttle = self.scaleAction(action)

        throttle  = self.mapRange(0., 1., self.lowAct[0], self.highAct[0], throttle)

        # not in simple mode
        # self.autopilot.target_pitch_and_heading(pitch.item(), heading.item())
        # self.autopilot.engage()

        # Block whilst the vessel orients

        # Sometimes this enters an infinite loop if we're waiting whilst the vessel
        # has crashed, so just wait 10 seconds instead and hope everything is okay

        # self.autopilot.wait()

        # thread = Thread(target = lambda : self.vessel.auto_pilot.wait())
        # thread.start()
        # thread.join(timeout = 5.0)

        # time.sleep(1)

        self.control.throttle = throttle.item()

    def calculateReward(self):

        exploded = self.exploded()

        reward = 0

        # Could require some experimentation
        # For now:
        # Give 1 unit per 1000 metres altitude descended past 100k

        if not exploded:

            reward += (1E5 - self.telemetry.surface_altitude) / 1E3

        # Give -100 units if we explode
        # Now less than reward for going slow
        # if self.exploded():
            # reward -= 10

        # Give 100 units for touching down safely

        touchedDown = False

        if not exploded and (self.totalVelocity() < 1 and self.telemetry.surface_altitude < 3) or self.landed():
            reward += 1000
            touchedDown = True

        # Give 100 * fraction of fuel left units if touched down safely

        if touchedDown and not exploded:
            frac = 0

            try:
                frac = self.propellant.current_amount / self.initFuel
            except:
                print('Could not get propellant amount')
                frac = self.stepCounter / 200.

            reward += 1000 * frac

        # Give -100 units for running out of fuel

        # if self.terminated and not self.exploded() and len(self.vessel.parts.engines) > 0 and self.propellant.current_amount < 1E-5:
            # reward -= 100

        # # Give some reward if we are slow when hitting the ground
        #
        # if self.exploded() or touchedDown:
        #
        #     # If using this when not on ground, limit to NEGATIVE velocity
        #     reward += 1000. * np.exp(0.001 * self.telemetry.velocity[0])

        # Give some reward if we are slow (double sided)

        # if not exploded:
        # Reward even if exploded

        if self.telemetry.velocity[0] < 0:
            reward += 1000. * np.exp(-0.01 * np.abs(self.telemetry.velocity[0])) # was 500

        reward += 1000. * np.exp(-0.0001 * self.telemetry.surface_altitude)

        # Give -100 if we go up too fast

        # if self.telemetry.velocity[0] > 250: # 100 -> 200, so it has time to act
        if self.telemetry.velocity[0] > 0 and self.telemetry.surface_altitude > 100: # Let it bounce a bit
            reward -= 5000

        # Give -100 if we went up too far

        if self.telemetry.surface_altitude > 40000:
            reward -= 5000

        reward /= 1000.

        return reward

    def step(self, action):

        print('Action:', action)

        self.stepCounter += 1

        if self.stepCounter == 10:
            # This is zero at the start, not sure why
            self.initFuel = self.propellant.current_amount

        self._takeAction(action)

        # Wait a bit for our actions to take effect
        time.sleep(0.1) # Was 0.5

        done = self.terminate()
        obs = self._nextObservation()
        reward = self.calculateReward()

        self.obs = obs
        self.reward = reward

        print('Obs:', obs)
        print('Reward:', reward)
        print('Done:', done)

        return obs, reward, done, {}

if __name__ == '__main__':

    import time

    lander = KerbalLanderEnvironment(saveName = 'lander100k')

    # print(lander.step([0, 0, 1]))

    while 1:
        print(lander.terminate())
        print(lander.exploded())
        time.sleep(0.5)

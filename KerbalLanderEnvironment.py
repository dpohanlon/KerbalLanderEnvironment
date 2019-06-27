import numpy as np

import gym
from gym import spaces, logger

class KerbalLanderEnvironment(gym.Env):

    def loadSave(self, name):
        self.conn.space_center.load(name)

    def initKRPC(self, serverAddress):

        self.conn = krpc.connect( name = 'LanderRL',
                                  address = serverAddress)

        self.vessel = self.conn.space_center.active_vessel

        self.frame = self.vessel.orbit.body.reference_frame

        self.telemetry = self.vessel.flight(self.frame)
        self.control = self.vessel.control

        self.engine = self.vessel.parts.engines[0]
        self.thrust = self.engine.available_thrust

        self.autopilot = self.vessel.auto_pilot
        self.autopilot.reference_frame = self.frame

        self.propellant = self.engine.propellants[0]
        self.currentFuel = self.propellant.current_amount
        self.hasFuel = self.engine.has_fuel

    def terminate(self):

        termAlt = self.altitude > 100000 or self.altitude < 3
        termFuel = not self.hasFuel
        termLanded = self.vessel.situation.landed

        # Also have a nSteps requirement in here?

        return termAlt or termFuel or termLanded

    def __init__(self, serverAddress = '192.168.1.104', saveName = 'lander'):

        super(KerbalLanderEnvironment, self).__init__()

        self.serverAddress = serverAddress
        self.saveName = saveName

        self.reward_range = (0, 1)

        self.stepCounter = 0

        # Observations are:
        # Current throttle (float)
        # Altitude above surface (float)
        # Velocity vector (3 floats)

        # Could add position vector (3 floats)(?)

        lowObs = np.array([
            0, # Throttle,
            0, # Altitude,
            -np.finfo(np.float32).max, # vx,
            -np.finfo(np.float32).max, # vy,
            -np.finfo(np.float32).max # vz
        ])

        highObs = np.array([
            1, # Throttle,
            100000, # Altitude,
            np.finfo(np.float32).max, # vx,
            np.finfo(np.float32).max, # vy,
            np.finfo(np.float32).max # vz
        ])

        self.observation_space = spaces.Box(lowObs, highObs, dtype = np.float32)

        # Actions are:
        # Point in a heading at pitch (2 floats)
        # Set Thrust (float)

        lowAct = np.array([
            -90, # pitch
            0, # heading
            0, # throttle
        ])

        highAct = np.array([
            90, # pitch
            360, # heading
            1, # throttle
        ])

        self.action_space = spaces.Box(lowAct, highAct, dtype = np.float32)

        loadSave(self.saveName)

        initKRPC(self.serverAddress)

    def reset(self):

        loadSave(self.saveName)
        initKRPC(self.serverAddress)

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

    def _takeAction(self, action):

        pitch, heading, throttle = action

        self.autopilot.target_pitch_and_heading(pitch, heading)
        self.autopilot.engage()

        # Block whilst the vessel orients
        # self.autopilot.wait()

        self.control.throttle = throttle

    def calculateReward(self):
        pass

    def step(self, action):

        self.stepCounter += 1

        self._takeAction(action)

        obs = self._nextObservation()
        reward = calculateReward()
        done = self.terminate()

        return obs, reward, done, {}

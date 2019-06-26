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

    def terminate(self):

        termAlt = self.altitude > 100000 or self.altitude < 3
        termFuel = self.fuel < 1E-3
        termLanded = self.vessel.situation.landed

        return termAlt or termFuel or termLanded

    def __init__(self, serverAddress = '192.168.1.104', saveName = 'lander'):

        super(KerbalLanderEnvironment, self).__init__()

        self.serverAddress = serverAddress
        self.saveName = saveName

        self.reward_range = (0, 1)

        # Observations are:
        # Current thrust (float)
        # Altitude above surface (float)
        # Velocity vector (3 floats)

        # Could add position vector (3 floats)(?)

        lowObs = np.array([
            0, # Thrust,
            0, # Altitude,
            -np.finfo(np.float32).max, # v1,
            -np.finfo(np.float32).max, # v2,
            -np.finfo(np.float32).max # v3
        ])

        highObs = np.array([
            1, # Thrust,
            100000, # Altitude,
            np.finfo(np.float32).max, # v1,
            np.finfo(np.float32).max, # v2,
            np.finfo(np.float32).max # v3
        ])

        self.observation_space = spaces.Box(lowObs, highObs, dtype = np.float32)

        # Actions are:
        # Point in a heading at pitch (2 floats)
        # Set Thrust (float)

        lowAct = np.array([
            -90, # pitch
            0, # heading
            0, # thrust
        ])

        highAct = np.array([
            90, # pitch
            360, # heading
            1, # thrust
        ])

        self.action_space = spaces.Box(lowAct, highAct, dtype = np.float32)

        loadSave(self.saveName)

        initKRPC(self.serverAddress)

    def reset(self):
        pass

    def _next_observation(self):
        pass

    def step(self):
        pass

    def _take_action(self):
        pass

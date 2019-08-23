import numpy as np
import gym

import tensorflow as tf

from stable_baselines.td3.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG, PPO2, TD3

import KerbalLanderEnvironment

env = gym.make('KerbalLanderSimple3D-v0')
env = DummyVecEnv([lambda: env])

n_actions = env.action_space.shape[-1]

# Noise for actions in range [-1, 1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=np.array([0.25, 0.05, 0.05]))

model = TD3('MlpPolicy', env, verbose=1, action_noise=action_noise, learning_rate = 1E-3, learning_starts = 1000000,
            policy_kwargs = dict(act_fun=tf.nn.sigmoid, layers=[32, 32, 32]))
model.learn(total_timesteps=5000000)
model.save("sb_ksp3d")

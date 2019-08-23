import numpy as np
import gym

import tensorflow as tf

# from stable_baselines.common.policies import MlpPolicy
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG, PPO2, TD3

import KerbalLanderEnvironment

env = gym.make('KerbalLanderSimple-v0')
# env = gym.make('KerbalLander-v0')
env = DummyVecEnv([lambda: env])

n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.25) * np.ones(n_actions))

model = TD3('MlpPolicy', env, verbose=1, action_noise=action_noise, learning_rate = 1E-3, learning_starts = 100000,
            policy_kwargs = dict(act_fun=tf.nn.sigmoid, layers=[16, 16, 16]))
model.learn(total_timesteps=2000000)
model.save("sb_ksp2")

# model = TD3.load("sb_ksp")
#
# obs = env.reset()
# done = False
# while not done:
#     action, _states = model.predict(obs)
#     obs, rewards, done, info = env.step(action)

import numpy as np
import gym

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess, GaussianWhiteNoiseProcess

import KerbalLanderEnvironment

ENV_NAME = ('KerbalLander-v0')

env = gym.make(ENV_NAME)
np.random.seed(42)

assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(16))
actor.add(keras.layers.LeakyReLU(alpha=0.3)) # Don't die
actor.add(Dense(16))
actor.add(keras.layers.LeakyReLU(alpha=0.3))
actor.add(Dense(16))
actor.add(keras.layers.LeakyReLU(alpha=0.3))
# actor.add(Dense(nb_actions, kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None)))
actor.add(Dense(nb_actions))
actor.add(Activation('sigmoid')) # Scale to action space
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
# x = Concatenate()([action_input, flattened_observation])
x = Dense(16)(flattened_observation)
# x = Dense(16)(x)
x = Activation('relu')(x)
x = Concatenate()([action_input, x])
x = Dense(16)(x)
x = Activation('relu')(x)
x = Dense(16)(x)
x = Activation('relu')(x)
# x = Dense(16)(x)
# x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

memory = SequentialMemory(limit=100000, window_length=1)
# Mean-converging process, has a momentum effect to minimise difference between the next
# and previous actions (mean = 0 -> have no effect on average)

# random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=[0.25, 0.25, 0.25], mu=[0.1, 0.0, 0.1], sigma=0.25)

# random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=0.2, mu=0.0, sigma=0.2)
random_process = GaussianWhiteNoiseProcess(size=nb_actions, mu = 0.0, sigma = 0.20, sigma_min = 0.01, n_steps_annealing = 5000)

agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=200, nb_steps_warmup_actor=500,
                  random_process=random_process, gamma=.99, target_model_update=1E-3)

# random_process = GaussianWhiteNoiseProcess(size=nb_actions, mu = 0.0, sigma = 0.1, sigma_min = 0.01, n_steps_annealing = 5000)
#
# agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
#                   memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
#                   random_process=random_process, gamma=.99, target_model_update=0.1)

agent.compile(Adam(lr=0.001, clipnorm=1.))

# agent.load_weights('ddpg_{}_SimpleG_ExplodeOkay_weights.h5f'.format(ENV_NAME))

agent.fit(env, nb_steps=5000, visualize=False, verbose=1, nb_max_episode_steps = 250,  log_interval = 50)

agent.save_weights('ddpg_{}_SimpleG_ExplodeOkayFaster_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
# agent.test(env, nb_episodes=5, visualize=False, nb_max_episode_steps=200)

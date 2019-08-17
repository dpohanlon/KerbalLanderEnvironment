import numpy as np
import gym

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

import keras.backend as K

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess, GaussianWhiteNoiseProcess

import KerbalLanderEnvironment

nCPU = 1

conf = K.tf.ConfigProto(device_count={'CPU': nCPU},
                        intra_op_parallelism_threads=nCPU,
                        inter_op_parallelism_threads=nCPU)
K.set_session(K.tf.Session(config=conf))

ENV_NAME = ('KerbalLander-v0')

env = gym.make(ENV_NAME)
np.random.seed(42)

assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(16, activation = 'relu'))
# actor.add(Dense(16, activation = 'relu'))
actor.add(Dense(16, activation = 'relu'))
actor.add(Dense(16, activation = 'relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('sigmoid')) # Scale to action space
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1, ) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
# x = Concatenate()([action_input, flattened_observation])
x = Dense(16)(flattened_observation)
x = Activation('relu')(x)
x = Concatenate()([action_input, x])
x = Dense(16)(x)
x = Activation('relu')(x)
# x = Dense(16)(x)
# x = Activation('relu')(x)
x = Dense(16)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

memory = SequentialMemory(limit=100000, window_length=1)

agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=25000, nb_steps_warmup_actor=25000,
                  random_process=None, gamma=.99, target_model_update=1E-3)

agent.compile(Adam(lr=0.001, clipnorm=1.))

# agent.fit(env, nb_steps=1000000, visualize=False, verbose=1, nb_max_episode_steps = 10000,  log_interval = 10000,
          # action_repetition = 10)
# agent.save_weights('ddpg_{}_SimpleSimTestFixFuel.h5f'.format(ENV_NAME), overwrite=True)

agent.load_weights('ddpg_KerbalLanderSimple-v0_SimpleSimFuelReward.h5f')
agent.test(env, nb_episodes=1, visualize=False, nb_max_episode_steps=10000)

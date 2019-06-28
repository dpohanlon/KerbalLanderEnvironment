from gym.envs.registration import register
 
register(id='KerbalLander-v0', 
    entry_point='KerbalLanderEnvironment.envs:KerbalLanderEnvironment', 
)

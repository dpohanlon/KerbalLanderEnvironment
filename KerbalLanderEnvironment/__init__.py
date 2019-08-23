
from gym.envs.registration import register

register(id='KerbalLander-v0',
    entry_point='KerbalLanderEnvironment.envs:KerbalLanderEnvironment',
)

register(id='KerbalLanderSimple-v0',
    entry_point='KerbalLanderEnvironment.envs:KerbalLanderSimpleEnvironment',
)

register(id='KerbalLanderSimple3D-v0',
    entry_point='KerbalLanderEnvironment.envs:KerbalLanderSimpleEnvironment3D',
         )

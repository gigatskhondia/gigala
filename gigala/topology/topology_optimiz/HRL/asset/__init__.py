from asset.topology_optimization import CantileverEnv
from gym.envs.registration import register


register(
    id="T0-h-v1",
    entry_point="asset:CantileverEnv",
)
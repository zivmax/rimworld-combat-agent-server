from gymnasium.envs.registration import register
from env.rimworld import RimWorldEnv
from env.rimworld import EnvOptions
from env.game import GameOptions

register(
    id="gymnasium_env/RimWorld-v0",
    entry_point=RimWorldEnv,
)


rimworld_env = "gymnasium_env/RimWorld-v0"

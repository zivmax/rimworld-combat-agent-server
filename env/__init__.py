from gymnasium.envs.registration import register
from env.rimworld import RimWorldEnv

register(
    id="gymnasium_env/RimWorld-v0",
    entry_point=RimWorldEnv,
)


rimworld_env = "gymnasium_env/RimWorld-v0"

from gymnasium.envs.registration import register
from env.rimworld import RimWorldEnv, EnvOptions, register_keyboard_interrupt
from env.game import GameOptions

register(
    id="gymnasium_env/RimWorld-v0",
    entry_point=RimWorldEnv,
)


rimworld_env = "gymnasium_env/RimWorld-v0"

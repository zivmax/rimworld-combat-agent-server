# this page is explicitly used fr storing trainning hyperparams

EPSILON = {"START": 1, "FINAL": 0.01, "DECAY": 0.999}
HIDDEN_SIZE = 128
TARGET_UPDATE = 10

MEMORY_SIZE = 100000
DEVICE = "cuda"

BATCH_SIZE = 64

BATCH_X = 8
BATCH_Y = 8

GAMMA = 0.99

LEARNING_RATE = 0.001

N_EPISODES = 100

EPISOLD_SAVE_INTERVAL = 25

EPISOLD_LOG_INTERVAL = 25

RE_TRAIN = True

LOAD_PATH = "agents/dqn/model_pth/dqn_model_episode_1000.pth"

OPTIONS = {
    "interval": 3.0,
    "speed": 4,
    "action_range": 4,
    "is_remote": False,
    "rewarding": {
        "original": 0,
        "ally_down": -10,
        "enemy_down": 10,
        "ally_danger_ratio": 0.5,
        "enemy_danger_ratio": -0.5,
    },
}

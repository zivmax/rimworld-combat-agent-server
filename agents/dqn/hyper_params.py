# this page is explicitly used fr storing trainning hyperparams

EPSILON = {"START": 1.0, "FINAL": 0.2, "DECAY": 0.995}
HIDDEN_SIZE1 = 256
HIDDEN_SIZE2 = 128
TARGET_UPDATE = 10

MEMORY_SIZE = 30000
DEVICE = "cuda"

BATCH_SIZE = 64

BATCH_X = 8
BATCH_Y = 8

GAMMA = 0.94

LEARNING_RATE = 0.01

N_EPISODES = 100

EPISOLD_SAVE_INTERVAL = 25

EPISOLD_LOG_INTERVAL = 10

LOAD_TEST_EPISODES = 50

TRAINING = True

LOAD_PATH = "agents/dqn/models/2024-12-19_20:08:09/episode_25.pth"

OPTIONS = {
    "interval": 1.0,
    "speed": 1,
    "action_range": 4,
    "is_remote": False,
    "rewarding": {
        "original": 0,
        "ally_down": -8,
        "enemy_down": 15,
        "ally_danger": 0.2,
        "enemy_danger": 1.5,
    },
}

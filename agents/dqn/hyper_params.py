# this page is explicitly used fr storing trainning hyperparams

EPSILON = {"START": 1.0, "FINAL": 0.1, "DECAY": 0.995}
HIDDEN_SIZE1 = 256
HIDDEN_SIZE2 = 128
TARGET_UPDATE = 10

MEMORY_SIZE = 30000
DEVICE = "cuda"

BATCH_SIZE = 64

BATCH_X = 8
BATCH_Y = 8

GAMMA = 0.96

LEARNING_RATE = 0.03

N_EPISODES = 600

EPISOLD_SAVE_INTERVAL = 100

EPISOLD_LOG_INTERVAL = 10

LOAD_TEST_EPISODES = 50

TRAINING = True

LOAD_PATH = "agents/dqn/models/2024-12-19_22:58:36/episode_400.pth"
CONTINUE_TRAINING_PATH = "agents/dqn/models/2024-12-19_22:58:36/episode_400.pth"
CONTINUE_NUM = 400
OPTIONS = {
    "interval": 1.0,
    "speed": 3,
    "action_range": 3,
    "is_remote": False,
    "rewarding": {
        "original": 0,
        "win": 100,
        "lose": -100,
        "ally_defeated": -20,
        "enemy_defeated": 15,
        "ally_danger": -1.6,
        "enemy_danger": 1.2,
        "ally_cover": 0.8,
    },
}

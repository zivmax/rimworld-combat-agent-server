# this page is explicitly used fr storing trainning hyperparams

EPSILON = {"START": 1, "FINAL": 0.01, "DECAY": 0.995}
HIDDEN_SIZE1 = 256
HIDDEN_SIZE2 = 128
TARGET_UPDATE = 10

MEMORY_SIZE = 500000
DEVICE = "cuda"

BATCH_SIZE = 64

BATCH_X = 8 
BATCH_Y = 8 

GAMMA = 0.99

LEARNING_RATE = 0.01

N_EPISODES = 200

EPISOLD_SAVE_INTERVAL = 50

EPISOLD_LOG_INTERVAL = 10

RE_TRAIN = True

LOAD_PATH = "agents/dqn/model_pth/dqn_model_episode_1000.pth"

OPTIONS = {
    "interval": 2.0,
    "speed": 3,
    "action_range": 4,
    "is_remote": False,
    "rewarding": {
        "original": 0,
        "ally_down": -8,
        "enemy_down": 10,
        "ally_danger": -0.7,
        "enemy_danger": 0.9,
    },
}

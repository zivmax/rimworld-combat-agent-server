# this page is explicitly used fr storing trainning hyperparams

EPSILON = {"START": 1.0, "FINAL": 0.2, "DECAY": 0.995}
HIDDEN_SIZE1 = 256
HIDDEN_SIZE2 = 128
TARGET_UPDATE = 10

MEMORY_SIZE = 10000
DEVICE = "cuda"

BATCH_SIZE = 64

BATCH_X = 8 
BATCH_Y = 8 

GAMMA = 0.94

LEARNING_RATE = 0.01

N_EPISODES = 1000

EPISOLD_SAVE_INTERVAL = 100

EPISOLD_LOG_INTERVAL = 10

RE_TRAIN = True

LOAD_PATH = "agents/dqn/model_pth/dqn_model_episode_1000.pth"

OPTIONS = {
    "interval": 1.0,
    "speed": 3,
    "action_range": 4,
    "is_remote": False,
    "rewarding": {
        "original": 0,
        "ally_down": -8,
        "enemy_down": 15,
        "ally_danger": -1.2,
        "enemy_danger": 1.5,
    },
}

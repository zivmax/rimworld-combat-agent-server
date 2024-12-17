# this page is explicitly used fr storing trainning hyperparams

EPSILON = {
    "START": 1,
    "FINAL": 0.01,
    "DECAY": 0.999
}
HIDDEN_SIZE = 128
TARGET_UPDATE = 10

MEMORY_SIZE = 10000
DEVICE = "cuda"

BATCH_SIZE = 64

GAMMA = 0.99

LEARNING_RATE = 0.001

N_EPISODES = 200

EPISOLD_SAVE_INTERVAL = 50

EPISOLD_LOG_INTERVAL = 10

RE_TRAIN = True
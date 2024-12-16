"""
this page is speceficlly for storing hyperparameters and rewards used in trainning process
"""

REWARD = {
    "original": 0,
    "ally_down":-10,
    "enemy_down":10,
    "ally_danger_ratio":0.5,
    "enemy_danger_ratio":-0.5
}

ACTION_SPACE_RADIUS_FACTOR = 1.
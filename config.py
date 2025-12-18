"""
Configuration file for CTF game parameters
"""

# Window settings
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60

# Grid settings
GRID_ROWS = 12
GRID_COLS = 16
CELL_SIZE = 50

# Colors (RGB)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
LIGHT_BLUE = (173, 216, 230)
LIGHT_RED = (255, 182, 193)

# Game settings
PLAYER_SPEED = 200  # milliseconds per move for human
AI_SPEED = 200  # milliseconds per move for AI

# Q-Learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.2  # Exploration rate
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.05

# Rewards
REWARD_CAPTURE_FLAG = 100
REWARD_DEFEND_FLAG = 20
REWARD_GET_CAUGHT = -10
REWARD_STEP = -1
REWARD_MOVE_TOWARD_FLAG = 5
REWARD_MOVE_AWAY_FLAG = -2

# Game modes
MODE_HUMAN_VS_AI = 1
MODE_AI_VS_AI = 2

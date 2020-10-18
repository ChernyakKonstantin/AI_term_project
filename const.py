#Here I define all constants

IMAGE_PATH = "collision_screen.png"
APP_TITLE = "Game"
RESET_BTN = "r"
MESSAGE_REGION = (80, 130, 280, 230)
SEQ_LEN = 3
VIEWPORT_SIZE = (360,360)

RESIZE = (64,64)

INPUT_SHAPE = [SEQ_LEN, *RESIZE]

DEVICE = "cuda" 

REPLAY_SIZE = 3 * 10**5
BATCH_SIZE = 64

REPLAY_START_SIZE = 1000

MEAN_REWARD_BOUND = 200
GAMMA = 0.99

LEARNING_RATE = 0.001

SYNC_TARGET_FRAMES = 10**3

EPSILON_DECAY_LAST_FRAME = 10**4
EPSILON_START = 1.0
EPSILON_FINAL = 0.02
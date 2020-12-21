from environment import Environment
from net import make_model
from agent import Agent

# Define all constants
APP_TITLE = "Game"
SEQ_LEN = 3
VIEWPORT_SIZE = (360, 360)

env = Environment(APP_TITLE, SEQ_LEN, VIEWPORT_SIZE)

N_ACTIONS = env.get_actions_num()

model = make_model()
model.load_weights(f'{10000}_frame_tgt_model')

agent = Agent(env, None)

total_rewards = []
best_mean_reward = None
frame_idx = 0

done = False
losses = []

while not done:
    reward = agent.eval_step(model)


# cv2.destroyAllWindows()
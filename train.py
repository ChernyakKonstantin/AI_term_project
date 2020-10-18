import torch
from torch.optim import Adam
import numpy as np
import cv2

from environment import Environment
from net import Net
from agent import Agent
from buffer import ExperienceBuffer
from loss_func import loss_func


# Define all constants
APP_TITLE = "Game"
SEQ_LEN = 3
VIEWPORT_SIZE = (360, 360)

RESIZE = (64, 64)

INPUT_SHAPE = [SEQ_LEN, *RESIZE]

DEVICE = "cuda"

REPLAY_SIZE = 3 * 10**5
BATCH_SIZE = 128

REPLAY_START_SIZE = 3 * 1000

MEAN_REWARD_BOUND = 200
GAMMA = 0.99

LEARNING_RATE = 0.001

SYNC_TARGET_FRAMES = 3 * 10**3

EPSILON_DECAY_LAST_FRAME = 10**4
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

env = Environment(APP_TITLE, SEQ_LEN, VIEWPORT_SIZE)

N_ACTIONS = env.get_actions_num()

net = Net(INPUT_SHAPE, N_ACTIONS).to(DEVICE)
tgt_net = Net(INPUT_SHAPE, N_ACTIONS).to(DEVICE)

buffer = ExperienceBuffer(True, REPLAY_SIZE, SEQ_LEN, 64, 64) # Hardcoded 64x64

agent = Agent(env, buffer, DEVICE)

optimizer = Adam(net.parameters())

epsilon = EPSILON_START

total_rewards = []
best_mean_reward = None
frame_idx = 0

done = False
losses = []

while not done :
    frame_idx += 1

    epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

    reward = agent.train_step(net, epsilon)

    if reward is not None:
        total_rewards.append(reward)
        mean_reward = np.mean(total_rewards[-10:]) # Cредняя награда за послдение 10 игр

        print("Frame %d: done %d games, reward %.3f, eps %.2f" \
              % (frame_idx ,len(total_rewards), total_rewards[-1], epsilon))

        if best_mean_reward is None or best_mean_reward < mean_reward:
            if best_mean_reward is not None:
                print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                model_save_name = 'dqn' + str(len(total_rewards)) + '.pt'
                torch.save(tgt_net.state_dict(), model_save_name)
                best_mean_reward = mean_reward
        if mean_reward > MEAN_REWARD_BOUND:
            print("Solved in %d frames!" % frame_idx)
            break

    if buffer.cur_index < REPLAY_START_SIZE:
        continue

    if frame_idx % SYNC_TARGET_FRAMES == 0:
        tgt_net.load_state_dict(net.state_dict())
        torch.save(tgt_net.state_dict(), "%d_frame_tgt_net.pt" % frame_idx)

        print("SYNCHRONIZED")

    optimizer.zero_grad()
    batch = buffer.sample(BATCH_SIZE)
    loss = loss_func(batch, net, tgt_net, GAMMA, DEVICE)
    losses.append(loss.item())
    if frame_idx % 100 == 0: # %1000
        print("MEAN_LOSS_VALUE: " + str(np.mean(losses[-1000:])))
    loss.backward()
    optimizer.step()
    
cv2.destroyAllWindows()

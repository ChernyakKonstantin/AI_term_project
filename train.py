import numpy as np
import cv2

from environment import Environment
from net import make_model
from agent import Agent
from buffer import ExperienceBuffer

# Define all constants
APP_TITLE = "Game"
SEQ_LEN = 3
VIEWPORT_SIZE = (360, 360)

REPLAY_SIZE = 7 * 10**3
BATCH_SIZE = 100

REPLAY_START_SIZE = 1500
MEAN_REWARD_BOUND = 10**4
GAMMA = 0.99

SYNC_TARGET_FRAMES = 2500

EPSILON_DECAY_LAST_FRAME = 3000
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

env = Environment(APP_TITLE, SEQ_LEN, VIEWPORT_SIZE)

N_ACTIONS = env.get_actions_num()

model = make_model()
tgt_model = make_model()

buffer = ExperienceBuffer(True, REPLAY_SIZE, SEQ_LEN, 325, 325, 1)  # Hard-coded!

agent = Agent(env, buffer)

epsilon = EPSILON_START

total_rewards = []
best_mean_reward = None
frame_idx = 0

done = False

losses = []
saved_loss = np.memmap("losses.npy", dtype=np.float32, mode='w+', shape=(10**6))
saved_loss_id = 0

while not done:
    frame_idx += 1

    epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

    reward = agent.train_step(model, epsilon)

    if reward is not None:
        total_rewards.append(reward)
        mean_reward = np.mean(total_rewards[-10:])  # Cредняя награда за последние 10 игр

        print("Frame %d: done %d games, reward %.3f, eps %.2f" \
              % (frame_idx ,len(total_rewards), total_rewards[-1], epsilon))

        if best_mean_reward is None or best_mean_reward < mean_reward:
            if best_mean_reward is not None:
                print(f"Best mean reward updated {best_mean_reward} -> {mean_reward}, model saved")
                tgt_model.save_weights(f'dqn_total_best_reward_{len(total_rewards)}')  # Check if file name is correct!
                best_mean_reward = mean_reward
        if mean_reward > MEAN_REWARD_BOUND:
            print("Solved in {frame_idx} frames!")
            break

    if buffer.cur_index < REPLAY_START_SIZE:
        continue

    if frame_idx % SYNC_TARGET_FRAMES == 0:
        model.save_weights(f'{frame_idx}_frame_tgt_model')
        tgt_model.load_weights(f'{frame_idx}_frame_tgt_model')
        print("SYNCHRONIZED")

    states, next_states, actions, rewards, dones = buffer.sample(BATCH_SIZE)
    X_train = states
    next_state_values = tgt_model.predict(next_states).max(axis=-1) # Вернет значения лучших Q
    next_state_values[dones] = 0.0 # Для завершившихся эпизодов следующее состояние имеет значение 0
    y_train = next_state_values * GAMMA + rewards # Ожидаемое значение Q согласно уравнению Белмана
    history = model.fit(X_train, y_train, epochs=1, verbose=2)  # !
    loss = history.history["loss"][0]
    losses.append(loss)
    saved_loss[saved_loss_id] = loss
    saved_loss_id += 1
    if frame_idx % 500 == 0:
        print("MEAN_LOSS_VALUE: " + str(np.mean(losses[-500:])))

cv2.destroyAllWindows()

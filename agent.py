import random
from sklearn.preprocessing import MinMaxScaler

class Agent:
    def __init__(self, environment, experience_buffer):
        self.env = environment
        self.exp_buffer = experience_buffer
        self.total_reward = 0.0
        self.state = None
        self.tmp_memory = []
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def train_step(self, net, epsilon):
        """
        The method perform a step in environment.

        Parameters
        ----------
        net : instance of class Net. (WRONG SINCE UPD)
        epsilon : float

        Returns
        -------
        done_reward : float
            Reward, obrained through the episode.
        """
        action = self._select_action(net, epsilon)
        next_state, reward, is_done = self.env.step(action)
        self.total_reward += reward
        # self.exp_buffer.append(self.state, next_state, action, reward, is_done)
        self.tmp_memory.append([self.state, next_state, action, reward, is_done])
        self.state = next_state
        if is_done is True:
            # Scale rewards
            rewards = [[step[3]] for step in self.tmp_memory]
            scaled_rewards = MinMaxScaler().fit_transform(rewards)
            # Fill buffer
            for data, rew in zip(self.tmp_memory, scaled_rewards):
                self.exp_buffer.append(data[0], data[1], data[2], rew[0], data[4])
            self.tmp_memory = []  # Clean temporary memory
            done_reward = self.total_reward
            self._reset()
            return done_reward
        else:
            return None

    def _select_action(self, model, epsilon):
        """
        The method selects random action with probability of epsilon.
        Otherwise it selects action, maximazing Q value.
        Self.state is preprocessed before convertation to torch.tensor.

        Parameters
        ----------
        model : instance of class Net.
        epsilon : float

        Returns
        -------
        action : int
        """
        if random.random() < epsilon:
            action = self.env.sample_action()
        else:
            q_vals = model.predict(self.state.reshape([1, *self.state.shape]))
            action = q_vals.argmax(axis=-1)[0]
        return action

    def eval_step(self, model):
        a = 1
        q_vals = model.predict(self.state.reshape([1, *self.state.shape]))
        action = q_vals.argmax(axis=-1)[0]

        next_state, reward, is_done = self.env.step(action)
        self.state = next_state
        self.total_reward += reward
        if is_done is True:
            done_reward = self.total_reward
            self._reset()
            return done_reward
        else:
            return None

import numpy as np


class ExperienceBuffer:
    """
    This class provides an experience buffer.
    To store large data files numpy memmap is used.
    """

    def __init__(self, is_new, capacity, seq_len, height, width):
        """
        Parameters
        ----------
        is_new : bool
            Define if a new storage required.
        capacity : int
        seq_len : int
            Number of subsequent steps to capture.
        height : int
            Image height.
        width : int
            Image height.

        Returns
        -------
        None.

        """

        if is_new is True:
            mode = "w+"
            cur_index = np.memmap("cur_index.npy", dtype=np.int32, mode=mode, shape=(1,))  # Create file
            is_full = np.memmap("is_full.npy", dtype=np.bool, mode=mode, shape=(1,))  # Create file
            del cur_index  # Save file
            del is_full
            self.cur_index = np.memmap("cur_index.npy", dtype=np.int32, mode="r+", shape=(1,))[0]  # only integer values
            self.cur_index = 0
            self.is_full = np.memmap("is_full.npy", dtype=np.bool, mode="r+", shape=(1,))[0]
            self.is_full = False
        else:
            mode = "r+"
            self.cur_index = np.memmap("cur_index.npy", dtype=np.int32, mode="r+", shape=(1,))[0]
            self.is_full = np.memmap("is_full.npy", dtype=np.bool, mode="r+", shape=(1,))[0]

        self.states = np.memmap("states.npy", dtype=np.float32, mode=mode,
                                shape=(capacity, seq_len, height, width))  # grayscale viewport no preprocessing
        self.next_states = np.memmap("next_states.npy", dtype=np.float32, mode=mode,
                                     shape=(capacity, seq_len, height, width))  # grayscale viewport no preprocessing
        self.actions = np.memmap("actions.npy", dtype=np.uint8, mode=mode, shape=(capacity,))  # enough for actions
        self.rewards = np.memmap("rewards.npy", dtype=np.float32, mode=mode,
                                 shape=(capacity,))  # тут могут быть дробные значения
        self.is_dones = np.memmap("is_dones.npy", dtype=np.bool, mode=mode, shape=(capacity,))  # only bool values
        self.capacity = capacity
        self.seq_len = seq_len

    def append(self, state, next_state, action, reward, is_done):
        """


        Parameters
        ----------
        state : numpy array
            numpy array of the following shape [seq_len, height, width].
        next_state : numpy array
            numpy array of the following shape [seq_len, height, width].
        action : int
        reward : float32
        is_done : bool

        Returns
        -------
        None.

        """
        # state is list of [np.array shape of [seq_len, height, width], np.array shape of [seq_len, height, width]]
        # The same for next_state
        if self.cur_index == self.capacity:
            self.cur_index = 0
            self.is_full = True
        self.states[self.cur_index] = state
        self.next_states[self.cur_index] = next_state
        self.actions[self.cur_index] = action
        self.rewards[self.cur_index] = reward
        self.is_dones[self.cur_index] = is_done
        self.cur_index += 1

    def __len__(self):
        """
        Returns
        -------
        int
            Returns the buffer capacity.

        """
        return self.capacity

    def sample(self, batch_size):
        """


        Parameters
        ----------
        batch_size : int
            Number of elements to create a batch.

        Returns
        -------
        states : tuple
            numpy array of the following shape[batch_size, seq_len, height, width].
        next_states : tuple
            numpy array of the following shape[batch_size, seq_len, height, width].
        actions : numpy array of ints
        rewards : numpy array of floats
        is_dones : numpy array of bools
        """
        if self.is_full is True:
            max_index = self.capacity
        else:
            max_index = self.cur_index
        indexes = np.random.choice(np.arange(max_index), size=batch_size, replace=False)
        states = self.states[indexes]
        next_states = self.next_states[indexes]
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        is_dones = self.is_dones[indexes]
        return states, next_states, actions, rewards, is_dones

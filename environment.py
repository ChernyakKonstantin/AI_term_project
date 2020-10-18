# Imports required
import pyautogui
import win32gui
import numpy as np
import random
import time
from pynput.keyboard import Controller, Key
from PIL import ImageOps
import cv2


class Environment:
    """
       This class finds an environment window by th given name and emulates keyboard pressings
       to interact within the window.
       Methods, the agent interacts with, are 'step(action)' and 'reset()'
    """

    def __init__(self, app_title, seq_len, viewport_size):
        """
           app_title: string
           seq_len: int - a number of subsequent frames to capture.
           viewport_size: tuple - width and height of a viewport.
        """
        print("Click on the app window!")
        time.sleep(1)

        # Define stereo_vision components. I found the parameters experementally.
        self.left_matcher = cv2.StereoBM_create(numDisparities=32, blockSize=5)
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
        self.wls_filter.setLambda(8000)
        self.wls_filter.setSigmaColor(1.2)

        self.next_observation = None
        self.actions = (0, 1, 2, 3, 4, 5)
        self.rewards = {
            0: 1.0,
            1: -50,
            2: 1.0,
            3: 1.0,
            4: -50,
            5: -50,  # all -1.5 were -0.5 before
            "Penalty": -100
        }

        self.seq_len = seq_len
        self.l_viewport_region, self.mid_viewport_region, self.r_viewport_region = self._get_viewports_region(
            *viewport_size)

        self.controller = Controller()  # Keyboard emulator
        self.buttons = {
            0: [Key.up],
            1: [Key.down],
            2: [Key.up, Key.left],
            3: [Key.up, Key.right],
            4: [Key.down, Key.left],
            5: [Key.down, Key.right],
            "reset": 'r'
        }

        env_window_id = win32gui.FindWindow(None, app_title)
        self.screen_region = self._get_window_rect(env_window_id)

    # Private methods
    def _perform_action(self, action_name):
        """
        The method emulates pressing
        the given keyboard button.

        Parameters
        ----------
        action_name : int

        Returns
        -------
        None.

        """
        buttons = self.buttons[action_name]
        for btn in buttons:
            self.controller.press(btn)

    def _stop_action(self, action_name):
        """
        The method emulates releasing
        the given keyboard button.

        Parameters
        ----------
        action_name : int

        Returns
        -------
        None.

        """
        buttons = self.buttons[action_name]
        for btn in buttons:
            self.controller.release(btn)

    def _get_observation(self):
        """
        The method performs environment window image
        grabbing and splits it into three viewport images.

        Returns
        -------
        observation : numpy array of shape [self.seq_len, 64, 64]. It is depth maps

        """
        is_done = False  # Initially it is assumed, that there is no collision
        observation = []  # np.empty([self.seq_len, 64, 64], dtype=np.float32)  # Hard-Coded 64x64
        prev_frame = np.ones([360, 360], dtype=np.uint8)
        for seq_index in range(self.seq_len):
            frame = pyautogui.screenshot(region=self.screen_region)
            frame = ImageOps.grayscale(frame)
            frame = np.asarray(frame, dtype=np.uint8)
            frame = frame[26:-3, 3:-3]  # Cut off some shit around the frame I CHANGE IT LATER

            left_image = frame[:, :360]
            right_image = frame[:, 360:]

            left_disp = self.left_matcher.compute(left_image, right_image)
            right_disp = self.right_matcher.compute(right_image, left_image)

            filtered_disp = self.wls_filter.filter(left_disp, left_image, disparity_map_right=right_disp)
            filtered_disp = filtered_disp[33:-2, 33:-2]  # Crop no information area and make frame looks rectangle
                                                        # Я оставил один пиксель (было 34 на отрез), чтобы после нормализации  выглядело лучше
            norm_image = cv2.normalize(filtered_disp, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                       dtype=cv2.CV_32F)
            
            #TO SHOW DEPTH MAP
            cv2.imshow('Frame',norm_image)
            cv2.moveWindow('Frame',600,500)
            cv2.waitKey(25)
            
            
            norm_image = cv2.resize(norm_image, (64, 64))  # Hard-Coded 64x64

            observation.append(norm_image)

            if np.all(prev_frame == frame):
                is_done = True
            prev_frame = frame.copy()

            time.sleep(0.01)  # Без этого костыля не работает
        observation = np.array(observation, dtype=np.float32)
        return observation, is_done

    def _get_reward(self, action_name, is_done):
        if is_done == True:
            return self.rewards["Penalty"]
        else:
            return self.rewards[action_name]

    def _get_window_rect(self, window_id):
        """This function returns a position of the window left top corner,
           and the window width and height.
           Input:
               int - the window id.
           Output:
                tuple of int - (left, top, width, height)
        """
        left, top, right, bottom = win32gui.GetWindowRect(window_id)
        width = right - left
        height = bottom - top
        return left, top, width, height

    def _get_viewports_region(self, viewport_width, viewport_height):  # I am not sure)
        """
            This function returns left top and right bottom points
            for the app viewports.
            For each viewport it returns the following tuple: (left, top, right, bottom).
            Input:
                viewport_width: int,
                viewport_height: int.
            Output:
                tuple of l_rect, mid_rect, r_tect
        """
        # I measured the constants
        TITLE_BAR = 25
        PADDING = 1

        left_1 = PADDING
        top_1 = PADDING + TITLE_BAR
        right_1 = left_1 + viewport_width
        bottom_1 = top_1 + viewport_height
        l_rect = (left_1, top_1, right_1, bottom_1)

        left_2 = right_1 + 1
        top_2 = top_1
        right_2 = left_2 + viewport_width
        bottom_2 = bottom_1
        mid_rect = (left_2, top_2, right_2, bottom_2)

        left_3 = right_2 + 1
        top_3 = top_1
        right_3 = left_3 + viewport_width
        bottom_3 = bottom_1
        r_rect = (left_3, top_3, right_3, bottom_3)
        return l_rect, mid_rect, r_rect

    # Public methods

    def step(self, action_name):
        """
        The method performs the given action.

        Parameters
        ----------
        action_name : int

        Returns
        -------
        tuple
            tuple of next_observation, reward, is_done.
        """
        self._perform_action(action_name)
        self.next_observation, is_done = self._get_observation()
        self._stop_action(action_name)
        reward = self._get_reward(action_name, is_done)
        return self.next_observation, reward, is_done

    def reset(self):
        # Press restart
        self.controller.press(self.buttons['reset'])
        time.sleep(0.15)  # To emulate human pressing
        self.controller.release(self.buttons['reset'])

        self.next_observation, _ = self._get_observation()
        return self.next_observation

    def get_actions_num(self):
        """
        The method returns number of possible actions

        Returns
        -------
        int
        """
        return len(self.actions)

    def sample_action(self):
        """
        The function returns a random action from all possible actions.

        Returns
        -------
        int

        """
        return random.choice(self.actions)

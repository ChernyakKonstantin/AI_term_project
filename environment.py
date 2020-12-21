import pyautogui
import win32gui
import numpy as np
from time import sleep
import cv2
from PIL import ImageOps


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
        self.state_file = r'environment_app\status.dat'
        self.action_file = r'environment_app\actions.dat'
        # Clear file to avoid previous commands
        with open(self.action_file, 'w') as f:
            f.write('')
        self.action_duration = 1 / 10  # Seconds
        self.actions = ('forward', 'left', 'right')
        # Define stereo_vision components. I found the parameters experementally.
        self.left_matcher = cv2.StereoBM_create(numDisparities=32, blockSize=5)
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
        self.wls_filter.setLambda(8000)
        self.wls_filter.setSigmaColor(1.2)
        # Define environment initial state
        self.next_observation = None

        self.seq_len = seq_len

        (self.l_viewport_region,
         self.mid_viewport_region,
         self.r_viewport_region) = self._get_viewports_region(*viewport_size)
        env_window_id = win32gui.FindWindow(None, app_title)
        self.screen_region = self._get_window_rect(env_window_id)

    # Private methods

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

    def _start_action(self, action_name):
        while True:
            try:
                with open(self.action_file, 'w') as f:
                    f.write(action_name)
                    break
            except:
                continue

    def _stop_action(self):
        while True:
            try:
                with open(self.action_file, 'w') as f:
                    f.write('')
                    break
            except:
                continue

    def _perform_action(self, action_name):
        self._start_action(action_name)
        screenshots = self._get_screenshot()
        self._stop_action()
        observation = self._get_observation(screenshots)
        return observation

    def _get_screenshot(self):
        screenshots = []
        for seq_index in range(self.seq_len):
            screenshot = pyautogui.screenshot(region=self.screen_region)
            screenshots.append(screenshot)
        return screenshots

    def _get_observation(self, raw_screenshots):
        """
        The method performs environment window image
        grabbing and splits it into three viewport images.

        Returns
        -------
        observation : numpy array of shape [self.seq_len, 64, 64]. It is depth maps

        """
        observation = []
        for frame in raw_screenshots:
            frame = ImageOps.grayscale(frame)
            frame = np.asarray(frame, dtype=np.uint8)
            frame = frame[26:-3, 3:-3]  # Crop irrelevant part

            left_image = frame[:, :360]
            right_image = frame[:, 360:]

            left_disp = self.left_matcher.compute(left_image, right_image)
            right_disp = self.right_matcher.compute(right_image, left_image)

            filtered_disp = self.wls_filter.filter(left_disp, left_image, disparity_map_right=right_disp)
            filtered_disp = filtered_disp[33:-2, 33:-2]  # Crop no information area and make frame looks rectangle
                                                         # Я оставил один пиксель (было 34 на отрез),
                                                         # чтобы после нормализации  выглядело лучше
            norm_image = cv2.normalize(filtered_disp, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                       dtype=cv2.CV_32F)
            norm_image = norm_image.reshape([*norm_image.shape, 1])

            #TO SHOW DEPTH MAP
            cv2.imshow('Frame', norm_image)
            cv2.moveWindow('Frame', 600, 500)
            cv2.waitKey(25)

            observation.append(norm_image)
        observation = np.array(observation, dtype=np.float32)
        return observation

    def _get_reward(self, observation):
        """
        Я пробую стимулировать агента к минимизации близких расстояний (чем ближе, тем белее, тем больше значение)
        """
        return 1 / observation.mean()

    def _get_is_done(self):
        while True:
            try:
                with open(self.state_file, 'r') as f:
                    status = f.read()
            except:
                continue
            if status == "Alive":
                return False
            elif status == "Dead":
                return True
            else:
                raise NameError(f'Wrong status in file: {self.state_file}')

    # Public methods

    def step(self, action_id):
        """
        The method performs the given action.

        Parameters
        ----------
        action_id : int

        Returns
        -------
        tuple
            tuple of next_observation, reward, is_done.
        """
        self.next_observation = self._perform_action(self.actions[action_id])
        reward = self._get_reward(self.next_observation)
        is_done = self._get_is_done()
        return self.next_observation, reward, is_done

    def reset(self):
        # Press restart
        self.next_observation = self._perform_action('reset')
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
        The function returns a random action_id from all possible actions.

        Returns
        -------
        int
        """
        return np.random.choice(np.arange(len(self.actions)))

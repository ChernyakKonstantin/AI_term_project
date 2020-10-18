#Imports required
import pyautogui
import win32gui
import numpy as np
import random
import time
from pynput.keyboard import Controller, Key
from PIL import ImageOps

class Environment:
    """
       This class runs the given application and emulates keyboard pressings to interact with the application.
       Methods, the agent interacts with, are 'step(action)' and 'reset()' 
    """
    def __init__(self, app_title, reset_btn, interruption_message, message_region, seq_len, viewport_size):
        """
           app_title: string
           reset_btn: string - a keyboard key which resets the app if a session is over.
           interruption_message: numpy array - a grayscale image, popping up if a session is over.
           message_region: tuple - Area of interruption message of the following shape: left, top, right, bottom.
           seq_len: int - a number of subsequent frames to capture.
           viewport_size: tuple - width and height of a viewport.
        """
        print("Click on the app window!")
        time.sleep(1)
        self.next_observation = None
        self.actions = (0,1,2,3,4,5)
        self.rewards = {
                        0: 1.0,
                        1: -1.5,
                        2: 1.0,
                        3: 1.0,
                        4: -1.5,
                        5: -1.5, #all -1.5 were -0.5 before
                        "Penalty": -100
                       }
        
        self.seq_len = seq_len
        self.l_viewport_region, self.mid_viewport_region, self.r_viewport_region = self._get_viewports_region(*viewport_size)
        
        self.interruption_message = interruption_message
        self.message_region = message_region
        
        self.controller = Controller() #Keyboard emulator
        self.buttons = {
                        0: [Key.up],
                        1: [Key.down],
                        2: [Key.up, Key.left],
                        3: [Key.up, Key.right],
                        4: [Key.down, Key.left],
                        5: [Key.down, Key.right]
                       }
        self.reset_btn = reset_btn
        
        hwnd = win32gui.FindWindow(None, app_title)
        self.screen_region = self._get_window_rect(hwnd)
           
    #Private methods
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
        observation : numpy array of shape [2, self.seq_len, 64, 64]

        """
        observation = np.empty([2, self.seq_len, 64, 64], dtype=np.uint8) #Hard-Coded 64x64
        self.right_viewport_frames = []
        for seq_index in range(self.seq_len):
            t1 = time.time()
            frame = pyautogui.screenshot(region=self.screen_region)
            l_viewport_frame = frame.crop(self.l_viewport_region)
            l_viewport_frame = ImageOps.grayscale(l_viewport_frame)
            l_viewport_frame = l_viewport_frame.resize((64,64)) #Hard-coded!
            l_viewport_frame = np.asarray(l_viewport_frame)
            
            mid_viewport_frame = frame.crop(self.mid_viewport_region)
            mid_viewport_frame = ImageOps.grayscale(mid_viewport_frame)
            mid_viewport_frame = mid_viewport_frame.resize((64,64)) #Hard-coded!
            mid_viewport_frame = np.asarray(mid_viewport_frame)
            
            r_viewport_frame = frame.crop(self.r_viewport_region)
            r_viewport_frame = ImageOps.grayscale(r_viewport_frame)
            r_viewport_frame = np.asarray(r_viewport_frame)

            observation[0, seq_index] = l_viewport_frame
            observation[1, seq_index] = mid_viewport_frame
            
            self.right_viewport_frames.append(r_viewport_frame)
            
            while time.time() - t1 < 0.1:
                pass
            
        return observation       
    

    def _get_is_done(self):
        """
           The method checks if collision message is shown of the 3rd viewport. 
        """
        for frame in self.right_viewport_frames:
            if np.all(self._get_message_area(frame) == self.interruption_message):
                return True
            else:
                return False
    

    def _get_message_area(self, observation):
        """Return numpy array of pixels from area of interruption message."""
        left = self.message_region[0]
        right = self.message_region[2]
        top = self.message_region[1]
        bottom = self.message_region[3]
        return observation[top: bottom, left: right] 
        
    
    def _get_reward(self, action_name, is_done):
        if is_done == True:
            return self.rewards["Penalty"]
        else:
            return self.rewards[action_name]
         
    
    def _get_window_rect(self, hwnd):
        """This function returns a position of the window left top corner,
           and the window width and height.
           Input:
               int - the window id.
           Output:
                tuple of int - (left, top, width, height)
        """
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top
        return left, top, width, height 
    
    
    def _get_viewports_region(self, viewport_width, viewport_height):
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
        #I measured the constants
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
    

    #Public methods

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
        self.next_observation = self._get_observation()
        self._stop_action((action_name))
        is_done = self._get_is_done()
        reward = self._get_reward(action_name, is_done)
        return self.next_observation, reward, is_done


    def reset(self):    
        #Press restart
        self.controller.press(self.reset_btn)
        time.sleep(0.15) #To emulate human pressing
        self.controller.release(self.reset_btn)
        
        self.next_observation = self._get_observation()
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

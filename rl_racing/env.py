from start import start_game
from queue import Queue, Empty
from pynput.keyboard import Key, Controller
import numpy as np
from PIL import Image
from mss import mss
import cv2
import subprocess

class Action:
    UP = {Key.up}
    LEFT = {Key.left}
    RIGHT = {Key.right}
    DOWN = {Key.down}
    UP_LEFT = {Key.up,Key.left}
    UP_RIGHT = {Key.up,Key.right}
    #DOWN_LEFT = {Key.down,Key.left}
    #DOWN_RIGHT = {Key.down,Key.right}
class DustRacingRL:
    def __init__(self):
        self.state_queue, self.pid, self.win_ids = start_game()
        self.keyboard = Controller()
        self.action_space = [Action.UP, Action.LEFT, Action.RIGHT, Action.UP_LEFT, Action.UP_RIGHT]
        self.last_action = None
        self.car_mon = {'top': 114+57-42-70, 'left': 322+58-31-70, 'width': 540, 'height': 540}
        #self.resize_dim = (256,256)
        self.resize_dim = (128,128)
        #self.resize_dim = (84,84)
        self.input_shape = self.resize_dim
        self.sct = mss()

    def reset(self):
        self.focus_window()
        self.last_action = None
        self.clean_up()
        self.keyboard.press(Key.esc)
        self.keyboard.release(Key.esc)
        self.keyboard.press(Key.enter)
        state = None
        while state != 'START':
            try:  
                state = self.state_queue.get_nowait()
                state = state.strip("\n")
            except:
                pass
        self.keyboard.release(Key.enter)
        return self.__observe()

    def clean_up(self):
        self.keyboard.release(Key.up)
        self.keyboard.release(Key.left)
        self.keyboard.release(Key.right)
        self.keyboard.release(Key.down)

    def __observe(self):
        frame = None
        car_box = self.sct.grab(self.car_mon)
        car_box =  np.array(Image.frombytes('RGB', (car_box.width, car_box.height), car_box.bgra, "raw", "BGRX"))
        frame = cv2.cvtColor(cv2.resize(car_box, self.resize_dim), cv2.COLOR_BGR2GRAY)
        #normalization
        return frame/255.0
        #zero mean
        #return frame-frame.mean()
        #normalization + zero mean
        """frame = frame/255.0
        return frame-frame.mean()"""
        #standardization
        #return (frame-frame.mean())/frame.std()
        #edge detection
        """frame = cv2.Canny(frame, threshold1 = 200, threshold2 = 300)
        frame = frame/255.0
        return frame"""

    def step(self, action):
        action = self.action_space[action]
        next_state = None
        reward = -0.03
        #get reward when go forward
        if Key.up in action:
            #print('up')
            reward = 0.5
        done = False
        info = None
        if self.last_action and self.last_action != action:
            for key in (self.last_action - action):
                self.keyboard.release(key)
        self.last_action = action
        for key in action:
            self.keyboard.press(key)
        next_state = self.__observe()
        try:  
            msg = self.state_queue.get_nowait() # or q.get(timeout=.1)
            msg = msg.strip("\n")
            if msg == 'LOSE':
                reward = -1
                done = True
            elif msg == 'FINISH_LAP':
                reward = +1
                info = msg
            elif msg == 'FINISH_GAME':
                reward = +1
                done = True
                info = msg
        except:
            pass
        if done:
            self.clean_up()
        return next_state, reward, done, info

    def focus_window(self):
        for win_id in self.win_ids:
            subprocess.run(["xdotool", "windowfocus", str(win_id)])

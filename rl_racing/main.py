from tqdm import trange
from start import start_game
from queue import Queue, Empty
from env import DustRacingRL, Action
import time
import cv2
import models
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

refresh_fps_time = 0.3
episodes = 5
window_name = 'rl_image'
cv2.namedWindow(window_name)
cv2.moveWindow(window_name, 800,800)
env = DustRacingRL()
#model = models.RandomModel(env)
#env.play(model)
#model = models.DoubleDqnModel(env)
#model = models.DqnModel(env, 'CnnDQN')
model = models.DqnModel(env, 'DDQN')
#model = models.DqnModel(env, 'CnnLstmDQN')
#model = models.DoubleDrqnModel(env)
#model.load('success-210825.pt')
#model.load('101.pt')
"""tot_reward, finish_lap, finish_game = model.play(episodes = 30)
print('finish_lap:{}'.format(finish_lap))
print('finish_game:{}'.format(finish_game))"""
model.train()
for i in range(5000):
    model.debug_buffer(i)

"""for episode in range(episodes):
    env.play(model)"""

env.clean_up()
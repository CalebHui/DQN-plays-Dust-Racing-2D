from tqdm import trange
from start import start_game
from queue import Queue, Empty
from env import DustRacingRL, Action
import time
import cv2
import models
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

window_name = 'rl_image'
cv2.namedWindow(window_name)
cv2.moveWindow(window_name, 800,800)
env = DustRacingRL()
#model = models.RandomModel(env)
#env.play(model)
#model = models.DoubleDqnModel(env)
#Random replay buffer
#model = models.DqnModel(env, 'CnnDQN')
#model = models.DqnModel(env, 'DDQN')
#prioritized replay buffer
model = models.PrioritizedDqnModel(env, 'CnnDQN')
#model = models.PrioritizedDqnModel(env, 'DDQN')

#The pytorch saving files is in the directory models/save/doubledqn
#the file name is <DQN type>-<training episodes>.pt
#e.g. model.load('500188.pt') will load models/save/doubledqn/CnnDQN-500188.pt
#model.load('500188.pt')
#model.load('success-514602.pt')
#tot_reward, finish_lap, finish_game = model.play(episodes = 30, plot_stat=True)
#demo3
#tot_reward, finish_lap, finish_game = model.play(episodes = 10, plot_stat=True, random_model=True)
#print('finish_lap:{}'.format(finish_lap))
#print('finish_game:{}'.format(finish_game))
#uncomment this to start training model
model.train()
#replay memory in replay buffer
for i in range(5000):
    model.debug_buffer(i)

env.clean_up()
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
model = models.DoubleDqnModel(env)
#model = models.DoubleDrqnModel(env)
#model.load('1500001.pt')
model.train()
"""for episode in range(episodes):
    env.play(model)"""

"""env.reset()
last_time = time.time()
cumulative = refresh_fps_time
tr = trange(episodes+1, desc='Agent training', leave=True)
for episode in tr:
    tr.set_description("Agent training")
    tr.refresh()
    loop_time = time.time()-last_time
    cumulative -= loop_time
    if cumulative <= 0:
        cumulative = refresh_fps_time
        print('{}fps'.format(round(1/(loop_time))))
    last_time = time.time()
    next_state, reward, done, info = env.step(Action.UP)
    #slow to show image
    cv2.imshow(window_name, next_state)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    if done:
        env.reset()"""
env.clean_up()
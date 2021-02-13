from tqdm import trange
import screen_capture
from start import start_game
from queue import Queue, Empty

episodes = 3000000
console_queue = start_game()
tr = trange(episodes+1, desc='Agent training', leave=True)
for episode in tr:
    tr.set_description("Agent training")
    tr.refresh() 
    status = None
    try:  
        status = console_queue.get_nowait() # or q.get(timeout=.1)
        print(status)
    except:
        pass
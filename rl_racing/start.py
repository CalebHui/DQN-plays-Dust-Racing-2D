#from subprocess import Popen, PIPE
import subprocess
import os
from queue import Queue, Empty
from threading  import Thread
import time

def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line.decode())
    out.close()

def start_game():
    program_path = "./dustrac-game"
    os.chdir("DustRacing2D/build")
    p = subprocess.Popen([program_path], stdout=subprocess.PIPE)
    #print(p.pid)
    time.sleep(3)
    win_id = subprocess.run(["xdotool", "search", "--pid", str(p.pid).strip()], capture_output=True, text=True).stdout.strip("\n")
    #print(win_id)
    subprocess.run(["xdotool", "windowmove", str(win_id), "0", "0"])
    q = Queue()
    t = Thread(target=enqueue_output, args=(p.stdout, q))
    t.daemon = True # thread dies with the program
    t.start()
    return q
    #result = p.stdout.readline().strip().decode()
    #print(result)
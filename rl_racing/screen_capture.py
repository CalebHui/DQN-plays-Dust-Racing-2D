import numpy as np
from PIL import Image
from mss import mss
import cv2
import time
from sys import getsizeof
from tqdm import trange
refresh_fps_time = 0.3

#full screen
#car_mon = {'top': 58, 'left': 68, 'width': 960, 'height': 540}
#box around car
#car_mon = {'top': 114+57-16, 'left': 322+58-16, 'width': 316+32, 'height': 316+32}
#car_mon = {'top': 114+57-42, 'left': 322+58-31, 'width': 400, 'height': 400}
car_mon = {'top': 114+57-42-70, 'left': 322+58-31-70, 'width': 540, 'height': 540}
#score box
#car_mon = {'top': 114+57-16, 'left': 322+58-16, 'width': 316+32, 'height': 316+32}
score_box = {'top': 400, 'left': 350, 'width': 50, 'height': 50}
resize_dim = (256,256)

"""def overlay_lines(image, lines):
    
    for line in lines:
        coordinates = line[0]
        cv2.line(image,(coordinates[0],coordinates[1]), \
                (coordinates[2],coordinates[3]),[255,255,255],3)

def edgeprocessed(image):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    
    edgeprocessed_img = cv2.Canny(gray_image, threshold1 = 200,\
                                  threshold2 = 300)    
    
    edgeprocessed_img = cv2.GaussianBlur(edgeprocessed_img,(5,5),0)
    
    lines = cv2.HoughLinesP(edgeprocessed_img, 1, np.pi/180, \
                            180, np.array([]), 100, 5)
    
    overlay_lines(edgeprocessed_img, lines)
    
    return edgeprocessed_img"""

def edgeprocessed(image):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edgeprocessed_img = cv2.Canny(gray_image,               \
                        threshold1 = 200, threshold2 = 300)
    return edgeprocessed_img

def screen_record(): 
    last_time = time.time()
    cumulative = refresh_fps_time
    with mss() as sct:
        while(True):
            car_box = sct.grab(car_mon)
            car_box =  np.array(Image.frombytes('RGB', (car_box.width, car_box.height), car_box.bgra, "raw", "BGRX"))
            #edge detection
            car_box = cv2.resize(car_box, resize_dim)
            edgeprocessed_img = edgeprocessed(car_box)
            cv2.imshow('window', edgeprocessed_img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

def screen_record1(): 
    last_time = time.time()
    cumulative = refresh_fps_time
    with mss() as sct:
        while(True):
            car_box = sct.grab(car_mon)
            car_box =  np.array(Image.frombytes('RGB', (car_box.width, car_box.height), car_box.bgra, "raw", "BGRX"))
            loop_time = time.time()-last_time
            #print('loop took {} seconds'.format(loop_time))
            cumulative -= loop_time
            if cumulative <= 0:
                cumulative = refresh_fps_time
                #print('{}fps'.format(round(1/(loop_time))))
            last_time = time.time()
            #downscale for training
            car_box = cv2.cvtColor(cv2.resize(car_box, resize_dim), cv2.COLOR_BGR2GRAY)
            #edge detection
            #car_box = cv2.cvtColor(car_box, cv2.COLOR_BGR2GRAY)
            #car_box = edgeprocessed(car_box)
            #upscale
            #car_box = cv2.resize(car_box, (540,540))
            cv2.imshow('window', car_box)
            #black white
            #ret,thresh1 = cv2.threshold(cv2.cvtColor(car_box, cv2.COLOR_BGR2GRAY),80,255,cv2.THRESH_BINARY)
            #black green
            #ret,thresh1 = cv2.threshold(car_box,127,255,cv2.THRESH_BINARY)
            #segmentation
            #ret, thresh1 = cv2.threshold(car_box,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            
            #cv2.imshow('window', thresh1)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            
#screen_record()
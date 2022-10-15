import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import cv2
import os
import sys
import socket  
import sound_edit as sound_music
import HandTrackingModule as htm
import time
from numpy.lib.function_base import append
import multiprocessing_test as MPI
from playsound import playsound

text=None
end_1=0
count=0

CLASS_MAP = {0:'fist',
   1:'five',
   2:'none',
   3:'okay',
   4:'L',
   5:'rad',
   6:'three',
 7:'thumbs'}

NUM_CLASSES = len(CLASS_MAP)

def mapper(val):
    return CLASS_MAP[val]

def list_com():
    # now that we have the background, we can segment the hand.
        
        cv2.putText(frame_copy, "Index 0: Fist", (330, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)
        cv2.putText(frame_copy, "Index 1: Five", (330, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)
        cv2.putText(frame_copy, "Index 2: None", (330, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)
        cv2.putText(frame_copy, "Index 3: Okay", (330, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)
        cv2.putText(frame_copy, "Index 4: L", (330, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)
        cv2.putText(frame_copy, "Index 5: Rad", (330, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)
        cv2.putText(frame_copy, "Index 6: THREE", (330, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)
        cv2.putText(frame_copy, "Index 7: Thumbs", (330, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)


             

def calc_accum_avg(frame, accumulated_weight):
    global background
    
    if background is None:
        background = frame.astype("float")
        ddd=frame
       
        cv2.imshow('backgrounder',ddd)
        
        return None
    cv2.accumulateWeighted(frame, background, accumulated_weight)
    


def camera(cami):
    ret, frame = cami.read()
    frame = cv2.flip(frame, 1)
    frame_copy = frame      
    gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    return frame,frame_copy,gray 

def segment(frame, threshold=20):
    global background
    diff = cv2.absdiff(background.astype("uint8"), frame)
    cv2.imshow('background',background)
    cv2.imshow('deffirance',diff)
    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    contours= cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(contours) ==0:
        calc_accum_avg(gray, accumulated_weight)
        cv2.putText(frame_copy, str('None'), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return None
    else:
        return thresholded

def comunication(msg) :          
    s = socket.socket()         # Create a socket object
    s.connect(("192.168.1.44",1234))
    if msg != None:
        ss=str(msg)
        s.sendall(ss.encode()) 
    s.close()


def thres_display(img):
    width=64
    height=64
    dim=(width,height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    test_img=image.img_to_array(resized)
    test_img=np.expand_dims(test_img,axis=0)
    test_img /= 255
    result= newmod.predict(test_img)
  
    
    #print("%s: %.2f%%" % (newmod.metrics_names[1], result[1]*100))    
    val=[index for index,value in enumerate(result[0]) if value ==1]
    return val

def descion(k):
 if k >= 4  :
                            print('two seconds have been past')
                            print(res[0])
                            cam.release()
                            cv2.destroyAllWindows()
                            try:
                                if res[0]==3:
                                    comunication(res[0])
                                if res[0]==3:
                                    comunication(res[0])    
                                
                                if res[0]== 1: 
                                        #playsound('trun on the light.mp3') 
                                        comunication(res[0])
                                if res[0]== 0: 
                                        #playsound('turn off.mp3')  
                                        comunication(res[0])
                                if res[0]== 3: 
                                        #playsound('turn off.mp3')  
                                        comunication(res[0])    
                                          
                                   
                                if res[0] == 5:
                                    #playsound('sound_system.mp3')
                                    sss=sound_music.main()
                                    sss
                                    #playsound('sound_mode_shut.mp3')
                                if res[0]== 7: 
                                        pass
                                       #playsound('turn on the fan.mp3')     
                                num_frames = 0
                                if num_frames==0:
                                    os.execv(sys.executable, ['python'] + sys.argv)           
                            except:
                                num_frames = 0                         
 else:
                    k=0   
tes=MPI.main()
tes
#playsound('open_system.mp3')



newmod=load_model('Gesture_classification.h5')


background = None
accumulated_weight = 0.3
zz=0
detector = htm.handDetector(detectionCon=0.8)
cam = cv2.VideoCapture(0) 
kl=0
n=0
num_frames = 0

cam.set(3, 720)
cam.set(4, 480)
#playsound('background.mp3')
while True:
    frame,frame_copy,gray=camera(cam)
    if num_frames < 60:
        calc_accum_avg(gray, accumulated_weight)
        if num_frames <= 59:
            cv2.putText(frame_copy, "GETTING BACKGROUND AVG", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)       
    else:  
        list_com()
        hand = segment(gray)
        img=frame
        img_2=frame
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img, draw=False)
        global test
        test=detector.handType(img)  
        if hand is not None:
            thresholded = hand
            start = time.time() 
            try:
               hans=detector.hands
               x_1,y_1,w_1,h_1=detector.roi_extractor(frame,hans)
               threshold=thresholded.copy()[y_1-20:y_1+h_1+10,x_1-20:x_1+w_1+10]
               if k == ord('s'):
                roi_64 = cv2.resize(threshold, (64, 64)) 
                save_path = os.path.join('finale_dataset/fist', '{}.jpg'.format(count + 1))
                cv2.imwrite(save_path, roi_64)
                count += 1
               end = time.time()
               end_1=end_1+end
               kl=((end_1-start)/10**10)
               descion(kl)      
               cv2.imshow("Threshol Image", threshold)
            except:
                cv2.destroyWindow('Threshol Image')
                pass
            try:
              res=thres_display(threshold)
              x=mapper(res[0])
              cv2.putText(frame_copy, str(res[0])+':'+x, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            except:
                res=[]
            if len(res)==0:
               cv2.putText(frame_copy,'None', (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
               res=[]
            else: 
                    pass
               
        if test =='Left':
            start=0           
            end_1=0
            kl=0  
            #num_frames=40

    num_frames += 1
    cv2.imshow("Hand Gestures", frame_copy)
    k = cv2.waitKey(10) & 0xFF
    if k == ord('r'):
        num_frames=0

    
            
    if k == ord('z'):
       
        break

cam.release()
cv2.destroyAllWindows()

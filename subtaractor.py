######################################################################## IMPORTING LIBRARIES ########################################################################
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
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




#Values initialization 
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

#Maping the values to the classes
def mapper(val):
    return CLASS_MAP[val]


#hand gesture recognition list
def list_com():
    
        cv2.putText(frame_copy, "Index 0: Fist", (330, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)
        cv2.putText(frame_copy, "Index 1: Five", (330, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)
        cv2.putText(frame_copy, "Index 2: None", (330, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)
        cv2.putText(frame_copy, "Index 3: Okay", (330, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)
        cv2.putText(frame_copy, "Index 4: L", (330, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)
        cv2.putText(frame_copy, "Index 5: Rad", (330, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)
        cv2.putText(frame_copy, "Index 6: THREE", (330, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)
        cv2.putText(frame_copy, "Index 7: Thumbs", (330, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)


             
#A function that calculates the accumulated average of the background
def calc_accum_avg(frame, accumulated_weight):
    #initializing the background
    global background
    #checking if the background is None
    if background is None:
        background = frame.astype("float")
        frame_data = frame
        # cv2.imshow('backgrounder',frame_data)
        return None
    #computing weighted average, accumulate it and update the background
    cv2.accumulateWeighted(frame, background, accumulated_weight)
    


#A function that reads the video feed and apply gray scale and gaussian blur then return the resulted frame
def camera(cami):
    ret, frame = cami.read()
    frame = cv2.flip(frame, 1)
    frame_copy = frame      
    gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    return frame,frame_copy,gray 


# A function that segment the hand region and returns the thresholded image
def segment(frame, threshold = 20):
    global background
    diff = cv2.absdiff(background.astype("uint8"), frame)
    cv2.imshow('background',background)
    cv2.imshow('deffirance',diff)
    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    contours= cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(contours) == 0:
        calc_accum_avg(gray, accumulated_weight)
        cv2.putText(frame_copy, str('None'), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return None
    else:
        return thresholded
    
    
# A function that communicates with the arduino and send the hand gesture to the raspberry pi using socket library 
def comunication(msg) :          
    s = socket.socket()  # Create a socket object
    s.connect(("192.168.1.44",1234))
    if msg != None:
        ss=str(msg)
        s.sendall(ss.encode()) 
    s.close()
    
    

#A function that feed the segmented hand to the model and predict the gesture
def thres_display(img):
    width = 64
    height = 64
    dim = (width,height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    test_img = image.img_to_array(resized)
    test_img = np.expand_dims(test_img,axis=0)
    test_img /= 255
    result= newmod.predict(test_img)
    #print("%s: %.2f%%" % (newmod.metrics_names[1], result[1]*100))    
    val=[index for index,value in enumerate(result[0]) if value ==1]
    return val



#A function apply the command based on the gesture
def descion(k):
 if k >= 4  :
                            print('two seconds have been past')
                            print(res[0])
                            cam.release()
                            cv2.destroyAllWindows()
                            try:
                                if res[0] == 0:
                                    comunication(res[0])    
                                if res[0] == 1: 
                                        #playsound('trun on the light.mp3') 
                                        comunication(res[0])
                                if res[0] == 2: 
                                        #playsound('turn off.mp3')  
                                        comunication(res[0])
                                if res[0] == 4: 
                                        #playsound('turn off.mp3')  
                                        comunication(res[0])        
                                if res[0] == 5:
                                    #playsound('sound_system.mp3')
                                    start_sound_manipulation=sound_music.main()
                                    start_sound_manipulation
                                    #playsound('sound_mode_shut.mp3')
                                if res[0] == 6:
                                    pass
                                if res[0] == 7:
                                    pass
                               
                                        
                                num_frames = 0
                                if num_frames == 0:
                                    os.execv(sys.executable, ['python'] + sys.argv)           
                            except:
                                num_frames = 0                         
 else:
        k=0   

######################################################################## MAIN ########################################################################

#playsound('open_system.mp3')
#creating an object of the class multiprocessing_test
tes=MPI.main()
# Runing object of the class multiprocessing_test to get access to the system   
tes
#loading classification model
newmod=load_model('model_colab_1.h5')
#initializing the accumulated weight and background
background = None
accumulated_weight = 0.3
#using handDetector class from HandTrackingModule
detector = htm.handDetector(detectionCon=0.8)
#Selecting camera 
cam = cv2.VideoCapture(0) 
kl=0
num_frames = 0
#Camera definition
cam.set(3, 720)
cam.set(4, 480)
#playsound('background.mp3')

######################################################################## PROGRAM LOOP ########################################################################
while True:
    #applying camera function to get the frame
    frame,frame_copy,gray=camera(cam)
    
    #setting the condition to start the hand detection
    if num_frames < 60:
        
        #calculating accumulated_weight average
        calc_accum_avg(gray, accumulated_weight)
        #getting the background 
        if num_frames <= 59:
            
            #getting background average message
            cv2.putText(frame_copy, "GETTING BACKGROUND AVG", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)       
    else: 
        
        #calling list of commands on screen  
        list_com()
        #segmenting moving objects 
        hand = segment(gray)
        img = frame
        img_2 = frame
        
        #finding hands using mediapipe library
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img, draw=False)
        #checking if the hand left  or right 
        global test
        test=detector.handType(img)  
        #starting counting time if the left hand detected
        if hand is not None:
            thresholded = hand
            start = time.time()
             
             
            try:
                
               #Detecting hands  
               hans=detector.hands
               #getting bounding box of the hand
               x_1,y_1,w_1,h_1 = detector.roi_extractor(frame,hans)
               #croping the  the hand from thresholded image 
               threshold  = thresholded.copy()[y_1-20:y_1+h_1+10,x_1-20:x_1+w_1+10]
               
               #capturing images using s key 
               if k == ord('s'):
                #resizing thresholded image    
                roi_64 = cv2.resize(threshold, (64, 64))
                #creating save path 
                save_path = os.path.join('finale_dataset/fist', '{}.jpg'.format(count + 1))
                #saving the image 
                cv2.imwrite(save_path, roi_64)
                count += 1
               
               #calcuting time using time library
               end = time.time()
               end_1=end_1+end
               kl=((end_1-start)/10**10)
               #making the decision after 2 seconds
               descion(kl)      
               cv2.imshow("Threshol Image", threshold)
               
            except:
                
                #destroy all windows      
                cv2.destroyWindow('Threshol Image')
                pass
            
            try:
               
              #getting the segmented hand to the model and predict the gesture 
              res=thres_display(threshold)
              x=mapper(res[0])
              cv2.putText(frame_copy, str(res[0])+':'+x, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
              
            except:
                res=[]
            
            #if results is empty write None
            if len(res)==0:
                #adding a text to the screen that display None
               cv2.putText(frame_copy,'None', (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
               res=[]
               
            else: 
                pass
                   
        #if the left hand detected reset the time      
        if test =='Left':
            start=0           
            end_1=0
            kl=0  

            
            

    num_frames += 1
    cv2.imshow("Hand Gestures", frame_copy)
    k = cv2.waitKey(10) & 0xFF
    #reset background average using r button 
    if k == ord('r'):
        num_frames = 0
    
    #exit the program using Z    
    if k == ord('z'):
        break




cam.release()
cv2.destroyAllWindows()

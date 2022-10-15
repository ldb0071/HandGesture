################################################################ IMPORTING LIBRARIES ########################################################################
import cv2
import time
import numpy as np
import HandTrackingModule as htm

#sounder class 
class sounder():
    def __init__(self):
       pass

######################################################################## MAIN ####################################################################################     

def main():
        ################################
        wCam, hCam = 1920,1080
        ################################
        length=100
        cam = cv2.VideoCapture(0)
        cam.set(3, wCam)
        cam.set(4, hCam)
        area = 0 
        detector = htm.handDetector(detectionCon=0.8, maxHands=1,trackCon=0.8)
        
        while True:
            #getting frame from camera feed 
            success, img = cam.read()
            # Find Hand
            img = detector.findHands(img)
            lmList, bbox = detector.findPosition(img, draw=False)
            
            if len(lmList) != 0:
                
                # Filter based on size
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100  
                # print(area)
                if 10 < area < 1000:
                    # Find Distance between index and Thumb
                    length, img, lineInfo = detector.findDistance(4,12, img)
             
            #condition for the distance between index and thumb       
            if length < 40 :
                 cv2.destroyAllWindows()
                 time.sleep(1)
                 return 
             
            cv2.imshow("Img", img)
            cv2.waitKey(1)

if __name__ == "__main__":
    main()          
################################################################ IMPORTING LIBRARIES ########################################################################
import cv2
import numpy as np
import HandTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class sounder():
    def __init__(self):
        pass

######################################################################## MAIN ########################################################################     

def main():
    ################################
    wCam, hCam = 1920, 1080
    ################################
    cap = cv2.VideoCapture(0)
    #setting up the camera definition
    cap.set(3, wCam)
    cap.set(4, hCam)
    #creating object from handDetector class
    detector = htm.handDetector(detectionCon=0.8, maxHands=1)
    #enabling sound volBar
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    #creating volume interface and intializing volbar  volume range and color and area  
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volBar = 400
    volPer = 0
    area = 0
    colorVol = (255, 0, 0)
    while True:
        success, img = cap.read()
        # Find Hand
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            # Filter based on size
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
            # print(area)
            if 40 < area < 1000:
                # Find Distance between index and Thumb
                length, img, lineInfo = detector.findDistance(8, 4, img,draw=True)
                # Convert Volume
                volBar = np.interp(length, [50, 200], [400, 150])
                volPer = np.interp(length, [50, 200], [0, 100])
                # Reduce Resolution to make it smoother
                smoothness = 10
                volPer = smoothness * round(volPer / smoothness)
                # Check fingers up
                fingers = detector.fingersUp()
                # If pinky is down set volume
                if not fingers[0]:
                    volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    colorVol = (0, 255, 0)           
                else:
                    colorVol = (255, 0, 0)
        # Drawings
            cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
            cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                        1, (255, 0, 0), 3)
            cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
            cv2.putText(img, f'Vol Set: {int(cVol)}', (400, 50), cv2.FONT_HERSHEY_COMPLEX,
                        1, colorVol, 3)
        # Frame rate      
            k=cv2.waitKey(10)
            if length <= 10 :
                    cv2.destroyAllWindows()
                    return
        cv2.imshow("Img", img)
        cv2.waitKey(1)
        
if __name__ == "__main__":
    main()          
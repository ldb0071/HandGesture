import cv2
import mediapipe as mp
import numpy as np 
import math

class handDetector():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.7, trackCon=0.8):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.lmList = []




    def findHands(self, img, draw=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
 
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img,draw=False):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                              (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)

        return self.lmList, bbox



    def findDistance(self, p1, p2, img, draw=True):
    
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 255, 150), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 255, 150), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 150), 3)
            cv2.circle(img, (cx, cy), 15, (255, 50, 200), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]
    


    def fingersUp(self):
        fingers = []
        if self.lmList[20][2] < self.lmList[18][2]:
                fingers.append(1)
        else:
                fingers.append(0)
        return fingers  



    def roi_extractor(self,img,hands):
         image_width, image_height = img.shape[1], img.shape[0]
         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
         results = hands.process(imgRGB)
         
        # print(results.multi_hand_landmarks)
         landmark_array = np.empty((0, 2), int)
         if results.multi_hand_landmarks:
            
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                        landmark_x = min(int(lm.x * image_width), image_width - 1)
                        landmark_y = min(int(lm.y * image_height), image_height - 1)
                        landmark_point = [np.array((landmark_x, landmark_y))]
                        landmark_array = np.append(landmark_array, landmark_point, axis=0)
                        x, y, w, h = cv2.boundingRect(landmark_array)
                        
            return  x, y, w, h
            


    def handType(self,img):
       
        self.results = self.hands.process(img)
        if self.results.multi_hand_landmarks:
            try:
                if self.lmList[17][1] <= self.lmList[5][1]:
                    return "Right"
                else:
                    return "Left"
            except:
                pass        
      
    

def main():
        wCam, hCam = 640, 480
        ################################

        cam = cv2.VideoCapture(1)
        cam.set(3, wCam)
        cam.set(4, hCam)
        pTime = 0

        detector = handDetector(detectionCon=0.8, maxHands=1)

        
        # volume.GetMute()
        # volume.GetMasterVolumeLevel()
      
        
        volBar = 400
        volPer = 0
        area = 0
        colorVol = (255, 0, 0)

        while True:
            success, img = cam.read()

            # Find Hand
            img = detector.findHands(img)
            lmList, bbox = detector.findPosition(img, draw=True)
            text=detector.handType(img)
            print(text)
            
            if len(lmList) != 0:   
                # Filter based on size
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
                # print(area)
                if 250 < area < 1000:
                    # Find Distance between index and Thumb
                    length, img, lineInfo = detector.findDistance(4, 8, img)
                    # print(length)
                    # Convert Volume
                    volBar = np.interp(length, [50, 200], [400, 150])
                    volPer = np.interp(length, [50, 200], [0, 100])
                    # Reduce Resolution to make it smoother
                    smoothness = 10
                    volPer = smoothness * round(volPer / smoothness)
                    # Check fingers up
                    fingers = detector.fingersUp()
                    # print(fingers)

                    # If pinky is down set volume
                    if not fingers[4]:
                        cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                        colorVol = (0, 255, 0)
                        break
                    else:
                        colorVol = (255, 0, 0)
            # Drawings
            cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
            cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
            cv2.imshow("Img", img)
            cv2.waitKey(1)


 

if __name__ == "__main__":
    main()
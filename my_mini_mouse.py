import cv2
import mediapipe as mp
from math import sqrt
import time
import numpy as np
import pyautogui as autopy

class HandDetector:
    def __init__(self, mode=False, maxHands=2, modelComplexity=1,detectionConf=0.5, trackConf=0.5):
        self.mode = mode 
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionConf = detectionConf
        self.trackConf = trackConf 

        self.mpHands = mp.solutions.hands #get base mphands
        #below is the module that allows us to locate the hands and we use the parms above in this
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionConf, self.trackConf) #use param from init
        self.mpDraw = mp.solutions.drawing_utils #drawing package
        self.tipIds = [4, 8, 12, 16, 20]
    
    #The below function gets the hands and if draw it will draw the landmarks
    def getHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #have to take the img and turn it to RGB
        self.results = self.hands.process(imgRGB) #find the hands(if there are any)
        
        if self.results.multi_hand_landmarks: #if there are hands
            for hand in self.results.multi_hand_landmarks: #for each hand in the frame
                if draw:
                    #mpDraw.draw_landmarks(img, hand) #draws only the landmarks
                    self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS) #draws landmarks and connector lines
        return img
    
    #this creates a list of all landmark positions on the hand
    def getPos(self, img, handNum=0, draw=True):
        lmList = [] #create the empty list

        x_max = 0
        y_max = 0
        x_min = 0
        y_min = 0

        if self.results.multi_hand_landmarks: #if there are hands
            #print(self.results.multi_hand_world_landmarks.__doc__)
            myHand = self.results.multi_hand_landmarks[handNum]
            
            h, w, c = img.shape #get size of img
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h

            #for each lm on the hand
            for id, lm in enumerate(myHand.landmark):
                #print(id, lm) #gives id and positions
                #h, w, c = img.shape #get size of img
                cx, cy = int(lm.x*w), int(lm.y*h) #turn x and y into pixel values
                #print(id, cx, cy) #print id and x and y values in pixels
                if cx > x_max:
                    x_max = cx
                if cx < x_min:
                    x_min = cx
                if cy > y_max:
                    y_max = cy
                if cy < y_min:
                    y_min = cy

                lmList.append([id, cx, cy]) #add positions to list with lm id

                if draw: #if draw then draw the extra circle 
                    cv2.circle(img, (cx, cy), 10, (0,255,0), cv2.FILLED) #adding an extra green circle
            
            cv2.rectangle(img, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)

        self.lmList = lmList
        #return the list of landmarks
        return lmList, [x_min - 20, y_min - 20, x_max + 20, y_max + 20]

    #function to test if fingers are up, returns a list of len 5 where 1 = up and 0 = down
    def fingersUp(self):
        fingers = []
        # Thumb
        if len(self.lmList) != 0:
            if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            # Fingers
            for id in range (1, 5):
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        return fingers
    
    def findDistance(self, p1, p2, img):
        length = sqrt((self.lmList[p1][1] - self.lmList[p2][1])**2 + (self.lmList[p1][2] - self.lmList[p2][2])**2)

        img = cv2.line(img, (self.lmList[p1][1], self.lmList[p1][2]), (self.lmList[p2][1], self.lmList[p2][2]), (0,0,255),2)

        return length, img
    
  

#set width and height of window
wCam, hCam = 640, 480
frameR = 100 #frame reduction
smoothening = 5
plocx, plocy = 0, 0
clocx, clocy = 0, 0

#set pTime
pTime = 0

#Set camera and dimensions
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

#Initalize Hand detector
detector = HandDetector(maxHands=1)

wScr, hScr = autopy.size()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1) #orignal soltion to flip whole image
    
    #get hand landmarks
    img = detector.getHands(img)
    lmlist, bbox = detector.getPos(img, draw=False)
    
    #We will need the tip of the index and middle finger
    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:] #index finger 
        x2, y2 = lmlist[12][1:] #middle finger 

        #check what fingers are up
        fingers = detector.fingersUp()
        #print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR), (255,0,255), 2)

        #Only index finger is moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            #convert coordinates and smooth values
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

            #smoothening
            clocx = plocx + (x3 - plocx)/smoothening
            clocy = plocy + (y3 - plocy)/smoothening

            #move mouse
            autopy.moveTo(clocx, clocy) #this work if whole image fliped
            #autopy.moveTo(wScr-x3, y3) #this works if whole image is NOT fliped

            plocx, plocy = clocx, clocy


        #Both fingers is click mode
        if fingers[1] == 1 and fingers[2] == 1:
            #find distance between fingers
            length, img = detector.findDistance(8, 12, img)
            #if short dist then click
            if length < 35:
                cv2.circle(img, (x1, y1), 4, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 4, (0, 255, 0), cv2.FILLED)
                autopy.click()

    #Adding frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

    #Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)

import cv2
import numpy as np
import time
#importing our pose module (can be used in many other applications)
import PoseEstimationModule as pm

#importing class from our pose module
detector = pm.poseDetector()

count = 0 #to count number of curls
direction = 0 #to check which direction (up and down)

pTime = 0

#stream video from live camera
cap = cv2.VideoCapture(0)

# Reading our video file
#cap = cv2.VideoCapture("AITrainerVideos/curls.mp4")

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))

    #img = cv2.imread("AITrainerVideos/plank.jpg")

    # to find pose using landmarks in image
    img = detector.findPose(img, False)
    # to get position array of landmarks
    lmList = detector.findPosition(img, False)
    #print(lmList)
    if len(lmList) > 0:
        # passing landmark of right arm
        #detector.findAngle(img, 12, 14, 16)

        # passing landmark of left arm
        angle = detector.findAngle(img, 11, 13, 15)

        #converting the minimum to maximum angle range to a 100 scale
        percentage = np.interp(angle, (40,160), (0,100)) #40 -> min angle, 160-> max angle
        #print(angle, percentage)

        # converting the minimum to maximum angle range to a bar scale from 0 to 100 percent
        bar = np.interp(angle, (40, 160), (650, 100))

        #check for bicep curl
        #color = (255, 0, 255) #to change color to green on reaching 100%
        if percentage == 0:
            #color = (0, 255, 0)
            if direction == 1:
                count += 0.5
            direction = 0
        if percentage == 100:
            #color = (0, 255, 0)
            if direction == 0:
                count += 0.5
            direction = 1

        #print(count)

        #to track count of bicep curls
        cv2.rectangle(img, (0,450), (250,720), (0,255,0), cv2.FILLED) #to display count in a box
        cv2.putText(img, str(int(count)), (45,670), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 28)

        #to display bar for percentage of curl
        cv2.rectangle(img, (1100, 100), (1175, 650), (0, 255, 0), 4)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(percentage)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 4)

        #frames per sec
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
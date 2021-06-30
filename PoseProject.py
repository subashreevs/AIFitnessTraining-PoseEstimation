import cv2
import time
#importing our pose module (can be used in many other applications)
import PoseEstimationModule as pm

cap = cv2.VideoCapture('PoseVideos/pv2.mp4')
#to capture live camera => cap = cv2.VideoCapture(0)

pTime = 0
detector = pm.poseDetector()  # creating object from imported module
while True:
    success, img = cap.read()
    img = detector.findPose(img)

    '''lmList = detector.findPosition(img)
    #print(lmList)'''

    #to show only a particular landmark say 14(elbow)
    lmList = detector.findPosition(img, draw=False)
    #does not print landmark if it is not present in a particular frame
    if len(lmList != 0):
        print(lmList[14])
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)
        #bigger red circle is used to track

    # to display frame per second in output video
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image:", img)
    cv2.waitKey(1)
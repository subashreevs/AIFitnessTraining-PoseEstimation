import cv2
import mediapipe as mp
import time
import math


class poseDetector():

    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):

        # initialising
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)  # Parameters are passed


    # method to find pose
    def findPose(self, img, draw=True):  # whether to display on img or not

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        # to show landmarks and connected lines
        if self.results.pose_landmarks:
            if draw:  # displays connections when draw is true
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    #method to find pose position data
    def findPosition(self, img, draw=True):
        self.lmList = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)

        return self.lmList

    #method that finds angle between 3 landmarks
    def findAngle(self, img, p1, p2, p3, draw=True):

        #getting the x and y coords of the required landmark
        x1, y1, = self.lmList[p1][1:]
        x2, y2, = self.lmList[p2][1:]
        x3, y3, = self.lmList[p3][1:]

        #calculating angle between the 2 lines connecting 3 landmarks
        angle = math.degrees( math.atan2(y1-y2, x1-x2) - math.atan2(y3-y2, x3-x2) )
        #print(angle)

        #for negative angles
        if(angle < 0):
            angle = angle + 360

        if draw:
            # highlighting the choosen landmarks
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)

            # highlighting the lines joining the choosen landmarks
            cv2.line(img, (x1,y1), (x2,y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)

            #to display calculated angle near (x2,y2) position on screen
            #cv2.putText(img, str(int(angle)), (x2-50, y2+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

        return angle


def main():
    cap = cv2.VideoCapture('PoseVideos/ps3.mp4')
    #to capture live camera => cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()  # creating object for the class
    while True:
        success, img = cap.read()
        img = detector.findPose(img)

        #lmList = detector.findPosition(img)
        #print(lmList)

        #to show only a particular landmark say 14(elbow)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255),
                       cv2.FILLED)  # bigger red circle is used to track


        # to display frame per second in output video
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image:", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

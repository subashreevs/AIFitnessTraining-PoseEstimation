import cv2
import mediapipe as mp
import time


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


    def findPosition(self, img, draw=True):
        lmList = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)

        return lmList


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

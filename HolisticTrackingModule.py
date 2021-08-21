import cv2
import mediapipe as mp


class bodyDetector():
    def __init__(self, mode=False, complexity=1, smooth_landmarks=True, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth_landmarks = smooth_landmarks
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpBody = mp.solutions.holistic
        self.body = self.mpBody.Holistic(self.mode, self.complexity, self.smooth_landmarks,
                                         self.detectionCon, self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils

    def processImage(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.body.process(imgRGB)
        return img

    def findPosePosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy, lm.z])
                if draw:
                    cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)
        return lmList

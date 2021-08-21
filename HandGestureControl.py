from collections import namedtuple
from datetime import datetime
from math import sqrt

import keyboard

import cv2
import numpy as np

import HolisticTrackingModule as holistic

wCam, hCam = 1920, 1080

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

arms = holistic.bodyDetector(detectionCon=0.60, trackingCon=0.80)

debug = True

Point = namedtuple("Point", ['x', 'y', 'z'])

CurrentRay = 0
left_bounds = False
right_bounds = False


def makeRay(pt1, pt2, length):
    first_point = Point(pt1[0], pt1[1], pt1[2])
    second_point = Point(pt2[0], pt2[1], pt2[2])
    new_point = Point(0, 0, 0)
    lenAB = sqrt(pow(first_point.x - second_point.x, 2.0) +
                 pow(first_point.y - second_point.y, 2.0) +
                 pow(first_point.z - second_point.z, 2.0))
    x = length / lenAB

    new_point = new_point._replace(x=second_point.x + (second_point.x - first_point.x) * x)
    new_point = new_point._replace(y=second_point.y + (second_point.y - first_point.y) * x)
    new_point = new_point._replace(z=second_point.z + (second_point.z - first_point.z) * x)
    return new_point


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[2] - b[2], c[1] - b[1]) - np.arctan2(a[2] - b[2], a[1] - b[1])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


print("**Calibration**")
print("Set left most point")
left_set = False
right_set = False

last_frame = datetime.now()

# Typical run loop
while True:
    success, img = cap.read()
    # Mirror image for debugging
    flipped = cv2.flip(img, 1)

    flipped = arms.processImage(flipped)
    armList = arms.findPosePosition(flipped, draw=False)

    # If person is in frame
    if len(armList) != 0:
        if armList[15][3] <= 0.08 and armList[13][3] <= 0.08:
            # Slice and dice
            if calculate_angle(armList[14], armList[12], armList[24]) < 35:
                CurrentRay = makeRay(armList[13], armList[15], 5000)
                if debug:
                    cv2.line(flipped, (int(CurrentRay.y), int(CurrentRay.z)),
                             (armList[13][1], armList[13][2]), (255, 0, 0), 8)
                    cv2.putText(flipped, str(int(CurrentRay.y)), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # If calibration is done
                if left_set and right_set:
                    fountain_point = int(np.interp(CurrentRay.y, [left_bounds, right_bounds], [1, 10]))
                    if debug:
                        cv2.putText(flipped, str(fountain_point), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                                    2)
            # Water height
            else:
                if armList[14][3] <= 0.08 and armList[16][3] <= 0.08:
                    height = sqrt(pow(armList[15][1] - armList[16][1], 2.0) +
                                  pow(armList[15][2] - armList[16][2], 2.0))
                    height_point = int(np.interp(height, [90, 500], [1, 50]))
                    if debug:
                        cv2.putText(flipped, str(height_point), (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    elapsed = datetime.now().__sub__(last_frame)
    last_frame = datetime.now()
    cv2.putText(flipped, str(int(1/elapsed.microseconds*1000000)), (500, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Gesture Tracking", flipped)
    cv2.waitKey(1)

    # Calibration setup
    if not left_set or not right_set:
        try:
            if keyboard.is_pressed(' '):
                if left_set:
                    print('Right bounds set.')
                    right_set = True
                    right_bounds = int(CurrentRay.y)
                else:
                    print('Left bounds set.')
                    left_set = True
                    left_bounds = int(CurrentRay.y)
                    print("Set right most point")
            continue
        except:
            continue

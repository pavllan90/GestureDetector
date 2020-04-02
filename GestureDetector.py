#!/usr/bin/env python

import numpy as np
import pickle
from hand_tracker import HandTracker
import cv2

class GestureDetector:
    def __init__(self):
        self.palm_model_path = "/home/pavel/ros_workspace/cv_tutorial/nodes/py2-RPS/models/palm_detection.tflite"
        self.landmark_model_path = "/home/pavel/ros_workspace/cv_tutorial/nodes/py2-RPS/models/hand_landmark.tflite"
        self.anchors_path = "/home/pavel/ros_workspace/cv_tutorial/nodes/py2-RPS/data/anchors.csv" 
        self.estimator_path = "/home/pavel/ros_workspace/cv_tutorial/nodes/py2-RPS/logregest_1200.pickle"
        self.detector = HandTracker(self.palm_model_path, self.landmark_model_path, self.anchors_path,
                           box_shift=0.2, box_enlarge=1.3)
        self.estimator = self.get_estimator()
        self.cap =   cv2.VideoCapture(0)

    def transform_frame(self, data):
        for i in range(21):
            data[i, 0] = data[i,0]- data[0,0]
            data[i, 1] = data[i,1]- data[0,1]
        data = data.reshape(1, 42)
        return data

    def draw_map(self, kp, img, box):
        for i in range(0,21):
            cv2.circle(img, (int(kp[i,0]), int(kp[i,1])), 10, (0,0,255))
        cv2.line(img, (int(box[0,0]), int(box[0,1])),(int(box[1,0]), int(box[1,1])),(0,255,0), 5)
        cv2.line(img, (int(box[1,0]), int(box[1,1])),(int(box[2,0]), int(box[2,1])),(0,255,0), 5)
        cv2.line(img, (int(box[2,0]), int(box[2,1])),(int(box[3,0]), int(box[3,1])),(0,255,0), 5)
        cv2.line(img, (int(box[3,0]), int(box[3,1])),(int(box[0,0]), int(box[0,1])),(0,255,0), 5)
        
        cv2.line(img, (int(kp[0,0]), int(kp[0,1])), (int(kp[1,0]), int(kp[1,1])),(255,0,255), 10)
        cv2.line(img, (int(kp[1,0]), int(kp[1,1])), (int(kp[2,0]), int(kp[2,1])),(255,0,255), 10)
        cv2.line(img, (int(kp[2,0]), int(kp[2,1])), (int(kp[3,0]), int(kp[3,1])),(255,0,255), 10)
        cv2.line(img, (int(kp[3,0]), int(kp[3,1])), (int(kp[4,0]), int(kp[4,1])),(255,0,255), 10)
        
        cv2.line(img, (int(kp[0,0]), int(kp[0,1])), (int(kp[5,0]), int(kp[5,1])),(255,0,0), 10)
        cv2.line(img, (int(kp[5,0]), int(kp[5,1])), (int(kp[6,0]), int(kp[6,1])),(255,0,0), 10)
        cv2.line(img, (int(kp[6,0]), int(kp[6,1])), (int(kp[7,0]), int(kp[7,1])),(255,0,0), 10)
        cv2.line(img, (int(kp[7,0]), int(kp[7,1])), (int(kp[8,0]), int(kp[8,1])),(255,0,0), 10)
        
        cv2.line(img, (int(kp[0,0]), int(kp[0,1])), (int(kp[9,0]), int(kp[9,1])),(0,0,0), 10)
        cv2.line(img, (int(kp[9,0]), int(kp[9,1])), (int(kp[10,0]), int(kp[10,1])),(0,0,0), 10)
        cv2.line(img, (int(kp[10,0]), int(kp[10,1])), (int(kp[11,0]), int(kp[11,1])),(0,0,0), 10)
        cv2.line(img, (int(kp[11,0]), int(kp[11,1])), (int(kp[12,0]), int(kp[12,1])),(0,0,0), 10)
        
        cv2.line(img, (int(kp[0,0]), int(kp[0,1])), (int(kp[13,0]), int(kp[13,1])),(100,50,0), 10)
        cv2.line(img, (int(kp[13,0]), int(kp[13,1])), (int(kp[14,0]), int(kp[14,1])),(100,50,0), 10)
        cv2.line(img, (int(kp[14,0]), int(kp[14,1])), (int(kp[15,0]), int(kp[15,1])),(100,50,0), 10)
        cv2.line(img, (int(kp[15,0]), int(kp[15,1])), (int(kp[16,0]), int(kp[16,1])),(100,50,0), 10)
        
        cv2.line(img, (int(kp[0,0]), int(kp[0,1])), (int(kp[17,0]), int(kp[17,1])),(25,0,100), 10)
        cv2.line(img, (int(kp[17,0]), int(kp[17,1])), (int(kp[18,0]), int(kp[18,1])),(25,0,100), 10)
        cv2.line(img, (int(kp[18,0]), int(kp[18,1])), (int(kp[19,0]), int(kp[19,1])),(25,0,100), 10)
        cv2.line(img, (int(kp[19,0]), int(kp[19,1])), (int(kp[20,0]), int(kp[20,1])),(25,0,100), 10)
            
    def get_estimator(self):
        with open(self.estimator_path, 'rb') as f:
            estimator = pickle.load(f)
        return estimator
       
    def get_gesture(self):
        while(1):
            try:
                ret, frame = self.cap.read()
                kp, box = self.detector(frame[:,:,::-1])
                break
            except ValueError:
                print("No hand found")
        self.draw_map(kp, frame,  box)
        player_move = self.estimator.predict(self.transform_frame(kp))
        cv2.putText(frame, str(player_move), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0),5)
        while(1):  
            cv2.imshow("Result", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        
detector = GestureDetector()
detector.get_gesture()
 


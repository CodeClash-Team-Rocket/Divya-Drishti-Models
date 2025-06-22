import cv2
import numpy as np
import torch
import pyttsx3
import threading
import time
import pandas as pd
from pathlib import Path

class ObstacleDetector:
    def __init__(self, model_path='yolov5s.pt', conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(self.device)

        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.9)
        self.last_announcement = ""
        self.last_announcement_time = 0
        self.announcement_cooldown = 2.0

        self.obstacle_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus',
            7: 'truck', 8: 'boat', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
            18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra',
            23: 'giraffe', 56: 'chair', 57: 'couch', 58: 'potted plant',
            59: 'bed', 61: 'toilet', 62: 'tv', 72: 'refrigerator' , 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
            45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
            50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
        }

    def detect_obstacles(self, frame):
        results = self.model(frame)
        df = results.pandas().xyxy[0]
        filtered = df[
            (df['confidence'] >= self.conf_threshold) &
            (df['class'].isin(self.obstacle_classes.keys()))
        ]
        return filtered

    def analyze_obstacles(self, detections, frame_width):
        if detections.empty:
            return 'clear', None

        frame_center = frame_width / 2
        obstacles = []

        for _, row in detections.iterrows():
            x1, x2 = row['xmin'], row['xmax']
            center_x = (x1 + x2) / 2
            width, height = x2 - x1, row['ymax'] - row['ymin']
            area = width * height

            position = (
                'left' if center_x < frame_center * 0.8 else
                'right' if center_x > frame_center * 1.2 else
                'center'
            )

            obstacles.append({
                'class': self.obstacle_classes.get(int(row['class']), 'obstacle'),
                'confidence': row['confidence'],
                'center_x': center_x,
                'position': position,
                'area': area,
                'bbox': (int(x1), int(row['ymin']), int(x2), int(row['ymax']))
            })

        obstacles.sort(key=lambda x: x['area'], reverse=True)
        closest = obstacles[0]

        center_obs = [o for o in obstacles if o['position'] == 'center']
        left_obs = [o for o in obstacles if o['position'] == 'left']
        right_obs = [o for o in obstacles if o['position'] == 'right']

        if center_obs:
            direction = 'left' if len(left_obs) < len(right_obs) else 'right'
        else:
            direction = 'right' if len(left_obs) else 'left' if len(right_obs) else 'clear'

        return direction, closest

    def speak_direction(self, direction, obstacle_info=None):
        now = time.time()
        if direction == 'clear':
            message = "Path is clear"
        elif direction in ['left', 'right']:
            obstacle = obstacle_info['class'] if obstacle_info else "obstacle"
            message = f"Move {direction} to avoid {obstacle}"
        else:
            message = "Stop! Multiple obstacles ahead"

        if (message != self.last_announcement or 
            now - self.last_announcement_time > self.announcement_cooldown):
            threading.Thread(target=lambda: self.tts_engine.say(message) or self.tts_engine.runAndWait()).start()
            self.last_announcement = message
            self.last_announcement_time = now
            print(f"🔊 {message}")

    def draw_detections(self, frame, detections, direction):
        for _, row in detections.iterrows():
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            label = f"{self.obstacle_classes.get(int(row['class']), 'object')}: {row['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        h, w = frame.shape[:2]
        cx, cy = w // 2, h - 50

        if direction == 'left':
            cv2.arrowedLine(frame, (cx, cy), (cx - 100, cy), (0, 255, 0), 5)
            cv2.putText(frame, "GO LEFT", (cx - 100, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif direction == 'right':
            cv2.arrowedLine(frame, (cx, cy), (cx + 100, cy), (0, 255, 0), 5)
            cv2.putText(frame, "GO RIGHT", (cx + 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif direction == 'clear':
            cv2.putText(frame, "PATH CLEAR", (cx - 80, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def run(self, source=0):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error: Cannot open video source")
            return

        self.speak_direction('clear')
        print("🚀 Obstacle Detection Started (press 'q' to quit)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            detections = self.detect_obstacles(frame)
            direction, obstacle = self.analyze_obstacles(detections, frame.shape[1])
            self.speak_direction(direction, obstacle)
            frame = self.draw_detections(frame, detections, direction)

            cv2.imshow('YOLOv5 Obstacle Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    detector = ObstacleDetector(conf_threshold=0.5)
    detector.run(source=0)

if __name__ == '__main__':
    main()

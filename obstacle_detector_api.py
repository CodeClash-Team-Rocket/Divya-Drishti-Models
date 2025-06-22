import cv2
import numpy as np
import torch
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import base64

app = FastAPI()

class ObstacleDetector:
    def __init__(self, model_path='yolov5m.pt', conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True).to(self.device)

        self.obstacle_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus',
            7: 'truck', 8: 'boat', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
            18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra',
            23: 'giraffe', 56: 'chair', 57: 'couch', 58: 'potted plant',
            59: 'bed', 61: 'toilet', 62: 'tv', 72: 'refrigerator', 39: 'bottle',
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

    def process_frame(self, frame):
      
        frame = cv2.flip(frame, 1)
       
        detections = self.detect_obstacles(frame)
       
        direction, obstacle = self.analyze_obstacles(detections, frame.shape[1])
       
        response = {
            'direction': direction,
            'obstacle': obstacle,
            'detections': detections.to_dict(orient='records')
        }
        return response


detector = ObstacleDetector(conf_threshold=0.5)

@app.post("/detect_obstacles")
async def detect_obstacles(file: UploadFile = File(...)):
    try:
      
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
       
        result = detector.process_frame(frame)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
import asyncio
import base64
import cv2
import numpy as np
import websockets
import json
from keras.models import load_model
import enchant
import pyttsx3
from cvzone.HandTrackingModule import HandDetector
import math
import os
import sys
import traceback
from string import ascii_uppercase

# Initialize dependencies
ddd = enchant.Dict("en-US")
hd = HandDetector(maxHands=1, detectionCon=0.7, minTrackCon=0.5)
hd2 = HandDetector(maxHands=1, detectionCon=0.7, minTrackCon=0.5)

class Application:
    def __init__(self, model_path="cnn8grps_rad1_model.h5"):
        try:
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                print("Model loaded successfully")
            else:
                print(f"Warning: Model file {model_path} not found. Using dummy predictions.")
                self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            
        try:
            self.speak_engine = pyttsx3.init()
            self.speak_engine.setProperty("rate", 100)
            voices = self.speak_engine.getProperty("voices")
            if voices:
                self.speak_engine.setProperty("voice", voices[0].id)
        except Exception as e:
            print(f"Error initializing TTS: {e}")
            self.speak_engine = None

        self.ct = {}
        self.ct["blank"] = 0
        self.blank_flag = 0
        self.space_flag = False
        self.next_flag = True
        self.prev_char = ""
        self.count = -1
        self.ten_prev_char = [" " for _ in range(10)]
        for i in ascii_uppercase:
            self.ct[i] = 0

        self.str = ""
        self.ccc = 0
        self.word = ""
        self.current_symbol = "Ready"
        self.photo = "Empty"
        self.word1 = ""
        self.word2 = ""
        self.word3 = ""
        self.word4 = ""
        self.pts = []

    def create_white_background(self, width=400, height=400):
        """Create a white background image"""
        return np.ones((height, width, 3), dtype=np.uint8) * 255

    def distance(self, x, y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def predict_simple(self, landmarks):
        """Simplified prediction based on hand landmarks"""
        if not landmarks or len(landmarks) < 21:
            return "Invalid"
        
        # Simple gesture recognition based on finger positions
        # This is a basic implementation - you should replace with your trained model
        
        # Get finger tip and pip positions
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        
        # Count extended fingers
        fingers = []
        
        # Thumb
        if thumb_tip[0] > thumb_ip[0]:  # Right hand
            fingers.append(1)
        else:
            fingers.append(0)
            
        # Other fingers
        fingers.append(1 if index_tip[1] < index_pip[1] else 0)
        fingers.append(1 if middle_tip[1] < middle_pip[1] else 0)
        fingers.append(1 if ring_tip[1] < ring_pip[1] else 0)
        fingers.append(1 if pinky_tip[1] < pinky_pip[1] else 0)
        
        # Simple gesture mapping
        total_fingers = sum(fingers)
        
        if total_fingers == 0:
            return "A"
        elif total_fingers == 1 and fingers[1] == 1:
            return "D"
        elif total_fingers == 2 and fingers[1] == 1 and fingers[2] == 1:
            return "V"
        elif total_fingers == 3 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
            return "W"
        elif total_fingers == 5:
            return "B"
        else:
            return "Unknown"

    def predict(self, test_image):
        """Main prediction function"""
        if self.model is None:
            # Use simple landmark-based prediction if no model
            return self.predict_simple(self.pts)
        
        try:
            # Prepare image for model
            white = test_image.copy()
            white = cv2.resize(white, (400, 400))
            white = white.reshape(1, 400, 400, 3)
            white = white.astype('float32') / 255.0
            
            # Get prediction
            prob = np.array(self.model.predict(white)[0], dtype="float32")
            ch1 = np.argmax(prob, axis=0)
            
            # Map prediction to character (simplified)
            alphabet = list(ascii_uppercase)
            if ch1 < len(alphabet):
                predicted_char = alphabet[ch1]
            else:
                predicted_char = "Unknown"
                
            return predicted_char
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self.predict_simple(self.pts)

    def process_prediction(self, predicted_char):
        """Process the predicted character and update the sentence"""
        if predicted_char == "Unknown" or predicted_char == "Invalid":
            return
            
        # Add character to string
        if predicted_char != self.prev_char:
            if predicted_char == "SPACE":
                self.str += " "
            elif predicted_char == "BACKSPACE":
                if len(self.str) > 0:
                    self.str = self.str[:-1]
            else:
                self.str += predicted_char
                
        self.prev_char = predicted_char
        self.current_symbol = predicted_char
        
        # Update word suggestions
        self.update_suggestions()

    def update_suggestions(self):
        """Update word suggestions based on current word"""
        if len(self.str.strip()) == 0:
            self.word1 = self.word2 = self.word3 = self.word4 = ""
            return
            
        # Get current word
        words = self.str.split()
        if words:
            current_word = words[-1].lower()
            self.word = current_word
            
            if len(current_word.strip()) > 0:
                try:
                    suggestions = ddd.suggest(current_word)
                    self.word1 = suggestions[0] if len(suggestions) > 0 else ""
                    self.word2 = suggestions[1] if len(suggestions) > 1 else ""
                    self.word3 = suggestions[2] if len(suggestions) > 2 else ""
                    self.word4 = suggestions[3] if len(suggestions) > 3 else ""
                except:
                    self.word1 = self.word2 = self.word3 = self.word4 = ""
            else:
                self.word1 = self.word2 = self.word3 = self.word4 = ""

    def draw_landmarks(self, image, landmarks):
        """Draw hand landmarks on image"""
        if not landmarks or len(landmarks) < 21:
            return image
            
        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        # Draw lines
        for connection in connections:
            start_point = (landmarks[connection[0]][0], landmarks[connection[0]][1])
            end_point = (landmarks[connection[1]][0], landmarks[connection[1]][1])
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)
        
        # Draw points
        for i, landmark in enumerate(landmarks):
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 255), -1)
            cv2.putText(image, str(i), (landmark[0] + 10, landmark[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        return image

async def handle_websocket(websocket):
    app_instance = Application()
    print("Client connected")
    
    try:
        async for message in websocket:
            try:
                # Parse JSON message
                data = json.loads(message)
                image_data = data['image'].split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                
                if image is None:
                    await websocket.send(json.dumps({
                        'result': app_instance.str,
                        'symbol': 'Error loading image',
                        'suggestions': [app_instance.word1, app_instance.word2, app_instance.word3, app_instance.word4]
                    }))
                    continue
                
                # Create white background for drawing
                white = app_instance.create_white_background()
                
                # Detect hands
                hands, img_with_hands = hd2.findHands(image, draw=True, flipType=True)
                
                if hands:
                    hand = hands[0]
                    landmarks = hand["lmList"]
                    app_instance.pts = landmarks
                    
                    # Draw landmarks on white background
                    # Scale landmarks to fit 400x400 canvas
                    h, w = image.shape[:2]
                    scaled_landmarks = []
                    for lm in landmarks:
                        x = int((lm[0] / w) * 400)
                        y = int((lm[1] / h) * 400)
                        scaled_landmarks.append([x, y])
                    
                    white_with_landmarks = app_instance.draw_landmarks(white, scaled_landmarks)
                    
                    # Make prediction
                    predicted_char = app_instance.predict(white_with_landmarks)
                    app_instance.process_prediction(predicted_char)
                    
                    response = {
                        'result': app_instance.str,
                        'symbol': app_instance.current_symbol,
                        'suggestions': [app_instance.word1, app_instance.word2, app_instance.word3, app_instance.word4]
                    }
                else:
                    response = {
                        'result': app_instance.str,
                        'symbol': 'No hand detected',
                        'suggestions': [app_instance.word1, app_instance.word2, app_instance.word3, app_instance.word4]
                    }
                
                await websocket.send(json.dumps(response))
                
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    'result': app_instance.str,
                    'symbol': 'Invalid JSON',
                    'suggestions': [app_instance.word1, app_instance.word2, app_instance.word3, app_instance.word4]
                }))
            except Exception as e:
                print(f"Processing error: {e}")
                await websocket.send(json.dumps({
                    'result': app_instance.str,
                    'symbol': f'Error: {str(e)}',
                    'suggestions': [app_instance.word1, app_instance.word2, app_instance.word3, app_instance.word4]
                }))
                
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("Connection closed")

async def main():
    print("Starting Sign Language Detection Server...")
    server = await websockets.serve(handle_websocket, "localhost", 8765)
    print("WebSocket server started on ws://localhost:8765")
    print("Waiting for client connections...")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
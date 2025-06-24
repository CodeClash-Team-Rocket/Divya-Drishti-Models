
```markdown


🧠 Divya Drishti: AI-Powered Assistance for the Visually & Hearing Impaired

Empowering visually and hearing-impaired individuals with real-time obstacle detection and sign language recognition using AI.

🎯 Project Overview
Divya Drishti (Sanskrit for Divine Vision) is a dual-module assistive technology suite designed to enhance independence for visually and hearing-impaired individuals. Powered by YOLOv5 for obstacle detection and TensorFlow/Keras for sign language recognition, it delivers real-time audio feedback for navigation and communication.

🦯 Obstacle Detection: Detects obstacles in real-time and provides voice-guided navigation instructions.
✋ Sign Language Detection: Recognizes A-Z hand signs, constructs sentences, and converts them to speech.
⚡ Accessible & Affordable: Built with open-source tools for widespread adoption.


🧭 Navigation

👁️ Obstacle Detection
✋ Sign Language Detection
⚙️ Setup Instructions
📈 Future Enhancements
🎥 Demo Videos
🤝 Team & Credits


👁️ Obstacle Detection

Real-time AI system that detects obstacles and provides audio navigation guidance like “Move Left,” “Go Right,” or “Path Clear.”

💡 Features

🎥 Live Video Feed: Captures and analyzes video via webcam.
🧠 YOLOv5 Model: Uses pretrained YOLOv5 (s/m variant) for robust object detection.
🔊 Voice Guidance: Offline Text-to-Speech (pyttsx3) for directional instructions.
🔍 Obstacle Positioning: Determines obstacle location (left/center/right).
⚙️ REST API: Upload images for detection results and navigation advice.
🚀 GPU Support: CUDA acceleration for faster processing.



📸 Sample Output
Detection: Person
Direction: Move Right
Voice Output: "Move right to avoid person"

Visual Output:

✅ Bounding boxes around detected objects
🏷️ Object class and confidence labels
➡️ Directional arrows for navigation guidance

🚀 Run Locally
git clone https://github.com/CodeClash-Team-Rocket/Divya-Drishti-Models.git
cd Divya-Drishti-Models/Obstacle\ Detection
pip install -r requirements.txt
python main.py



✋ Sign Language Detection

Real-time hand sign recognition (A-Z) with sentence construction and speech output for seamless communication.

🔍 Features

✋ Live Sign Recognition: Detects A-Z hand gestures using webcam.
📃 Sentence Construction: Builds meaningful sentences from sequential signs.
🔈 Speech Synthesis: Converts sentences to audio using pyttsx3.
🧠 Smart Suggestions: PyEnchant-based spell-checker for accurate word output.
🛑 Emergency SOS: Optional button for critical alerts.



📸 Sample Output
Signs Detected: H-E-L-L-O
Sentence: Hello
Voice Output: "Hello"

🚀 Run Locally
cd Divya-Drishti-Models/Sign\ Language\ Detection
pip install -r requirements.txt
python final_preds.py


⚙️ Setup Instructions

Clone the Repository:
git clone https://github.com/CodeClash-Team-Rocket/Divya-Drishti-Models.git


Install Dependencies:
cd Divya-Drishti-Models
pip install -r requirements.txt


Run Modules:

Obstacle Detection: cd Obstacle\ Detection && python main.py
Sign Language Detection: cd Sign\ Language\ Detection && python final_preds.py
API Server: cd Obstacle\ Detection && uvicorn api:app --reload



Requirements:

Python 3.8+ (Obstacle Detection), Python 3.10 (Sign Language Detection)
Webcam for real-time detection
Optional: CUDA-enabled GPU for faster processing


📈 Future Enhancements

🌍 GPS Integration: Enable outdoor navigation with path planning.
🗣️ Multilingual TTS: Support for regional languages.
📱 Mobile App: Deploy API to Android for portable access.
📏 Depth Estimation: Calculate obstacle distances for precise guidance.
.



## 👁️ Obstacle Detection

> **Real-time AI system that detects obstacles and provides audio navigation guidance like “Move Left,” “Go Right,” or “Path Clear.”**

### 💡 Features

- 🎥 **Live Video Feed**: Captures and analyzes video via webcam.
- 🧠 **YOLOv5 Model**: Uses pretrained YOLOv5 (s/m variant) for robust object detection.
- 🔊 **Voice Guidance**: Offline Text-to-Speech (pyttsx3) for directional instructions.
- 🔍 **Obstacle Positioning**: Determines obstacle location (left/center/right).
- ⚙️ **REST API**: Upload images for detection results and navigation advice.
- 🚀 **GPU Support**: CUDA acceleration for faster processing.

### 📦 Tech Stack

| Component       | Technology                     |
|-----------------|-------------------------------|
| Model           | YOLOv5 (PyTorch)              |
| Voice Engine    | pyttsx3                      |
| Backend API     | FastAPI, Uvicorn             |
| Libraries       | OpenCV, Pillow, NumPy, Pandas|
| Deployment      | Local / Cloud-ready          |

### 📸 Sample Output

```text
Detection: Person
Direction: Move Right
Voice Output: "Move right to avoid person"
```

**Visual Output**:
- ✅ Bounding boxes around detected objects
- 🏷️ Object class and confidence labels
- ➡️ Directional arrows for navigation guidance

### 🚀 Run Locally

```bash
git clone https://github.com/CodeClash-Team-Rocket/Divya-Drishti-Models.git
cd Divya-Drishti-Models/Obstacle\ Detection
pip install -r requirements.txt
python main.py
```



## ✋ Sign Language Detection

> **Real-time hand sign recognition (A-Z) with sentence construction and speech output for seamless communication.**

### 🔍 Features

- ✋ **Live Sign Recognition**: Detects A-Z hand gestures using webcam.
- 📃 **Sentence Construction**: Builds meaningful sentences from sequential signs.
- 🔈 **Speech Synthesis**: Converts sentences to audio using pyttsx3.
- 🧠 **Smart Suggestions**: PyEnchant-based spell-checker for accurate word output.
- 🛑 **Emergency SOS**: Optional button for critical alerts.

### ⚙️ Tech Stack

| Component       | Technology                     |
|-----------------|-------------------------------|
| Model           | TensorFlow/Keras (CNN)        |
| Hand Detection  | CVZone, OpenCV               |
| GUI             | Tkinter                      |
| Voice Engine    | pyttsx3                      |
| Spell Check     | PyEnchant                    |

### 📸 Sample Output

```text
Signs Detected: H-E-L-L-O
Sentence: Hello
Voice Output: "Hello"
```

### 🚀 Run Locally

```bash
cd Divya-Drishti-Models/Sign\ Language\ Detection
pip install -r requirements.txt
python final_preds.py
```

---

## ⚙️ Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/CodeClash-Team-Rocket/Divya-Drishti-Models.git
   ```

2. **Install Dependencies**:
   ```bash
   cd Divya-Drishti-Models
   pip install -r requirements.txt
   ```

3. **Run Modules**:
   - Obstacle Detection: `cd Obstacle\ Detection && python main.py`
   - Sign Language Detection: `cd Sign\ Language\ Detection && python final_preds.py`
   

**Requirements**:
- Python 3.8+ (Obstacle Detection), Python 3.10 (Sign Language Detection)
- Webcam for real-time detection
- Optional: CUDA-enabled GPU for faster processing

---

## 📈 Future Enhancements

- 🌍 **GPS Integration**: Enable outdoor navigation with path planning.
- 🗣️ **Multilingual TTS**: Support for regional languages.
- 📱 **Mobile App**: Deploy API to Android for portable access.
- 📏 **Depth Estimation**: Calculate obstacle distances for precise guidance.
- 🤖 **Wearable Device**: Raspberry Pi-based compact solution for portability.

---

## 🎥 Demo Videos

| Module                     | Demo Link                                      |
|----------------------------|-----------------------------------------------|
| 🦯 Obstacle Detection      | [Watch Now](https://youtu.be/phupkcV_204)    |
| ✋ Sign Language Detection  | [Watch Now](https://youtu.be/NAT0WzSdmXQ)    |

---





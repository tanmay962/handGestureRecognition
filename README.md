# Gesture Detection v1.0

Real-time hand gesture recognition using MediaPipe Holistic (hands + face + body) with IoT architecture.

## Features
- MediaPipe Holistic tracking — 41-feature vector (hands + face + body)
- MLP neural network for static gestures (A-Z, 0-9)
- LSTM with temporal attention for dynamic gestures (Hello, Thank You...)
- Adaptive NLP with personal corpus learning
- IoT mode — phone as sensor device via MQTT
- PWA — installable, works on mobile

## Run Locally
```bash
cd backend
pip install -r requirements.txt
python3 main.py
# Open http://localhost:8000
```

## IoT Mode
Open `http://localhost:8000/sensor` on your phone to use it as a remote sensor.

## Deploy to Hugging Face Spaces
1. Create a new Space (Docker SDK)
2. Push this repository
3. App runs at `https://your-space.hf.space`

## Architecture
```
Phone (sensor.html) → MediaPipe Holistic → MQTT → Backend → Prediction → Display
```

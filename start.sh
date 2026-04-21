#!/bin/bash
echo "==================================="
echo " Gesture Detection v1.0"
echo " MediaPipe Holistic | 41 features"
echo "==================================="
echo ""
cd "$(dirname "$0")/backend"
pip install -r requirements.txt -q 2>/dev/null
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)" 2>/dev/null
python3 main.py

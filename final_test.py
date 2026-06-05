import os
import sys

# Project root
ROOT = os.getcwd()
sys.path.append(ROOT)

from backend.audio.fusion_predict import predict_audio_emotion

# Dummy wav creation or use existing if any? 
# I'll just try to predict any file.
test_file = r"C:\Users\M.DHAARANI\Downloads\AMCEF\backend\audio\tmpxa9wqa0w.wav" # from log

if os.path.exists(test_file):
    print(f"Testing prediction on {test_file}...")
    res = predict_audio_emotion(test_file)
    print(f"Prediction Result: {res}")
else:
    print("No test file found to run prediction test.")

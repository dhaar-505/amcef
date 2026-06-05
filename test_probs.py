import os
import sys
import numpy as np

# Project root
ROOT = os.getcwd()
sys.path.append(ROOT)

from backend.audio.fusion_predict import fusion_prediction, MODELS_DIR, get_models, COMMON_EMOTIONS

def test_file(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    print(f"\n--- Testing: {os.path.basename(file_path)} ---")
    try:
        emotion, confidence, probs = fusion_prediction(file_path)
        print(f"Final Emotion: {emotion} ({confidence:.4f})")
        
        # Breakdown
        m = get_models()
        from backend.audio.fusion_predict import map_to_common, emodb_labels, emovo_labels, shemo_labels, extract_features
        
        vec, mel = extract_features(file_path)
        p1 = m['emodb'].predict(vec, verbose=0)
        p2 = m['emovo'].predict(vec, verbose=0)
        p3 = m['shemo'].predict(mel, verbose=0)
        
        print(f"EMODB raw top: {emodb_labels[np.argmax(p1[0])]} ({np.max(p1[0]):.4f})")
        print(f"EMOVO raw top: {emovo_labels[np.argmax(p2[0])]} ({np.max(p2[0]):.4f})")
        print(f"SHEMO raw top: {shemo_labels[np.argmax(p3[0])]} ({np.max(p3[0]):.4f})")
        
        print("Common Probs:")
        for idx, emo in enumerate(COMMON_EMOTIONS):
            print(f"  {emo:10}: {probs[idx]:.4f}")
            
    except Exception as e:
        import traceback
        traceback.print_exc()

# Create a small dummy wav if needed, but better use a real one if exists
# Try to find any wav in the assets or current dir
import glob
wavs = glob.glob("**/*.wav", recursive=True)
if wavs:
    test_file(wavs[0])
    if len(wavs) > 1:
        test_file(wavs[1])
else:
    print("No wav files found in the project directory.")

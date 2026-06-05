# AMCEF: Adaptive Multi-Dataset Confidence-based Emotion Fusion System

## Overview
AMCEF is a robust emotion recognition system that leverages multiple independently trained speech models to provide accurate and culturally diverse emotion predictions.

## Documentation
For a detailed technical explanation of the architecture, feature extraction pipeline, and fusion algorithm, please refer to the primary documentation:

👉 **[walkthrough.md](./walkthrough.md)**

## Core Components
- **Audio-Only Inference**: Focuses exclusively on high-fidelity speech emotion recognition.
- **Multi-Model Fusion**: Combines insights from EMO-DB, EMOVO, and SHEMO datasets.
- **Adaptive Confidence**: Uses weighted probabilities to ensure the most reliable predictions.

## Quick Start
1. Ensure `requirements.txt` dependencies are installed.
2. Place audio files for prediction in the input directory.
3. Run `fusion_predict.py` to obtain results.

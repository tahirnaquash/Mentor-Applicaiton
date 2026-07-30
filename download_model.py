# download_model.py
import os
import shutil

# 1. Clear any broken partial files if they exist
if os.path.exists('clinical_model_v1'):
    try:
        shutil.rmtree('clinical_model_v1')
    except Exception:
        pass

print("Downloading model from Hugging Face online hub...")
from sentence_transformers import SentenceTransformer

# Load model cleanly into memory
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Saving model weights to clean folder directory...")
# Saving to a fresh, unlocked path targets
model.save('clinical_model_v1')

print("[SUCCESS] Model files successfully saved to 'clinical_model_v1'!")
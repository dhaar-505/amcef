import os
import tensorflow as tf

# Define the patched Dense class
original_dense = tf.keras.layers.Dense
class PatchDense(original_dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

# Patch both public and known internal paths
tf.keras.layers.Dense = PatchDense
try:
    import keras.src.layers.core.dense
    keras.src.layers.core.dense.Dense = PatchDense
except:
    pass

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope

MODELS_DIR = r"c:\Users\M.DHAARANI\Downloads\AMCEF\assets\models"

try:
    with custom_object_scope({"Dense": PatchDense}):
        print("Testing EMODB model...")
        m1 = load_model(os.path.join(MODELS_DIR, "emodb_model.keras"))
        print("EMODB OK")
    
    with custom_object_scope({"Dense": PatchDense}):
        print("Testing EMOVO model...")
        m2 = load_model(os.path.join(MODELS_DIR, "emovo_model.keras"))
        print("EMOVO OK")
    
    with custom_object_scope({"Dense": PatchDense}):
        print("Testing SHEMO model...")
        m3 = load_model(os.path.join(MODELS_DIR, "shemo_model.keras"))
        print("SHEMO OK")
    
except Exception as e:
    import traceback
    traceback.print_exc()

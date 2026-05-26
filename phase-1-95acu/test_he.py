import utils
import os
import tensorflow as tf
from tensorflow import keras

print("Loading model...")
model = keras.models.load_model('phase1_1dcnn_final.keras')

print("Extracting HE matrices...")
try:
    matrices = utils.extract_he_matrices(model)
    print("Success. Shapes:")
    for k, v in matrices.items():
        print(f"  {k}: {len(v)} x {len(v[0]) if isinstance(v[0], list) else 1}")
except Exception as e:
    import traceback
    traceback.print_exc()

print("DONE.")

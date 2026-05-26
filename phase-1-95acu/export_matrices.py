import pickle
from tensorflow import keras
import utils

print("Loading model...")
model = keras.models.load_model('phase1_1dcnn_final.keras')

print("Extracting HE matrices (this might take a minute)...")
he_matrices = utils.extract_he_matrices(model)

print("Saving to he_matrices.pkl...")
with open('he_matrices.pkl', 'wb') as f:
    pickle.dump(he_matrices, f)

print("Done! File size:")
import os
print(os.path.getsize('he_matrices.pkl') / 1024 / 1024, "MB")

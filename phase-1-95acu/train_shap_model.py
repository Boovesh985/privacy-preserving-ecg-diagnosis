import os, json, warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
tf.random.set_seed(42)

from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.utils import to_categorical

X_test_feat_scaled = np.load('X_test_feat_scaled.npy')
y_test = np.load('y_test.npy').astype(int)

N_FEATURES = 45
num_classes = 5

def build_feat_cnn(input_length=45, num_classes=5):
    inp = layers.Input(shape=(input_length, 1), name='feat_input')
    x   = layers.Conv1D(32, 5, padding='same', activation='relu', name='conv1')(inp)
    x   = layers.MaxPooling1D(2, name='maxpool1')(x)
    x   = layers.Conv1D(64, 3, padding='same', activation='relu', name='conv2')(x)
    x   = layers.MaxPooling1D(2, name='maxpool2')(x)
    x   = layers.Conv1D(128, 3, padding='same', activation='relu', name='conv3')(x)
    x   = layers.MaxPooling1D(2, name='maxpool3')(x)
    x   = layers.Flatten(name='flatten')(x)
    x   = layers.Dense(128, activation='relu', name='dense1')(x)
    x   = layers.Dropout(0.5, name='dropout')(x)
    out = layers.Dense(num_classes, activation='softmax', name='output')(x)
    return keras.Model(inp, out, name='ECG_FEAT_CNN_SHAP')

cnn_feat = build_feat_cnn(N_FEATURES, num_classes)
cnn_feat.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy', # Changed from tf.keras.losses...
    metrics=['accuracy']
)

X_train_feat_cnn = X_test_feat_scaled.reshape(-1, N_FEATURES, 1)
y_train_cat = to_categorical(y_test, 5)

from sklearn.utils.class_weight import compute_class_weight
cw_vals = compute_class_weight('balanced', classes=np.arange(5), y=y_test)
class_weights = dict(enumerate(cw_vals))

print("Training SHAP feature proxy model...")
hist_feat = cnn_feat.fit(
    X_train_feat_cnn, y_train_cat,
    batch_size=64, epochs=15,
    class_weight=class_weights,
    verbose=1
)

cnn_feat.save('cnn_feat_best.keras')
print("Model saved to cnn_feat_best.keras")

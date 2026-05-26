import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import json
import time
import numpy as np
import tenseal as ts
import keras
import shap

N_CLASSES = 5
SIGNAL_LEN = 187
POLY_MOD_DEGREE = 8192
COEFF_MOD_BITS = [60, 40, 40, 60]
SCALE = 2 ** 40

def create_ckks_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MOD_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BITS
    )
    context.global_scale = SCALE
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context.serialize(save_secret_key=True)

def load_cnn_model():
    """Load the 1D-CNN model for fast plaintext inference."""
    return keras.models.load_model('phase1_1dcnn_final.keras')

def run_encrypted_inference_demo(x_flat, ctx_bytes):
    """Demonstrate real CKKS encryption/decryption cycle + fast CNN inference.
    
    The encryption and decryption are REAL (proving CKKS works on ECG data).
    The CNN inference uses plaintext for speed on free-tier hardware.
    
    Paper reports: 110ms encrypt + 240ms cloud infer + 70ms decrypt = 420ms total
    on dedicated server hardware. On HF free tier (2 vCPU), full HE matmul on 
    5984x5952 matrices takes 10+ minutes, making live demos impractical.
    """
    timings = {}
    
    # ---- REAL: Client encrypts ECG signal using CKKS ----
    t0 = time.perf_counter()
    ctx = ts.context_from(ctx_bytes)
    enc_vector = ts.ckks_vector(ctx, x_flat.tolist())
    ciphertext_bytes = enc_vector.serialize()
    timings['encrypt_ms'] = (time.perf_counter() - t0) * 1000
    timings['ciphertext_size_kb'] = len(ciphertext_bytes) / 1024
    
    # ---- CNN inference (plaintext for demo speed) ----
    t1 = time.perf_counter()
    model = load_cnn_model()
    x_input = x_flat.reshape(1, -1, 1)
    logits = model.predict(x_input, verbose=0)[0]
    timings['infer_ms'] = (time.perf_counter() - t1) * 1000
    
    # ---- REAL: Client decrypts (proves round-trip works) ----
    t2 = time.perf_counter()
    decrypted = np.array(enc_vector.decrypt())
    timings['decrypt_ms'] = (time.perf_counter() - t2) * 1000
    
    # Verify encryption fidelity
    timings['encryption_error'] = float(np.max(np.abs(decrypted[:len(x_flat)] - x_flat)))
    timings['total_ms'] = timings['encrypt_ms'] + timings['infer_ms'] + timings['decrypt_ms']
    
    return logits, timings, ciphertext_bytes

def get_shap_explainer(model_path, background_path):
    model = keras.models.load_model(model_path)
    shap_bg = np.load(background_path)
    
    def predict_fn(x):
        if x.ndim == 2:
            x = x.reshape(-1, x.shape[1], 1)
        return model.predict(x, verbose=0)
    
    explainer = shap.KernelExplainer(predict_fn, shap_bg[:50])
    return explainer

def get_shap_values(explainer, x_feat, num_classes=5):
    if x_feat.ndim == 1:
        x_feat = x_feat.reshape(1, -1)
    elif x_feat.ndim == 3:
        x_feat = x_feat.squeeze(-1)
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_vals = explainer.shap_values(x_feat, nsamples=100)
        
    if isinstance(shap_vals, list):
        return np.array(shap_vals)
    return shap_vals

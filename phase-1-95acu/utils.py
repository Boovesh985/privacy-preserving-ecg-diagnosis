import os
import json
import time
import pickle
import numpy as np
import tenseal as ts
import tensorflow as tf
from tensorflow import keras
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

def conv1d_to_matrix(kernel, input_len, padding='same'):
    k_size, in_ch, out_ch = kernel.shape
    pad = k_size // 2 if padding == 'same' else 0
    W = np.zeros((input_len * in_ch, input_len * out_ch), dtype=np.float64)
    # Vectorized block assignment eliminates 2M python loops
    for ko in range(k_size):
        offset = ko - pad
        if offset >= 0:
            t_start, t_end = 0, input_len - offset
        else:
            t_start, t_end = -offset, input_len
        ker_matrix = kernel[ko, :, :]
        for t in range(t_start, t_end):
            t_in = t + offset
            W[t_in*in_ch : (t_in+1)*in_ch, t*out_ch : (t+1)*out_ch] += ker_matrix
    return W

def avgpool_matrix(input_len, channels, pool_size=2):
    output_len = input_len // pool_size
    W = np.zeros((input_len * channels, output_len * channels), dtype=np.float64)
    # Vectorized assignment
    for t_out in range(output_len):
        t_in_start = t_out * pool_size
        for p in range(pool_size):
            t_in = t_in_start + p
            row_idx = np.arange(t_in*channels, (t_in+1)*channels)
            col_idx = np.arange(t_out*channels, (t_out+1)*channels)
            W[row_idx, col_idx] = 1.0 / pool_size
    return W

def extract_he_matrices(model):
    def get_layer_weights(model, *names):
        for name in names:
            try:
                return model.get_layer(name).get_weights()
            except ValueError:
                continue
        raise ValueError(f"None of {names} found.")

    c1w, c1b = get_layer_weights(model, 'conv1', 'conv1d', 'conv1d_1')
    c2w, c2b = get_layer_weights(model, 'conv2', 'conv1d_1', 'conv1d_2')
    c3w, c3b = get_layer_weights(model, 'conv3', 'conv1d_2', 'conv1d_3')
    d1w, d1b = get_layer_weights(model, 'dense1', 'dense', 'dense_1')
    ow, ob = get_layer_weights(model, 'output', 'dense_1', 'dense1', 'dense_2')

    k1, _, ch1 = c1w.shape
    k2, _, ch2 = c2w.shape
    k3, _, ch3 = c3w.shape

    L1 = SIGNAL_LEN
    L1p = L1 // 2
    L2p = L1p // 2
    L3p = L2p // 2
    flatten_dim = L3p * ch3

    W_conv1 = conv1d_to_matrix(c1w, L1)
    b_conv1 = np.tile(c1b, L1).astype(np.float64)
    W_pool1 = avgpool_matrix(L1, ch1)

    W_conv2 = conv1d_to_matrix(c2w, L1p)
    b_conv2 = np.tile(c2b, L1p).astype(np.float64)
    W_pool2 = avgpool_matrix(L1p, ch2)

    W_conv3 = conv1d_to_matrix(c3w, L2p)
    b_conv3 = np.tile(c3b, L2p).astype(np.float64)
    W_pool3 = avgpool_matrix(L2p, ch3)

    W_dense1 = d1w.astype(np.float64)
    b_dense1 = d1b.astype(np.float64)
    W_out = ow.astype(np.float64)
    b_out = ob.astype(np.float64)

    W_fused_12 = W_pool1 @ W_conv2
    W_fused_tail = W_pool2 @ W_conv3 @ W_pool3 @ W_dense1 @ W_out
    b_tail = (b_conv3 @ W_pool3 @ W_dense1 @ W_out) + (b_dense1 @ W_out) + b_out

    return {
        'W_conv1': W_conv1, 'b_conv1': b_conv1,
        'W_fused_12': W_fused_12, 'b_conv2': b_conv2,
        'W_fused_tail': W_fused_tail, 'b_tail': b_tail
    }

def ckks_poly_relu(enc):
    return enc * enc

def run_encrypted_inference(x_flat, ctx, he_matrices):
    mat = he_matrices
    timings = {}

    # Eq (3.2): Client encrypts
    t0 = time.perf_counter()
    enc = ts.ckks_vector(ctx, x_flat.tolist())
    timings['encrypt_ms'] = (time.perf_counter() - t0) * 1000

    # Eq (3.3): Cloud inference
    t1 = time.perf_counter()
    enc = enc.mm(mat['W_conv1'].tolist()) + mat['b_conv1'].tolist()
    enc = ckks_poly_relu(enc)
    enc = enc.mm(mat['W_fused_12'].tolist()) + mat['b_conv2'].tolist()
    enc = ckks_poly_relu(enc)
    enc = enc.mm(mat['W_fused_tail'].tolist()) + mat['b_tail'].tolist()
    timings['infer_ms'] = (time.perf_counter() - t1) * 1000

    # Eq (3.4): Client decrypts
    t2 = time.perf_counter()
    out = np.array(enc.decrypt()[:N_CLASSES], dtype=np.float64)
    timings['decrypt_ms'] = (time.perf_counter() - t2) * 1000
    timings['total_ms'] = timings['encrypt_ms'] + timings['infer_ms'] + timings['decrypt_ms']

    return out, timings

def get_shap_explainer(model_path, background_path):
    model = keras.models.load_model(model_path)
    shap_bg = np.load(background_path)
    # DeepExplainer expects tensor background
    explainer = shap.DeepExplainer(model, shap_bg[:100])
    return explainer

def get_shap_values(explainer, x_feat, num_classes=5):
    # x_feat shape (1, 45) -> needs (1, 45, 1) if CNN expects it
    if x_feat.ndim == 2:
        x_feat = np.expand_dims(x_feat, -1)
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_vals = explainer.shap_values(x_feat)
        
    if isinstance(shap_vals, list):
        return np.array(shap_vals)
    return shap_vals

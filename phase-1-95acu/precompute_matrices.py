"""Pre-compute HE matrices locally and save as pickle.
This avoids the 10+ minute computation on HF Spaces free tier.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pickle
import numpy as np
import keras

# --- Same matrix functions from utils.py ---
SIGNAL_LEN = 187

def conv1d_to_matrix(kernel, input_len, padding='same'):
    k_size, in_ch, out_ch = kernel.shape
    pad = k_size // 2 if padding == 'same' else 0
    W = np.zeros((input_len * in_ch, input_len * out_ch), dtype=np.float64)
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
    for t_out in range(output_len):
        t_in_start = t_out * pool_size
        for p in range(pool_size):
            t_in = t_in_start + p
            row_idx = np.arange(t_in*channels, (t_in+1)*channels)
            col_idx = np.arange(t_out*channels, (t_out+1)*channels)
            W[row_idx, col_idx] = 1.0 / pool_size
    return W

def get_layer_weights(model, *names):
    for name in names:
        try:
            return model.get_layer(name).get_weights()
        except ValueError:
            continue
    raise ValueError(f"None of {names} found.")

print("Loading model...")
model = keras.models.load_model('phase1_1dcnn_final.keras')

print("Extracting layer weights...")
c1w, c1b = get_layer_weights(model, 'conv1', 'conv1d', 'conv1d_1')
c2w, c2b = get_layer_weights(model, 'conv2', 'conv1d_1', 'conv1d_2')
c3w, c3b = get_layer_weights(model, 'conv3', 'conv1d_2', 'conv1d_3')
d1w, d1b = get_layer_weights(model, 'dense1', 'dense', 'dense_1')
ow, ob   = get_layer_weights(model, 'output', 'dense_1', 'dense1', 'dense_2')

k1, _, ch1 = c1w.shape
k2, _, ch2 = c2w.shape
k3, _, ch3 = c3w.shape

L1 = SIGNAL_LEN
L1p = L1 // 2   # 93
L2p = L1p // 2  # 46
L3p = L2p // 2  # 23

print(f"Building conv1 matrix ({L1}x1 -> {L1}x{ch1})...")
W_conv1 = conv1d_to_matrix(c1w, L1)
b_conv1 = np.tile(c1b, L1).astype(np.float64)
W_pool1 = avgpool_matrix(L1, ch1)

print(f"Building conv2 matrix ({L1p}x{ch1} -> {L1p}x{ch2})...")
W_conv2 = conv1d_to_matrix(c2w, L1p)
b_conv2 = np.tile(c2b, L1p).astype(np.float64)
W_pool2 = avgpool_matrix(L1p, ch2)

print(f"Building conv3 matrix ({L2p}x{ch2} -> {L2p}x{ch3})...")
W_conv3 = conv1d_to_matrix(c3w, L2p)
b_conv3 = np.tile(c3b, L2p).astype(np.float64)
W_pool3 = avgpool_matrix(L2p, ch3)

W_dense1 = d1w.astype(np.float64)
b_dense1 = d1b.astype(np.float64)
W_out = ow.astype(np.float64)
b_out = ob.astype(np.float64)

print("Fusing matrices...")
W_fused_12 = W_pool1 @ W_conv2
W_fused_tail = W_pool2 @ W_conv3 @ W_pool3 @ W_dense1 @ W_out
b_tail = (b_conv3 @ W_pool3 @ W_dense1 @ W_out) + (b_dense1 @ W_out) + b_out

# Also pre-compute .tolist() versions for TenSEAL (avoids slow conversion at runtime)
print("Converting to lists for TenSEAL (this is the slowest part)...")
he_matrices = {
    'W_conv1': W_conv1.tolist(),
    'b_conv1': b_conv1.tolist(),
    'W_fused_12': W_fused_12.tolist(),
    'b_conv2': b_conv2.tolist(),
    'W_fused_tail': W_fused_tail.tolist(),
    'b_tail': b_tail.tolist(),
}

output_path = os.path.join(os.path.dirname(__file__), '..', 'hf-deploy', 'he_matrices.pkl')
print(f"Saving to {output_path}...")
with open(output_path, 'wb') as f:
    pickle.dump(he_matrices, f, protocol=pickle.HIGHEST_PROTOCOL)

size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"Done! File size: {size_mb:.2f} MB")
print(f"\nUpload he_matrices.pkl to your HF Space.")

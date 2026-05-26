import numpy as np

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
        ker_matrix = kernel[ko, :, :] # (in_ch, out_ch)
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

def extract_he_matrices_mock():
    import time
    # Model shapes mock
    c1w = np.random.randn(5, 1, 32)
    L1 = 187
    ch1 = 32
    
    W_conv1 = conv1d_to_matrix(c1w, L1)
    W_pool1 = avgpool_matrix(L1, ch1)
    
    t0 = time.time()
    W_fused = np.random.randn(5984, 5952)
    print("Testing tolist() memory allocation for 35M floats...")
    fused_list = W_fused.tolist()
    t1 = time.time()
    print(f"tolist() took {t1 - t0:.2f} seconds")

extract_he_matrices_mock()

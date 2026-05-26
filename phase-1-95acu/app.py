import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import json
import base64
import sys

# Ensure utils is loaded correctly
import utils

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Secure Cloud-Based ECG Analysis",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS STYLING ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .stButton>button {
        background-color: #2e6c80;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1a4a5a;
        transform: translateY(-2px);
    }
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2e6c80;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
CLASS_NAMES = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']

# --- RESOURCE LOADING ---
@st.cache_resource(show_spinner="Loading AI Models and Initializing Homomorphic Context...")
def load_all_resources():
    model_cnn = keras.models.load_model('phase1_1dcnn_final.keras')
    X_test_cnn = np.load('X_test_cnn.npy')
    X_test_feat = np.load('X_test_feat_scaled.npy')
    y_test = np.load('y_test.npy').astype(int)
    
    with open('feature_names.json', 'r') as f:
        feature_names = json.load(f)
        
    ctx = utils.create_ckks_context()
    he_matrices = utils.extract_he_matrices(model_cnn)
    
    explainer = utils.get_shap_explainer('cnn_feat_best.keras', 'shap_background_feat.npy')
    
    return model_cnn, X_test_cnn, X_test_feat, y_test, feature_names, he_matrices, explainer

try:
    model_cnn, X_test_cnn, X_test_feat, y_test, feature_names, he_matrices, explainer = load_all_resources()
    if 'ckks_ctx_bytes' not in st.session_state:
        st.session_state['ckks_ctx_bytes'] = utils.create_ckks_context()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# --- SIDEBAR: PATIENT SELECTION ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Ecg_normal.svg/1024px-Ecg_normal.svg.png", use_container_width=True)
    st.title("Patient Control Panel")
    
    selected_class = st.selectbox(
        "Filter by True Condition",
        options=["All"] + CLASS_NAMES
    )
    
    if selected_class == "All":
        available_indices = np.arange(len(y_test))
    else:
        class_idx = CLASS_NAMES.index(selected_class)
        available_indices = np.where(y_test == class_idx)[0]
    
    if len(available_indices) == 0:
        st.warning("No patients found with the selected condition.")
        st.stop()
        
    patient_id = st.selectbox(
        "Select Patient ID",
        options=available_indices[:100], # limit to 100 for UI responsiveness
        format_func=lambda x: f"Patient #{x}"
    )
    
    x_patient_raw = X_test_cnn[patient_id].squeeze(-1)
    x_patient_feat = X_test_feat[patient_id]
    y_true_label = y_test[patient_id]
    y_true_name = CLASS_NAMES[y_true_label]

    st.markdown("---")
    st.markdown(f"**Ground Truth:** `{y_true_name}`")
    st.markdown(f"**Raw Signal Length:** `187 samples`")
    st.markdown(f"**Extracted Features:** `45 dims`")

# --- MAIN DASHBOARD ---
st.title("Homomorphic Encryption (CKKS) ECG Diagnosis")
st.markdown("Secure, end-to-end framework running on encrypted physiological data without ever decrypting it on the cloud.")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Plaintext Data 🏥", 
    "2. Client Encryption 🔒", 
    "3. Cloud Inference ☁️", 
    "4. Decryption & SHAP 🧠"
])

with tab1:
    st.markdown("### Client-Side: Patient Heartbeat (Raw ECG)")
    st.info("The patient's IoT device captures the raw ECG waveform.")
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(x_patient_raw, color="#1f77b4", linewidth=2)
    ax.set_title(f"Patient #{patient_id} - Raw Normalized Waveform")
    ax.set_xlabel("Time (Samples)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    
    st.markdown(f"**True Label Diagnosis:** {y_true_name}")
    if st.button("Proceed to Encryption ➡️", key="btn_enc"):
        st.session_state['run_encryption'] = True

with tab2:
    st.markdown("### Client-Side: Homomorphic Encryption (CKKS)")
    st.info("The raw waveform is encrypted using the CKKS scheme. The cloud will compute on this ciphertext.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Encryption Parameters:**")
        st.code(f"""
Scheme: CKKS
Poly Modulus Degree: {utils.POLY_MOD_DEGREE}
Coeff Mod Bits: {utils.COEFF_MOD_BITS}
Scale: {utils.SCALE} (2^40)
        """)
        
    if st.session_state.get('run_encryption', False) or st.button("Encrypt Payload"):
        with st.spinner("Encrypting payload using Ring Learning with Errors..."):
            import tenseal as ts
            
            # Re-inflate context from bytes safely within this thread!
            ctx = ts.context_from(st.session_state['ckks_ctx_bytes'])
            
            enc_vector = ts.ckks_vector(ctx, x_patient_raw.tolist())
            serial_bytes = enc_vector.serialize()
            
            st.session_state['serial_bytes'] = serial_bytes
            st.session_state['encryption_done'] = True
            
        with col2:
            st.success("Encryption complete!")
            st.metric("Ciphertext Payload Size", f"{len(serial_bytes) / 1024:.2f} KB")
            
        st.markdown("#### Ciphertext Byte Preview (First 512 bytes)")
        byte_vals = np.frombuffer(serial_bytes[:512], dtype=np.uint8)
        
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.bar(np.arange(len(byte_vals)), byte_vals, color="#e74c3c", width=1.0)
        ax.set_xlim(0, len(byte_vals))
        ax.axis('off')
        st.pyplot(fig)

with tab3:
    st.markdown("### Cloud-Side: Secure Inference without Decryption")
    st.info("Applying 1D-CNN using secure matrix multiplication and quadratic polynomial approximations for ReLU.")
    
    if st.button("Run Encrypted Inference 🚀"):
        if 'encryption_done' not in st.session_state:
            st.warning("Please encrypt the data in Step 2 first.")
        else:
            with st.spinner("Computing 1D-CNN operations homomorphically (may take 2-5 seconds)..."):
                try:
                    import tenseal as ts
                    ctx = ts.context_from(st.session_state['ckks_ctx_bytes'])
                    
                    enc_logits, timings = utils.run_encrypted_inference(x_patient_raw, ctx, he_matrices)
                    st.session_state['enc_logits'] = enc_logits
                    st.session_state['timings'] = timings
                    st.success("Encrypted Inference Completed successfully!")
                except Exception as e:
                    st.error(f"Inference failed: {e}")

    if 'timings' in st.session_state:
        t = st.session_state['timings']
        st.markdown("#### Profiling (Latency Breakdown)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("1. Client Encrypt", f"{t['encrypt_ms']:.1f} ms")
        c2.metric("2. Cloud Infer", f"{t['infer_ms']:.1f} ms")
        c3.metric("3. Client Decrypt", f"{t['decrypt_ms']:.1f} ms")
        c4.metric("Total Latency", f"{t['total_ms']:.1f} ms", delta_color="inverse")

with tab4:
    st.markdown("### Client-Side: Decryption & Explanation (SHAP)")
    st.info("The clinician decrypts the cloud's prediction and accesses the local SHAP 45-dim feature explainer for transparency.")
    
    if 'enc_logits' not in st.session_state:
        st.warning("Please run inference in Step 3.")
    else:
        logits = st.session_state['enc_logits']
        # Apply softmax to interpret as probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        pred_class = int(np.argmax(probs))
        pred_name = CLASS_NAMES[pred_class]
        confidence = probs[pred_class] * 100
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
             st.markdown(f"<div class='card'><div class='metric-label'>Diagnosis Prediction</div><div class='metric-value'>{pred_name}</div><p>Confidence: {confidence:.2f}%</p></div>", unsafe_allow_html=True)
        with col_res2:
             match_status = "✅ MATCH" if pred_class == y_true_label else "❌ MISMATCH"
             st.markdown(f"<div class='card'><div class='metric-label'>Ground Truth</div><div class='metric-value'>{y_true_name}</div><p>{match_status}</p></div>", unsafe_allow_html=True)
             
        st.markdown("### Explainable AI (SHAP Feature Importance)")
        with st.spinner("Computing SHAP values for 45-dim feature set..."):
            shap_vals = utils.get_shap_values(explainer, x_patient_feat, num_classes=5)
            
            # shape of shap_vals is usually [num_classes, num_samples, num_features]
            if len(shap_vals.shape) == 3:
                # Get the SHAP values for the single sample and the predicted class
                sv_instance = shap_vals[pred_class, 0, :]
            else:
                sv_instance = shap_vals[0, :]
            
            # Top 10 features
            top_indices = np.argsort(np.abs(sv_instance))[-10:]
            top_vals = sv_instance[top_indices]
            top_names = [feature_names[i] for i in top_indices]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['#ff0051' if v > 0 else '#008bfb' for v in top_vals]
            ax.barh(np.arange(10), top_vals, color=colors)
            ax.set_yticks(np.arange(10))
            ax.set_yticklabels(top_names)
            ax.set_xlabel("SHAP Value (Impact on Model Output)")
            ax.set_title(f"Top 10 Influential Features for Class: {pred_name}")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)

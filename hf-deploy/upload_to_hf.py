"""
Upload all files to a Hugging Face Space using the huggingface_hub API.
No Git or Git LFS required.

USAGE:
  1. Set your HF_TOKEN below (or pass as env var)
  2. Set your HF_USERNAME below
  3. Run:  python upload_to_hf.py
"""

import os
from huggingface_hub import HfApi, create_repo

# ============================================================
#  CONFIGURE THESE TWO VALUES
# ============================================================
HF_USERNAME = os.environ.get("HF_USERNAME", "YOUR_USERNAME_HERE")
HF_TOKEN    = os.environ.get("HF_TOKEN",    "YOUR_TOKEN_HERE")
# ============================================================

SPACE_NAME = "secure-ecg-analysis"
REPO_ID    = f"{HF_USERNAME}/{SPACE_NAME}"
SOURCE_DIR = os.path.join(os.path.dirname(__file__), "..", "phase-1-95acu")
DEPLOY_DIR = os.path.dirname(__file__)

# Files to upload: (local_path, path_in_repo)
FILES_TO_UPLOAD = [
    # Deployment configs (from hf-deploy/)
    (os.path.join(DEPLOY_DIR, "app.py"),            "app.py"),
    (os.path.join(DEPLOY_DIR, "utils.py"),           "utils.py"),
    (os.path.join(DEPLOY_DIR, "requirements.txt"),   "requirements.txt"),
    (os.path.join(DEPLOY_DIR, "packages.txt"),       "packages.txt"),
    
    # Models (from source)
    (os.path.join(SOURCE_DIR, "phase1_1dcnn_final.keras"), "phase1_1dcnn_final.keras"),
    (os.path.join(SOURCE_DIR, "cnn_feat_best.keras"),       "cnn_feat_best.keras"),
    
    # Data files (from source)
    (os.path.join(SOURCE_DIR, "X_test_cnn.npy"),            "X_test_cnn.npy"),
    (os.path.join(SOURCE_DIR, "X_test_feat_scaled.npy"),     "X_test_feat_scaled.npy"),
    (os.path.join(SOURCE_DIR, "y_test.npy"),                 "y_test.npy"),
    (os.path.join(SOURCE_DIR, "shap_background_feat.npy"),   "shap_background_feat.npy"),
    
    # Config files (from source)
    (os.path.join(SOURCE_DIR, "feature_names.json"),  "feature_names.json"),
    (os.path.join(SOURCE_DIR, "scaler_feat.pkl"),     "scaler_feat.pkl"),
    (os.path.join(SOURCE_DIR, "scaler_raw.pkl"),      "scaler_raw.pkl"),
]

def main():
    api = HfApi(token=HF_TOKEN)
    
    # Step 1: Create the Space (if it doesn't exist)
    print(f"{'='*60}")
    print(f"  Creating Space: {REPO_ID}")
    print(f"{'='*60}")
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="space",
            space_sdk="streamlit",
            token=HF_TOKEN,
            exist_ok=True,  # Don't error if it already exists
        )
        print(f"  ✅ Space created/verified: https://huggingface.co/spaces/{REPO_ID}")
    except Exception as e:
        print(f"  ⚠️  Space creation note: {e}")
    
    # Step 2: Upload each file
    print(f"\n{'='*60}")
    print(f"  Uploading {len(FILES_TO_UPLOAD)} files...")
    print(f"{'='*60}")
    
    for local_path, repo_path in FILES_TO_UPLOAD:
        abs_path = os.path.abspath(local_path)
        if not os.path.exists(abs_path):
            print(f"  ❌ MISSING: {abs_path}")
            continue
        
        size_mb = os.path.getsize(abs_path) / (1024 * 1024)
        print(f"  📤 Uploading {repo_path} ({size_mb:.2f} MB)...", end=" ", flush=True)
        
        try:
            api.upload_file(
                path_or_fileobj=abs_path,
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="space",
                token=HF_TOKEN,
            )
            print("✅")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n{'='*60}")
    print(f"  🎉 DEPLOYMENT COMPLETE!")
    print(f"  🌐 App URL: https://huggingface.co/spaces/{REPO_ID}")
    print(f"  📋 Check build logs at the URL above (click 'Logs' tab)")
    print(f"  ⏱️  First build takes ~5-10 min (TenSEAL compilation)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

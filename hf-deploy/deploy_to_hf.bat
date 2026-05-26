@echo off
REM ============================================================
REM  Deploy Secure ECG Analysis App to Hugging Face Spaces
REM ============================================================
REM  USAGE:
REM    1. Edit the USERNAME variable below
REM    2. Run:  deploy_to_hf.bat
REM ============================================================

SET USERNAME=YOUR_HF_USERNAME
SET SPACE_NAME=secure-ecg-analysis
SET SOURCE_DIR=E:\phase-1-95acu\phase-1-95acu
SET DEPLOY_DIR=E:\phase-1-95acu\hf-deploy

echo.
echo ============================================================
echo  Step 1: Install Hugging Face CLI
echo ============================================================
pip install huggingface_hub

echo.
echo ============================================================
echo  Step 2: Login to Hugging Face
echo  (Paste your access token when prompted)
echo ============================================================
huggingface-cli login

echo.
echo ============================================================
echo  Step 3: Create the Space
echo ============================================================
huggingface-cli repo create %SPACE_NAME% --type space --space-sdk streamlit
if errorlevel 1 (
    echo Space may already exist, continuing...
)

echo.
echo ============================================================
echo  Step 4: Clone the Space
echo ============================================================
cd /d E:\phase-1-95acu
if exist "hf-space" rmdir /s /q hf-space
git clone https://huggingface.co/spaces/%USERNAME%/%SPACE_NAME% hf-space
cd hf-space

echo.
echo ============================================================
echo  Step 5: Install Git LFS
echo ============================================================
git lfs install

echo.
echo ============================================================
echo  Step 6: Copy deployment files
echo ============================================================
copy /Y "%DEPLOY_DIR%\.gitattributes"    .
copy /Y "%DEPLOY_DIR%\packages.txt"      .
copy /Y "%DEPLOY_DIR%\requirements.txt"  .
copy /Y "%DEPLOY_DIR%\app.py"            .
copy /Y "%DEPLOY_DIR%\utils.py"          .

REM Model files
copy /Y "%SOURCE_DIR%\phase1_1dcnn_final.keras"  .
copy /Y "%SOURCE_DIR%\cnn_feat_best.keras"        .

REM Data files
copy /Y "%SOURCE_DIR%\X_test_cnn.npy"             .
copy /Y "%SOURCE_DIR%\X_test_feat_scaled.npy"      .
copy /Y "%SOURCE_DIR%\y_test.npy"                  .
copy /Y "%SOURCE_DIR%\shap_background_feat.npy"    .

REM Config files
copy /Y "%SOURCE_DIR%\feature_names.json"          .
copy /Y "%SOURCE_DIR%\scaler_feat.pkl"             .
copy /Y "%SOURCE_DIR%\scaler_raw.pkl"              .

echo.
echo ============================================================
echo  Step 7: Git add, commit, and push
echo ============================================================
git add -A
git commit -m "Deploy Secure ECG Analysis with HE + SHAP"
git push origin main

echo.
echo ============================================================
echo  DONE! Your app will be live at:
echo  https://huggingface.co/spaces/%USERNAME%/%SPACE_NAME%
echo ============================================================
echo.
echo  Check the Logs tab on the Space page to monitor the build.
echo  First build takes 5-10 minutes (TenSEAL compilation).
echo ============================================================
pause

# CropScan — AI Precision Agriculture App
## Setup & Run (3 steps)

### Step 1 — Install packages
```
pip install -r requirements.txt
```

### Step 2 — Make sure your model files are here:
```
C:\Users\rshri\Downloads\agri_outputs\mobilenetv3_plantdisease.onnx
C:\Users\rshri\Downloads\agri_outputs\class_names.json
```
The app auto-detects them. No config needed.

### Step 3 — Run
```
python app.py
```

### Open in browser
- Same PC   →  http://localhost:5000
- Phone     →  http://<your-ip>:5000  (same WiFi)

Find your IP: run `ipconfig` in Command Prompt, look for IPv4 Address.

---
Model auto-detected: app prints "Real model loaded" or "Demo mode" on startup.

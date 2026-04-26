"""
CropScan Flask Backend — app.py
Preprocessing matches the standalone notebook val_transform exactly:
  EXIF-transpose -> resize(224, 224, BILINEAR) -> /255 -> (x - mean) / std
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageOps
import json, os, io, random

app = Flask(__name__, static_folder='static')
CORS(app)

# Model search order: prefer the merged self-contained ONNX shipped with the
# Flutter app, fall back to the notebook output dir, then user paths.
SEARCH_PATHS = [
    os.path.join(os.path.dirname(__file__), 'models'),
]

MODEL_PATH = None
LABELS_PATH = None
for path in SEARCH_PATHS:
    m = os.path.join(path, 'mobilenetv3_plantdisease.onnx')
    l = os.path.join(path, 'class_names.json')
    if os.path.exists(m) and os.path.exists(l):
        MODEL_PATH = m
        LABELS_PATH = l
        break

# Optional rembg for background removal. Only used when the client *explicitly*
# requests bg_removal=true. Default is OFF so inputs match the notebook exactly.
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
    print("rembg available (used only when bg_removal=true)")
except ImportError:
    REMBG_AVAILABLE = False
    print("rembg not installed (bg_removal=true will be a no-op)")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TREATMENTS = {
    "Apple___Apple_scab": {"action": "Apply fungicide (captan or myclobutanil) every 7-10 days. Remove infected leaves.", "severity": "moderate"},
    "Apple___Black_rot": {"action": "Prune dead wood. Apply copper-based fungicide. Remove mummified fruit.", "severity": "high"},
    "Apple___Cedar_apple_rust": {"action": "Apply myclobutanil before symptoms appear. Remove nearby juniper trees.", "severity": "moderate"},
    "Apple___healthy": {"action": "No treatment needed. Continue routine monitoring.", "severity": "none"},
    "Blueberry___healthy": {"action": "No treatment needed. Maintain soil pH 4.5-5.5.", "severity": "none"},
    "Cherry_(including_sour)___Powdery_mildew": {"action": "Apply sulfur spray. Improve air circulation. Avoid overhead irrigation.", "severity": "moderate"},
    "Cherry_(including_sour)___healthy": {"action": "No treatment needed. Monitor for pests.", "severity": "none"},
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {"action": "Apply strobilurin fungicide. Rotate crops. Use resistant hybrids.", "severity": "high"},
    "Corn_(maize)___Common_rust_": {"action": "Apply propiconazole fungicide. Plant resistant varieties.", "severity": "moderate"},
    "Corn_(maize)___Northern_Leaf_Blight": {"action": "Apply azoxystrobin at tasseling. Use certified disease-free seed.", "severity": "high"},
    "Corn_(maize)___healthy": {"action": "No treatment needed. Maintain proper spacing.", "severity": "none"},
    "Grape___Black_rot": {"action": "Apply mancozeb at bud break. Remove mummified berries.", "severity": "high"},
    "Grape___Esca_(Black_Measles)": {"action": "No chemical cure. Remove infected wood. Protect pruning wounds.", "severity": "high"},
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {"action": "Apply copper fungicide early season. Improve airflow.", "severity": "moderate"},
    "Grape___healthy": {"action": "No treatment needed. Continue standard management.", "severity": "none"},
    "Orange___Haunglongbing_(Citrus_greening)": {"action": "IMMEDIATE: No cure. Remove infected trees. Control psyllid. Notify agriculture dept.", "severity": "critical"},
    "Peach___Bacterial_spot": {"action": "Apply copper bactericide at petal fall, repeat every 14 days.", "severity": "moderate"},
    "Peach___healthy": {"action": "No treatment needed. Standard thinning program sufficient.", "severity": "none"},
    "Pepper,_bell___Bacterial_spot": {"action": "Apply copper hydroxide. Remove infected plants.", "severity": "moderate"},
    "Pepper,_bell___healthy": {"action": "No treatment needed. Monitor for aphids.", "severity": "none"},
    "Potato___Early_blight": {"action": "Apply chlorothalonil fungicide. Remove lower infected leaves.", "severity": "moderate"},
    "Potato___Late_blight": {"action": "URGENT: Apply metalaxyl immediately. Destroy infected plants.", "severity": "critical"},
    "Potato___healthy": {"action": "No treatment needed. Monitor for Colorado potato beetle.", "severity": "none"},
    "Raspberry___healthy": {"action": "No treatment needed. Prune old canes after harvest.", "severity": "none"},
    "Soybean___healthy": {"action": "No treatment needed. Monitor for soybean aphid.", "severity": "none"},
    "Squash___Powdery_mildew": {"action": "Apply potassium bicarbonate or neem oil. Increase plant spacing.", "severity": "moderate"},
    "Strawberry___Leaf_scorch": {"action": "Remove infected leaves. Apply captan fungicide.", "severity": "moderate"},
    "Strawberry___healthy": {"action": "No treatment needed. Renew beds every 3 years.", "severity": "none"},
    "Tomato___Bacterial_spot": {"action": "Apply copper bactericide weekly. Avoid splashing water on leaves.", "severity": "moderate"},
    "Tomato___Early_blight": {"action": "Apply chlorothalonil. Remove lower leaves. Mulch around base.", "severity": "moderate"},
    "Tomato___Late_blight": {"action": "URGENT: Apply metalaxyl immediately. Bag all infected material.", "severity": "critical"},
    "Tomato___Leaf_Mold": {"action": "Reduce humidity below 85%. Apply mancozeb. Prune for airflow.", "severity": "moderate"},
    "Tomato___Septoria_leaf_spot": {"action": "Apply copper fungicide. Remove infected lower leaves.", "severity": "moderate"},
    "Tomato___Spider_mites Two-spotted_spider_mite": {"action": "Apply miticide or neem oil. Increase humidity.", "severity": "moderate"},
    "Tomato___Target_Spot": {"action": "Apply azoxystrobin. Remove infected debris. Rotate crops.", "severity": "moderate"},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {"action": "No cure. Remove plants. Control whitefly with imidacloprid.", "severity": "critical"},
    "Tomato___Tomato_mosaic_virus": {"action": "No cure. Remove plants immediately. Disinfect tools.", "severity": "high"},
    "Tomato___healthy": {"action": "No treatment needed. Continue standard IPM scouting.", "severity": "none"},
}

CLASS_NAMES = list(TREATMENTS.keys())
session = None
INPUT_NAME = None
DEMO_MODE = True

if MODEL_PATH:
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        INPUT_NAME = session.get_inputs()[0].name
        with open(LABELS_PATH, 'r') as f:
            data = json.load(f)
            CLASS_NAMES = data.get('classes', CLASS_NAMES)
        DEMO_MODE = False
        print(f"Real model loaded from {MODEL_PATH}")
        print(f"  classes: {len(CLASS_NAMES)}  input: {INPUT_NAME}")
    except Exception as e:
        print(f"Model load failed: {e}")


def remove_background_rembg(img):
    """Replace background with neutral gray (130,130,130) — closer to PlantVillage
    training distribution than pure white, which is OOD for the model."""
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    cut = Image.open(io.BytesIO(rembg_remove(buf.getvalue()))).convert('RGBA')
    bg = Image.new('RGB', cut.size, (130, 130, 130))
    bg.paste(cut, mask=cut.split()[3])
    return bg


def preprocess(image_bytes, use_bg_removal=False):
    """Match the notebook's val_transform exactly:
       Image.open -> exif_transpose -> resize(224, 224, BILINEAR) -> /255 -> (x-mean)/std
    Optional rembg-based background swap runs *before* the canonical pipeline."""
    img = Image.open(io.BytesIO(image_bytes))
    # CRITICAL: bake EXIF orientation into pixels. Phone photos store rotation
    # as a tag, not as rotated pixels — without this the model sees sideways
    # leaves and confidence collapses.
    img = ImageOps.exif_transpose(img).convert('RGB')
    print(f"  size after EXIF transpose: {img.size}")

    if use_bg_removal and REMBG_AVAILABLE:
        try:
            img = remove_background_rembg(img)
            print("  bg removed via rembg (gray fill)")
        except Exception as e:
            print(f"  bg removal failed, using raw image: {e}")

    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    # Diagnostic: dump first 3 normalized pixels per channel so the offline
    # path and the standalone notebook can be diffed against these numbers.
    print(f"  tensor R[0..2]={arr[0,0,0]:.4f},{arr[0,0,1]:.4f},{arr[0,0,2]:.4f} "
          f"G[0..2]={arr[1,0,0]:.4f},{arr[1,0,1]:.4f},{arr[1,0,2]:.4f} "
          f"B[0..2]={arr[2,0,0]:.4f},{arr[2,0,1]:.4f},{arr[2,0,2]:.4f}")
    return np.expand_dims(arr, axis=0)


def weather_stress(temp, hum):
    return float(min(0.6 * abs(temp - 25) / 25 + 0.4 * (1 - hum / 100), 1.0))


def health_score(pd, vn, ws):
    return float(np.clip(0.45 * (1 - pd) + 0.35 * vn + 0.20 * (1 - ws), 0, 1))


def caf(pd, vn, ws, conf):
    rc = float(conf)
    rn = float(vn) if vn > 0.15 else 0.1
    rw = float(1 - ws)
    tot = rc + rn + rw + 1e-9
    w1, w2, w3 = rc / tot, rn / tot, rw / tot
    H = float(np.clip(w1 * (1 - pd) + w2 * vn + w3 * (1 - ws), 0, 1))
    return {
        "H_adaptive": round(H, 4),
        "w1": round(w1, 3), "w2": round(w2, 3), "w3": round(w3, 3),
        "r_cnn": round(rc, 3), "r_ndvi": round(rn, 3), "r_weather": round(rw, 3),
    }


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/status')
def status():
    return jsonify({
        "model_loaded": not DEMO_MODE,
        "demo_mode": DEMO_MODE,
        "model_path": MODEL_PATH,
        "classes": len(CLASS_NAMES),
        "bg_removal_available": REMBG_AVAILABLE,
        "version": "CropScan 3.1",
    })


@app.route('/weather')
def get_weather():
    """Real-time weather (Open-Meteo, no key) + NDVI (NASA MODIS via ORNL DAAC)."""
    try:
        import requests as req
        lat = float(request.args.get('lat', 20.5937))
        lon = float(request.args.get('lon', 78.9629))

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,weather_code"
            f"&timezone=auto"
        )
        w_resp = req.get(weather_url, timeout=5)
        current = w_resp.json().get("current", {})
        temperature = round(float(current.get("temperature_2m", 28.0)), 1)
        humidity = round(float(current.get("relative_humidity_2m", 65.0)), 1)
        weather_code = int(current.get("weather_code", 0))

        ndvi = 0.45
        ndvi_source = "fallback"
        try:
            d_resp = req.get(
                f"https://modis.ornl.gov/rst/api/v1/MOD13Q1/dates"
                f"?latitude={lat}&longitude={lon}",
                timeout=3,
            )
            if d_resp.status_code == 200:
                dates = d_resp.json().get("dates", [])
                if dates:
                    latest_date = dates[-1]["modis_date"]
                    s_resp = req.get(
                        f"https://modis.ornl.gov/rst/api/v1/MOD13Q1/subset"
                        f"?latitude={lat}&longitude={lon}"
                        f"&startDate={latest_date}&endDate={latest_date}"
                        f"&kmAboveBelow=0&kmLeftRight=0",
                        timeout=4,
                    )
                    if s_resp.status_code == 200:
                        for band in s_resp.json().get("subset", []):
                            if band.get("band") == "_250m_16_days_NDVI":
                                raw = band.get("data", [None])[0]
                                if raw is not None and raw != -3000:
                                    ndvi = max(-1.0, min(1.0, round(float(raw) * 0.0001, 4)))
                                    ndvi_source = "NASA MODIS MOD13Q1"
                                break
        except Exception as ndvi_err:
            print(f"  NDVI fetch failed: {ndvi_err}, using fallback")

        ws = weather_stress(temperature, humidity)
        return jsonify({
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "temperature": temperature,
            "humidity": humidity,
            "ndvi": ndvi,
            "weather_stress": round(ws, 3),
            "weather_code": weather_code,
            "ndvi_source": ndvi_source,
            "data_sources": "Open-Meteo (weather) + " + ndvi_source + " (NDVI)",
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({"error": "No image uploaded"}), 400

        temp = float(request.form.get('temperature', 30))
        hum = float(request.form.get('humidity', 60))
        vn = float(request.form.get('ndvi', 0.45))
        # Default OFF: matches the notebook's val_transform pipeline. Clients
        # that want background removal must opt in explicitly.
        use_bg = request.form.get('bg_removal', 'false').lower() == 'true'
        img_bytes = file.read()

        print(f"\nScan — temp={temp} hum={hum} ndvi={vn} bg={use_bg}")

        if not DEMO_MODE:
            tensor = preprocess(img_bytes, use_bg_removal=use_bg)
            logits = session.run(None, {INPUT_NAME: tensor})[0][0]
            exp_l = np.exp(logits - np.max(logits))
            probs = exp_l / exp_l.sum()
            pred_idx = int(np.argmax(probs))
            disease_label = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx])
            top5_idx = np.argsort(probs)[::-1][:5]
            top5 = [
                {"label": CLASS_NAMES[i].replace('___', ' — ').replace('_', ' '),
                 "prob": round(float(probs[i]) * 100, 1)}
                for i in top5_idx
            ]
        else:
            idx = random.randint(0, len(CLASS_NAMES) - 1)
            disease_label = CLASS_NAMES[idx]
            confidence = round(random.uniform(0.58, 0.97), 4)
            pf = np.random.dirichlet(np.ones(len(CLASS_NAMES)) * 0.3)
            pf[idx] = confidence
            pf = pf / pf.sum()
            top5_idx = np.argsort(pf)[::-1][:5]
            top5 = [
                {"label": CLASS_NAMES[i].replace('___', ' — ').replace('_', ' '),
                 "prob": round(float(pf[i]) * 100, 1)}
                for i in top5_idx
            ]

        ws = weather_stress(temp, hum)
        is_healthy = 'healthy' in disease_label.lower()
        pd = (1 - confidence) if is_healthy else confidence
        H = health_score(pd, vn, ws)
        caf_result = caf(pd, vn, ws, confidence)
        status_str = "Healthy" if H > 0.6 else ("Moderate Risk" if H >= 0.3 else "High Risk")
        tx = TREATMENTS.get(disease_label, {"action": "Consult a local agronomist.", "severity": "unknown"})

        print(f"  -> {disease_label} ({confidence*100:.1f}%) H={H*100:.1f}%")

        return jsonify({
            "disease_label": disease_label.replace('___', ' — ').replace('_', ' '),
            "confidence": round(confidence * 100, 1),
            "health_score": round(H * 100, 1),
            "health_status": status_str,
            "weather_stress": round(ws, 3),
            "ndvi_used": round(vn, 2),
            "temperature": temp,
            "humidity": hum,
            "top5": top5,
            "treatment": tx["action"],
            "severity": tx["severity"],
            "caf": caf_result,
            "is_healthy": is_healthy,
            "demo_mode": DEMO_MODE,
            "bg_removal_used": use_bg and REMBG_AVAILABLE,
            "bg_method": "rembg" if (use_bg and REMBG_AVAILABLE) else "none",
            "pd": round(pd, 4),
            "ws": round(ws, 4),
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  CropScan AI v3.1")
    print("=" * 50)
    print(f"  Mode      : {'REAL MODEL' if not DEMO_MODE else 'DEMO'}")
    print(f"  Model path: {MODEL_PATH or '(none)'}")
    print(f"  BG removal: {'rembg available (opt-in)' if REMBG_AVAILABLE else 'unavailable'}")
    print(f"  Classes   : {len(CLASS_NAMES)}")
    print("  URL       : http://localhost:5001")
    print("=" * 50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)

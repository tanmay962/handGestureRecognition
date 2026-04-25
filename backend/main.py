import asyncio
import base64
import json
import math
import os
import time

# Load backend/.env automatically so GEMINI_API_KEY etc. work without manual export
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from database import (
    STATIC_FEATURES, DYNAMIC_FEATURES, DYNAMIC_FRAMES,
    MIN_STATIC_SAMPLES, MIN_DYNAMIC_SAMPLES,
    get_db, init_db, log_session, score_sample_quality,
    simulate_static, simulate_dynamic,
)
from ml_models import UnifiedLSTMModel, DEVICE, FEATURE_VERSION
import nlp as _nlp_module
from nlp import (
    SentenceState,
    build_base_model, build_personal_model, load_personal_corpus_from_db,
    kn_suggest, gesture_context_suggest, word_prefix_suggest, offline_grammar_correct,
)

try:
    import paho.mqtt.client as mqtt_lib
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

try:
    from mediapipe.tasks.python import BaseOptions as MP_BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

FRONTEND = Path(__file__).parent.parent / "frontend"
MP_MODEL  = Path(__file__).parent / "hand_landmarker.task"

unified_nn      = UnifiedLSTMModel()
_conf_threshold = 0.65
_train_lock     = asyncio.Lock()
sentence_state  = SentenceState()

_ADMIN_PIN       = os.environ.get("ADMIN_PIN", "").strip()
_GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "").strip()
_AUTO_RETRAIN_EVERY = int(os.environ.get("AUTO_RETRAIN_EVERY", "10"))

if not _ADMIN_PIN:
    print("[WARN] ADMIN_PIN not set — destructive endpoints are unprotected.")
if _GEMINI_API_KEY:
    print("[Gemini] Server-side API key loaded from GEMINI_API_KEY env var.")

def require_admin(x_admin_pin: str = Header(default="")):
    if _ADMIN_PIN and x_admin_pin != _ADMIN_PIN:
        raise HTTPException(status_code=401, detail="Invalid or missing admin PIN")


class ConnectionManager:
    def __init__(self):
        self.active = []

    async def connect(self, ws):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

manager = ConnectionManager()

async def push(event, data=None):
    await manager.broadcast({"event": event, **(data or {})})


class PredictionBuffer:
    # window=5, min_votes=3: requires 3/5 agreement — faster response than 4/7
    # while still rejecting single-frame noise. Frontend ensemble handles the
    # deeper temporal smoothing, so backend only needs to catch obvious jitter.
    def __init__(self, window: int = 5, min_votes: int = 3):
        self.window    = window
        self.min_votes = min_votes
        self._buf: list = []

    def push(self, name: str, conf: float):
        self._buf.append((name, conf))
        if len(self._buf) > self.window:
            self._buf.pop(0)

    def smooth(self, raw: dict) -> dict:
        if not self._buf:
            return raw
        names = [n for n, _ in self._buf]
        best  = max(set(names), key=names.count)
        votes = names.count(best)
        if votes < self.min_votes:
            return {**raw, "below_threshold": True}
        avg_conf = sum(c for n, c in self._buf if n == best) / votes
        return {**raw, "name": best, "conf": round(avg_conf, 4)}

    def reset(self):
        self._buf.clear()


_pred_buffer        = PredictionBuffer()
_gesture_thresholds: dict = {}


# MediaPipe (optional, server-side)

_mp_ready      = False
_mp_landmarker = None

def _init_mediapipe():
    global _mp_ready, _mp_landmarker
    if not MP_AVAILABLE:
        print("[MediaPipe] Not installed — using browser-side landmarks")
        return
    if not MP_MODEL.exists():
        print("[MediaPipe] Model file not found — using browser-side landmarks")
        return
    try:
        opts = HandLandmarkerOptions(
            base_options=MP_BaseOptions(model_asset_path=str(MP_MODEL)),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
        )
        _mp_landmarker = HandLandmarker.create_from_options(opts)
        _mp_ready      = True
        print("[MediaPipe] Server-side ready")
    except Exception as e:
        print(f"[MediaPipe] Init failed: {e} — using browser-side")


def _landmarks_to_features(lm):
    tips  = [4,  8, 12, 16, 20]
    pips  = [3,  6, 10, 14, 18]
    mcps  = [2,  5,  9, 13, 17]
    curls = []
    for i in range(5):
        t2m = math.sqrt(
            (lm[tips[i]].x - lm[mcps[i]].x) ** 2 +
            (lm[tips[i]].y - lm[mcps[i]].y) ** 2 +
            (lm[tips[i]].z - lm[mcps[i]].z) ** 2
        )
        p2m = math.sqrt(
            (lm[pips[i]].x - lm[mcps[i]].x) ** 2 +
            (lm[pips[i]].y - lm[mcps[i]].y) ** 2 +
            (lm[pips[i]].z - lm[mcps[i]].z) ** 2
        )
        curl = max(0, min(1, 1 - (t2m / (p2m * 2.5) if p2m > 0.001 else 0)))
        curls.append(curl)

    dx = lm[9].x - lm[0].x
    dy = lm[9].y - lm[0].y
    dz = lm[9].z - lm[0].z
    l  = math.sqrt(dx**2 + dy**2 + dz**2) or 0.001
    hd = [dx/l, dy/l, dz/l]

    sx = lm[17].x - lm[5].x
    sy = lm[17].y - lm[5].y
    sz = lm[17].z - lm[5].z
    ls = math.sqrt(sx**2 + sy**2 + sz**2) or 0.001

    dom   = curls + hd + [sx/ls * 0.1, sy/ls * 0.1, sz/ls * 0.1]
    aux   = [0.0] * 11
    face  = [0.0] * 10
    pose  = [0.0] * 6
    flags = [1.0, 0.0, 0.0]
    return (dom + aux + face + pose + flags)[:STATIC_FEATURES]


# MQTT subscriber

_mqtt_enabled = False
_mqtt_client  = None
_mqtt_loop: Optional[asyncio.AbstractEventLoop] = None
_mqtt_topic   = os.environ.get("MQTT_TOPIC", "gesture-detection/sensor/features")


def _mqtt_on_connect(client, userdata, flags, reason_code, properties):
    if not reason_code.is_failure:
        client.subscribe(_mqtt_topic)
        print(f"[MQTT] Connected — subscribed to '{_mqtt_topic}'")
    else:
        print(f"[MQTT] Connection refused: {reason_code}")


def _mqtt_on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
    if reason_code.is_failure:
        print(f"[MQTT] Unexpected disconnect: {reason_code} — will auto-reconnect")


def _mqtt_on_message(client, userdata, msg):
    global _mqtt_loop
    try:
        payload    = json.loads(msg.payload.decode())
        features   = payload.get("features", [])
        model_type = payload.get("model_type", "static")
        if not features:
            return

        if not unified_nn.trained:
            return

        result = unified_nn.predict(features, _conf_threshold)

        if _mqtt_loop and not _mqtt_loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                push("mqtt_prediction", {
                    "model":           model_type,
                    "gesture":         result["name"],
                    "conf":            result["conf"],
                    "below_threshold": result["below_threshold"],
                    "source":          "mqtt",
                }),
                _mqtt_loop,
            )

        if result["conf"] > 0.3:
            with get_db() as db:
                db.execute(
                    "INSERT INTO prediction_log(gesture,confidence,model_type,created_at) VALUES(?,?,?,?)",
                    (result["name"], result["conf"], model_type, time.time()),
                )

    except Exception as e:
        print(f"[MQTT] Message handler error: {e}")


def start_mqtt_subscriber():
    global _mqtt_enabled, _mqtt_client, _mqtt_loop

    if not MQTT_AVAILABLE:
        print("[MQTT] paho-mqtt not installed — subscriber disabled")
        return

    broker   = os.environ.get("MQTT_BROKER", "broker.hivemq.com")
    port     = int(os.environ.get("MQTT_PORT", "1883"))
    username = os.environ.get("MQTT_USER", "")
    password = os.environ.get("MQTT_PASS", "")

    try:
        _mqtt_loop = asyncio.get_running_loop()
    except RuntimeError:
        _mqtt_loop = asyncio.get_event_loop()

    try:
        client = mqtt_lib.Client(
            client_id=f"gesture-backend-{int(time.time())}",
            callback_api_version=mqtt_lib.CallbackAPIVersion.VERSION2,
        )
        if username:
            client.username_pw_set(username, password)
        client.on_connect    = _mqtt_on_connect
        client.on_disconnect = _mqtt_on_disconnect
        client.on_message    = _mqtt_on_message
        client.reconnect_delay_set(min_delay=2, max_delay=30)
        client.connect_async(broker, port, keepalive=60)
        client.loop_start()
        _mqtt_client  = client
        _mqtt_enabled = True
        print(f"[MQTT] Connecting to {broker}:{port}  topic='{_mqtt_topic}'")
    except Exception as e:
        print(f"[MQTT] Failed to start subscriber: {e}")
        _mqtt_enabled = False


# App lifespan

@asynccontextmanager
async def lifespan(app):
    global _conf_threshold, _gesture_thresholds

    init_db()
    build_base_model()
    load_personal_corpus_from_db()
    _init_mediapipe()

    with get_db() as db:
        row = db.execute("SELECT value FROM settings WHERE key='conf_threshold'").fetchone()
        if row:
            _conf_threshold = float(row['value'])

        for r in db.execute(
            "SELECT key, value FROM settings WHERE key LIKE 'conf_threshold_%'"
        ).fetchall():
            gesture = r['key'][len('conf_threshold_'):]
            if gesture:
                _gesture_thresholds[gesture] = float(r['value'])

        model_row = db.execute(
            "SELECT data FROM models WHERE id LIKE 'unified_%' ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()

    if model_row:
        try:
            unified_nn.from_json(json.loads(model_row['data']))
            # Restore auto-calibrated per-gesture thresholds into runtime map
            if unified_nn.per_gesture_threshold:
                for gname, thresh in unified_nn.per_gesture_threshold.items():
                    _gesture_thresholds.setdefault(gname, thresh)
            print(
                f"[Gesture Detection] Auto-loaded saved model | "
                f"gestures={len(unified_nn.gestures)} acc={unified_nn.accuracy:.3f} "
                f"T={unified_nn.temperature:.2f}"
            )
        except Exception as e:
            print(f"[Gesture Detection] Auto-load failed (will need manual train/load): {e}")
    else:
        print("[Gesture Detection] No saved model found — train and save to persist")

    start_mqtt_subscriber()
    print(
        f"[Gesture Detection] Ready | device={DEVICE} | "
        f"conf_threshold={_conf_threshold} | Unified BiLSTM (static + dynamic)"
    )
    yield

    if _mqtt_client:
        _mqtt_client.loop_stop()
        _mqtt_client.disconnect()
        print("[MQTT] Disconnected")


# App setup

app = FastAPI(title="Gesture Detection v1.0", lifespan=lifespan)

_CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request models

class NNInitRequest(BaseModel):
    model_type:     str
    input_size:     int  = 0
    hidden_sizes:   list = []
    output_size:    int  = 0
    gestures:       list = []
    gesture_types:  dict = {}
    feature_version: str = "1.0"

class NNTrainRequest(BaseModel):
    model_type: str
    inputs:     list
    labels:     list
    epochs:     int   = 30
    lr:         float = 0.001

class PredictRequest(BaseModel):
    features:     list
    aux_features: list = []   # hand-swapped vector for independent aux-hand prediction
    model_type:   str  = 'static'

class SampleSaveRequest(BaseModel):
    gesture:     str
    samples:     list
    sample_type: str = 'static'
    source:      str = 'camera'

class SimulateRequest(BaseModel):
    gesture:     str
    count:       int = 35
    sample_type: str = 'static'

class SentenceRequest(BaseModel):
    action: str
    word:   Optional[str] = None
    letter: Optional[str] = None
    text:   Optional[str] = None

class SettingRequest(BaseModel):
    key:   str
    value: str

class GestureRegisterRequest(BaseModel):
    name:         str
    gesture_type: str = 'static'
    category:     str = 'custom'

class DeleteSamplesRequest(BaseModel):
    gesture:     str
    sample_type: str = 'all'
    source:      str = 'all'

class ConfThresholdRequest(BaseModel):
    threshold: float

class PredLogRequest(BaseModel):
    gesture:    str
    confidence: float
    model_type: str


# Auto-retrain helper

async def _do_auto_retrain():
    """Load all DB samples, retrain, and save. Runs as a background task."""
    if _train_lock.locked():
        return

    with get_db() as db:
        gestures = [r['name'] for r in db.execute(
            "SELECT name FROM gesture_registry ORDER BY name"
        )]
        if not gestures:
            return
        gesture_types = {r['name']: r['gesture_type'] for r in db.execute(
            "SELECT name, gesture_type FROM gesture_registry"
        )}
        gesture_idx = {name: i for i, name in enumerate(gestures)}

        all_inputs, all_labels = [], []
        for row in db.execute(
            "SELECT gesture, sample FROM static_samples WHERE quality >= 0.3"
        ):
            g = row['gesture']
            if g in gesture_idx:
                all_inputs.append(json.loads(row['sample']))
                all_labels.append(gesture_idx[g])
        for row in db.execute(
            "SELECT gesture, frames FROM dynamic_samples WHERE quality >= 0.3"
        ):
            g = row['gesture']
            if g in gesture_idx:
                all_inputs.append(json.loads(row['frames']))
                all_labels.append(gesture_idx[g])

    if not all_inputs:
        return

    await push("train_progress", {"model": "unified", "progress_pct": 0, "complete": False, "auto": True})
    unified_nn.initialize(len(gestures), gestures, gesture_types)

    async with _train_lock:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: unified_nn.train_batch(all_inputs, all_labels, 200, 0.001),
        )

    with get_db() as db:
        db.execute(
            "INSERT OR REPLACE INTO models(id,data,source,updated_at) VALUES(?,?,?,?)",
            ('unified_camera', json.dumps(unified_nn.to_json()), 'camera', time.time()),
        )
    await push("train_progress", {
        "model":           "unified",
        "accuracy":        unified_nn.accuracy,
        "val_accuracy":    unified_nn.val_accuracy,
        "loss":            unified_nn.loss,
        "epochs":          unified_nn.epochs,
        "per_gesture_acc": unified_nn.per_gesture_acc,
        "progress_pct":    100,
        "complete":        True,
        "auto":            True,
    })
    log_session("auto_retrain", detail=f"acc={unified_nn.accuracy:.3f}")


# Routes

@app.get("/favicon.ico")
async def favicon():
    from fastapi.responses import FileResponse
    ico = FRONTEND / "favicon.ico"
    if ico.exists():
        return FileResponse(str(ico), media_type="image/x-icon")
    return Response(status_code=204)

@app.get("/api/health")
def health():
    return {
        "status":           "ok",
        "version":          "1.0",
        "device":           str(DEVICE),
        "torch":            __import__("torch").__version__,
        "mediapipe_server": _mp_ready,
        "model_type":       "UnifiedBiLSTM",
    }

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_json({"event": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(ws)


# Neural network

@app.post("/api/nn/init")
def nn_init(req: NNInitRequest):
    unified_nn.initialize(req.output_size, req.gestures, req.gesture_types or {})
    unified_nn.feature_version = req.feature_version
    return {"ok": True, "model": req.model_type}

@app.post("/api/nn/train")
async def nn_train(req: NNTrainRequest):
    if unified_nn.model is None:
        return {"error": "model not initialised — call /api/nn/init first",
                "accuracy": 0, "loss": 1, "epochs": 0, "per_gesture_acc": {}}

    _loop = asyncio.get_running_loop()

    def _progress(stats):
        asyncio.run_coroutine_threadsafe(
            push("train_progress", {
                "model":        req.model_type,
                "epoch":        stats["epoch"],
                "total_epochs": stats["total_epochs"],
                "progress_pct": stats["progress_pct"],
                "train_loss":   stats.get("train_loss"),
                "val_acc":      stats.get("val_acc"),
                "complete":     False,
            }),
            _loop,
        )

    async with _train_lock:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: unified_nn.train_batch(req.inputs, req.labels, req.epochs, req.lr, _progress),
        )

    # Sync auto-calibrated per-gesture thresholds to the runtime threshold map
    if unified_nn.per_gesture_threshold:
        with get_db() as db:
            for gname, thresh in unified_nn.per_gesture_threshold.items():
                _gesture_thresholds[gname] = thresh
                db.execute("INSERT OR REPLACE INTO settings VALUES(?,?)",
                           (f"conf_threshold_{gname}", str(thresh)))

    await push("train_progress", {
        "model":                   req.model_type,
        "accuracy":                unified_nn.accuracy,
        "val_accuracy":            unified_nn.val_accuracy,
        "loss":                    unified_nn.loss,
        "epochs":                  unified_nn.epochs,
        "per_gesture_acc":         unified_nn.per_gesture_acc,
        "temperature":             unified_nn.temperature,
        "per_gesture_threshold":   unified_nn.per_gesture_threshold,
        "progress_pct":            100,
        "complete":                True,
    })
    log_session("train_batch", detail=f"{req.model_type} acc={unified_nn.accuracy:.3f} val={unified_nn.val_accuracy:.3f} T={unified_nn.temperature:.2f}")
    return {
        "accuracy":              unified_nn.accuracy,
        "val_accuracy":          unified_nn.val_accuracy,
        "loss":                  unified_nn.loss,
        "epochs":                unified_nn.epochs,
        "per_gesture_acc":       unified_nn.per_gesture_acc,
        "temperature":           unified_nn.temperature,
        "per_gesture_threshold": unified_nn.per_gesture_threshold,
    }

@app.post("/api/nn/predict")
def nn_predict(req: PredictRequest):
    if not unified_nn.trained:
        return {"idx": -1, "conf": 0.0, "probs": [], "name": "Unknown", "below_threshold": True, "aux": None}

    raw = unified_nn.predict(req.features, conf_threshold=0.0)

    threshold = _gesture_thresholds.get(raw["name"], _conf_threshold)
    raw["below_threshold"] = raw["conf"] < threshold

    _pred_buffer.push(raw["name"], raw["conf"])
    result = _pred_buffer.smooth(raw)

    if result["conf"] > 0.3:
        with get_db() as db:
            db.execute(
                "INSERT INTO prediction_log(gesture,confidence,model_type,created_at) VALUES(?,?,?,?)",
                (result["name"], result["conf"], req.model_type, time.time()),
            )

    # Aux-hand independent prediction using hand-swapped feature vector
    if req.aux_features and len(req.aux_features) >= 41:
        aux_raw = unified_nn.predict(req.aux_features, conf_threshold=0.0)
        aux_thresh = _gesture_thresholds.get(aux_raw["name"], _conf_threshold)
        aux_raw["below_threshold"] = aux_raw["conf"] < aux_thresh
        result["aux"] = aux_raw
    else:
        result["aux"] = None

    return result

@app.get("/api/nn/status")
def nn_status():
    info = {
        "trained":         unified_nn.trained,
        "accuracy":        unified_nn.accuracy,
        "val_accuracy":    unified_nn.val_accuracy,
        "loss":            unified_nn.loss,
        "epochs":          unified_nn.epochs,
        "per_gesture_acc": unified_nn.per_gesture_acc,
    }
    return {
        "static":  info,
        "dynamic": {**info, "model_type": "UnifiedBiLSTM"},
    }

@app.post("/api/nn/save/{model_type}")
async def nn_save(model_type: str, source: str = 'camera'):
    model_id = f"unified_{source}"
    with get_db() as db:
        db.execute(
            "INSERT OR REPLACE INTO models(id,data,source,updated_at) VALUES(?,?,?,?)",
            (model_id, json.dumps(unified_nn.to_json()), source, time.time()),
        )
    await push("model_saved", {"model": model_type, "source": source})
    log_session("model_saved", detail=f"unified_{source}")
    return {"ok": True, "model_id": model_id}

@app.post("/api/nn/load/{model_type}")
def nn_load(model_type: str, source: str = 'camera'):
    model_id = f"unified_{source}"
    with get_db() as db:
        row = db.execute("SELECT data FROM models WHERE id=?", (model_id,)).fetchone()

    if not row:
        return {"ok": False, "error": f"no saved unified model for '{model_id}'"}

    unified_nn.from_json(json.loads(row['data']))
    _pred_buffer.reset()
    return {"ok": True, "accuracy": unified_nn.accuracy, "val_accuracy": unified_nn.val_accuracy,
            "loss": unified_nn.loss, "epochs": unified_nn.epochs,
            "gestures": list(unified_nn.gestures.items())}

@app.post("/api/nn/reset/{model_type}")
async def nn_reset(model_type: str):
    unified_nn.reset()
    _pred_buffer.reset()
    with get_db() as db:
        db.execute("DELETE FROM models WHERE id LIKE 'unified_%'")
    await push("model_reset", {"model": model_type})
    log_session("model_reset", detail=model_type)
    return {"ok": True}

@app.post("/api/nn/reset_all")
async def nn_reset_all(_: None = Depends(require_admin)):
    unified_nn.reset()
    _pred_buffer.reset()
    with get_db() as db:
        db.execute("DELETE FROM models")
    await push("model_reset", {"model": "all"})
    log_session("model_reset_all")
    return {"ok": True}

@app.get("/api/nn/per_gesture_accuracy")
def per_gesture_accuracy():
    return {"static": unified_nn.per_gesture_acc, "dynamic": unified_nn.per_gesture_acc}

@app.get("/api/nn/confusion_matrix")
def get_confusion_matrix():
    """Confusion matrix from last training's val set (true → predicted counts)."""
    return {
        "matrix":   unified_nn.confusion_matrix,
        "gestures": list(unified_nn.gestures.values()),
    }

@app.get("/api/nn/calibration")
def get_calibration():
    """Post-training calibration state: temperature, per-gesture thresholds, margins."""
    return {
        "temperature":             unified_nn.temperature,
        "per_gesture_threshold":   unified_nn.per_gesture_threshold,
        "margin_threshold":        unified_nn.margin_threshold,
        "uncertainty_threshold":   unified_nn.uncertainty_threshold,
    }


# Confidence threshold

@app.post("/api/settings/conf_threshold")
def set_conf_threshold(req: ConfThresholdRequest):
    global _conf_threshold
    _conf_threshold = max(0.0, min(1.0, req.threshold))
    with get_db() as db:
        db.execute(
            "INSERT OR REPLACE INTO settings VALUES(?,?)",
            ("conf_threshold", str(_conf_threshold)),
        )
    return {"ok": True, "threshold": _conf_threshold}

@app.get("/api/settings/conf_threshold")
def get_conf_threshold():
    return {"threshold": _conf_threshold}


# Samples

@app.post("/api/samples/simulate")
def api_simulate(req: SimulateRequest):
    if req.sample_type == 'static':
        samples  = []
        mirrored = []
        for _ in range(req.count):
            vec, mir = simulate_static(req.gesture, mirror_aug=True)
            samples.append(vec)
            if mir:
                mirrored.append(mir)
        return {"gesture": req.gesture, "samples": samples, "mirrored": mirrored, "count": len(samples)}
    else:
        samples = [s for _ in range(req.count) if (s := simulate_dynamic(req.gesture)) is not None]
        return {"gesture": req.gesture, "samples": samples, "mirrored": [], "count": len(samples)}

@app.post("/api/samples/save")
async def save_samples(req: SampleSaveRequest, background_tasks: BackgroundTasks):
    now = time.time()
    with get_db() as db:
        if req.sample_type == "static":
            for s in req.samples:
                q = score_sample_quality(s)
                db.execute(
                    "INSERT INTO static_samples(gesture,sample,quality,source,created_at) VALUES(?,?,?,?,?)",
                    (req.gesture, json.dumps(s), q, req.source, now),
                )
            count = db.execute(
                "SELECT COUNT(*) FROM static_samples WHERE gesture=?", (req.gesture,)
            ).fetchone()[0]
        else:
            for s in req.samples:
                q = score_sample_quality(s[:5] if len(s) >= 5 else s)
                db.execute(
                    "INSERT INTO dynamic_samples(gesture,frames,quality,source,created_at) VALUES(?,?,?,?,?)",
                    (req.gesture, json.dumps(s), q, req.source, now),
                )
            count = db.execute(
                "SELECT COUNT(*) FROM dynamic_samples WHERE gesture=?", (req.gesture,)
            ).fetchone()[0]
        db.execute("UPDATE gesture_registry SET updated_at=? WHERE name=?", (now, req.gesture))

        # Count new samples since last train/save to decide if auto-retrain is due
        new_since = db.execute(
            "SELECT COUNT(*) FROM session_history "
            "WHERE event_type='sample_saved' "
            "AND created_at > COALESCE("
            "  (SELECT MAX(created_at) FROM session_history "
            "   WHERE event_type IN ('model_saved', 'auto_retrain')), 0)"
        ).fetchone()[0]

    log_session("sample_saved", req.gesture, f"{req.sample_type} x{len(req.samples)}")

    # Auto-retrain: trigger every N new samples when a trained model already exists
    if (unified_nn.trained and _AUTO_RETRAIN_EVERY > 0
            and not _train_lock.locked()
            and new_since > 0 and new_since % _AUTO_RETRAIN_EVERY == 0):
        background_tasks.add_task(_do_auto_retrain)

    return {"ok": True, "total": count}

@app.get("/api/samples/load")
def load_samples(source: str = 'all', min_quality: float = 0.3):
    result = {}
    with get_db() as db:
        if source == 'all':
            s_rows = db.execute(
                "SELECT gesture,sample FROM static_samples WHERE quality >= ? ORDER BY created_at",
                (min_quality,)
            )
            d_rows = db.execute(
                "SELECT gesture,frames FROM dynamic_samples WHERE quality >= ? ORDER BY created_at",
                (min_quality,)
            )
        else:
            s_rows = db.execute(
                "SELECT gesture,sample FROM static_samples WHERE source=? AND quality >= ? ORDER BY created_at",
                (source, min_quality)
            )
            d_rows = db.execute(
                "SELECT gesture,frames FROM dynamic_samples WHERE source=? AND quality >= ? ORDER BY created_at",
                (source, min_quality)
            )

        for row in s_rows:
            s = json.loads(row['sample'])
            if len(s) < STATIC_FEATURES:
                s = s + [0.0] * (STATIC_FEATURES - len(s) - 2) + [1, 0]
            elif len(s) > STATIC_FEATURES:
                s = s[:STATIC_FEATURES]
            result.setdefault(row['gesture'], {}).setdefault('static', []).append(s)

        for row in d_rows:
            f = json.loads(row['frames'])
            expected = DYNAMIC_FRAMES * DYNAMIC_FEATURES
            if len(f) < expected:
                f = f + [0.0] * (expected - len(f))
            elif len(f) > expected:
                f = f[:expected]
            result.setdefault(row['gesture'], {}).setdefault('dynamic', []).append(f)

    return result

@app.get("/api/samples/meta")
def samples_meta():
    meta = {}
    with get_db() as db:
        for row in db.execute("SELECT gesture,COUNT(*) as cnt,AVG(quality) as avg_q FROM static_samples GROUP BY gesture"):
            meta.setdefault(row['gesture'], {
                'static': 0, 'dynamic': 0,
                'static_quality': 0, 'dynamic_quality': 0,
                'static_camera': 0, 'static_glove': 0,
                'dynamic_camera': 0, 'dynamic_glove': 0,
            })
            meta[row['gesture']]['static']         = row['cnt']
            meta[row['gesture']]['static_quality'] = round(row['avg_q'] or 0, 2)

        for row in db.execute("SELECT gesture,source,COUNT(*) as cnt FROM static_samples GROUP BY gesture,source"):
            meta.setdefault(row['gesture'], {
                'static': 0, 'dynamic': 0,
                'static_quality': 0, 'dynamic_quality': 0,
                'static_camera': 0, 'static_glove': 0,
                'dynamic_camera': 0, 'dynamic_glove': 0,
            })
            src = row['source'] or 'camera'
            meta[row['gesture']]['static_' + src] = row['cnt']

        for row in db.execute("SELECT gesture,COUNT(*) as cnt,AVG(quality) as avg_q FROM dynamic_samples GROUP BY gesture"):
            meta.setdefault(row['gesture'], {
                'static': 0, 'dynamic': 0,
                'static_quality': 0, 'dynamic_quality': 0,
                'static_camera': 0, 'static_glove': 0,
                'dynamic_camera': 0, 'dynamic_glove': 0,
            })
            meta[row['gesture']]['dynamic']         = row['cnt']
            meta[row['gesture']]['dynamic_quality'] = round(row['avg_q'] or 0, 2)

        for row in db.execute("SELECT gesture,source,COUNT(*) as cnt FROM dynamic_samples GROUP BY gesture,source"):
            meta.setdefault(row['gesture'], {
                'static': 0, 'dynamic': 0,
                'static_quality': 0, 'dynamic_quality': 0,
                'static_camera': 0, 'static_glove': 0,
                'dynamic_camera': 0, 'dynamic_glove': 0,
            })
            src = row['source'] or 'camera'
            meta[row['gesture']]['dynamic_' + src] = row['cnt']

    return meta

@app.delete("/api/samples/clear")
async def clear_samples(_: None = Depends(require_admin)):
    with get_db() as db:
        db.execute("DELETE FROM static_samples")
        db.execute("DELETE FROM dynamic_samples")
        db.execute("DELETE FROM models")
    unified_nn.reset()
    await push("samples_cleared", {})
    log_session("samples_cleared_all")
    return {"ok": True}

@app.delete("/api/samples/gesture")
async def delete_gesture_samples(req: DeleteSamplesRequest):
    with get_db() as db:
        src = req.source
        if req.sample_type in ('static', 'all'):
            if src == 'all':
                db.execute("DELETE FROM static_samples WHERE gesture=?", (req.gesture,))
            else:
                db.execute("DELETE FROM static_samples WHERE gesture=? AND source=?", (req.gesture, src))
        if req.sample_type in ('dynamic', 'all'):
            if src == 'all':
                db.execute("DELETE FROM dynamic_samples WHERE gesture=?", (req.gesture,))
            else:
                db.execute("DELETE FROM dynamic_samples WHERE gesture=? AND source=?", (req.gesture, src))
    await push("gesture_samples_deleted", {"gesture": req.gesture, "type": req.sample_type})
    log_session("gesture_samples_deleted", req.gesture, req.sample_type)
    return {"ok": True}

@app.get("/api/samples/preview/{gesture}")
def sample_preview(gesture: str):
    with get_db() as db:
        s_rows = db.execute(
            "SELECT sample,quality,created_at FROM static_samples WHERE gesture=? ORDER BY created_at",
            (gesture,)
        ).fetchall()
        d_rows = db.execute(
            "SELECT frames,quality,created_at FROM dynamic_samples WHERE gesture=? ORDER BY created_at",
            (gesture,)
        ).fetchall()

    sq = [r['quality'] for r in s_rows]
    dq = [r['quality'] for r in d_rows]
    return {
        "staticCount":        len(s_rows),
        "dynamicCount":       len(d_rows),
        "staticPreview":      [json.loads(r['sample']) for r in s_rows[-3:]],
        "staticQualities":    sq[-10:],
        "dynamicQualities":   dq[-10:],
        "avgStaticQuality":   round(sum(sq) / len(sq), 2) if sq else 0,
        "avgDynamicQuality":  round(sum(dq) / len(dq), 2) if dq else 0,
        "updatedAt":          s_rows[-1]['created_at'] * 1000 if s_rows else (
                               d_rows[-1]['created_at'] * 1000 if d_rows else None),
    }

# Gesture registry

@app.get("/api/gestures")
def get_gestures():
    with get_db() as db:
        rows = db.execute("SELECT * FROM gesture_registry ORDER BY category, name").fetchall()
    return [dict(r) for r in rows]

@app.post("/api/gestures/register")
def register_gesture(req: GestureRegisterRequest):
    now = time.time()
    with get_db() as db:
        db.execute(
            "INSERT OR REPLACE INTO gesture_registry(name,gesture_type,category,created_at,updated_at) "
            "VALUES(?,?,?,COALESCE((SELECT created_at FROM gesture_registry WHERE name=?),?),?)",
            (req.name, req.gesture_type, req.category, req.name, now, now),
        )
    log_session("gesture_registered", req.name, req.gesture_type)
    return {"ok": True}

@app.delete("/api/gestures/{name}")
async def delete_gesture(name: str):
    with get_db() as db:
        db.execute("DELETE FROM gesture_registry WHERE name=?", (name,))
        db.execute("DELETE FROM static_samples WHERE gesture=?", (name,))
        db.execute("DELETE FROM dynamic_samples WHERE gesture=?", (name,))
    await push("gesture_deleted", {"gesture": name})
    log_session("gesture_deleted", name)
    return {"ok": True}

@app.get("/api/gestures/readiness")
def gesture_readiness(source: str = 'all'):
    meta = {}
    with get_db() as db:
        if source == 'all':
            s_rows = db.execute("SELECT gesture,COUNT(*) as cnt FROM static_samples GROUP BY gesture")
            d_rows = db.execute("SELECT gesture,COUNT(*) as cnt FROM dynamic_samples GROUP BY gesture")
        else:
            s_rows = db.execute(
                "SELECT gesture,COUNT(*) as cnt FROM static_samples WHERE source=? GROUP BY gesture", (source,)
            )
            d_rows = db.execute(
                "SELECT gesture,COUNT(*) as cnt FROM dynamic_samples WHERE source=? GROUP BY gesture", (source,)
            )
        for row in s_rows:
            meta.setdefault(row['gesture'], {})['static']  = row['cnt']
        for row in d_rows:
            meta.setdefault(row['gesture'], {})['dynamic'] = row['cnt']
        reg = {r['name']: r['gesture_type'] for r in db.execute("SELECT name,gesture_type FROM gesture_registry")}

    result = {}
    for name, gtype in reg.items():
        s = meta.get(name, {}).get('static',  0)
        d = meta.get(name, {}).get('dynamic', 0)
        if gtype == 'static':
            result[name] = {"static": s, "dynamic": d, "ready": s >= MIN_STATIC_SAMPLES,
                            "needed": max(0, MIN_STATIC_SAMPLES - s), "type": gtype}
        else:
            result[name] = {"static": s, "dynamic": d, "ready": d >= MIN_DYNAMIC_SAMPLES,
                            "needed": max(0, MIN_DYNAMIC_SAMPLES - d), "type": gtype}
    return result


# History

@app.get("/api/history")
def get_history(limit: int = 50):
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM session_history ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]

@app.get("/api/history/stats")
def history_stats():
    with get_db() as db:
        total_captures = db.execute(
            "SELECT COUNT(*) FROM session_history WHERE event_type='sample_saved'"
        ).fetchone()[0]
        total_trains = db.execute(
            "SELECT COUNT(*) FROM session_history WHERE event_type='train_batch'"
        ).fetchone()[0]
        last_train = db.execute(
            "SELECT MAX(created_at) FROM session_history WHERE event_type='model_saved'"
        ).fetchone()[0]
        new_since = db.execute(
            "SELECT COUNT(*) FROM session_history "
            "WHERE event_type='sample_saved' "
            "AND created_at > COALESCE((SELECT MAX(created_at) FROM session_history WHERE event_type='model_saved'), 0)"
        ).fetchone()[0]

    return {
        "totalCaptures":            total_captures,
        "totalTrains":              total_trains,
        "lastTrainAt":              last_train * 1000 if last_train else None,
        "newSamplesSinceLastTrain": new_since,
        "needsRetrain":             new_since >= 5,
    }


# Prediction log

@app.post("/api/predictions/log")
def log_prediction(req: PredLogRequest):
    with get_db() as db:
        db.execute(
            "INSERT INTO prediction_log(gesture,confidence,model_type,created_at) VALUES(?,?,?,?)",
            (req.gesture, req.confidence, req.model_type, time.time()),
        )
    return {"ok": True}

@app.get("/api/predictions/recent")
def recent_predictions(limit: int = 20):
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM prediction_log ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# Export / Import

@app.get("/api/export/json")
def export_json():
    data = {
        "version":          "1.0",
        "feature_version":  FEATURE_VERSION,
        "static_features":  STATIC_FEATURES,
        "dynamic_features": DYNAMIC_FEATURES,
        "mqtt_enabled":     _mqtt_enabled,
        "exported_at":      time.time(),
        "static_samples":   {},
        "dynamic_samples":  {},
        "gestures":         [],
    }
    with get_db() as db:
        for row in db.execute("SELECT gesture,sample,quality,created_at FROM static_samples ORDER BY created_at"):
            data["static_samples"].setdefault(row['gesture'], []).append({
                "sample":     json.loads(row['sample']),
                "quality":    row['quality'],
                "created_at": row['created_at'],
            })
        for row in db.execute("SELECT gesture,frames,quality,created_at FROM dynamic_samples ORDER BY created_at"):
            data["dynamic_samples"].setdefault(row['gesture'], []).append({
                "frames":     json.loads(row['frames']),
                "quality":    row['quality'],
                "created_at": row['created_at'],
            })
        for row in db.execute("SELECT * FROM gesture_registry"):
            data["gestures"].append(dict(row))

    return Response(
        content=json.dumps(data, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=signlens_dataset.json"},
    )

@app.post("/api/import/json")
async def import_json(body: dict, _: None = Depends(require_admin)):
    body_fv = body.get("feature_version", "unknown")
    if body_fv not in ("unknown", FEATURE_VERSION, "6.0"):
        return {"error": f"feature_version mismatch: dataset has {body_fv}, current is {FEATURE_VERSION}. Retrain first."}

    now      = time.time()
    imported = {"static": 0, "dynamic": 0, "gestures": 0}

    with get_db() as db:
        for name, samples in body.get("static_samples", {}).items():
            for entry in samples:
                s = entry.get("sample", entry) if isinstance(entry, dict) else entry
                q = entry.get("quality", 0.8)  if isinstance(entry, dict) else 0.8
                db.execute(
                    "INSERT INTO static_samples(gesture,sample,quality,created_at) VALUES(?,?,?,?)",
                    (name, json.dumps(s), q, now),
                )
                imported["static"] += 1

        for name, samples in body.get("dynamic_samples", {}).items():
            for entry in samples:
                f = entry.get("frames", entry) if isinstance(entry, dict) else entry
                q = entry.get("quality", 0.7)  if isinstance(entry, dict) else 0.7
                db.execute(
                    "INSERT INTO dynamic_samples(gesture,frames,quality,created_at) VALUES(?,?,?,?)",
                    (name, json.dumps(f), q, now),
                )
                imported["dynamic"] += 1

        for g in body.get("gestures", []):
            db.execute(
                "INSERT OR IGNORE INTO gesture_registry(name,gesture_type,category,created_at,updated_at) VALUES(?,?,?,?,?)",
                (g['name'], g.get('gesture_type', 'static'), g.get('category', 'custom'), now, now),
            )
            imported["gestures"] += 1

    log_session("dataset_imported", detail=str(imported))
    return {"ok": True, "imported": imported}


# MediaPipe endpoints

@app.get("/api/mediapipe/status")
def mp_status():
    return {"server_side": _mp_ready, "model_exists": MP_MODEL.exists()}

@app.post("/api/mediapipe/process")
async def mp_process(body: dict):
    if not _mp_ready:
        return {"error": "server_mp_unavailable", "fallback": "use_browser_mediapipe"}
    try:
        import mediapipe as mp
        img_bytes = base64.b64decode(body.get("frame", ""))
        arr       = np.frombuffer(img_bytes, dtype=np.uint8)
        mp_img    = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr.reshape(1, -1, 3),
        )
        res = _mp_landmarker.detect(mp_img)
        if not res.hand_landmarks:
            return {"detected": False, "features": None}
        feats = _landmarks_to_features(res.hand_landmarks[0])
        await push("features_extracted", {"features": feats, "source": "server_mp"})
        return {"detected": True, "features": feats}
    except Exception as e:
        return {"error": str(e)}


# MQTT status

@app.get("/api/mqtt/status")
def mqtt_status():
    return {
        "enabled":   _mqtt_enabled,
        "available": MQTT_AVAILABLE,
        "topic":     _mqtt_topic,
        "broker":    os.environ.get("MQTT_BROKER", "broker.hivemq.com"),
        "port":      int(os.environ.get("MQTT_PORT", "1883")),
        "connected": bool(_mqtt_client and _mqtt_client.is_connected()),
    }


# NLP

@app.post("/api/nlp/suggestions")
def get_suggestions(body: dict):
    return {"suggestions": kn_suggest(body.get("words", []), use_personal=True)}

@app.post("/api/nlp/context_suggestions")
def get_context_suggestions(body: dict):
    import nlp as nlp_module
    words           = body.get("words", [])
    recent_gestures = body.get("recent_gestures", [])
    suggestions     = gesture_context_suggest(words, recent_gestures)
    return {
        "suggestions":     suggestions,
        "personal_active": nlp_module._personal_lm is not None,
        "corpus_size":     len(nlp_module._personal_corpus),
    }

@app.post("/api/nlp/word_suggestions")
def get_word_suggestions(body: dict):
    return {"suggestions": word_prefix_suggest(
        body.get("prefix", ""),
        user_vocab=body.get("user_vocab"),
    )}

@app.post("/api/nlp/learn")
async def nlp_learn(body: dict):
    """Save a spoken sentence to the personal corpus and update the gesture→word map."""
    import nlp as nlp_module
    sentence         = body.get("sentence", "").strip()
    gesture_sequence = body.get("gesture_sequence", [])

    if not sentence:
        return {"ok": False, "error": "empty sentence"}

    nlp_module._personal_corpus.append(sentence)
    if len(nlp_module._personal_corpus) > 500:
        nlp_module._personal_corpus = nlp_module._personal_corpus[-500:]

    words = sentence.lower().split()
    for i, g in enumerate(gesture_sequence):
        if g and i < len(words):
            nlp_module._gesture_map.setdefault(g, []).append(words[i])
            if len(nlp_module._gesture_map[g]) > 100:
                nlp_module._gesture_map[g] = nlp_module._gesture_map[g][-100:]

    with get_db() as db:
        db.execute(
            "INSERT INTO personal_corpus(sentence,gesture_sequence,created_at) VALUES(?,?,?)",
            (sentence, json.dumps(gesture_sequence), time.time()),
        )
    log_session("sentence_learned", detail=sentence[:50])
    return {"ok": True, "corpus_size": len(nlp_module._personal_corpus)}

@app.post("/api/nlp/retrain")
async def nlp_retrain():
    import threading
    threading.Thread(target=build_personal_model, daemon=True).start()
    return {"ok": True, "sentences": len(_nlp_module._personal_corpus), "message": "Retraining in background"}

@app.get("/api/nlp/stats")
def nlp_stats():
    import nlp as nlp_module
    return {
        "corpus_size":         len(nlp_module._personal_corpus),
        "personal_model_active": nlp_module._personal_lm is not None,
        "gesture_map_size":    len(nlp_module._gesture_map),
        "base_vocab_size":     len(nlp_module._base_vocab),
    }

@app.post("/api/nlp/grammar")
def nlp_grammar(body: dict):
    sentence  = body.get("sentence", "")
    corrected = offline_grammar_correct(sentence)
    return {"ok": True, "original": sentence, "corrected": corrected, "changed": corrected is not None}


# Sentence

@app.get("/api/sentence")
def get_sentence():
    return sentence_state.to_dict()

@app.post("/api/sentence")
async def update_sentence(req: SentenceRequest):
    s = sentence_state
    if   req.action == 'add_word'               and req.word:   s.add_word(req.word)
    elif req.action == 'remove_last':
        if s.spelling: s.spelling = s.spelling[:-1]
        elif s.words:  s.words.pop()
    elif req.action == 'clear':
        s.words = []; s.spelling = ''; s.completion = None; s.word_suggestions = []
    elif req.action == 'add_letter'             and req.letter: s.spelling += req.letter.lower()
    elif req.action == 'accept_word_suggestion' and req.word:   s.add_word(req.word); s.spelling = ''
    elif req.action == 'accept_completion':
        if s.completion:
            s.words = s.completion.lower().split(); s.completion = None
    elif req.action == 'replace_corrected'      and req.text:   s.words = req.text.lower().split()
    elif req.action == 'set_suggestions'        and req.word:   s.suggestions = req.word.split(',')
    elif req.action == 'set_completion':         s.completion = req.text

    await push("sentence_updated", s.to_dict())
    return s.to_dict()


# Settings

@app.get("/api/settings/conf_thresholds")
def get_all_gesture_thresholds():
    return {"global": _conf_threshold, "per_gesture": _gesture_thresholds}

@app.get("/api/settings/conf_threshold/{gesture}")
def get_gesture_threshold(gesture: str):
    return {"gesture": gesture, "threshold": _gesture_thresholds.get(gesture, _conf_threshold)}

@app.post("/api/settings/conf_threshold/{gesture}")
def set_gesture_threshold(gesture: str, req: ConfThresholdRequest):
    global _gesture_thresholds
    val = max(0.0, min(1.0, req.threshold))
    _gesture_thresholds[gesture] = val
    with get_db() as db:
        db.execute(
            "INSERT OR REPLACE INTO settings VALUES(?,?)",
            (f"conf_threshold_{gesture}", str(val)),
        )
    return {"ok": True, "gesture": gesture, "threshold": val}

@app.get("/api/settings/gemini_key")
def get_gemini_key():
    """Tell the frontend whether a Gemini key is available (env or DB). Never expose the key itself."""
    key = _get_effective_gemini_key()
    return {"available": bool(key)}

@app.get("/api/settings/{key}")
def get_setting(key: str):
    with get_db() as db:
        row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return {"value": row['value'] if row else None}

@app.post("/api/settings")
def save_setting(req: SettingRequest):
    with get_db() as db:
        db.execute("INSERT OR REPLACE INTO settings VALUES(?,?)", (req.key, req.value))
    return {"ok": True}

@app.delete("/api/settings/{key}")
def delete_setting(key: str):
    with get_db() as db:
        db.execute("DELETE FROM settings WHERE key=?", (key,))
    return {"ok": True}


# ── Gemini proxy ─────────────────────────────────────────────────────────────
# All Gemini calls go through here so the API key never reaches the browser.

def _get_effective_gemini_key() -> str:
    """Env var takes priority; fall back to user-supplied key stored in DB."""
    if _GEMINI_API_KEY:
        return _GEMINI_API_KEY
    with get_db() as db:
        row = db.execute("SELECT value FROM settings WHERE key='apiKey'").fetchone()
    if row and row['value']:
        try:
            import json as _json
            return _json.loads(row['value']).strip()
        except Exception:
            return str(row['value']).strip()
    return ""

_GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

async def _gemini_call(prompt: str, temperature: float = 0.4, max_tokens: int = 50) -> Optional[str]:
    key = _get_effective_gemini_key()
    if not key:
        return None
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens, "topP": 0.8},
    }
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(f"{_GEMINI_ENDPOINT}?key={key}", json=payload)
        if not r.is_success:
            return None
        d = r.json()
        return d["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"[Gemini] call failed: {e}")
        return None

class GeminiSuggestRequest(BaseModel):
    sentence: str = ""
    last_word: str = ""
    context: str = ""

class GeminiCompleteRequest(BaseModel):
    sentence: str

class GeminiGrammarRequest(BaseModel):
    sentence: str

class GeminiKeyRequest(BaseModel):
    key: str

@app.post("/api/gemini/suggestions")
async def gemini_suggestions(req: GeminiSuggestRequest):
    prompt = (
        f'You are a predictive text assistant for sign language. User builds sentences word by word.\n'
        f'Context: "{req.context}"\nSentence: "{req.sentence or "(empty)"}"\n'
        + (f'Last word: "{req.last_word}"\n' if req.last_word else '')
        + 'Suggest exactly 5 next words. Respond ONLY with JSON array like ["w1","w2","w3","w4","w5"]'
    )
    text = await _gemini_call(prompt, temperature=0.4, max_tokens=50)
    if not text:
        return {"suggestions": None}
    import re
    m = re.search(r'\[[\s\S]*?\]', text.replace("```json", "").replace("```", ""))
    if m:
        try:
            words = json.loads(m.group(0))
            if isinstance(words, list) and words:
                return {"suggestions": [str(w).lower() for w in words[:5]]}
        except Exception:
            pass
    return {"suggestions": None}

@app.post("/api/gemini/complete")
async def gemini_complete(req: GeminiCompleteRequest):
    if not req.sentence:
        return {"completion": None}
    text = await _gemini_call(
        f'Complete naturally: "{req.sentence}"\nReturn ONLY completed sentence, no quotes, under 10 words.',
        temperature=0.3, max_tokens=30,
    )
    return {"completion": text}

@app.post("/api/gemini/grammar")
async def gemini_grammar(req: GeminiGrammarRequest):
    if not req.sentence:
        return {"corrected": None}
    text = await _gemini_call(
        f'Fix grammar: "{req.sentence}"\nReturn ONLY corrected sentence.',
        temperature=0.1, max_tokens=50,
    )
    return {"corrected": text}

@app.post("/api/gemini/test")
async def gemini_test(req: GeminiKeyRequest):
    """Store a user-supplied key in DB and verify it works."""
    # Save to DB first so _get_effective_gemini_key picks it up
    with get_db() as db:
        db.execute("INSERT OR REPLACE INTO settings VALUES(?,?)", ("apiKey", json.dumps(req.key)))
    text = await _gemini_call("Say OK", temperature=0.1, max_tokens=5)
    return {"ok": bool(text)}


# Frontend static files (must be last)

@app.get("/service-worker.js")
async def sw_route():
    from fastapi.responses import FileResponse
    return FileResponse(str(FRONTEND / "service-worker.js"), media_type="application/javascript")

@app.get("/sensor")
@app.get("/sensor.html")
async def sensor_route():
    from fastapi.responses import FileResponse
    return FileResponse(str(FRONTEND / "sensor.html"), media_type="text/html")

app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

"""
Gesture Detection v1.0 — Python Backend
MediaPipe Holistic: 41 features (hands + face + body)
Optimised MLP + LSTM, adaptive NLP, SQLite persistence
"""


import asyncio, json, math, random, sqlite3, os, time, base64
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from nltk.lm import KneserNeyInterpolated
try:
    import paho.mqtt.client as mqtt_client
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
from nltk.lm.preprocessing import padded_everygram_pipeline

MP_AVAILABLE = False
try:
    from mediapipe.tasks.python import vision as mp_vision, BaseOptions as MP_BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    MP_AVAILABLE = True
except Exception:
    pass

BASE     = Path(__file__).parent
DB_PATH  = BASE / "signlens.db"
FRONTEND = BASE.parent / "frontend"
MP_MODEL = BASE / "hand_landmarker.task"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DYNAMIC_FRAMES   = 45
DYNAMIC_FEATURES = 41   # Phase 2A: 11+11+2 presence flags
STATIC_FEATURES  = 41   # Phase 2A: 11+11+2 presence flags
FEATURE_VERSION  = "1.0"
MIN_STATIC_SAMPLES  = 10
MIN_DYNAMIC_SAMPLES = 5

# ===========================================================================
# Admin auth
# ===========================================================================
_ADMIN_PIN = os.environ.get("ADMIN_PIN", "").strip()
if not _ADMIN_PIN:
    print("[WARN] ADMIN_PIN not set — destructive endpoints are unprotected. Set ADMIN_PIN in .env.")

def require_admin(x_admin_pin: str = Header(default="")):
    """Enforce X-Admin-Pin header on destructive endpoints when ADMIN_PIN is configured."""
    if _ADMIN_PIN and x_admin_pin != _ADMIN_PIN:
        raise HTTPException(status_code=401, detail="Invalid or missing admin PIN")

# ===========================================================================
# WebSocket manager
# ===========================================================================
class ConnectionManager:
    def __init__(self): self.active = []
    async def connect(self, ws):
        await ws.accept(); self.active.append(ws)
    def disconnect(self, ws):
        if ws in self.active: self.active.remove(ws)
    async def broadcast(self, data):
        dead = []
        for ws in self.active:
            try: await ws.send_json(data)
            except: dead.append(ws)
        for ws in dead: self.disconnect(ws)

manager = ConnectionManager()
async def push(event, data={}):
    await manager.broadcast({"event": event, **data})

# ===========================================================================
# DB schema v6
# ===========================================================================
GESTURE_TEMPLATES = {
    'Hello':    {'curls':[.1,.1,.1,.1,.1],'ori':[0,-.8,.1,.02,-.01,.01],'type':'dynamic'},
    'Thank You':{'curls':[.2,.3,.7,.7,.5],'ori':[.1,-.6,.3,.01,.02,-.01],'type':'dynamic'},
    'Yes':      {'curls':[.4,.8,.8,.8,.7],'ori':[0,-.9,0,0,.05,0],'type':'dynamic'},
    'No':       {'curls':[.3,.2,.8,.8,.7],'ori':[0,-.3,.1,.05,0,0],'type':'dynamic'},
    'Help':     {'curls':[.1,.1,.1,.1,.8],'ori':[-.3,-.5,.5,.01,-.02,.02],'type':'dynamic'},
    'Please':   {'curls':[.6,.2,.2,.2,.6],'ori':[.1,-.5,-.2,-.01,.01,.03],'type':'dynamic'},
    'Sorry':    {'curls':[.5,.5,.5,.5,.5],'ori':[0,-.3,.6,-.01,-.02,.01],'type':'dynamic'},
    'Stop':     {'curls':[0,0,0,0,0],    'ori':[0,-1,0,0,0,0],          'type':'static'},
    'Go':       {'curls':[.7,.1,.8,.8,.8],'ori':[.3,-.4,.1,.02,-.03,.01],'type':'dynamic'},
    'Water':    {'curls':[.3,.7,.7,.7,.7],'ori':[.1,-.5,-.2,.01,.01,-.02],'type':'dynamic'},
    'A':{'curls':[.3,1,1,1,1],    'ori':[0,-.7,0,0,0,0],'type':'static'},
    'B':{'curls':[.8,0,0,0,0],    'ori':[0,-1,0,0,0,0], 'type':'static'},
    'C':{'curls':[.4,.4,.4,.4,.4],'ori':[.2,-.7,.1,0,0,0],'type':'static'},
    'D':{'curls':[.2,.1,.8,.8,.8],'ori':[0,-.8,0,0,0,0],'type':'static'},
    'E':{'curls':[.7,.8,.8,.8,.8],'ori':[0,-.8,0,0,0,0],'type':'static'},
    'F':{'curls':[.5,.1,.8,.8,.8],'ori':[0,-.7,0,0,0,0],'type':'static'},
    'G':{'curls':[.3,.1,.9,.9,.9],'ori':[0,-.5,.2,0,0,0],'type':'static'},
    'H':{'curls':[.8,.1,.1,.9,.9],'ori':[0,-.6,0,0,0,0],'type':'static'},
    'I':{'curls':[.8,.9,.9,.9,.1],'ori':[0,-.8,0,0,0,0],'type':'static'},
    'J':{'curls':[.8,.9,.9,.9,.1],'ori':[0,-.8,0,0,0,.1],'type':'dynamic'},
    'K':{'curls':[.3,0,0,.9,.9],  'ori':[0,-.8,.1,0,0,0],'type':'static'},
    'L':{'curls':[.1,.1,.9,.9,.9],'ori':[0,-.6,0,0,0,0],'type':'static'},
    'M':{'curls':[.6,.8,.8,.8,.9],'ori':[0,-.5,-.3,0,0,0],'type':'static'},
    'N':{'curls':[.6,.8,.8,.9,.9],'ori':[0,-.5,-.3,0,0,0],'type':'static'},
    'O':{'curls':[.5,.5,.5,.5,.5],'ori':[0,-.7,0,0,0,0],'type':'static'},
    'P':{'curls':[.3,0,0,.9,.9],  'ori':[.3,-.3,-.3,.02,0,0],'type':'static'},
    'Q':{'curls':[.3,0,.9,.9,.9], 'ori':[.3,-.2,-.4,.02,0,0],'type':'static'},
    'R':{'curls':[.8,0,0,.9,.9],  'ori':[0,-.8,0,0,0,0],'type':'static'},
    'S':{'curls':[.5,1,1,1,1],    'ori':[0,-.8,0,0,0,0],'type':'static'},
    'T':{'curls':[.4,.9,.9,.9,.9],'ori':[0,-.8,0,0,0,0],'type':'static'},
    'U':{'curls':[.8,0,0,.9,.9],  'ori':[0,-1,0,0,0,0],'type':'static'},
    'V':{'curls':[.8,0,0,.9,.9],  'ori':[0,-.9,.15,0,0,0],'type':'static'},
    'W':{'curls':[.7,.1,.1,.1,.9],'ori':[0,-.8,0,0,0,0],'type':'static'},
    'X':{'curls':[.8,.5,.9,.9,.9],'ori':[0,-.8,0,0,0,0],'type':'static'},
    'Y':{'curls':[.1,.9,.9,.9,.1],'ori':[0,-.7,0,0,0,0],'type':'static'},
    'Z':{'curls':[.8,0,.9,.9,.9], 'ori':[0,-.7,0,.04,0,.05],'type':'dynamic'},
    '0':{'curls':[.6,.6,.6,.6,.6],'ori':[0,-.8,0,0,0,0],'type':'static'},
    '1':{'curls':[.8,0,.9,.9,.9], 'ori':[0,-.9,0,0,0,0],'type':'static'},
    '2':{'curls':[.8,0,0,.9,.9],  'ori':[0,-.9,0,0,0,0],'type':'static'},
    '3':{'curls':[.8,0,0,0,.9],   'ori':[0,-.9,0,0,0,0],'type':'static'},
    '4':{'curls':[.8,0,0,0,0],    'ori':[0,-.9,0,0,0,0],'type':'static'},
    '5':{'curls':[0,0,0,0,0],     'ori':[0,-.9,0,0,0,0],'type':'static'},
    '6':{'curls':[.7,.9,.9,0,0],  'ori':[0,-.8,0,0,0,0],'type':'static'},
    '7':{'curls':[.7,.9,0,.9,0],  'ori':[0,-.8,0,0,0,0],'type':'static'},
    '8':{'curls':[.7,0,.9,0,0],   'ori':[0,-.8,0,0,0,0],'type':'static'},
    '9':{'curls':[.7,0,.9,.9,.9], 'ori':[0,-.8,.1,0,0,0],'type':'static'},
}
DEFAULT_GESTURES = ['Hello','Thank You','Yes','No','Help','Please','Sorry','Stop','Go','Water']

def clamp(v,a,b): return max(a,min(b,v))
def noise(s=0.1): return (random.random()-.5)*s

def _sim_one_hand(name, mirror=False):
    """Simulate 11 features for one hand."""
    t=GESTURE_TEMPLATES.get(name)
    if not t: return [random.random()*.5 for _ in range(11)]
    curls=[clamp(c+noise(.12),0,1) for c in t['curls']]
    ori  =[o+noise(.03) for o in t['ori']]
    if mirror: ori[0]=-ori[0]; ori[3]=-ori[3]
    return curls+ori

def _sim_face_features(dom=None):
    """Simulate 8 face features: nose_x, nose_y, eye_dist, h1_nose_dx, h1_nose_dy, h2_nose_dx, h2_nose_dy, tilt"""
    nose_x = random.uniform(0.4, 0.6)
    nose_y = random.uniform(0.3, 0.6)
    eye_dist = random.uniform(0.3, 0.5)
    h1nx = clamp((dom[5] - nose_x + noise(0.05)) if dom else noise(0.1), -1, 1)
    h1ny = clamp((dom[6] - nose_y + noise(0.05)) if dom else noise(0.1), -1, 1)
    tilt = noise(0.05)
    return [nose_x, nose_y, eye_dist, h1nx, h1ny, 0.0, 0.0, tilt]

def _sim_pose_features():
    """Simulate 6 pose features: sh_mid_x, sh_mid_y, h1_sh_dy, h2_sh_dy, elbow_angle, body_visible"""
    return [
        random.uniform(0.45, 0.55),  # shoulder mid x
        random.uniform(0.4, 0.6),   # shoulder mid y
        clamp(noise(0.2), -1, 1),   # hand1 to shoulder dy
        0.0,                         # hand2 absent
        random.uniform(0.3, 0.9),   # elbow angle
        1.0,                         # body visible
    ]

def simulate_static(name, mirror_aug=False):
    """Produce 41-feature Holistic vector: [dom×11]+[aux×11]+[face×8]+[pose×6]+[flags×3]."""
    dom   = _sim_one_hand(name)
    aux   = [0]*11
    face  = _sim_face_features(dom)
    pose  = _sim_pose_features()
    flags = [1, 0, 1]  # dom_present, aux_absent, face_present
    vec   = dom+aux+face+pose+flags
    # Ensure exactly 41
    while len(vec) < STATIC_FEATURES: vec.append(0.0)
    vec = vec[:STATIC_FEATURES]
    if mirror_aug:
        dom_m  = _sim_one_hand(name, mirror=True)
        face_m = _sim_face_features(dom_m)
        vec_m  = dom_m+aux+face_m+pose+flags
        while len(vec_m) < STATIC_FEATURES: vec_m.append(0.0)
        return vec, vec_m[:STATIC_FEATURES]
    return vec, None

def simulate_dynamic(name,frames=DYNAMIC_FRAMES):
    """Produce flat sequence: frames × 41 features = 1845 floats (Holistic)."""
    t=GESTURE_TEMPLATES.get(name)
    if not t: return None
    out=[]
    face_base=_sim_face_features()
    pose_base=_sim_pose_features()
    for f in range(frames):
        p=f/frames; mn=math.sin(p*math.pi*2)*.1
        curls=[clamp(c*p+noise(.08),0,1) for c in t['curls']]
        ori  =[o*p+(mn if i<3 else noise(.02))*.3 for i,o in enumerate(t['ori'])]
        dom  =curls+ori          # 11 features dominant hand
        aux  =[0]*11             # 11 zeros (no aux hand)
        flags=[1,0]              # dom_present=1, aux_present=0
        out.extend(dom+aux+flags)
    return out

def score_sample_quality(vec):
    arr=np.array(vec[:5])
    if len(arr)==0: return 0.5
    variance=float(np.var(arr))
    clipping=float(np.mean((arr<0.02)|(arr>0.98)))
    return round(min(1.0,max(0.1,0.5+variance*4-clipping*0.3)),3)

def get_db():
    conn=sqlite3.connect(DB_PATH); conn.row_factory=sqlite3.Row; return conn

def _get_cols(db, table):
    try: return [r[1] for r in db.execute(f"PRAGMA table_info({table})")]
    except: return []

def _migrate_old_samples(db, now):
    """Migrate old blob-style schema (gesture PK, samples JSON-array) to v6 row-per-sample schema."""
    sc = _get_cols(db, 'static_samples')
    dc = _get_cols(db, 'dynamic_samples')
    old_static, old_dynamic = {}, {}

    # Old schema: gesture TEXT PK, samples TEXT (json array of arrays)
    if sc and 'id' not in sc and 'sample' not in sc:
        print("[DB] Migrating old static_samples schema...")
        try:
            for row in db.execute("SELECT gesture, samples FROM static_samples"):
                arr = json.loads(row[0 if not row['gesture'] else 'samples'] if 'samples' in sc else row[1] or '[]')
                if isinstance(arr, list): old_static[row['gesture']] = arr
        except Exception as e: print(f"[DB] static backup error: {e}")
        db.execute("DROP TABLE static_samples")

    if dc and 'id' not in dc and 'frames' not in dc:
        print("[DB] Migrating old dynamic_samples schema...")
        try:
            col = 'samples' if 'samples' in dc else (dc[1] if len(dc)>1 else None)
            if col:
                for row in db.execute(f"SELECT gesture, {col} FROM dynamic_samples"):
                    arr = json.loads(row[1] or '[]')
                    if isinstance(arr, list): old_dynamic[row['gesture']] = arr
        except Exception as e: print(f"[DB] dynamic backup error: {e}")
        db.execute("DROP TABLE dynamic_samples")

    return old_static, old_dynamic

def _restore_migrated(db, old_static, old_dynamic, now):
    for gesture, samples in old_static.items():
        for s in samples:
            if isinstance(s, list) and len(s) >= 5:
                q = score_sample_quality(s)
                db.execute("INSERT INTO static_samples(gesture,sample,quality,created_at) VALUES(?,?,?,?)",
                           (gesture, json.dumps(s), q, now))
    for gesture, samples in old_dynamic.items():
        for s in samples:
            if isinstance(s, list) and len(s) >= 5:
                q = score_sample_quality(s[:5])
                db.execute("INSERT INTO dynamic_samples(gesture,frames,quality,created_at) VALUES(?,?,?,?)",
                           (gesture, json.dumps(s), q, now))
    if old_static or old_dynamic:
        print(f"[DB] Restored {sum(len(v) for v in old_static.values())} static, "
              f"{sum(len(v) for v in old_dynamic.values())} dynamic samples")

def init_db():
    now = time.time()
    with get_db() as db:
        # Run migration BEFORE CREATE TABLE IF NOT EXISTS (tables may be old-schema)
        old_static, old_dynamic = _migrate_old_samples(db, now)

        db.executescript("""
            CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT);
            CREATE TABLE IF NOT EXISTS gesture_registry(
                name TEXT PRIMARY KEY,gesture_type TEXT NOT NULL DEFAULT 'static',
                category TEXT NOT NULL DEFAULT 'custom',created_at REAL,updated_at REAL);
            CREATE TABLE IF NOT EXISTS static_samples(
                id INTEGER PRIMARY KEY AUTOINCREMENT,gesture TEXT NOT NULL,
                sample TEXT NOT NULL,quality REAL DEFAULT 1.0,
                source TEXT DEFAULT 'camera',created_at REAL);
            CREATE TABLE IF NOT EXISTS dynamic_samples(
                id INTEGER PRIMARY KEY AUTOINCREMENT,gesture TEXT NOT NULL,
                frames TEXT NOT NULL,quality REAL DEFAULT 1.0,
                source TEXT DEFAULT 'camera',created_at REAL);
            CREATE TABLE IF NOT EXISTS models(id TEXT PRIMARY KEY,data TEXT,source TEXT DEFAULT 'camera',updated_at REAL);
            CREATE TABLE IF NOT EXISTS session_history(
                id INTEGER PRIMARY KEY AUTOINCREMENT,event_type TEXT NOT NULL,
                gesture TEXT,detail TEXT,created_at REAL);
            CREATE TABLE IF NOT EXISTS prediction_log(
                id INTEGER PRIMARY KEY AUTOINCREMENT,gesture TEXT NOT NULL,
                confidence REAL NOT NULL,model_type TEXT NOT NULL,created_at REAL);
            CREATE INDEX IF NOT EXISTS idx_static_g  ON static_samples(gesture);
            CREATE INDEX IF NOT EXISTS idx_dynamic_g ON dynamic_samples(gesture);
            CREATE TABLE IF NOT EXISTS personal_corpus(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sentence TEXT NOT NULL,
                gesture_sequence TEXT,
                created_at REAL
            );
        """)

        # Column-level migrations for partially-updated DBs
        sc = _get_cols(db, 'static_samples')
        dc = _get_cols(db, 'dynamic_samples')
        gc = _get_cols(db, 'gesture_registry')
        mc = _get_cols(db, 'models')
        if 'quality'      not in sc: db.execute("ALTER TABLE static_samples  ADD COLUMN quality REAL DEFAULT 1.0")
        if 'quality'      not in dc: db.execute("ALTER TABLE dynamic_samples ADD COLUMN quality REAL DEFAULT 1.0")
        if 'source'       not in sc: db.execute("ALTER TABLE static_samples  ADD COLUMN source TEXT DEFAULT 'camera'")
        if 'source'       not in dc: db.execute("ALTER TABLE dynamic_samples ADD COLUMN source TEXT DEFAULT 'camera'")
        if 'gesture_type' not in gc: db.execute("ALTER TABLE gesture_registry ADD COLUMN gesture_type TEXT NOT NULL DEFAULT 'static'")
        if 'category'     not in gc: db.execute("ALTER TABLE gesture_registry ADD COLUMN category     TEXT NOT NULL DEFAULT 'custom'")
        if 'updated_at'   not in mc: db.execute("ALTER TABLE models ADD COLUMN updated_at REAL")
        if 'source'       not in mc: db.execute("ALTER TABLE models ADD COLUMN source TEXT DEFAULT 'camera'")

        # Restore migrated data
        _restore_migrated(db, old_static, old_dynamic, now)

    _seed_registry()
    _load_personal_corpus()

def _load_personal_corpus():
    """Load saved personal corpus from DB on startup."""
    global _personal_corpus,_gesture_map
    try:
        with get_db() as db:
            rows=db.execute("SELECT sentence,gesture_sequence FROM personal_corpus ORDER BY created_at").fetchall()
        for row in rows:
            sentence=row['sentence']
            _personal_corpus.append(sentence)
            gestures=json.loads(row['gesture_sequence'] or '[]')
            words=sentence.lower().split()
            for i,g in enumerate(gestures):
                if g and i<len(words):
                    if g not in _gesture_map: _gesture_map[g]=[]
                    _gesture_map[g].append(words[i])
        if _personal_corpus:
            print(f"[NLP] Loaded {len(_personal_corpus)} personal sentences from DB")
            _build_personal_lm()
    except Exception as e:
        print(f"[NLP] Personal corpus load skipped: {e}")

def _seed_registry():
    now=time.time()
    defaults=[]
    for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        defaults.append((ch,'dynamic' if ch in('J','Z') else 'static','alphabet'))
    for d in range(10):
        defaults.append((str(d),'static','number'))
    for name,meta in GESTURE_TEMPLATES.items():
        if name not in [x[0] for x in defaults]:
            defaults.append((name,meta.get('type','static'),'word'))
    with get_db() as db:
        for name,gtype,cat in defaults:
            db.execute("INSERT OR IGNORE INTO gesture_registry(name,gesture_type,category,created_at,updated_at) VALUES(?,?,?,?,?)",
                       (name,gtype,cat,now,now))

def log_session(event_type,gesture=None,detail=None):
    with get_db() as db:
        db.execute("INSERT INTO session_history(event_type,gesture,detail,created_at) VALUES(?,?,?,?)",
                   (event_type,gesture,detail,time.time()))

# ===========================================================================
# Static Model — MLP with BatchNorm
# ===========================================================================
class GestureNet(nn.Module):
    def __init__(self,input_size,hidden_sizes,output_size):
        super().__init__()
        layers=[]; prev=input_size
        for h in hidden_sizes:
            layers+=[nn.Linear(prev,h),nn.BatchNorm1d(h),nn.ReLU(),nn.Dropout(0.15)]
            prev=h
        layers.append(nn.Linear(prev,output_size))
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class PyTorchNN:
    def __init__(self,nn_id='static'):
        self.id=nn_id; self.model=None; self.gestures={}
        self.feature_version=FEATURE_VERSION
        self.trained=False; self.accuracy=0.0; self.loss=1.0
        self.epochs=0; self._input_size=0; self._hidden_sizes=[]; self._output_size=0
        self.per_gesture_acc={}

    def initialize(self,input_size,hidden_sizes,output_size):
        self._input_size=input_size; self._hidden_sizes=hidden_sizes; self._output_size=output_size
        self.model=GestureNet(input_size,hidden_sizes,output_size).to(DEVICE)

    def train_batch(self,inputs,labels,epochs=10,lr=0.008):
        if not inputs or self.model is None: return
        # Normalise all inputs to expected input size — handles mixed old/new samples
        normed=[]
        for s in inputs:
            if len(s)<self._input_size:
                s=list(s)+[0.0]*(self._input_size-len(s))
            normed.append(s[:self._input_size])
        X=torch.tensor(normed,dtype=torch.float32).to(DEVICE)
        y=torch.tensor(labels,dtype=torch.long).to(DEVICE)
        opt=optim.Adam(self.model.parameters(),lr=lr,weight_decay=1e-4)
        sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,epochs))
        # Weighted loss for class balance (Phase 2A)
        try:
            counts=torch.bincount(y,minlength=len(self.gestures)).float()
            w=1.0/(counts+1e-6); w=w/w.sum()*len(self.gestures); w=w.to(DEVICE)
        except: w=None
        crit=nn.CrossEntropyLoss(weight=w) if w is not None else nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(epochs):
            opt.zero_grad(); loss=crit(self.model(X),y); loss.backward(); opt.step(); sched.step()
        self.model.eval()
        with torch.no_grad():
            out=self.model(X); preds=out.argmax(dim=1)
            self.accuracy=(preds==y).float().mean().item(); self.loss=crit(out,y).item()
            self.per_gesture_acc={}
            for idx,name in self.gestures.items():
                mask=(y==idx)
                if mask.sum()>0: self.per_gesture_acc[name]=(preds[mask]==y[mask]).float().mean().item()
        self.epochs+=epochs; self.trained=True

    def predict(self,features,conf_threshold=0.0):
        if not self.trained or self.model is None:
            return {"idx":-1,"conf":0.0,"probs":[],"name":"Unknown","below_threshold":False}
        self.model.eval()
        pad=features+[0.0]*max(0,self._input_size-len(features))
        x=torch.tensor([pad[:self._input_size]],dtype=torch.float32).to(DEVICE)
        with torch.no_grad(): probs=torch.softmax(self.model(x),dim=1)[0]
        idx=int(probs.argmax().item()); conf=float(probs[idx].item())
        return {"idx":idx,"conf":conf,"probs":probs.tolist(),"name":self.gestures.get(idx,"Unknown"),"below_threshold":conf<conf_threshold}

    def add_gesture(self,name,idx): self.gestures[idx]=name
    def reset(self):
        self.model=None; self.trained=False; self.gestures={}
        self.accuracy=0.0; self.loss=1.0; self.epochs=0; self.per_gesture_acc={}

    def to_json(self):
        state={k:v.tolist() for k,v in self.model.state_dict().items()} if self.model else {}
        return {"id":self.id,"state_dict":state,"gestures":list(self.gestures.items()),
                "accuracy":self.accuracy,"loss":self.loss,"epochs":self.epochs,
                "input_size":self._input_size,"hidden_sizes":self._hidden_sizes,
                "output_size":self._output_size,"per_gesture_acc":self.per_gesture_acc}

    def from_json(self,d):
        self._input_size=d.get("input_size",0); self._hidden_sizes=d.get("hidden_sizes",[])
        self._output_size=d.get("output_size",0)
        if self._input_size and self._output_size:
            self.initialize(self._input_size,self._hidden_sizes,self._output_size)
            if d.get("state_dict"):
                self.model.load_state_dict({k:torch.tensor(v) for k,v in d["state_dict"].items()})
                self.model.to(DEVICE)
        self.gestures={int(k):v for k,v in d.get("gestures",[])}
        self.accuracy=d.get("accuracy",0.0); self.loss=d.get("loss",1.0)
        self.epochs=d.get("epochs",0); self.trained=bool(self._input_size and d.get("state_dict"))
        self.per_gesture_acc=d.get("per_gesture_acc",{})

# ===========================================================================
# Dynamic Model — LSTM with temporal attention (v6 upgrade)
# ===========================================================================
class GestureLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size,dropout=0.3):
        super().__init__()
        self.hidden_size=hidden_size; self.num_layers=num_layers
        self.lstm=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,
                          dropout=dropout if num_layers>1 else 0.0,bidirectional=False)
        self.attn=nn.Linear(hidden_size,1)
        self.dropout=nn.Dropout(dropout)
        self.norm=nn.LayerNorm(hidden_size)
        self.fc=nn.Linear(hidden_size,output_size)
    def forward(self,x):
        h0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(x.device)
        c0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(x.device)
        out,_=self.lstm(x,(h0,c0))
        attn_w=torch.softmax(self.attn(out),dim=1)
        ctx=(attn_w*out).sum(dim=1)
        return self.fc(self.dropout(self.norm(ctx)))

class LSTMModel:
    def __init__(self):
        self.model=None; self.gestures={}; self.trained=False
        self.accuracy=0.0; self.loss=1.0; self.epochs=0
        self.hidden_size=128; self.num_layers=2
        self._output_size=0; self.per_gesture_acc={}

    def initialize(self,num_classes,gesture_list):
        self._output_size=num_classes
        self.model=GestureLSTM(DYNAMIC_FEATURES,self.hidden_size,self.num_layers,num_classes).to(DEVICE)
        self.gestures={i:name for i,name in enumerate(gesture_list)}

    def _prepare(self,raw_sequences):
        seqs=[]
        for s in raw_sequences:
            arr=np.array(s,dtype=np.float32)
            frames=len(arr)//DYNAMIC_FEATURES
            if frames<DYNAMIC_FRAMES:
                pad=np.zeros(DYNAMIC_FRAMES*DYNAMIC_FEATURES,dtype=np.float32)
                pad[:len(arr)]=arr; arr=pad; frames=DYNAMIC_FRAMES
            seqs.append(arr[:DYNAMIC_FRAMES*DYNAMIC_FEATURES].reshape(DYNAMIC_FRAMES,DYNAMIC_FEATURES))
        return np.array(seqs,dtype=np.float32)

    def train_batch(self,inputs,labels,epochs=10,lr=0.001):
        if not inputs or self.model is None: return
        X=torch.tensor(self._prepare(inputs),dtype=torch.float32).to(DEVICE)
        y=torch.tensor(labels,dtype=torch.long).to(DEVICE)
        opt=optim.Adam(self.model.parameters(),lr=lr,weight_decay=1e-4)
        sched=optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5,factor=0.5,verbose=False)
        # Weighted loss for class balance (Phase 2A)
        try:
            counts=torch.bincount(y,minlength=len(self.gestures)).float()
            w=1.0/(counts+1e-6); w=w/w.sum()*len(self.gestures); w=w.to(DEVICE)
        except: w=None
        crit=nn.CrossEntropyLoss(weight=w) if w is not None else nn.CrossEntropyLoss()
        self.model.train()
        for ep in range(epochs):
            opt.zero_grad(); loss=crit(self.model(X),y); loss.backward(); opt.step()
            if ep%5==0: sched.step(loss)
        self.model.eval()
        with torch.no_grad():
            out=self.model(X); preds=out.argmax(dim=1)
            self.accuracy=(preds==y).float().mean().item(); self.loss=crit(out,y).item()
            self.per_gesture_acc={}
            for idx,name in self.gestures.items():
                mask=(y==idx)
                if mask.sum()>0: self.per_gesture_acc[name]=(preds[mask]==y[mask]).float().mean().item()
        self.epochs+=epochs; self.trained=True

    def predict(self,flat_sequence,conf_threshold=0.0):
        if not self.trained or self.model is None:
            return {"idx":-1,"conf":0.0,"probs":[],"name":"Unknown","below_threshold":False}
        self.model.eval()
        arr=np.array(flat_sequence,dtype=np.float32)
        if len(arr)<DYNAMIC_FRAMES*DYNAMIC_FEATURES:
            pad=np.zeros(DYNAMIC_FRAMES*DYNAMIC_FEATURES,dtype=np.float32)
            pad[:len(arr)]=arr; arr=pad
        arr=arr[:DYNAMIC_FRAMES*DYNAMIC_FEATURES].reshape(1,DYNAMIC_FRAMES,DYNAMIC_FEATURES)
        X=torch.tensor(arr,dtype=torch.float32).to(DEVICE)
        with torch.no_grad(): probs=torch.softmax(self.model(X),dim=1)[0]
        idx=int(probs.argmax().item()); conf=float(probs[idx].item())
        return {"idx":idx,"conf":conf,"probs":probs.tolist(),"name":self.gestures.get(idx,"Unknown"),"below_threshold":conf<conf_threshold}

    def add_gesture(self,name,idx): self.gestures[idx]=name
    def reset(self):
        self.model=None; self.trained=False; self.gestures={}
        self.accuracy=0.0; self.loss=1.0; self.epochs=0; self.per_gesture_acc={}

    def to_json(self):
        state={k:v.tolist() for k,v in self.model.state_dict().items()} if self.model else {}
        return {"type":"lstm","state_dict":state,"gestures":list(self.gestures.items()),
                "accuracy":self.accuracy,"loss":self.loss,"epochs":self.epochs,
                "hidden_size":self.hidden_size,"num_layers":self.num_layers,
                "output_size":self._output_size,"per_gesture_acc":self.per_gesture_acc}

    def from_json(self,d):
        gesture_list=[v for _,v in sorted(d.get("gestures",[]),key=lambda x:int(x[0]))]
        if not gesture_list: return
        self.hidden_size=d.get("hidden_size",128); self.num_layers=d.get("num_layers",2)
        self._output_size=d.get("output_size",len(gesture_list))
        self.initialize(self._output_size,gesture_list)
        if d.get("state_dict"):
            try: self.model.load_state_dict({k:torch.tensor(v) for k,v in d["state_dict"].items()}); self.model.to(DEVICE)
            except Exception as e: print(f"[LSTM] load warning: {e}")
        self.gestures={int(k):v for k,v in d.get("gestures",[])}
        self.accuracy=d.get("accuracy",0.0); self.loss=d.get("loss",1.0)
        self.epochs=d.get("epochs",0); self.trained=bool(d.get("state_dict"))
        self.per_gesture_acc=d.get("per_gesture_acc",{})

static_nn=PyTorchNN("static")
dynamic_nn=LSTMModel()
_conf_threshold=0.65
# One lock per model — prevents concurrent training from corrupting weights
_static_train_lock=asyncio.Lock()
_dynamic_train_lock=asyncio.Lock()

# ===========================================================================
# NLP
# ===========================================================================
WORD_LIST=("the,be,to,of,and,a,in,that,have,I,it,for,not,on,with,he,as,you,do,at,this,but,"
    "his,by,from,they,we,her,she,or,an,will,my,one,all,would,there,their,what,so,up,"
    "out,if,about,who,get,which,go,me,when,make,can,like,time,no,just,him,know,take,"
    "people,into,year,your,good,some,could,them,see,other,than,then,now,look,only,"
    "come,its,over,think,also,back,after,use,two,how,our,work,first,well,way,even,"
    "new,want,because,any,these,give,day,most,us,great,need,help,home,water,food,"
    "right,please,thank,yes,stop,go,sorry,hello,name,where,here,open,close,hot,cold,"
    "big,small,happy,sad,hungry,thirsty,tired,sick,pain,doctor,family,friend,love,"
    "feel,understand,speak,hear,see,wait,eat,drink,sleep,walk,sit,stand,more,less,"
    "very,much,many,few,old,young,man,woman,child,boy,girl,mother,father,brother,"
    "sister,son,daughter,husband,wife,baby,hand,head,eye,ear,nose,mouth,face,arm,"
    "leg,foot,body,heart,back,door,window,house,room,school,work,car,bus,train,road,"
    "today,tomorrow,yesterday,morning,afternoon,evening,night,always,never,sometimes,"
    "soon,already,again,still,before,after,during,while,because,since,until,safe,ready").split(',')

NLP_PHRASES=["hello how are you today","thank you very much please",
    "please help me I need water","yes I understand you completely",
    "no thank you sorry about that","I am hungry please help me",
    "good morning how are you feeling","please wait I need help now",
    "I want to go home now","stop please listen to me carefully",
    "I feel sick I need doctor","open the door for me please",
    "I am very happy today thank","can you hear me please help",
    "I need food and water please","where is the bathroom please help",
    "I do not understand please repeat","can you speak more slowly please",
    "I am tired and need rest","please call my family for me",
    "I need to go to hospital","thank you for your help today",
    "yes please I would like that","no I do not want that",
    "I am sorry I did not","please tell me where to go",
    "I feel better thank you very","can I have some water please",
    "I need help right now please","good morning I am feeling well"]

# ── Base NLP model ─────────────────────────────────────────────────────────
_kn_lm=None; _kn_vocab=set()

# ── Personal corpus model (Phase 2B) ───────────────────────────────────────
_personal_lm=None; _personal_corpus=[]; _gesture_map={}
# _gesture_map: {gesture_name: [words_that_followed]}

def _build_nlp():
    global _kn_lm,_kn_vocab
    tokenized=[p.lower().split() for p in NLP_PHRASES]
    train_data,vocab=padded_everygram_pipeline(3,tokenized)
    lm=KneserNeyInterpolated(3); lm.fit(train_data,vocab)
    _kn_lm=lm; _kn_vocab=set(vocab)
    print(f"[NLP] Kneser-Ney trigram ready")

def _build_personal_lm():
    """Retrain personal model from accumulated corpus sentences."""
    global _personal_lm
    if len(_personal_corpus) < 3:
        _personal_lm=None; return
    try:
        tokenized=[s.lower().split() for s in _personal_corpus if s.strip()]
        train_data,vocab=padded_everygram_pipeline(3,tokenized)
        lm=KneserNeyInterpolated(3); lm.fit(train_data,vocab)
        _personal_lm=lm
        print(f"[NLP] Personal model retrained: {len(_personal_corpus)} sentences")
    except Exception as e:
        print(f"[NLP] Personal retrain failed: {e}")
        _personal_lm=None

def kn_suggest(words, n=5, use_personal=True):
    """Suggest next words using personal model (if available) then base model."""
    ctx=list(words[-2:]) if len(words)>=2 else (list(words[-1:]) if words else [])
    candidates=set(WORD_LIST)|(_kn_vocab-{'<s>','</s>','<UNK>'})
    scores={}

    # Score with personal model (double weight)
    if use_personal and _personal_lm is not None:
        for w in candidates:
            try:
                s=_personal_lm.score(w,ctx)
                if s>0: scores[w]=scores.get(w,0)+s*2.0
            except: pass

    # Score with base model
    if _kn_lm is not None:
        for w in candidates:
            try:
                s=_kn_lm.score(w,ctx)
                if s>0: scores[w]=scores.get(w,0)+s
            except: pass

    if not scores: return['hello','I','please','help','thank']
    return[w for w,_ in sorted(scores.items(),key=lambda x:-x[1])[:n]]

def gesture_context_suggest(words, recent_gestures, n=5):
    """Suggest words weighted by gesture history context."""
    base=kn_suggest(words, n=n*3, use_personal=True)
    if not recent_gestures or not _gesture_map: return base[:n]
    # Boost words that tend to follow these gestures
    boost={}
    for g in recent_gestures:
        for w in _gesture_map.get(g,[]):
            boost[w]=boost.get(w,0)+1
    if not boost: return base[:n]
    scored=[]
    for w in base:
        scored.append((w, boost.get(w,0)))
    # Merge boosted words not in base
    for w,sc in sorted(boost.items(),key=lambda x:-x[1]):
        if w not in[x[0] for x in scored]: scored.append((w,sc))
    scored.sort(key=lambda x:-x[1])
    result=[w for w,_ in scored[:n]]
    return result if result else base[:n]

def word_prefix_suggest(prefix, n=5):
    p=prefix.lower()
    return[w for w in WORD_LIST if w.startswith(p) and w!=p][:n]

def offline_grammar_correct(sentence):
    """Rule-based grammar correction — no Gemini needed."""
    if not sentence or not sentence.strip(): return None
    words=sentence.strip().lower().split()
    changed=False
    ACTION_VERBS={'want','need','like','love','hate','go','see','hear','feel','know','think','eat','drink'}
    # Rule 1: missing "I" at start before action verb
    if words and words[0] in ACTION_VERBS:
        words.insert(0,'i'); changed=True
    # Rule 2: "me" at start → "i"
    if words and words[0]=='me':
        words[0]='i'; changed=True
    # Rule 3: inline "i" → "I"
    words=['I' if w=='i' else w for w in words]
    # Rule 4: capitalise first word
    if words: words[0]=words[0].capitalize()
    # Rule 5: deduplicate consecutive words
    deduped=[words[0]] if words else []
    for i in range(1,len(words)):
        if words[i].lower()!=words[i-1].lower(): deduped.append(words[i])
        else: changed=True
    words=deduped
    # Rule 6: add period if missing
    if words and words[-1][-1] not in '.!?':
        words[-1]+='.' ; changed=True
    return ' '.join(words) if changed else None

# ===========================================================================
# MediaPipe
# ===========================================================================
_mp_ready=False; _mp_landmarker=None
def _init_mp():
    global _mp_ready,_mp_landmarker
    if not MP_AVAILABLE: print("[MediaPipe] not installed — using browser-side"); return
    if not MP_MODEL.exists(): print("[MediaPipe] model not found — using browser-side"); return
    try:
        opts=HandLandmarkerOptions(base_options=MP_BaseOptions(model_asset_path=str(MP_MODEL)),running_mode=RunningMode.IMAGE,num_hands=1)
        _mp_landmarker=HandLandmarker.create_from_options(opts); _mp_ready=True
        print("[MediaPipe] server-side ready")
    except Exception as e: print(f"[MediaPipe] {e} — using browser-side")

def _landmarks_to_features(lm):
    tips=[4,8,12,16,20]; pips=[3,6,10,14,18]; mcps=[2,5,9,13,17]; curls=[]
    for i in range(5):
        t2m=math.sqrt((lm[tips[i]].x-lm[mcps[i]].x)**2+(lm[tips[i]].y-lm[mcps[i]].y)**2+(lm[tips[i]].z-lm[mcps[i]].z)**2)
        p2m=math.sqrt((lm[pips[i]].x-lm[mcps[i]].x)**2+(lm[pips[i]].y-lm[mcps[i]].y)**2+(lm[pips[i]].z-lm[mcps[i]].z)**2)
        curls.append(clamp(1-(t2m/(p2m*2.5) if p2m>0.001 else 0),0,1))
    dx=lm[9].x-lm[0].x; dy=lm[9].y-lm[0].y; dz=lm[9].z-lm[0].z
    l=math.sqrt(dx**2+dy**2+dz**2) or 0.001; hd=[dx/l,dy/l,dz/l]
    sx=lm[17].x-lm[5].x; sy=lm[17].y-lm[5].y; sz=lm[17].z-lm[5].z
    ls=math.sqrt(sx**2+sy**2+sz**2) or 0.001
    return curls+hd+[sx/ls*.1,sy/ls*.1,sz/ls*.1]

# ===========================================================================
# Sentence
# ===========================================================================
class SentenceState:
    def __init__(self):
        self.words=[]; self.spelling=''; self.context=[]
        self.suggestions=['hello','I','please','help','thank']
        self.word_suggestions=[]; self.completion=None
    def add_word(self,w):
        self.words.append(w.lower()); self.context.append(w.lower())
        if len(self.context)>50: self.context=self.context[-30:]
        self.spelling=''; self.word_suggestions=[]
    def to_dict(self):
        disp=' '.join(self.words)+((' ' if self.words else '')+self.spelling+'_' if self.spelling else '')
        return{"words":self.words,"spelling":self.spelling,"context":self.context,
               "suggestions":self.suggestions,"wordSuggestions":self.word_suggestions,
               "completion":self.completion,"sentence":' '.join(self.words),"displayText":disp}

sentence_state=SentenceState()

# ===========================================================================
# MQTT Subscriber
# ===========================================================================
_mqtt_enabled   = False
_mqtt_client    = None
_mqtt_loop: asyncio.AbstractEventLoop | None = None
_mqtt_topic     = os.environ.get("MQTT_TOPIC", "gesture-detection/sensor/features")

def _mqtt_on_connect(client, userdata, flags, reason_code, properties):
    if not reason_code.is_failure:
        client.subscribe(_mqtt_topic)
        print(f"[MQTT] Connected — subscribed to '{_mqtt_topic}'")
    else:
        print(f"[MQTT] Connection refused: {reason_code}")

def _mqtt_on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
    if reason_code.is_failure:
        print(f"[MQTT] Unexpected disconnect: {reason_code} — paho will auto-reconnect")

def _mqtt_on_message(client, userdata, msg):
    """
    Expected payload (JSON):
        {"features": [...], "model_type": "static"|"dynamic"}
    Publishes a 'mqtt_prediction' WebSocket event to all connected clients.
    """
    global _mqtt_loop
    try:
        payload    = json.loads(msg.payload.decode())
        features   = payload.get("features", [])
        model_type = payload.get("model_type", "static")
        if not features:
            return
        nn = static_nn if model_type == "static" else dynamic_nn
        if not nn.trained:
            return
        result = nn.predict(features, _conf_threshold)
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
        # Log high-confidence predictions
        if result["conf"] > 0.3:
            with get_db() as db:
                db.execute(
                    "INSERT INTO prediction_log(gesture,confidence,model_type,created_at) VALUES(?,?,?,?)",
                    (result["name"], result["conf"], model_type, time.time()),
                )
    except Exception as e:
        print(f"[MQTT] message handler error: {e}")

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
        client = mqtt_client.Client(
            client_id=f"gesture-backend-{int(time.time())}",
            callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2,
        )
        if username:
            client.username_pw_set(username, password)
        client.on_connect    = _mqtt_on_connect
        client.on_disconnect = _mqtt_on_disconnect
        client.on_message    = _mqtt_on_message
        client.reconnect_delay_set(min_delay=2, max_delay=30)
        client.connect_async(broker, port, keepalive=60)
        client.loop_start()          # runs paho I/O in a background thread
        _mqtt_client  = client
        _mqtt_enabled = True
        print(f"[MQTT] Connecting to {broker}:{port}  topic='{_mqtt_topic}'")
    except Exception as e:
        print(f"[MQTT] Failed to start subscriber: {e}")
        _mqtt_enabled = False

# ===========================================================================
# Lifespan
# ===========================================================================
@asynccontextmanager
async def lifespan(app):
    init_db(); _build_nlp(); _init_mp()
    # restore conf threshold from DB
    global _conf_threshold
    with get_db() as db:
        row=db.execute("SELECT value FROM settings WHERE key='conf_threshold'").fetchone()
        if row: _conf_threshold=float(row['value'])
    start_mqtt_subscriber()
    print(f"[Gesture Detection v1.0] Ready | device={DEVICE} | 41-feature Holistic | LSTM dynamic | conf_threshold={_conf_threshold}")
    yield
    # --- shutdown ---
    if _mqtt_client:
        _mqtt_client.loop_stop()
        _mqtt_client.disconnect()
        print("[MQTT] Disconnected")

app=FastAPI(title="Gesture Detection v1.0",lifespan=lifespan)
_CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS","http://localhost:8000,http://127.0.0.1:8000").split(",") if o.strip()]
app.add_middleware(CORSMiddleware,allow_origins=_CORS_ORIGINS,allow_methods=["*"],allow_headers=["*"])

# ===========================================================================
# Pydantic
# ===========================================================================
class NNInitRequest(BaseModel):
    model_type:str; input_size:int=0; hidden_sizes:list=[]; output_size:int=0; gestures:list=[]; feature_version:str="6.1"
class NNTrainRequest(BaseModel):
    model_type:str; inputs:list; labels:list; epochs:int=10; lr:float=0.001
class PredictRequest(BaseModel):
    features:list; model_type:str='static'
class SampleSaveRequest(BaseModel):
    gesture:str; samples:list; sample_type:str='static'; source:str='camera'
class SimulateRequest(BaseModel):
    gesture:str; count:int=35; sample_type:str='static'
class GenerateDemoRequest(BaseModel):
    gestures:list
class SentenceRequest(BaseModel):
    action:str; word:Optional[str]=None; letter:Optional[str]=None; text:Optional[str]=None
class SettingRequest(BaseModel):
    key:str; value:str
class GestureRegisterRequest(BaseModel):
    name:str; gesture_type:str='static'; category:str='custom'
class DeleteSamplesRequest(BaseModel):
    gesture:str; sample_type:str='all'; source:str='all'
class ConfThresholdRequest(BaseModel):
    threshold:float
class PredLogRequest(BaseModel):
    gesture:str; confidence:float; model_type:str

# ===========================================================================
# Routes
# ===========================================================================
@app.get("/favicon.ico")
async def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)  # no content — stops 404 noise

@app.get("/api/health")
def health():
    return{"status":"ok","version":"1.0","device":str(DEVICE),"torch":torch.__version__,
           "mediapipe_server":_mp_ready,"dynamic_model":"LSTM"}

@app.websocket("/ws")
async def ws_endpoint(ws:WebSocket):
    await manager.connect(ws)
    try:
        while True:
            msg=await ws.receive_text()
            if msg=="ping": await ws.send_json({"event":"pong"})
    except WebSocketDisconnect: manager.disconnect(ws)

# -- Neural Network --
@app.post("/api/nn/init")
def nn_init(req:NNInitRequest):
    if req.model_type=='dynamic':
        dynamic_nn.initialize(req.output_size,req.gestures)
        dynamic_nn.feature_version=req.feature_version
    else:
        static_nn.initialize(req.input_size,req.hidden_sizes,req.output_size)
        for i,name in enumerate(req.gestures): static_nn.add_gesture(name,i)
        static_nn.feature_version=req.feature_version
    return{"ok":True,"model":req.model_type,"feature_version":req.feature_version}

@app.post("/api/nn/train")
async def nn_train(req:NNTrainRequest, _:None=Depends(require_admin)):
    nn   = static_nn   if req.model_type=='static' else dynamic_nn
    lock = _static_train_lock if req.model_type=='static' else _dynamic_train_lock
    if nn.model is None:
        return{"error":"model not initialised — call /api/nn/init first","accuracy":0,"loss":1,"epochs":0,"per_gesture_acc":{}}
    # Lock prevents concurrent requests from corrupting the same model object.
    # run_in_executor offloads the blocking PyTorch work to a thread so the
    # event loop stays responsive during training.
    async with lock:
        loop=asyncio.get_running_loop()
        await loop.run_in_executor(None, nn.train_batch, req.inputs, req.labels, req.epochs, req.lr)
    await push("train_progress",{"model":req.model_type,"accuracy":nn.accuracy,"loss":nn.loss,
                                  "epochs":nn.epochs,"per_gesture_acc":nn.per_gesture_acc})
    log_session("train_batch",detail=f"{req.model_type} acc={nn.accuracy:.3f}")
    return{"accuracy":nn.accuracy,"loss":nn.loss,"epochs":nn.epochs,"per_gesture_acc":nn.per_gesture_acc}

@app.post("/api/nn/predict")
def nn_predict(req:PredictRequest):
    nn=static_nn if req.model_type=='static' else dynamic_nn
    if not nn.trained:
        return{"idx":-1,"conf":0.0,"probs":[],"name":"Unknown","below_threshold":True}
    # Feature size mismatch check
    expected=getattr(nn,'_input_size',None)
    if expected and len(req.features) not in (expected, expected*DYNAMIC_FRAMES):
        return{"idx":-1,"conf":0.0,"probs":[],"name":"Unknown","below_threshold":True,
               "error":f"feature_mismatch:expected {expected} got {len(req.features)}"}
    result=nn.predict(req.features,_conf_threshold)
    if result["conf"]>0.3:
        with get_db() as db:
            db.execute("INSERT INTO prediction_log(gesture,confidence,model_type,created_at) VALUES(?,?,?,?)",
                       (result["name"],result["conf"],req.model_type,time.time()))
    return result

@app.get("/api/nn/status")
def nn_status():
    return{"static":{"trained":static_nn.trained,"accuracy":static_nn.accuracy,"loss":static_nn.loss,
                     "epochs":static_nn.epochs,"per_gesture_acc":static_nn.per_gesture_acc},
           "dynamic":{"trained":dynamic_nn.trained,"accuracy":dynamic_nn.accuracy,"loss":dynamic_nn.loss,
                      "epochs":dynamic_nn.epochs,"model_type":"LSTM","per_gesture_acc":dynamic_nn.per_gesture_acc}}

@app.post("/api/nn/save/{model_type}")
async def nn_save(model_type:str, source:str='camera'):
    nn=static_nn if model_type=='static' else dynamic_nn
    model_id=f"{model_type}_{source}"  # e.g. "static_camera", "static_glove"
    with get_db() as db:
        db.execute("INSERT OR REPLACE INTO models(id,data,source,updated_at) VALUES(?,?,?,?)",
                   (model_id,json.dumps(nn.to_json()),source,time.time()))
    await push("model_saved",{"model":model_type,"source":source})
    log_session("model_saved",detail=f"{model_type}_{source}")
    return{"ok":True,"model_id":model_id}

@app.post("/api/nn/load/{model_type}")
def nn_load(model_type:str, source:str='camera'):
    model_id=f"{model_type}_{source}"
    with get_db() as db:
        row=db.execute("SELECT data FROM models WHERE id=?",(model_id,)).fetchone()
        used_id=model_id
        if not row:
            # Fall back to legacy key (saved before source was tracked)
            row=db.execute("SELECT data FROM models WHERE id=?",(model_type,)).fetchone()
            used_id=model_type
    if not row: return{"ok":False,"error":f"no saved model for '{model_id}'"}
    if used_id != model_id:
        print(f"[WARN] nn_load: '{model_id}' not found — loaded legacy model '{used_id}' instead")
    nn=static_nn if model_type=='static' else dynamic_nn
    nn.from_json(json.loads(row['data']))
    return{"ok":True,"accuracy":nn.accuracy,"epochs":nn.epochs,"gestures":list(nn.gestures.items())}

@app.post("/api/nn/reset/{model_type}")
async def nn_reset(model_type:str):
    nn=static_nn if model_type=='static' else dynamic_nn
    nn.reset()
    with get_db() as db: db.execute("DELETE FROM models WHERE id=?",(model_type,))
    await push("model_reset",{"model":model_type}); log_session("model_reset",detail=model_type)
    return{"ok":True}

@app.post("/api/nn/reset_all")
async def nn_reset_all(_:None=Depends(require_admin)):
    static_nn.reset(); dynamic_nn.reset()
    with get_db() as db: db.execute("DELETE FROM models")
    await push("model_reset",{"model":"all"}); log_session("model_reset_all")
    return{"ok":True}

@app.get("/api/nn/per_gesture_accuracy")
def per_gesture_accuracy():
    return{"static":static_nn.per_gesture_acc,"dynamic":dynamic_nn.per_gesture_acc}

# -- Confidence threshold --
@app.post("/api/settings/conf_threshold")
def set_conf_threshold(req:ConfThresholdRequest):
    global _conf_threshold
    _conf_threshold=max(0.0,min(1.0,req.threshold))
    with get_db() as db:
        db.execute("INSERT OR REPLACE INTO settings VALUES(?,?)",("conf_threshold",str(_conf_threshold)))
    return{"ok":True,"threshold":_conf_threshold}

@app.get("/api/settings/conf_threshold")
def get_conf_threshold_route():
    return{"threshold":_conf_threshold}

# -- Samples --
@app.post("/api/samples/simulate")
def api_simulate(req:SimulateRequest):
    if req.sample_type=='static':
        samples=[]; mirrored=[]
        for _ in range(req.count):
            vec,mir=simulate_static(req.gesture,mirror_aug=True)
            samples.append(vec)
            if mir: mirrored.append(mir)
        return{"gesture":req.gesture,"samples":samples,"mirrored":mirrored,"count":len(samples)}
    else:
        samples=[s for _ in range(req.count) if(s:=simulate_dynamic(req.gesture)) is not None]
        return{"gesture":req.gesture,"samples":samples,"mirrored":[],"count":len(samples)}

@app.post("/api/samples/save")
def save_samples(req:SampleSaveRequest):
    now=time.time()
    with get_db() as db:
        if req.sample_type=="static":
            for s in req.samples:
                q=score_sample_quality(s)
                db.execute("INSERT INTO static_samples(gesture,sample,quality,source,created_at) VALUES(?,?,?,?,?)",
                           (req.gesture,json.dumps(s),q,req.source,now))
            count=db.execute("SELECT COUNT(*) FROM static_samples WHERE gesture=?",(req.gesture,)).fetchone()[0]
        else:
            for s in req.samples:
                q=score_sample_quality(s[:5] if len(s)>=5 else s)
                db.execute("INSERT INTO dynamic_samples(gesture,frames,quality,source,created_at) VALUES(?,?,?,?,?)",
                           (req.gesture,json.dumps(s),q,req.source,now))
            count=db.execute("SELECT COUNT(*) FROM dynamic_samples WHERE gesture=?",(req.gesture,)).fetchone()[0]
        db.execute("UPDATE gesture_registry SET updated_at=? WHERE name=?",(now,req.gesture))
    log_session("sample_saved",req.gesture,f"{req.sample_type} x{len(req.samples)}")
    return{"ok":True,"total":count}

@app.get("/api/samples/load")
def load_samples(source:str='all'):
    """Load samples optionally filtered by source (camera|glove|all)."""
    result={}
    with get_db() as db:
        if source == 'all':
            s_rows = db.execute("SELECT gesture,sample FROM static_samples ORDER BY created_at")
            d_rows = db.execute("SELECT gesture,frames FROM dynamic_samples ORDER BY created_at")
        else:
            s_rows = db.execute("SELECT gesture,sample FROM static_samples WHERE source=? ORDER BY created_at", (source,))
            d_rows = db.execute("SELECT gesture,frames FROM dynamic_samples WHERE source=? ORDER BY created_at", (source,))

        for row in s_rows:
            s=json.loads(row['sample'])
            if len(s)<STATIC_FEATURES:
                s=s+[0.0]*(STATIC_FEATURES-len(s)-2)+[1,0]
            elif len(s)>STATIC_FEATURES:
                s=s[:STATIC_FEATURES]
            result.setdefault(row['gesture'],{}).setdefault('static',[]).append(s)

        for row in d_rows:
            f=json.loads(row['frames'])
            expected=DYNAMIC_FRAMES*DYNAMIC_FEATURES
            if len(f)<expected: f=f+[0.0]*(expected-len(f))
            elif len(f)>expected: f=f[:expected]
            result.setdefault(row['gesture'],{}).setdefault('dynamic',[]).append(f)
    return result

@app.get("/api/samples/meta")
def samples_meta():
    meta={}
    with get_db() as db:
        # Total counts + per-source counts for static
        for row in db.execute("SELECT gesture,COUNT(*) as cnt,AVG(quality) as avg_q FROM static_samples GROUP BY gesture"):
            meta.setdefault(row['gesture'],{'static':0,'dynamic':0,'static_quality':0,'dynamic_quality':0,
                                            'static_camera':0,'static_glove':0,'dynamic_camera':0,'dynamic_glove':0})
            meta[row['gesture']]['static']=row['cnt']
            meta[row['gesture']]['static_quality']=round(row['avg_q'] or 0,2)
        # Per-source breakdown for static
        for row in db.execute("SELECT gesture,source,COUNT(*) as cnt FROM static_samples GROUP BY gesture,source"):
            meta.setdefault(row['gesture'],{'static':0,'dynamic':0,'static_quality':0,'dynamic_quality':0,
                                            'static_camera':0,'static_glove':0,'dynamic_camera':0,'dynamic_glove':0})
            src=row['source'] or 'camera'
            meta[row['gesture']]['static_'+src]=row['cnt']
        # Total counts + per-source for dynamic
        for row in db.execute("SELECT gesture,COUNT(*) as cnt,AVG(quality) as avg_q FROM dynamic_samples GROUP BY gesture"):
            meta.setdefault(row['gesture'],{'static':0,'dynamic':0,'static_quality':0,'dynamic_quality':0,
                                            'static_camera':0,'static_glove':0,'dynamic_camera':0,'dynamic_glove':0})
            meta[row['gesture']]['dynamic']=row['cnt']
            meta[row['gesture']]['dynamic_quality']=round(row['avg_q'] or 0,2)
        for row in db.execute("SELECT gesture,source,COUNT(*) as cnt FROM dynamic_samples GROUP BY gesture,source"):
            meta.setdefault(row['gesture'],{'static':0,'dynamic':0,'static_quality':0,'dynamic_quality':0,
                                            'static_camera':0,'static_glove':0,'dynamic_camera':0,'dynamic_glove':0})
            src=row['source'] or 'camera'
            meta[row['gesture']]['dynamic_'+src]=row['cnt']
    return meta

@app.delete("/api/samples/clear")
async def clear_samples(_:None=Depends(require_admin)):
    with get_db() as db:
        db.execute("DELETE FROM static_samples"); db.execute("DELETE FROM dynamic_samples"); db.execute("DELETE FROM models")
    static_nn.reset(); dynamic_nn.reset()
    await push("samples_cleared",{}); log_session("samples_cleared_all")
    return{"ok":True}

@app.delete("/api/samples/gesture")
async def delete_gesture_samples(req:DeleteSamplesRequest):
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
    await push("gesture_samples_deleted",{"gesture":req.gesture,"type":req.sample_type})
    log_session("gesture_samples_deleted",req.gesture,req.sample_type)
    return{"ok":True}

@app.get("/api/samples/preview/{gesture}")
def sample_preview(gesture:str):
    with get_db() as db:
        s_rows=db.execute("SELECT sample,quality,created_at FROM static_samples WHERE gesture=? ORDER BY created_at",(gesture,)).fetchall()
        d_rows=db.execute("SELECT frames,quality,created_at FROM dynamic_samples WHERE gesture=? ORDER BY created_at",(gesture,)).fetchall()
    ss=[json.loads(r['sample']) for r in s_rows]; ds=[json.loads(r['frames']) for r in d_rows]
    sq=[r['quality'] for r in s_rows]; dq=[r['quality'] for r in d_rows]
    return{"staticCount":len(ss),"dynamicCount":len(ds),"staticPreview":ss[-3:],
           "staticQualities":sq[-10:],"dynamicQualities":dq[-10:],
           "avgStaticQuality":round(sum(sq)/len(sq),2) if sq else 0,
           "avgDynamicQuality":round(sum(dq)/len(dq),2) if dq else 0,
           "updatedAt":s_rows[-1]['created_at']*1000 if s_rows else(d_rows[-1]['created_at']*1000 if d_rows else None)}

@app.post("/api/samples/generate_demo")
async def generate_demo(req:GenerateDemoRequest):
    result={}; now=time.time()
    for name in req.gestures:
        t=GESTURE_TEMPLATES.get(name,{})
        static=[simulate_static(name) for _ in range(35)]
        dynamic=[s for _ in range(25) if(s:=simulate_dynamic(name,DYNAMIC_FRAMES)) is not None] if t.get('type')=='dynamic' else []
        with get_db() as db:
            for s in static:
                q=score_sample_quality(s)
                db.execute("INSERT INTO static_samples(gesture,sample,quality,created_at) VALUES(?,?,?,?)",(name,json.dumps(s),q,now))
            for d in dynamic:
                db.execute("INSERT INTO dynamic_samples(gesture,frames,quality,created_at) VALUES(?,?,?,?)",(name,json.dumps(d),0.7,now))
            db.execute("INSERT OR IGNORE INTO gesture_registry(name,gesture_type,category,created_at,updated_at) VALUES(?,?,?,?,?)",
                       (name,t.get('type','static'),'word',now,now))
        result[name]={'static':len(static),'dynamic':len(dynamic)}
        await push("demo_progress",{"gesture":name,"done":list(result.keys())})
    return result

# -- Gesture registry --
@app.get("/api/gestures")
def get_gestures():
    with get_db() as db:
        rows=db.execute("SELECT * FROM gesture_registry ORDER BY category,name").fetchall()
    return[dict(r) for r in rows]

@app.post("/api/gestures/register")
def register_gesture(req:GestureRegisterRequest):
    now=time.time()
    with get_db() as db:
        db.execute("INSERT OR REPLACE INTO gesture_registry(name,gesture_type,category,created_at,updated_at) VALUES(?,?,?,COALESCE((SELECT created_at FROM gesture_registry WHERE name=?),?),?)",
                   (req.name,req.gesture_type,req.category,req.name,now,now))
    log_session("gesture_registered",req.name,req.gesture_type)
    return{"ok":True}

@app.delete("/api/gestures/{name}")
async def delete_gesture(name:str):
    with get_db() as db:
        db.execute("DELETE FROM gesture_registry WHERE name=?",(name,))
        db.execute("DELETE FROM static_samples WHERE gesture=?",(name,))
        db.execute("DELETE FROM dynamic_samples WHERE gesture=?",(name,))
    await push("gesture_deleted",{"gesture":name}); log_session("gesture_deleted",name)
    return{"ok":True}

@app.get("/api/gestures/readiness")
def gesture_readiness(source:str='all'):
    """Readiness filtered by source so UI shows correct progress per input mode."""
    meta={}
    with get_db() as db:
        if source == 'all':
            s_count_rows = db.execute("SELECT gesture,COUNT(*) as cnt FROM static_samples GROUP BY gesture")
            d_count_rows = db.execute("SELECT gesture,COUNT(*) as cnt FROM dynamic_samples GROUP BY gesture")
        else:
            s_count_rows = db.execute("SELECT gesture,COUNT(*) as cnt FROM static_samples WHERE source=? GROUP BY gesture", (source,))
            d_count_rows = db.execute("SELECT gesture,COUNT(*) as cnt FROM dynamic_samples WHERE source=? GROUP BY gesture", (source,))
        for row in s_count_rows:
            meta.setdefault(row['gesture'],{})['static']=row['cnt']
        for row in d_count_rows:
            meta.setdefault(row['gesture'],{})['dynamic']=row['cnt']
        reg={r['name']:r['gesture_type'] for r in db.execute("SELECT name,gesture_type FROM gesture_registry")}
    result={}
    for name,gtype in reg.items():
        s=meta.get(name,{}).get('static',0); d=meta.get(name,{}).get('dynamic',0)
        if gtype=='static':
            result[name]={"static":s,"dynamic":d,"ready":s>=MIN_STATIC_SAMPLES,"needed":max(0,MIN_STATIC_SAMPLES-s),"type":gtype}
        else:
            result[name]={"static":s,"dynamic":d,"ready":d>=MIN_DYNAMIC_SAMPLES,"needed":max(0,MIN_DYNAMIC_SAMPLES-d),"type":gtype}
    return result

# -- History --
@app.get("/api/history")
def get_history(limit:int=50):
    with get_db() as db:
        rows=db.execute("SELECT * FROM session_history ORDER BY created_at DESC LIMIT ?",(limit,)).fetchall()
    return[dict(r) for r in rows]

@app.get("/api/history/stats")
def history_stats():
    with get_db() as db:
        total_captures=db.execute("SELECT COUNT(*) FROM session_history WHERE event_type='sample_saved'").fetchone()[0]
        total_trains=db.execute("SELECT COUNT(*) FROM session_history WHERE event_type='train_batch'").fetchone()[0]
        last_train=db.execute("SELECT MAX(created_at) FROM session_history WHERE event_type='model_saved'").fetchone()[0]
        new_since=db.execute("SELECT COUNT(*) FROM session_history WHERE event_type='sample_saved' AND created_at>COALESCE((SELECT MAX(created_at) FROM session_history WHERE event_type='model_saved'),0)").fetchone()[0]
    return{"totalCaptures":total_captures,"totalTrains":total_trains,
           "lastTrainAt":last_train*1000 if last_train else None,
           "newSamplesSinceLastTrain":new_since,"needsRetrain":new_since>=5}

# -- Prediction log --
@app.post("/api/predictions/log")
def log_prediction(req:PredLogRequest):
    with get_db() as db:
        db.execute("INSERT INTO prediction_log(gesture,confidence,model_type,created_at) VALUES(?,?,?,?)",
                   (req.gesture,req.confidence,req.model_type,time.time()))
    return{"ok":True}

@app.get("/api/predictions/recent")
def recent_predictions(limit:int=20):
    with get_db() as db:
        rows=db.execute("SELECT * FROM prediction_log ORDER BY created_at DESC LIMIT ?",(limit,)).fetchall()
    return[dict(r) for r in rows]

# -- Export / Import --
@app.get("/api/export/json")
def export_json():
    data={"version":"1.0","mqtt_enabled":_mqtt_enabled,"feature_version":FEATURE_VERSION,
          "static_features":STATIC_FEATURES,"dynamic_features":DYNAMIC_FEATURES,
          "exported_at":time.time(),"static_samples":{},"dynamic_samples":{},"gestures":[]}
    with get_db() as db:
        for row in db.execute("SELECT gesture,sample,quality,created_at FROM static_samples ORDER BY created_at"):
            data["static_samples"].setdefault(row['gesture'],[]).append({"sample":json.loads(row['sample']),"quality":row['quality'],"created_at":row['created_at']})
        for row in db.execute("SELECT gesture,frames,quality,created_at FROM dynamic_samples ORDER BY created_at"):
            data["dynamic_samples"].setdefault(row['gesture'],[]).append({"frames":json.loads(row['frames']),"quality":row['quality'],"created_at":row['created_at']})
        for row in db.execute("SELECT * FROM gesture_registry"):
            data["gestures"].append(dict(row))
    return Response(content=json.dumps(data,indent=2),media_type="application/json",
                    headers={"Content-Disposition":"attachment; filename=signlens_dataset.json"})

@app.post("/api/import/json")
async def import_json(body:dict, _:None=Depends(require_admin)):
    # Version check
    body_fv=body.get("feature_version","unknown")
    if body_fv not in("unknown",FEATURE_VERSION,"6.0"):
        return{"error":f"feature_version_mismatch: dataset has {body_fv}, current is {FEATURE_VERSION}. Retrain first."}
    now=time.time(); imported={"static":0,"dynamic":0,"gestures":0}
    with get_db() as db:
        for name,samples in body.get("static_samples",{}).items():
            for entry in samples:
                s=entry.get("sample",entry) if isinstance(entry,dict) else entry
                q=entry.get("quality",0.8) if isinstance(entry,dict) else 0.8
                db.execute("INSERT INTO static_samples(gesture,sample,quality,created_at) VALUES(?,?,?,?)",(name,json.dumps(s),q,now))
                imported["static"]+=1
        for name,samples in body.get("dynamic_samples",{}).items():
            for entry in samples:
                f=entry.get("frames",entry) if isinstance(entry,dict) else entry
                q=entry.get("quality",0.7) if isinstance(entry,dict) else 0.7
                db.execute("INSERT INTO dynamic_samples(gesture,frames,quality,created_at) VALUES(?,?,?,?)",(name,json.dumps(f),q,now))
                imported["dynamic"]+=1
        for g in body.get("gestures",[]):
            db.execute("INSERT OR IGNORE INTO gesture_registry(name,gesture_type,category,created_at,updated_at) VALUES(?,?,?,?,?)",
                       (g['name'],g.get('gesture_type','static'),g.get('category','custom'),now,now))
            imported["gestures"]+=1
    log_session("dataset_imported",detail=str(imported))
    return{"ok":True,"imported":imported}

# -- MediaPipe --
@app.get("/api/mediapipe/status")
def mp_status(): return{"server_side":_mp_ready,"model_exists":MP_MODEL.exists()}

# -- MQTT --
@app.get("/api/mqtt/status")
def mqtt_status():
    return{
        "enabled":    _mqtt_enabled,
        "available":  MQTT_AVAILABLE,
        "topic":      _mqtt_topic,
        "broker":     os.environ.get("MQTT_BROKER","broker.hivemq.com"),
        "port":       int(os.environ.get("MQTT_PORT","1883")),
        "connected":  bool(_mqtt_client and _mqtt_client.is_connected()),
    }

@app.post("/api/mediapipe/process")
async def mp_process(body:dict):
    if not _mp_ready: return{"error":"server_mp_unavailable","fallback":"use_browser_mediapipe"}
    try:
        import mediapipe as mp
        img_bytes=base64.b64decode(body.get("frame",""))
        arr=np.frombuffer(img_bytes,dtype=np.uint8)
        mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=arr.reshape(-1,arr.shape[-1]) if arr.ndim>1 else arr.reshape(1,-1,3))
        res=_mp_landmarker.detect(mp_img)
        if not res.hand_landmarks: return{"detected":False,"features":None}
        feats=_landmarks_to_features(res.hand_landmarks[0])
        await push("features_extracted",{"features":feats,"source":"server_mp"})
        return{"detected":True,"features":feats}
    except Exception as e: return{"error":str(e)}

# -- NLP Phase 2B --
@app.post("/api/nlp/suggestions")
def get_suggestions(body:dict):
    return{"suggestions":kn_suggest(body.get("words",[]),use_personal=True)}

@app.post("/api/nlp/context_suggestions")
def get_context_suggestions(body:dict):
    """Gesture-aware suggestions using personal corpus + gesture history."""
    words=body.get("words",[])
    recent_gestures=body.get("recent_gestures",[])
    suggestions=gesture_context_suggest(words,recent_gestures)
    return{"suggestions":suggestions,"personal_active":_personal_lm is not None,
           "corpus_size":len(_personal_corpus)}

@app.post("/api/nlp/word_suggestions")
def get_word_suggestions(body:dict):
    return{"suggestions":word_prefix_suggest(body.get("prefix",""))}

@app.post("/api/nlp/learn")
async def nlp_learn(body:dict):
    """Save a sentence to personal corpus and update gesture map."""
    global _personal_corpus,_gesture_map
    sentence=body.get("sentence","").strip()
    gesture_sequence=body.get("gesture_sequence",[])
    if not sentence: return{"ok":False,"error":"empty sentence"}
    # Add to personal corpus
    _personal_corpus.append(sentence)
    if len(_personal_corpus)>500: _personal_corpus=_personal_corpus[-500:]
    # Update gesture→word map
    words=sentence.lower().split()
    for i,g in enumerate(gesture_sequence):
        if g and i<len(words):
            if g not in _gesture_map: _gesture_map[g]=[]
            _gesture_map[g].append(words[i])
            if len(_gesture_map[g])>100: _gesture_map[g]=_gesture_map[g][-100:]
    # Persist to DB
    with get_db() as db:
        db.execute("INSERT INTO personal_corpus(sentence,gesture_sequence,created_at) VALUES(?,?,?)",
                   (sentence,json.dumps(gesture_sequence),time.time()))
    log_session("sentence_learned",detail=sentence[:50])
    return{"ok":True,"corpus_size":len(_personal_corpus)}

@app.post("/api/nlp/retrain")
async def nlp_retrain():
    """Background retrain of personal NLP model."""
    import threading
    def _retrain():
        _build_personal_lm()
    threading.Thread(target=_retrain,daemon=True).start()
    return{"ok":True,"sentences":len(_personal_corpus),
           "message":"Retraining in background"}

@app.get("/api/nlp/stats")
def nlp_stats():
    return{"corpus_size":len(_personal_corpus),"personal_model_active":_personal_lm is not None,
           "gesture_map_size":len(_gesture_map),"base_vocab_size":len(_kn_vocab)}

@app.post("/api/nlp/grammar")
def nlp_grammar(body:dict):
    """Offline grammar correction — no Gemini needed."""
    sentence=body.get("sentence","")
    corrected=offline_grammar_correct(sentence)
    return{"ok":True,"original":sentence,"corrected":corrected,
           "changed":corrected is not None}

# -- Sentence --
@app.get("/api/sentence")
def get_sentence(): return sentence_state.to_dict()

@app.post("/api/sentence")
async def update_sentence(req:SentenceRequest):
    s=sentence_state
    if   req.action=='add_word'               and req.word:  s.add_word(req.word)
    elif req.action=='remove_last':
        if s.spelling: s.spelling=s.spelling[:-1]
        elif s.words:  s.words.pop()
    elif req.action=='clear': s.words=[]; s.spelling=''; s.completion=None; s.word_suggestions=[]
    elif req.action=='add_letter'             and req.letter: s.spelling+=req.letter.lower()
    elif req.action=='accept_word_suggestion' and req.word:   s.add_word(req.word); s.spelling=''
    elif req.action=='accept_completion':
        if s.completion: s.words=s.completion.lower().split(); s.completion=None
    elif req.action=='replace_corrected'      and req.text:   s.words=req.text.lower().split()
    elif req.action=='set_suggestions'        and req.word:   s.suggestions=req.word.split(',')
    elif req.action=='set_completion':         s.completion=req.text
    await push("sentence_updated",s.to_dict())
    return s.to_dict()

# -- Settings --
@app.get("/api/settings/{key}")
def get_setting(key:str):
    with get_db() as db:
        row=db.execute("SELECT value FROM settings WHERE key=?",(key,)).fetchone()
    return{"value":row['value'] if row else None}

@app.post("/api/settings")
def save_setting(req:SettingRequest):
    with get_db() as db: db.execute("INSERT OR REPLACE INTO settings VALUES(?,?)",(req.key,req.value))
    return{"ok":True}

@app.delete("/api/settings/{key}")
def delete_setting(key:str):
    with get_db() as db: db.execute("DELETE FROM settings WHERE key=?",(key,))
    return{"ok":True}

# -- Frontend (must be last) --

@app.get("/service-worker.js")
async def sw_route():
    from fastapi.responses import FileResponse
    return FileResponse(str(FRONTEND / "service-worker.js"), media_type="application/javascript")

@app.get("/sensor")
@app.get("/sensor.html")
async def sensor_route():
    from fastapi.responses import FileResponse
    return FileResponse(str(FRONTEND / "sensor.html"), media_type="text/html")

app.mount("/",StaticFiles(directory=str(FRONTEND),html=True),name="frontend")

if __name__=="__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

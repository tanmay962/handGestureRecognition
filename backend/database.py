import json
import math
import random
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "signlens.db"

STATIC_FEATURES  = 41
DYNAMIC_FEATURES = 41
DYNAMIC_FRAMES   = 45
MIN_STATIC_SAMPLES  = 10
MIN_DYNAMIC_SAMPLES = 5

GESTURE_TEMPLATES = {
    'Hello':    {'curls':[.1,.1,.1,.1,.1], 'ori':[0,-.8,.1,.02,-.01,.01],  'type':'dynamic'},
    'Thank You':{'curls':[.2,.3,.7,.7,.5], 'ori':[.1,-.6,.3,.01,.02,-.01], 'type':'dynamic'},
    'Yes':      {'curls':[.4,.8,.8,.8,.7], 'ori':[0,-.9,0,0,.05,0],         'type':'dynamic'},
    'No':       {'curls':[.3,.2,.8,.8,.7], 'ori':[0,-.3,.1,.05,0,0],        'type':'dynamic'},
    'Help':     {'curls':[.1,.1,.1,.1,.8], 'ori':[-.3,-.5,.5,.01,-.02,.02], 'type':'dynamic'},
    'Please':   {'curls':[.6,.2,.2,.2,.6], 'ori':[.1,-.5,-.2,-.01,.01,.03], 'type':'dynamic'},
    'Sorry':    {'curls':[.5,.5,.5,.5,.5], 'ori':[0,-.3,.6,-.01,-.02,.01],  'type':'dynamic'},
    'Stop':     {'curls':[0,0,0,0,0],      'ori':[0,-1,0,0,0,0],            'type':'static'},
    'Go':       {'curls':[.7,.1,.8,.8,.8], 'ori':[.3,-.4,.1,.02,-.03,.01],  'type':'dynamic'},
    'Water':    {'curls':[.3,.7,.7,.7,.7], 'ori':[.1,-.5,-.2,.01,.01,-.02], 'type':'dynamic'},
    # Alphabet
    'A':{'curls':[.3,1,1,1,1],    'ori':[0,-.7,0,0,0,0],     'type':'static'},
    'B':{'curls':[.8,0,0,0,0],    'ori':[0,-1,0,0,0,0],      'type':'static'},
    'C':{'curls':[.4,.4,.4,.4,.4],'ori':[.2,-.7,.1,0,0,0],   'type':'static'},
    'D':{'curls':[.2,.1,.8,.8,.8],'ori':[0,-.8,0,0,0,0],     'type':'static'},
    'E':{'curls':[.7,.8,.8,.8,.8],'ori':[0,-.8,0,0,0,0],     'type':'static'},
    'F':{'curls':[.5,.1,.8,.8,.8],'ori':[0,-.7,0,0,0,0],     'type':'static'},
    'G':{'curls':[.3,.1,.9,.9,.9],'ori':[0,-.5,.2,0,0,0],    'type':'static'},
    'H':{'curls':[.8,.1,.1,.9,.9],'ori':[0,-.6,0,0,0,0],     'type':'static'},
    'I':{'curls':[.8,.9,.9,.9,.1],'ori':[0,-.8,0,0,0,0],     'type':'static'},
    'J':{'curls':[.8,.9,.9,.9,.1],'ori':[0,-.8,0,0,0,.1],    'type':'dynamic'},
    'K':{'curls':[.3,0,0,.9,.9],  'ori':[0,-.8,.1,0,0,0],    'type':'static'},
    'L':{'curls':[.1,.1,.9,.9,.9],'ori':[0,-.6,0,0,0,0],     'type':'static'},
    'M':{'curls':[.6,.8,.8,.8,.9],'ori':[0,-.5,-.3,0,0,0],   'type':'static'},
    'N':{'curls':[.6,.8,.8,.9,.9],'ori':[0,-.5,-.3,0,0,0],   'type':'static'},
    'O':{'curls':[.5,.5,.5,.5,.5],'ori':[0,-.7,0,0,0,0],     'type':'static'},
    'P':{'curls':[.3,0,0,.9,.9],  'ori':[.3,-.3,-.3,.02,0,0],'type':'static'},
    'Q':{'curls':[.3,0,.9,.9,.9], 'ori':[.3,-.2,-.4,.02,0,0],'type':'static'},
    'R':{'curls':[.8,0,0,.9,.9],  'ori':[0,-.8,0,0,0,0],     'type':'static'},
    'S':{'curls':[.5,1,1,1,1],    'ori':[0,-.8,0,0,0,0],     'type':'static'},
    'T':{'curls':[.4,.9,.9,.9,.9],'ori':[0,-.8,0,0,0,0],     'type':'static'},
    'U':{'curls':[.8,0,0,.9,.9],  'ori':[0,-1,0,0,0,0],      'type':'static'},
    'V':{'curls':[.8,0,0,.9,.9],  'ori':[0,-.9,.15,0,0,0],   'type':'static'},
    'W':{'curls':[.7,.1,.1,.1,.9],'ori':[0,-.8,0,0,0,0],     'type':'static'},
    'X':{'curls':[.8,.5,.9,.9,.9],'ori':[0,-.8,0,0,0,0],     'type':'static'},
    'Y':{'curls':[.1,.9,.9,.9,.1],'ori':[0,-.7,0,0,0,0],     'type':'static'},
    'Z':{'curls':[.8,0,.9,.9,.9], 'ori':[0,-.7,0,.04,0,.05], 'type':'dynamic'},
    # Numbers
    '0':{'curls':[.6,.6,.6,.6,.6],'ori':[0,-.8,0,0,0,0],    'type':'static'},
    '1':{'curls':[.8,0,.9,.9,.9], 'ori':[0,-.9,0,0,0,0],    'type':'static'},
    '2':{'curls':[.8,0,0,.9,.9],  'ori':[0,-.9,0,0,0,0],    'type':'static'},
    '3':{'curls':[.8,0,0,0,.9],   'ori':[0,-.9,0,0,0,0],    'type':'static'},
    '4':{'curls':[.8,0,0,0,0],    'ori':[0,-.9,0,0,0,0],    'type':'static'},
    '5':{'curls':[0,0,0,0,0],     'ori':[0,-.9,0,0,0,0],    'type':'static'},
    '6':{'curls':[.7,.9,.9,0,0],  'ori':[0,-.8,0,0,0,0],    'type':'static'},
    '7':{'curls':[.7,.9,0,.9,0],  'ori':[0,-.8,0,0,0,0],    'type':'static'},
    '8':{'curls':[.7,0,.9,0,0],   'ori':[0,-.8,0,0,0,0],    'type':'static'},
    '9':{'curls':[.7,0,.9,.9,.9], 'ori':[0,-.8,.1,0,0,0],   'type':'static'},
}

DEFAULT_GESTURES = ['Hello','Thank You','Yes','No','Help','Please','Sorry','Stop','Go','Water']


# Helpers

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def noise(scale=0.1):
    return (random.random() - 0.5) * scale


# Sample simulation (used in demo/testing)

def _sim_one_hand(name, mirror=False):
    t = GESTURE_TEMPLATES.get(name)
    if not t:
        return [random.random() * 0.5 for _ in range(11)]
    curls = [clamp(c + noise(0.12), 0, 1) for c in t['curls']]
    ori   = [o + noise(0.03) for o in t['ori']]
    if mirror:
        ori[0] = -ori[0]
        ori[3] = -ori[3]
    return curls + ori


def _sim_face_features(dominant_hand=None):
    nose_x    = random.uniform(0.4, 0.6)
    nose_y    = random.uniform(0.3, 0.6)
    eye_scale = clamp(random.uniform(0.3, 0.5), 0.0, 1.0)
    if dominant_hand:
        h1nx = clamp(noise(0.25) + 0.1, -1, 1)
        h1ny = clamp(noise(0.25) + 0.3, -1, 1)
    else:
        h1nx = noise(0.1)
        h1ny = noise(0.1)
    tilt       = noise(0.05)
    mouth_open = clamp(noise(0.04), 0.0, 1.0)
    eye_open   = clamp(0.5 + noise(0.08), 0.0, 1.0)
    return [nose_x, nose_y, eye_scale, h1nx, h1ny, 0.0, 0.0, tilt, mouth_open, eye_open]


def _sim_pose_features():
    return [
        random.uniform(0.45, 0.55),
        random.uniform(0.4, 0.6),
        clamp(noise(0.2), -1, 1),
        0.0,
        random.uniform(0.3, 0.9),
        1.0,
    ]


def simulate_static(name, mirror_aug=False):
    dom   = _sim_one_hand(name)
    aux   = [0] * 11
    face  = _sim_face_features(dom)
    pose  = _sim_pose_features()
    flags = [1, 0, 1]  # dom_present, aux_absent, face_present

    vec = (dom + aux + face + pose + flags)[:STATIC_FEATURES]
    while len(vec) < STATIC_FEATURES:
        vec.append(0.0)

    if mirror_aug:
        dom_m  = _sim_one_hand(name, mirror=True)
        face_m = _sim_face_features(dom_m)
        vec_m  = (dom_m + aux + face_m + pose + flags)[:STATIC_FEATURES]
        while len(vec_m) < STATIC_FEATURES:
            vec_m.append(0.0)
        return vec, vec_m

    return vec, None


def simulate_dynamic(name, frames=DYNAMIC_FRAMES):
    t = GESTURE_TEMPLATES.get(name)
    if not t:
        return None

    out = []
    for f in range(frames):
        p  = f / frames
        mn = math.sin(p * math.pi * 2) * 0.1
        curls = [clamp(c * p + noise(0.08), 0, 1) for c in t['curls']]
        ori   = [o * p + (mn if i < 3 else noise(0.02)) * 0.3 for i, o in enumerate(t['ori'])]
        dom   = curls + ori
        aux   = [0.0] * 11
        face  = _sim_face_features()
        pose  = _sim_pose_features()
        flags = [1.0, 0.0, 1.0]
        frame_vec = (dom + aux + face + pose + flags)[:DYNAMIC_FEATURES]
        while len(frame_vec) < DYNAMIC_FEATURES:
            frame_vec.append(0.0)
        out.extend(frame_vec)

    return out


def score_sample_quality(vec):
    import numpy as np

    arr      = np.array(vec[:5])
    variance = float(np.var(arr))
    clipping = float(np.mean((arr < 0.02) | (arr > 0.98)))
    return round(min(1.0, max(0.1, 0.5 + variance * 4 - clipping * 0.3)), 3)


# Database

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def log_session(event_type, gesture=None, detail=None):
    with get_db() as db:
        db.execute(
            "INSERT INTO session_history(event_type,gesture,detail,created_at) VALUES(?,?,?,?)",
            (event_type, gesture, detail, time.time())
        )


# Schema & migrations

def _get_cols(db, table):
    try:
        return [r[1] for r in db.execute(f"PRAGMA table_info({table})")]
    except Exception:
        return []


def _migrate_old_samples(db, now):
    sc = _get_cols(db, 'static_samples')
    dc = _get_cols(db, 'dynamic_samples')
    old_static, old_dynamic = {}, {}

    if sc and 'id' not in sc and 'sample' not in sc:
        print("[DB] Migrating old static_samples schema...")
        try:
            for row in db.execute("SELECT gesture, samples FROM static_samples"):
                arr = json.loads(row['samples'] if 'samples' in sc else row[1] or '[]')
                if isinstance(arr, list):
                    old_static[row['gesture']] = arr
        except Exception as e:
            print(f"[DB] static migration error: {e}")
        db.execute("DROP TABLE static_samples")

    if dc and 'id' not in dc and 'frames' not in dc:
        print("[DB] Migrating old dynamic_samples schema...")
        try:
            col = 'samples' if 'samples' in dc else (dc[1] if len(dc) > 1 else None)
            if col:
                for row in db.execute(f"SELECT gesture, {col} FROM dynamic_samples"):
                    arr = json.loads(row[1] or '[]')
                    if isinstance(arr, list):
                        old_dynamic[row['gesture']] = arr
        except Exception as e:
            print(f"[DB] dynamic migration error: {e}")
        db.execute("DROP TABLE dynamic_samples")

    return old_static, old_dynamic


def _restore_migrated(db, old_static, old_dynamic, now):
    for gesture, samples in old_static.items():
        for s in samples:
            if isinstance(s, list) and len(s) >= 5:
                q = score_sample_quality(s)
                db.execute(
                    "INSERT INTO static_samples(gesture,sample,quality,created_at) VALUES(?,?,?,?)",
                    (gesture, json.dumps(s), q, now)
                )
    for gesture, samples in old_dynamic.items():
        for s in samples:
            if isinstance(s, list) and len(s) >= 5:
                q = score_sample_quality(s[:5])
                db.execute(
                    "INSERT INTO dynamic_samples(gesture,frames,quality,created_at) VALUES(?,?,?,?)",
                    (gesture, json.dumps(s), q, now)
                )
    if old_static or old_dynamic:
        ns = sum(len(v) for v in old_static.values())
        nd = sum(len(v) for v in old_dynamic.values())
        print(f"[DB] Restored {ns} static, {nd} dynamic samples from old schema")


def _seed_registry():
    now = time.time()
    entries = []
    for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        gtype = 'dynamic' if ch in ('J', 'Z') else 'static'
        entries.append((ch, gtype, 'alphabet'))
    for d in range(10):
        entries.append((str(d), 'static', 'number'))
    for name, meta in GESTURE_TEMPLATES.items():
        if name not in [e[0] for e in entries]:
            entries.append((name, meta.get('type', 'static'), 'word'))

    with get_db() as db:
        for name, gtype, cat in entries:
            db.execute(
                "INSERT OR IGNORE INTO gesture_registry(name,gesture_type,category,created_at,updated_at) VALUES(?,?,?,?,?)",
                (name, gtype, cat, now, now)
            )


def init_db():
    now = time.time()
    with get_db() as db:
        old_static, old_dynamic = _migrate_old_samples(db, now)

        db.executescript("""
            CREATE TABLE IF NOT EXISTS settings(
                key   TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE TABLE IF NOT EXISTS gesture_registry(
                name         TEXT PRIMARY KEY,
                gesture_type TEXT NOT NULL DEFAULT 'static',
                category     TEXT NOT NULL DEFAULT 'custom',
                created_at   REAL,
                updated_at   REAL
            );
            CREATE TABLE IF NOT EXISTS static_samples(
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                gesture    TEXT NOT NULL,
                sample     TEXT NOT NULL,
                quality    REAL DEFAULT 1.0,
                source     TEXT DEFAULT 'camera',
                created_at REAL
            );
            CREATE TABLE IF NOT EXISTS dynamic_samples(
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                gesture    TEXT NOT NULL,
                frames     TEXT NOT NULL,
                quality    REAL DEFAULT 1.0,
                source     TEXT DEFAULT 'camera',
                created_at REAL
            );
            CREATE TABLE IF NOT EXISTS models(
                id         TEXT PRIMARY KEY,
                data       TEXT,
                source     TEXT DEFAULT 'camera',
                updated_at REAL
            );
            CREATE TABLE IF NOT EXISTS session_history(
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                gesture    TEXT,
                detail     TEXT,
                created_at REAL
            );
            CREATE TABLE IF NOT EXISTS prediction_log(
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                gesture    TEXT NOT NULL,
                confidence REAL NOT NULL,
                model_type TEXT NOT NULL,
                created_at REAL
            );
            CREATE TABLE IF NOT EXISTS personal_corpus(
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                sentence         TEXT NOT NULL,
                gesture_sequence TEXT,
                created_at       REAL
            );
            CREATE INDEX IF NOT EXISTS idx_static_g  ON static_samples(gesture);
            CREATE INDEX IF NOT EXISTS idx_dynamic_g ON dynamic_samples(gesture);
        """)

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

        _restore_migrated(db, old_static, old_dynamic, now)

    _seed_registry()

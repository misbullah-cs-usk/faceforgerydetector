import cv2
import numpy as np
import mediapipe as mp

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

# ----------- FEATURE DEFINITIONS -------------

# 1. Raw landmark points (choose key landmarks)
RAW_KEYS = [
    33, 263,   # eye corners
    1, 152,    # forehead, chin
    61, 291    # mouth corners
]

# 2. Geometry pairs (distances)
GEOMETRY_PAIRS = [
    (33, 263),
    (133, 362),
    (1, 152),
    (61, 291),
    (13, 14),
    (234, 454),
]

# 3. Symmetry (dx, dy)
SYMMETRY_PAIRS = [
    (33, 263),
    (133, 362),
    (234, 454),
    (61, 291),
]

# ----------- FULL FEATURE EXTRACTION -------------

def compute_mediapipe_all_features(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = mp_face.process(img_rgb)

    if not result.multi_face_landmarks:
        return None  # face not detected

    face = result.multi_face_landmarks[0]
    h, w, _ = img_bgr.shape

    pts = np.array([[lm.x * w, lm.y * h] for lm in face.landmark])

    # ===========================
    # 1. RAW FEATURES (12 dims)
    # ===========================
    raw_feats = pts[RAW_KEYS].flatten().astype(np.float32)  # shape (12,)

    # ===========================
    # 2. GEOMETRY FEATURES (6 dims)
    # ===========================
    geom_feats = []
    for i, j in GEOMETRY_PAIRS:
        geom_feats.append(np.linalg.norm(pts[i] - pts[j]))
    geom_feats = np.array(geom_feats, dtype=np.float32)     # shape (6,)

    # ===========================
    # 3. SYMMETRY FEATURES (dx, dy) per pair (8 dims)
    # ===========================
    sym_feats = []
    for L, R in SYMMETRY_PAIRS:
        diff = np.abs(pts[L] - pts[R])  # (dx, dy)
        sym_feats.extend(diff)
    sym_feats = np.array(sym_feats, dtype=np.float32)       # shape (8,)

    # ===========================
    # 4. CONCATENATE → (26 dims)
    # ===========================
    all_features = np.concatenate([raw_feats, geom_feats, sym_feats])

    return all_features  # shape (26,)

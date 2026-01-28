import os, io, tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import streamlit as st

try:
    from inference_sdk import InferenceHTTPClient
except Exception:
    InferenceHTTPClient = None

# ---------------------------
# STREAMLIT COMPAT (Cloud: versions différentes)
# ---------------------------
def image_in(container, img, **kwargs):
    """
    Affiche une image dans un container (st, colonne, expander, etc.)
    Compatible anciennes/nouvelles versions Streamlit:
    - use_container_width (récent)
    - use_column_width (ancien)
    """
    try:
        return container.image(img, use_container_width=True, **kwargs)
    except TypeError:
        return container.image(img, use_column_width=True, **kwargs)

def download_in(container, label, data, file_name, mime, **kwargs):
    """download_button compatible anciennes/nouvelles versions Streamlit."""
    try:
        return container.download_button(
            label,
            data=data,
            file_name=file_name,
            mime=mime,
            use_container_width=True,
            **kwargs,
        )
    except TypeError:
        return container.download_button(
            label,
            data=data,
            file_name=file_name,
            mime=mime,
            use_column_width=True,
            **kwargs,
        )

def button_in(container, label, **kwargs):
    """button compatible anciennes/nouvelles versions Streamlit."""
    try:
        return container.button(label, use_container_width=True, **kwargs)
    except TypeError:
        return container.button(label, use_column_width=True, **kwargs)

# ---------------------------
# UI / THEME
# ---------------------------
st.set_page_config(page_title="Floor Plan Analyzer (Demo)", page_icon="🏠", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 1.25rem; }
.h-title { font-size: 40px; font-weight: 850; letter-spacing: -0.02em; margin: 0; }
.h-sub { opacity: .80; margin-top: .25rem; }
.card {
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 16px;
  padding: 14px 16px;
  background: rgba(255,255,255,.03);
}
.small { font-size: 12px; opacity: .75; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<p class="h-title">🏠 Floor Plan Analyzer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="h-sub">Upload → Roboflow portes/fenêtres → masques + overlay → emprise lots (K-means) → calibration auto → surfaces/pourtours + image "Intérieur (vert)" → exports.</p>',
    unsafe_allow_html=True,
)
st.write("")

# ---------------------------
# CONFIG
# ---------------------------
DEFAULT_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "cubicasa5k-2-qpmsa/6").strip()
API_KEY = os.getenv("ROBOFLOW_API_KEY", "").strip()
API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com").strip()

if InferenceHTTPClient is None:
    st.error("Le package `inference-sdk` n'est pas installé. Vérifie requirements.txt.")
    st.stop()

if not API_KEY:
    st.warning("⚠️ ROBOFLOW_API_KEY n'est pas défini. Ajoute-le dans Streamlit Cloud → Settings → Secrets, puis reboot.")
    st.stop()

# ---------------------------
# IMAGE UTILS
# ---------------------------
def ensure_rgb_u8(img_pil: Image.Image) -> np.ndarray:
    im = img_pil.convert("RGB")
    arr = np.array(im)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] != 3:
        arr = arr[:, :, :3]
    return arr

def rgb_to_png_bytes(rgb_u8: np.ndarray) -> bytes:
    img = Image.fromarray(rgb_u8.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def gray_to_png_bytes(gray_u8: np.ndarray) -> bytes:
    img = Image.fromarray(gray_u8.astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ---------------------------
# HELPERS
# ---------------------------
def clamp_box(x1, y1, x2, y2, W, H):
    return max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)

def is_normalized_coords(vals: np.ndarray) -> bool:
    return float(np.max(vals)) <= 1.5

def iter_tiles(W, H, tile, overlap):
    step = tile - overlap
    xs = list(range(0, max(1, W - tile + 1), step))
    ys = list(range(0, max(1, H - tile + 1), step))
    if xs[-1] != max(0, W - tile):
        xs.append(max(0, W - tile))
    if ys[-1] != max(0, H - tile):
        ys.append(max(0, H - tile))
    for y0 in ys:
        for x0 in xs:
            yield x0, y0, min(W, x0 + tile), min(H, y0 + tile)

def clean_mask(mask: np.ndarray, min_area: int, close_k: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(close_k), int(close_k)))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    num, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    out = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= int(min_area):
            out[lab == i] = 255
    return out

@st.cache_resource
def get_client(api_url: str, api_key: str):
    return InferenceHTTPClient(api_url=api_url, api_key=api_key)

# ---------------------------
# PARAMS
# ---------------------------
@dataclass
class Params:
    # Roboflow
    model_id: str = DEFAULT_MODEL_ID
    pass1_tile: int = 2048
    pass1_over: int = 512
    pass2_tile: int = 1024
    pass2_over: int = 256
    conf_min_door: float = 0.05
    conf_min_win: float = 0.15
    clean_close_k_door: int = 3
    clean_close_k_win: int = 5
    min_area_door_px: int = 6
    min_area_win_px: int = 15

    # Emprise / lots (K-means)
    kmeans_K: int = 8
    kmeans_min_area_ratio: float = 0.01
    kmeans_smooth_k: int = 5
    kmeans_work_max: int = 2000

    # Calibration auto (portes)
    assumed_door_width_m: float = 0.90

    # Surfaces
    wall_thickness_m: float = 0.20
    habitable_perimeter_mode: str = "total"  # "total" ou "external"

    # Visu intérieur (vert)
    interior_alpha: float = 0.28  # transparence du vert

# ---------------------------
# ROBOFLOW INFERENCE
# ---------------------------
def infer_pass(
    client: InferenceHTTPClient,
    model_id: str,
    img_pil: Image.Image,
    tile_size: int,
    overlap: int,
    write_rooms: bool,
    conf_min_door: float,
    conf_min_win: float,
):
    W, H = img_pil.size
    rooms_index = np.zeros((H, W), np.int32) if write_rooms else None
    legend: Dict[str, str] = {}
    cls2id: Dict[str, int] = {}
    nxt = 1

    def rid_for(lbl: str) -> int:
        nonlocal nxt
        lbl = str(lbl).lower().strip() or "unknown"
        if lbl not in cls2id:
            cls2id[lbl] = nxt
            legend[str(nxt)] = lbl
            nxt += 1
        return cls2id[lbl]

    m_doors = np.zeros((H, W), np.uint8)
    m_wins = np.zeros((H, W), np.uint8)
    rows: List[dict] = []
    kept_doors = kept_wins = preds_total = tiles = 0

    with tempfile.TemporaryDirectory() as td:
        for (x0, y0, x1, y1) in iter_tiles(W, H, tile_size, overlap):
            tiles += 1
            tile = img_pil.crop((x0, y0, x1, y1))
            tw, th = tile.size
            tile_path = os.path.join(td, f"tile_{tile_size}_{x0}_{y0}.png")
            tile.save(tile_path)

            res = client.infer(tile_path, model_id=model_id)
            preds = res.get("predictions", []) or res.get("data", [])
            if isinstance(preds, dict) and "predictions" in preds:
                preds = preds["predictions"]
            if not isinstance(preds, list) or len(preds) == 0:
                continue

            preds_total += len(preds)

            for p in preds:
                lbl = str(p.get("class", "")).lower()
                conf = float(p.get("confidence", 1.0))

                is_door = any(k in lbl for k in ["door", "doors", "porte", "portes", "doorway"])
                is_win = any(k in lbl for k in ["window", "windows", "fen", "fenetre", "fenetres"])

                if is_door and conf < conf_min_door:
                    continue
                if is_win and conf < conf_min_win:
                    continue

                # polygon
                if "points" in p and isinstance(p["points"], list) and len(p["points"]) >= 3:
                    pts = np.array([[float(x), float(y)] for x, y in p["points"]], dtype=np.float32)
                    if is_normalized_coords(pts):
                        pts[:, 0] *= tw
                        pts[:, 1] *= th
                    pts[:, 0] = np.clip(pts[:, 0], 0, tw - 1)
                    pts[:, 1] = np.clip(pts[:, 1], 0, th - 1)
                    pts[:, 0] += x0
                    pts[:, 1] += y0
                    poly = pts.astype(np.int32)

                    xmn, ymn = float(poly[:, 0].min()), float(poly[:, 1].min())
                    xmx, ymx = float(poly[:, 0].max()), float(poly[:, 1].max())
                    cxc, cyc = (xmn + xmx) / 2, (ymn + ymx) / 2

                    if is_door:
                        cv2.fillPoly(m_doors, [poly], 255)
                        kept_doors += 1
                    elif is_win:
                        cv2.fillPoly(m_wins, [poly], 255)
                        kept_wins += 1
                    elif write_rooms and rooms_index is not None:
                        cv2.fillPoly(rooms_index, [poly], rid_for(lbl))

                    rows.append(
                        {
                            "label": lbl,
                            "type": "polygon",
                            "x_px": cxc,
                            "y_px": cyc,
                            "width_px": (xmx - xmn),
                            "height_px": (ymx - ymn),
                            "confidence": conf,
                            "pass_tile": tile_size,
                        }
                    )

                # bbox
                elif all(k in p for k in ("x", "y", "width", "height")):
                    cx, cy = float(p["x"]), float(p["y"])
                    bw, bh = float(p["width"]), float(p["height"])
                    if max(cx, cy, bw, bh) <= 1.5:
                        cx *= tw
                        cy *= th
                        bw *= tw
                        bh *= th

                    x1t = int(cx - bw / 2)
                    y1t = int(cy - bh / 2)
                    x2t = int(cx + bw / 2)
                    y2t = int(cy + bh / 2)
                    x1t, y1t, x2t, y2t = clamp_box(x1t, y1t, x2t, y2t, tw, th)

                    x1g, y1g = x1t + x0, y1t + y0
                    x2g, y2g = x2t + x0, y2t + y0
                    x1g, y1g, x2g, y2g = clamp_box(x1g, y1g, x2g, y2g, W, H)

                    cxc, cyc = (x1g + x2g) / 2, (y1g + y2g) / 2

                    if is_door:
                        cv2.rectangle(m_doors, (x1g, y1g), (x2g, y2g), 255, -1)
                        kept_doors += 1
                    elif is_win:
                        cv2.rectangle(m_wins, (x1g, y1g), (x2g, y2g), 255, -1)
                        kept_wins += 1
                    elif write_rooms and rooms_index is not None:
                        cv2.rectangle(rooms_index, (x1g, y1g), (x2g, y2g), rid_for(lbl), -1)

                    rows.append(
                        {
                            "label": lbl,
                            "type": "bbox",
                            "x_px": cxc,
                            "y_px": cyc,
                            "width_px": (x2g - x1g),
                            "height_px": (y2g - y1g),
                            "confidence": conf,
                            "pass_tile": tile_size,
                        }
                    )

    stats = dict(
        tile_size=tile_size,
        overlap=overlap,
        tiles=tiles,
        preds=preds_total,
        kept_doors=kept_doors,
        kept_windows=kept_wins,
    )
    return rooms_index, legend, m_doors, m_wins, rows, stats

def walls_from_rooms_index(rooms_index: np.ndarray) -> np.ndarray:
    a = rooms_index
    H, W = a.shape[:2]
    walls = np.zeros((H, W), np.uint8)
    walls[1:, :] |= (a[1:, :] != a[:-1, :])
    walls[:-1, :] |= (a[:-1, :] != a[1:, :])
    walls[:, 1:] |= (a[:, 1:] != a[:, :-1])
    walls[:, :-1] |= (a[:, :-1] != a[:, 1:])
    walls = (walls.astype(np.uint8) * 255)
    walls = cv2.morphologyEx(
        walls,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    return walls

def overlay_openings(base_rgb: np.ndarray, doors: np.ndarray, wins: np.ndarray, walls: Optional[np.ndarray] = None) -> np.ndarray:
    bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)

    if walls is not None:
        try:
            kernel_e = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
            closed = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, kernel_e, iterations=3)
            inv = cv2.bitwise_not(closed)
            flood = np.zeros((inv.shape[0] + 2, inv.shape[1] + 2), np.uint8)
            cv2.floodFill(inv, flood, (0, 0), 255)
            filled = cv2.bitwise_not(inv)
            cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cnt = max(cnts, key=cv2.contourArea)
                cv2.drawContours(bgr, [cnt], -1, (255, 0, 0), 3)
        except Exception:
            pass

    cs_d, _ = cv2.findContours(doors, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs_w, _ = cv2.findContours(wins, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fill = np.zeros_like(bgr)
    if cs_d:
        cv2.fillPoly(fill, cs_d, (255, 0, 255))
    if cs_w:
        cv2.fillPoly(fill, cs_w, (255, 255, 0))
    out_bgr = cv2.addWeighted(fill, 0.25, bgr, 0.75, 0)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

def df_openings_from_masks(doors: np.ndarray, wins: np.ndarray, min_area_d: int, min_area_w: int) -> pd.DataFrame:
    rows = []

    def extract(mask: np.ndarray, label: str, min_area: int):
        num, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < int(min_area):
                continue
            rows.append(
                {
                    "class": label,
                    "x_px": float(x + w / 2),
                    "y_px": float(y + h / 2),
                    "width_px": float(w),
                    "height_px": float(h),
                    "length_px": float(max(w, h)),
                    "area_px2": float(area),
                }
            )

    extract(doors, "door", min_area_d)
    extract(wins, "window", min_area_w)
    return pd.DataFrame(rows)

# ---------------------------
# EMPRISE / LOTS (K-means)
# ---------------------------
def detect_units_kmeans(
    base_rgb: np.ndarray,
    K: int = 8,
    min_area_ratio: float = 0.01,
    smooth_k: int = 5,
    work_max: int = 2000,
):
    img_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]
    min_area = int(min_area_ratio * H * W)

    scale = 1.0
    if max(H, W) > work_max:
        scale = work_max / max(H, W)
        img_small = cv2.resize(img_bgr, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_small = img_bgr.copy()

    hs, ws = img_small.shape[:2]

    img_lab = cv2.cvtColor(img_small, cv2.COLOR_BGR2LAB)
    X = img_lab.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    _, labels, centers = cv2.kmeans(X, int(K), None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(hs, ws)
    centers = centers.astype(np.uint8)

    bg_cluster = int(np.argmax(centers[:, 0]))

    mask_all_units = np.zeros((hs, ws), np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(smooth_k), int(smooth_k)))

    min_area_small = int(min_area * (scale**2)) if scale != 1.0 else min_area

    for c in range(int(K)):
        if c == bg_cluster:
            continue

        m = (labels == c).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)

        num, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area_small:
                continue
            comp = (lab == i).astype(np.uint8) * 255
            mask_all_units = cv2.bitwise_or(mask_all_units, comp)

    if scale != 1.0:
        mask_units = cv2.resize(mask_all_units, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        mask_units = mask_all_units

    num, lab, stats, _ = cv2.connectedComponentsWithStats((mask_units > 0).astype(np.uint8), 8)

    overlay = img_bgr.copy()
    lots_rows = []
    idx = 0

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        comp = (lab == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)

        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        cx = cy = None
        if M["m00"] > 1e-6:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 3)
        if cx is not None and cy is not None:
            cv2.putText(
                overlay,
                f"LOT {idx}",
                (max(0, cx - 40), max(0, cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        lots_rows.append(
            {
                "lot_id": idx,
                "area_px2": float(area),
                "bbox_x": float(x),
                "bbox_y": float(y),
                "bbox_w": float(w),
                "bbox_h": float(h),
                "cx": float(cx) if cx is not None else np.nan,
                "cy": float(cy) if cy is not None else np.nan,
            }
        )

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    df_lots = pd.DataFrame(lots_rows).sort_values("area_px2", ascending=False) if lots_rows else pd.DataFrame()

    meta = {
        "K": int(K),
        "bg_cluster": int(bg_cluster),
        "scale": float(scale),
        "min_area_px2": int(min_area),
        "lots": int(idx),
    }
    return mask_units.astype(np.uint8), overlay_rgb.astype(np.uint8), df_lots, meta

# ---------------------------
# CALIBRATION AUTO (portes)
# ---------------------------
def point_in_mask(mask_u8: np.ndarray, x: float, y: float) -> bool:
    H, W = mask_u8.shape[:2]
    xi = int(np.clip(round(x), 0, W - 1))
    yi = int(np.clip(round(y), 0, H - 1))
    return mask_u8[yi, xi] > 0

def estimate_scale_from_doors_using_units(
    df_open: pd.DataFrame,
    mask_units: Optional[np.ndarray],
    assumed_door_width_m: float = 0.90,
    min_doors: int = 2,
):
    if df_open is None or df_open.empty:
        return None, {"reason": "df_open empty"}

    doors = df_open[df_open["class"] == "door"].copy()
    if doors.empty:
        return None, {"reason": "no doors"}

    if mask_units is not None:
        inside = []
        for _, r in doors.iterrows():
            inside.append(point_in_mask(mask_units, float(r["x_px"]), float(r["y_px"])))
        doors["inside_units"] = inside
        doors_in = doors[doors["inside_units"]].copy()
        if len(doors_in) >= min_doors:
            doors = doors_in

    vals = doors["length_px"].astype(float).values
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 2]

    if vals.size < min_doors:
        return None, {"reason": f"not enough doors ({vals.size})"}

    q20, q80 = np.quantile(vals, [0.20, 0.80])
    core = vals[(vals >= q20) & (vals <= q80)]
    if core.size < 1:
        core = vals

    door_px_med = float(np.median(core))
    if door_px_med <= 1:
        return None, {"reason": "median too small"}

    m_per_px = float(assumed_door_width_m) / door_px_med

    spread = float(np.std(core) / (np.mean(core) + 1e-9))
    meta = {
        "assumed_door_width_m": float(assumed_door_width_m),
        "doors_total_used_for_filtering": int(vals.size),
        "doors_used_core": int(core.size),
        "door_px_median": door_px_med,
        "variation_cv": spread,
        "quality": "good" if core.size >= 3 and spread < 0.25 else "ok" if spread < 0.40 else "weak",
        "m_per_px": m_per_px,
        "px_per_m": 1.0 / m_per_px if m_per_px > 0 else None,
    }
    return m_per_px, meta

# ---------------------------
# SURFACES & POURTOURS + IMAGE "Intérieur (vert)"
# ---------------------------
def _px_metrics_from_mask(mask_u8: np.ndarray, external_only: bool = True):
    m = (mask_u8 > 0).astype(np.uint8) * 255
    area_px2 = float(np.count_nonzero(m))

    mode = cv2.RETR_EXTERNAL if external_only else cv2.RETR_CCOMP
    cnts, _ = cv2.findContours(m, mode, cv2.CHAIN_APPROX_SIMPLE)

    perim_px = 0.0
    for c in cnts:
        perim_px += float(cv2.arcLength(c, True))
    return area_px2, perim_px

def compute_surfaces_pourtours(
    mask_units: np.ndarray,
    m_per_px: float,
    wall_thickness_m: float = 0.20,
    habitable_perimeter_mode: str = "total",  # "total" ou "external"
):
    if mask_units is None or mask_units.size == 0 or m_per_px is None or m_per_px <= 0:
        return None

    area_px2, perim_px_ext = _px_metrics_from_mask(mask_units, external_only=True)
    area_m2 = area_px2 * (m_per_px ** 2)
    perim_m_ext = perim_px_ext * m_per_px

    t_px = max(1, int(round(wall_thickness_m / m_per_px)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * t_px + 1, 2 * t_px + 1))

    m = (mask_units > 0).astype(np.uint8) * 255
    hab = cv2.erode(m, k, iterations=1)

    hab_area_px2 = float(np.count_nonzero(hab))
    hab_area_m2 = hab_area_px2 * (m_per_px ** 2)
    wall_area_m2 = max(0.0, area_m2 - hab_area_m2)

    hab_external_only = (habitable_perimeter_mode == "external")
    _, hab_perim_px = _px_metrics_from_mask(hab, external_only=hab_external_only)
    hab_perim_m = hab_perim_px * m_per_px

    return {
        "surface_emprise_m2": float(area_m2),
        "pourtour_emprise_m": float(perim_m_ext),
        "surface_murs_m2": float(wall_area_m2),
        "wall_thickness_m": float(wall_thickness_m),
        "wall_thickness_px": int(t_px),
        "surface_habitable_m2": float(hab_area_m2),
        "pourtour_habitable_m": float(hab_perim_m),
        "mask_habitable": hab.astype(np.uint8),
    }

def overlay_interior_green(base_rgb: np.ndarray, mask_habitable_u8: np.ndarray, alpha: float = 0.28) -> np.ndarray:
    """
    Génère l'image comme ton exemple:
    "Intérieur (vert) = surface habitable approx"
    """
    rgb = base_rgb.copy()
    green = np.zeros_like(rgb)
    green[:, :, 1] = 255  # canal vert

    m = (mask_habitable_u8 > 0).astype(np.uint8)
    if m.ndim == 2:
        m3 = np.stack([m, m, m], axis=-1)
    else:
        m3 = m

    out = rgb.copy().astype(np.float32)
    out[m3 > 0] = (1 - alpha) * out[m3 > 0] + alpha * green[m3 > 0]
    return np.clip(out, 0, 255).astype(np.uint8)

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.markdown("### ⚙️ Paramètres")

params = Params(
    model_id=st.sidebar.text_input("Roboflow model_id", value=DEFAULT_MODEL_ID),
    pass1_tile=st.sidebar.selectbox("Pass 1 tile", [1536, 2048, 2560], index=1),
    pass1_over=st.sidebar.selectbox("Pass 1 overlap", [256, 384, 512, 640], index=2),
    pass2_tile=st.sidebar.selectbox("Pass 2 tile", [768, 1024, 1280], index=1),
    pass2_over=st.sidebar.selectbox("Pass 2 overlap", [128, 192, 256, 320], index=2),
    conf_min_door=float(st.sidebar.slider("Conf min door", 0.0, 0.5, 0.05, 0.01)),
    conf_min_win=float(st.sidebar.slider("Conf min window", 0.0, 0.8, 0.15, 0.01)),
    clean_close_k_door=int(st.sidebar.selectbox("Close K door", [1, 3, 5, 7], index=1)),
    clean_close_k_win=int(st.sidebar.selectbox("Close K window", [3, 5, 7, 9], index=1)),
    min_area_door_px=int(st.sidebar.selectbox("Min area door (px)", [1, 6, 15, 30], index=1)),
    min_area_win_px=int(st.sidebar.selectbox("Min area window (px)", [5, 15, 30, 60], index=1)),

    kmeans_K=int(st.sidebar.slider("Emprise: K-means K", 6, 14, 8, 1)),
    kmeans_min_area_ratio=float(st.sidebar.slider("Emprise: min area ratio", 0.001, 0.05, 0.01, 0.001)),
    kmeans_smooth_k=int(st.sidebar.selectbox("Emprise: smooth K", [3, 5, 7, 9], index=1)),
    kmeans_work_max=int(st.sidebar.selectbox("Emprise: work_max", [1200, 1600, 2000, 2400], index=2)),

    assumed_door_width_m=float(st.sidebar.selectbox("Calibration auto: largeur porte (m)", [0.73, 0.80, 0.90, 1.00], index=2)),

    wall_thickness_m=float(st.sidebar.slider("Murs: épaisseur (m)", 0.10, 0.35, 0.20, 0.01)),
    habitable_perimeter_mode=str(st.sidebar.selectbox("Pourtour habitable", ["total", "external"], index=0)),

    interior_alpha=float(st.sidebar.slider("Visu intérieur: alpha vert", 0.05, 0.60, 0.28, 0.01)),
)

st.sidebar.markdown(
    '<div class="small">Astuce: si tu rates des portes, baisse <b>Conf min door</b> à 0.03 et garde <b>Pass 2</b> à 1024.</div>',
    unsafe_allow_html=True,
)

# ---------------------------
# MAIN LAYOUT
# ---------------------------
left, right = st.columns([1.05, 1.0], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1) Upload du plan")
    file = st.file_uploader("PNG / JPG", type=["png", "jpg", "jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("2) Résultats")
    st.write("Upload une image puis clique sur **Analyze**.")
    st.markdown("</div>", unsafe_allow_html=True)

if file is not None:
    img_pil = Image.open(file)
    base_rgb = ensure_rgb_u8(img_pil)

    with left:
        image_in(
            st,
            rgb_to_png_bytes(base_rgb),
            caption=f"Input • {base_rgb.shape[1]}×{base_rgb.shape[0]} px",
        )

    colA, colB = st.columns([1, 1])
    run = button_in(colA, "🚀 Analyze")
    download_in(colB, "⬇️ Télécharger l'image input", rgb_to_png_bytes(base_rgb), "plan_input.png", "image/png")

    if run:
        client = get_client(API_URL, API_KEY)

        with st.spinner("Inference (multi-scale)…"):
            rooms_index, legend, m_doors_1, m_wins_1, rows_1, _ = infer_pass(
                client, params.model_id, img_pil.convert("RGB"),
                params.pass1_tile, params.pass1_over,
                write_rooms=True,
                conf_min_door=params.conf_min_door,
                conf_min_win=params.conf_min_win,
            )
            m_doors_1 = clean_mask(m_doors_1, params.min_area_door_px, params.clean_close_k_door)
            m_wins_1 = clean_mask(m_wins_1, params.min_area_win_px, params.clean_close_k_win)

            _, _, m_doors_2, m_wins_2, rows_2, _ = infer_pass(
                client, params.model_id, img_pil.convert("RGB"),
                params.pass2_tile, params.pass2_over,
                write_rooms=False,
                conf_min_door=params.conf_min_door,
                conf_min_win=params.conf_min_win,
            )
            m_doors_2 = clean_mask(m_doors_2, params.min_area_door_px, params.clean_close_k_door)
            m_wins_2 = clean_mask(m_wins_2, params.min_area_win_px, params.clean_close_k_win)

            doors = cv2.bitwise_or(m_doors_1, m_doors_2)
            wins = cv2.bitwise_or(m_wins_1, m_wins_2)

            walls = walls_from_rooms_index(rooms_index) if rooms_index is not None else None
            overlay = overlay_openings(base_rgb, doors, wins, walls)

            df_det = pd.DataFrame(rows_1 + rows_2)
            df_open = df_openings_from_masks(doors, wins, params.min_area_door_px, params.min_area_win_px)

            mask_units, lots_overlay, df_lots, lots_meta = detect_units_kmeans(
                base_rgb,
                K=params.kmeans_K,
                min_area_ratio=params.kmeans_min_area_ratio,
                smooth_k=params.kmeans_smooth_k,
                work_max=params.kmeans_work_max,
            )

            m_per_px, scale_meta = estimate_scale_from_doors_using_units(
                df_open=df_open,
                mask_units=mask_units,
                assumed_door_width_m=params.assumed_door_width_m,
                min_doors=2,
            )
            if m_per_px is not None:
                st.session_state["m_per_px"] = float(m_per_px)
            else:
                st.session_state.pop("m_per_px", None)

            surfaces = None
            interior_green = None

            if "m_per_px" in st.session_state and st.session_state["m_per_px"] > 0:
                surfaces = compute_surfaces_pourtours(
                    mask_units=mask_units,
                    m_per_px=float(st.session_state["m_per_px"]),
                    wall_thickness_m=params.wall_thickness_m,
                    habitable_perimeter_mode=params.habitable_perimeter_mode,
                )

                # ✅ IMAGE "Intérieur (vert) = surface habitable approx"
                if surfaces is not None:
                    interior_green = overlay_interior_green(
                        base_rgb=base_rgb,
                        mask_habitable_u8=surfaces["mask_habitable"],
                        alpha=params.interior_alpha,
                    )

                mpp = float(st.session_state["m_per_px"])
                if not df_open.empty:
                    df_open = df_open.copy()
                    df_open["length_m"] = df_open["length_px"].astype(float) * mpp
                    df_open["area_m2"] = df_open["area_px2"].astype(float) * (mpp ** 2)
                if not df_lots.empty:
                    df_lots = df_lots.copy()
                    df_lots["area_m2"] = df_lots["area_px2"].astype(float) * (mpp ** 2)

        # ---------------------------
        # RIGHT PANEL
        # ---------------------------
        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("🚪 Portes", int((df_open["class"] == "door").sum()) if not df_open.empty else 0)
            c2.metric("🪟 Fenêtres", int((df_open["class"] == "window").sum()) if not df_open.empty else 0)
            c3.metric("🧩 Lots", int(lots_meta.get("lots", 0)) if isinstance(lots_meta, dict) else 0)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Overlay portes/fenêtres")
            image_in(st, rgb_to_png_bytes(overlay))
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📏 Calibration auto (portes)")
            if m_per_px is None:
                st.warning(f"Calibration auto impossible: {scale_meta.get('reason','?')}")
            else:
                st.success(
                    f"1 px = {m_per_px*1000:.2f} mm | 1 m = {scale_meta['px_per_m']:.1f} px "
                    f"(quality={scale_meta['quality']}, doors_core={scale_meta['doors_used_core']}, cv={scale_meta['variation_cv']:.2f})"
                )
                if scale_meta.get("quality") == "weak":
                    st.warning("Qualité faible → préfère calibration manuelle (2 points) si tu veux du fiable.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Surfaces & pourtours")
            if surfaces is None:
                st.info("Surfaces indisponibles (échelle non calibrée).")
            else:
                st.write(
                    f"📐 Surface emprise: **{surfaces['surface_emprise_m2']:.2f} m²** | "
                    f"Pourtour: **{surfaces['pourtour_emprise_m']:.2f} m**"
                )
                st.write(
                    f"🧱 Surface murs: **{surfaces['surface_murs_m2']:.2f} m²** "
                    f"(épaisseur ~ **{surfaces['wall_thickness_m']:.2f} m** ≈ {surfaces['wall_thickness_px']} px)"
                )
                st.write(
                    f"✅ Surface habitable: **{surfaces['surface_habitable_m2']:.2f} m²** | "
                    f"Pourtour habitable: **{surfaces['pourtour_habitable_m']:.2f} m**"
                )
                with st.expander("Voir mask_habitable (debug)"):
                    image_in(st, gray_to_png_bytes(surfaces["mask_habitable"]), caption="mask_habitable")
            st.markdown("</div>", unsafe_allow_html=True)

            # ✅ AJOUT: l'image "Intérieur (vert)"
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Intérieur (vert) = surface habitable approx")
            if interior_green is None:
                st.info("Image intérieur indisponible (calibration requise).")
            else:
                image_in(st, rgb_to_png_bytes(interior_green))
                download_in(st, "⬇️ interior_green.png", rgb_to_png_bytes(interior_green), "interior_green.png", "image/png")
            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------
        # MASKS PREVIEW
        # ---------------------------
        st.markdown("### 🧪 Masques (portes / fenêtres / murs)")
        p1, p2, p3 = st.columns(3)
        image_in(p1, gray_to_png_bytes(doors), caption="mask_doors")
        image_in(p2, gray_to_png_bytes(wins), caption="mask_windows")
        if walls is not None:
            image_in(p3, gray_to_png_bytes(walls), caption="mask_walls (from rooms)")
        else:
            p3.info("Walls indisponibles")

        # ---------------------------
        # EMPRISE / LOTS
        # ---------------------------
        st.markdown("### 🧩 Emprise / Lots (K-means)")
        e1, e2 = st.columns(2)
        with e1:
            st.subheader("Overlay lots")
            image_in(st, rgb_to_png_bytes(lots_overlay))
        with e2:
            st.subheader("Mask emprise (units)")
            image_in(st, gray_to_png_bytes(mask_units))

        st.caption(
            f"K={lots_meta.get('K')} | bg_cluster={lots_meta.get('bg_cluster')} | scale={lots_meta.get('scale'):.3f} | "
            f"min_area={lots_meta.get('min_area_px2')} px² | lots={lots_meta.get('lots')}"
        )

        # ---------------------------
        # DOWNLOADS IMAGES
        # ---------------------------
        st.markdown("### ⬇️ Downloads images")
        dl1, dl2, dl3, dl4 = st.columns(4)
        download_in(dl1, "⬇️ overlay_openings.png", rgb_to_png_bytes(overlay), "overlay_openings.png", "image/png")
        download_in(dl2, "⬇️ mask_doors.png", gray_to_png_bytes(doors), "mask_doors.png", "image/png")
        download_in(dl3, "⬇️ mask_windows.png", gray_to_png_bytes(wins), "mask_windows.png", "image/png")
        if walls is not None:
            download_in(dl4, "⬇️ mask_walls.png", gray_to_png_bytes(walls), "mask_walls.png", "image/png")
        else:
            try:
                dl4.download_button("⬇️ mask_walls.png", data=b"", file_name="mask_walls.png", mime="image/png", disabled=True, use_container_width=True)
            except TypeError:
                dl4.download_button("⬇️ mask_walls.png", data=b"", file_name="mask_walls.png", mime="image/png", disabled=True, use_column_width=True)

        st.markdown("### ⬇️ Downloads emprise / lots")
        dlu1, dlu2 = st.columns(2)
        download_in(dlu1, "⬇️ lots_overlay.png", rgb_to_png_bytes(lots_overlay), "lots_overlay.png", "image/png")
        download_in(dlu2, "⬇️ mask_units.png", gray_to_png_bytes(mask_units), "mask_units.png", "image/png")

        # ---------------------------
        # TABLES
        # ---------------------------
        st.markdown("### Detections (brutes)")
        if not df_det.empty:
            st.dataframe(df_det.sort_values("confidence", ascending=False), height=260)
        else:
            st.info("Aucune détection brute.")

        st.markdown("### Openings (depuis masques)")
        if not df_open.empty:
            st.dataframe(df_open.sort_values(["class", "area_px2"], ascending=[True, False]), height=260)
        else:
            st.info("Aucune ouverture trouvée.")

        st.markdown("### Lots (depuis emprise)")
        if not df_lots.empty:
            st.dataframe(df_lots, height=260)
        else:
            st.info("Aucun lot détecté (essaie K=10 ou min_area_ratio=0.005).")

        # ---------------------------
        # CSV DOWNLOADS
        # ---------------------------
        st.markdown("### ⬇️ Downloads CSV")
        csv1 = df_det.to_csv(index=False).encode("utf-8")
        csv2 = df_open.to_csv(index=False).encode("utf-8")
        csv3 = df_lots.to_csv(index=False).encode("utf-8") if not df_lots.empty else b""

        cdl1, cdl2, cdl3 = st.columns(3)
        download_in(cdl1, "⬇️ doors_windows_detections.csv", csv1, "doors_windows_detections.csv", "text/csv")
        download_in(cdl2, "⬇️ openings_from_masks.csv", csv2, "openings_from_masks.csv", "text/csv")
        if not df_lots.empty:
            download_in(cdl3, "⬇️ lots.csv", csv3, "lots.csv", "text/csv")
        else:
            try:
                cdl3.download_button("⬇️ lots.csv", data=b"", file_name="lots.csv", mime="text/csv", disabled=True, use_container_width=True)
            except TypeError:
                cdl3.download_button("⬇️ lots.csv", data=b"", file_name="lots.csv", mime="text/csv", disabled=True, use_column_width=True)

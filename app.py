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
# UI / THEME
# ---------------------------
st.set_page_config(page_title="Floor Plan Analyzer (Demo)", page_icon="🏠", layout="wide")

st.markdown("""
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
""", unsafe_allow_html=True)

st.markdown('<p class="h-title">🏠 Floor Plan Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="h-sub">Démo Streamlit : upload d’un plan → détection portes/fenêtres → masques + overlay + export CSV.</p>', unsafe_allow_html=True)
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
    if xs[-1] != max(0, W - tile): xs.append(max(0, W - tile))
    if ys[-1] != max(0, H - tile): ys.append(max(0, H - tile))
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

@dataclass
class Params:
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
    m_wins  = np.zeros((H, W), np.uint8)
    rows: List[dict] = []
    kept_doors = kept_wins = preds_total = tiles = 0

    with tempfile.TemporaryDirectory() as td:
        for (x0, y0, x1, y1) in iter_tiles(W, H, tile_size, overlap):
            tiles += 1
            tile = img_pil.crop((x0, y0, x1, y1))
            tw, th = tile.size
            tile_path = os.path.join(td, f"tile_{tile_size}_{x0}_{y0}.png")
            tile.save(tile_path)

            # ✅ IMPORTANT: utiliser model_id choisi
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
                is_win  = any(k in lbl for k in ["window", "windows", "fen", "fenetre", "fenetres"])

                if is_door and conf < conf_min_door: continue
                if is_win  and conf < conf_min_win:  continue

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
                        cv2.fillPoly(m_doors, [poly], 255); kept_doors += 1
                    elif is_win:
                        cv2.fillPoly(m_wins,  [poly], 255); kept_wins += 1
                    elif write_rooms and rooms_index is not None:
                        cv2.fillPoly(rooms_index, [poly], rid_for(lbl))

                    rows.append({
                        "label": lbl, "type": "polygon",
                        "x_px": cxc, "y_px": cyc,
                        "width_px": (xmx - xmn), "height_px": (ymx - ymn),
                        "confidence": conf, "pass_tile": tile_size
                    })

                # bbox
                elif all(k in p for k in ("x", "y", "width", "height")):
                    cx, cy = float(p["x"]), float(p["y"])
                    bw, bh = float(p["width"]), float(p["height"])
                    if max(cx, cy, bw, bh) <= 1.5:
                        cx *= tw; cy *= th; bw *= tw; bh *= th

                    x1t = int(cx - bw / 2); y1t = int(cy - bh / 2)
                    x2t = int(cx + bw / 2); y2t = int(cy + bh / 2)
                    x1t, y1t, x2t, y2t = clamp_box(x1t, y1t, x2t, y2t, tw, th)

                    x1g, y1g = x1t + x0, y1t + y0
                    x2g, y2g = x2t + x0, y2t + y0
                    x1g, y1g, x2g, y2g = clamp_box(x1g, y1g, x2g, y2g, W, H)

                    cxc, cyc = (x1g + x2g) / 2, (y1g + y2g) / 2

                    if is_door:
                        cv2.rectangle(m_doors, (x1g, y1g), (x2g, y2g), 255, -1); kept_doors += 1
                    elif is_win:
                        cv2.rectangle(m_wins,  (x1g, y1g), (x2g, y2g), 255, -1); kept_wins += 1
                    elif write_rooms and rooms_index is not None:
                        cv2.rectangle(rooms_index, (x1g, y1g), (x2g, y2g), rid_for(lbl), -1)

                    rows.append({
                        "label": lbl, "type": "bbox",
                        "x_px": cxc, "y_px": cyc,
                        "width_px": (x2g - x1g), "height_px": (y2g - y1g),
                        "confidence": conf, "pass_tile": tile_size
                    })

    stats = dict(tile_size=tile_size, overlap=overlap, tiles=tiles, preds=preds_total, kept_doors=kept_doors, kept_windows=kept_wins)
    return rooms_index, legend, m_doors, m_wins, rows, stats


def walls_from_rooms_index(rooms_index: np.ndarray) -> np.ndarray:
    a = rooms_index
    H, W = a.shape[:2]
    walls = np.zeros((H, W), np.uint8)
    walls[1:, :]  |= (a[1:, :]  != a[:-1, :])
    walls[:-1, :] |= (a[:-1, :] != a[1:, :])
    walls[:, 1:]  |= (a[:, 1:]  != a[:, :-1])
    walls[:, :-1] |= (a[:, :-1] != a[:, 1:])
    walls = (walls.astype(np.uint8) * 255)
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
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
    cs_w, _ = cv2.findContours(wins,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fill = np.zeros_like(bgr)
    if cs_d: cv2.fillPoly(fill, cs_d, (255, 0, 255))   # doors magenta
    if cs_w: cv2.fillPoly(fill, cs_w, (255, 255, 0))   # windows cyan-ish
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
            rows.append({
                "class": label,
                "x_px": float(x + w / 2),
                "y_px": float(y + h / 2),
                "width_px": float(w),
                "height_px": float(h),
                "length_px": float(max(w, h)),
                "area_px2": float(area),
            })

    extract(doors, "door", min_area_d)
    extract(wins,  "window", min_area_w)
    return pd.DataFrame(rows)

def to_png_bytes(arr: np.ndarray) -> bytes:
    if arr.ndim == 2:
        img = Image.fromarray(arr.astype(np.uint8))
    else:
        img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

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
)
st.sidebar.markdown('<div class="small">Astuce: si tu rates des portes, baisse <b>Conf min door</b> à 0.03 et garde <b>Pass 2</b> à 1024.</div>', unsafe_allow_html=True)

# ---------------------------
# MAIN
# ---------------------------
left, right = st.columns([1.05, 1.0], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1) Upload du plan")
    file = st.file_uploader("PNG / JPG", type=["png", "jpg", "jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("2) Résultats")
    st.write("Upload une image puis clique sur **Analyze**.")
    st.markdown('</div>', unsafe_allow_html=True)

if file is not None:
    img_pil = Image.open(file).convert("RGB")
    base_rgb = np.array(img_pil)

    with left:
        st.image(img_pil, caption=f"Input • {img_pil.size[0]}×{img_pil.size[1]} px", use_container_width=True)

    colA, colB = st.columns([1, 1])
    run = colA.button("🚀 Analyze", use_container_width=True)
    colB.download_button("⬇️ Télécharger l'image input", data=to_png_bytes(base_rgb), file_name="plan_input.png", mime="image/png", use_container_width=True)

    if run:
        client = get_client(API_URL, API_KEY)

        with st.spinner("Inference (multi-scale)…"): 
            rooms_index, legend, m_doors_1, m_wins_1, rows_1, _ = infer_pass(
                client, params.model_id, img_pil,
                params.pass1_tile, params.pass1_over,
                write_rooms=True,
                conf_min_door=params.conf_min_door,
                conf_min_win=params.conf_min_win
            )
            m_doors_1 = clean_mask(m_doors_1, params.min_area_door_px, params.clean_close_k_door)
            m_wins_1  = clean_mask(m_wins_1,  params.min_area_win_px,  params.clean_close_k_win)

            _, _, m_doors_2, m_wins_2, rows_2, _ = infer_pass(
                client, params.model_id, img_pil,
                params.pass2_tile, params.pass2_over,
                write_rooms=False,
                conf_min_door=params.conf_min_door,
                conf_min_win=params.conf_min_win
            )
            m_doors_2 = clean_mask(m_doors_2, params.min_area_door_px, params.clean_close_k_door)
            m_wins_2  = clean_mask(m_wins_2,  params.min_area_win_px,  params.clean_close_k_win)

            doors = cv2.bitwise_or(m_doors_1, m_doors_2)
            wins  = cv2.bitwise_or(m_wins_1,  m_wins_2)

            walls = walls_from_rooms_index(rooms_index) if rooms_index is not None else None
            overlay = overlay_openings(base_rgb, doors, wins, walls)

            df_det = pd.DataFrame(rows_1 + rows_2)
            df_open = df_openings_from_masks(doors, wins, params.min_area_door_px, params.min_area_win_px)

        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("🚪 Portes", int((df_open["class"] == "door").sum()) if not df_open.empty else 0)
            c2.metric("🪟 Fenêtres", int((df_open["class"] == "window").sum()) if not df_open.empty else 0)
            c3.metric("🧱 Rooms classes", len(legend) if isinstance(legend, dict) else 0)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Overlay")
            st.image(overlay, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        p1, p2, p3 = st.columns(3)
        p1.image(doors, caption="mask_doors", use_container_width=True, clamp=True)
        p2.image(wins, caption="mask_windows", use_container_width=True, clamp=True)
        if walls is not None:
            p3.image(walls, caption="mask_walls (from rooms)", use_container_width=True, clamp=True)
        else:
            p3.info("Walls indisponibles")

        dl1, dl2, dl3, dl4 = st.columns(4)
        dl1.download_button("⬇️ overlay_openings.png", data=to_png_bytes(overlay), file_name="overlay_openings.png", mime="image/png", use_container_width=True)
        dl2.download_button("⬇️ mask_doors.png", data=to_png_bytes(doors), file_name="mask_doors.png", mime="image/png", use_container_width=True)
        dl3.download_button("⬇️ mask_windows.png", data=to_png_bytes(wins), file_name="mask_windows.png", mime="image/png", use_container_width=True)
        if walls is not None:
            dl4.download_button("⬇️ mask_walls.png", data=to_png_bytes(walls), file_name="mask_walls.png", mime="image/png", use_container_width=True)
        else:
            dl4.download_button("⬇️ mask_walls.png", data=b"", file_name="mask_walls.png", mime="image/png", disabled=True, use_container_width=True)

        st.markdown("### Detections (brutes)")
        if not df_det.empty:
            st.dataframe(df_det.sort_values("confidence", ascending=False), use_container_width=True, height=260)
        else:
            st.info("Aucune détection brute.")

        st.markdown("### Openings (depuis masques)")
        if not df_open.empty:
            st.dataframe(df_open.sort_values(["class", "area_px2"], ascending=[True, False]), use_container_width=True, height=260)
        else:
            st.info("Aucune ouverture trouvée.")

        csv1 = df_det.to_csv(index=False).encode("utf-8")
        csv2 = df_open.to_csv(index=False).encode("utf-8")
        cdl1, cdl2 = st.columns(2)
        cdl1.download_button("⬇️ doors_windows_detections.csv", data=csv1, file_name="doors_windows_detections.csv", mime="text/csv", use_container_width=True)
        cdl2.download_button("⬇️ openings_from_masks.csv", data=csv2, file_name="openings_from_masks.csv", mime="text/csv", use_container_width=True)

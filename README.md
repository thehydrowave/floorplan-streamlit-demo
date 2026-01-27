# Floor Plan Analyzer (Streamlit demo)

## 1) Configure
Set environment variables:
- `ROBOFLOW_API_KEY` (required)
- `ROBOFLOW_MODEL_ID` (optional, default: cubicasa5k-2-qpmsa/6)

## 2) Install
```bash
pip install -r requirements.txt
```

## 3) Run
```bash
streamlit run app.py
```

## Notes
- The Roboflow API key must stay server-side (Streamlit Cloud secrets / Render env vars).
- The app does multi-scale tiling (2048 + 1024) to improve door/window recall.

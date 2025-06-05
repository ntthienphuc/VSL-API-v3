# SPOTER ONNX FastAPI Server 🚀

Real-time (and offline) Vietnamese Sign-Language (VSL) recognition API powered by **MediaPipe Holistic**, **ONNX Runtime**, **FastAPI**, and **Redis Streams**.

## ✨ Key Features

- Thread-safe ONNX inference wrapper with CPU/GPU support
- Offline and chunked real-time inference endpoints
- Arm-pose heuristic for determining valid sign segments
- Auto-cleanup stale buffers in Redis
- Simple configuration using `dataclass` based models
- Lightweight, production-ready design

---

## 📂 Project Structure

```
.
├── app.py                 # FastAPI entry-point
├── inference.py           # Inference logic on videos
├── configs/
│   └── arguments.py       # Model & Inference configurations
├── data/
│   └── utils.py           # Arm class, angle calculation, state management
├── pipelines/
│   ├── spoter_onnx_inference.py   # ONNX Runtime wrapper and transforms
│   └── update.py          # Save uploaded videos by label
└── utils/
    ├── constants.py       # LANDMARK lists and config constants
    └── loggers.py         # Unified logger
```

---

## 📅 Requirements

- Python 3.10–3.12
- `fastapi`, `uvicorn`, `opencv-python`, `mediapipe`, `onnxruntime[(-gpu)]`, `redis`

```bash
# CPU
pip install -r requirements.txt

# Or GPU support (if CUDA >= 11.8)
pip install onnxruntime-gpu==1.18.0
```

---

## 🛠️ Setup

```bash
# 1. Clone
$ git clone <repo_url> && cd spoter-api

# 2. Create virtual environment
$ python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
$ pip install -r requirements.txt

# 4. Download model + gloss
$ mkdir -p models && cp <model.onnx> models/spoter_v3.0.onnx
$ cp <gloss.csv> gloss.csv

# 5. Start Redis
$ docker run -d --name redis -p 6379:6379 redis:7-alpine

# 6. Launch API
$ uvicorn app:app --reload --port 8000
```

---

## 🔌 API Endpoints

### `/upload` `POST`
Upload and archive video by label.

| Field       | Type | Required | Notes |
|-------------|------|----------|-------|
| `label`     | str  | Yes      | Gloss/word |
| `video_file`| file | Yes      | .mp4/.avi etc |

---

### `/spoter` `POST`
One-shot inference on a whole video.

| Field            | Type  | Default | Notes |
|------------------|-------|---------|-------|
| `video_file`     | file  |         | Required |
| `angle_threshold`| int   | 130     | Arm "up" threshold |
| `top_k`          | int   | 3       | Top-k labels |
| `rotate`         | bool  | False   | Rotate 90° |

**Response:**
```json
{
  "results_merged": [
    {
      "predictions": [
        {"gloss": "Xin_chào", "score": 0.91},
        {"gloss": "Tạm_biệt", "score": 0.04}
      ],
      "inference_time": 0.12,
      "start_time": 1.52,
      "end_time": 2.71
    }
  ],
  "message": "Inference completed successfully."
}
```

---

### `/spoter_segmented` `POST`
Stream-friendly chunked inference with Redis-based buffering.

| Field         | Type | Required | Notes |
|---------------|------|----------|-------|
| `video_file`  | file | Yes      | Chunk of video |
| `clientId`    | str  | Yes      | Session/client key |
| `angle_threshold` | int | 130 | Optional |
| `rotate`      | bool | False    | Optional |

**Behavior:**
- Collects signs when arm is "up"
- Once arms "drop", run inference
- Stores intermediate frames in Redis

**Returns:**
- `Buffered`
- `Hand action ended – inference now`
- `Hand down – run inference`

---

## 🎓 Pose Detection Logic

```text
if angle < threshold:
    arm is "up" → collect frames
else:
    arm is "down" → may infer if previously "up"
```

Frame window is inferred when both arms return to "down".

---

## 💪 Production Tips

- Use multiple workers: `uvicorn app:app --workers 4`
- Offload frame uploads to `/spoter_segmented`
- Deploy Redis separately
- Use `/healthcheck` for container liveness

---

## 🔮 License

MIT License © 2025 Tuan (Kevin) Le & Contributors

import os
import time
import tempfile
import asyncio
import traceback
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import threading
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import ORJSONResponse
import redis
import orjson

from data.utils import Arm
from configs.arguments import ModelConfig, InferenceConfig
from inference import run_inference, rotate_frame
from pipelines.spoter_onnx_inference import SPOTERONNXInferer, load_id2label
from pipelines.update import save_video

logging.basicConfig(level=logging.INFO)
app = FastAPI()

# Thread-local storage (chỉ giữ pool cho inference)
thread_local = threading.local()


# Định nghĩa các lớp để tái tạo cấu trúc Mediapipe (để reconstruct từ dict)
class Landmark:
    def __init__(self, x, y, z=0.0, visibility=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class Landmarks:
    def __init__(self, landmarks):
        self.landmark = landmarks


class MediapipeResult:
    def __init__(self, pose_landmarks=None, left_hand_landmarks=None, right_hand_landmarks=None):
        self.pose_landmarks = pose_landmarks
        self.left_hand_landmarks = left_hand_landmarks
        self.right_hand_landmarks = right_hand_landmarks


# Hàm chuyển đổi Mediapipe result sang dict
def mediapipe_to_dict(mp_result: MediapipeResult):
    def landmarks_to_list(landmarks):
        if landmarks is None:
            return None
        return [
            {'x': float(lm.x), 'y': float(lm.y), 'z': float(lm.z), 'visibility': float(lm.visibility)}
            for lm in landmarks.landmark
        ]

    return {
        'pose_landmarks': landmarks_to_list(mp_result.pose_landmarks),
        'left_hand_landmarks': landmarks_to_list(mp_result.left_hand_landmarks),
        'right_hand_landmarks': landmarks_to_list(mp_result.right_hand_landmarks),
    }


def dict_to_mediapipe(data):
    def list_to_landmarks(landmarks_list):
        if landmarks_list is None:
            return None
        landmarks = [Landmark(**lm) for lm in landmarks_list]
        return Landmarks(landmarks)

    return MediapipeResult(
        pose_landmarks=list_to_landmarks(data['pose_landmarks']),
        left_hand_landmarks=list_to_landmarks(data['left_hand_landmarks']),
        right_hand_landmarks=list_to_landmarks(data['right_hand_landmarks']),
    )


# Exception handler
@app.exception_handler(Exception)
async def _exc_handler(request: Request, exc: Exception):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logging.error(tb)
    return ORJSONResponse(status_code=500, content={"detail": tb})


# Background cleaner cho Redis (dọn các buffer cũ)
async def _cleaner():
    THRESH = 30  # giây
    while True:
        await asyncio.sleep(10)
        now = time.time()
        redis_client = app.state.redis_client
        for key in redis_client.scan_iter("buffer:*"):
            last_update = redis_client.hget(key, "last_update")
            if last_update and now - float(last_update.decode('utf-8')) > THRESH:
                redis_client.delete(key)
                logging.info(f"[Cleaner] cleared stale buffer for {key}")


# Startup event
@app.on_event("startup")
async def _startup():
    logging.info("[Startup] loading model ...")
    # 1. Tạo Inferer ONNX tại startup, tái sử dụng xuyên suốt
    app.state.spoter_inferer = SPOTERONNXInferer(
        "models/spoter_v3.0.onnx",
        load_id2label("gloss.csv"),
        num_frames=70,
        top_k=3,
    )
    logging.info("Providers: %s", app.state.spoter_inferer.session.get_providers())

    # 2. Khởi tạo Redis client (localhost:6379)
    app.state.redis_client = redis.Redis(host='localhost', port=6379, db=0)

    # 3. Thiết lập ThreadPoolExecutor cho inference (tối đa 4 thread)
    thread_local.pool = ThreadPoolExecutor(max_workers=4)

    # 4. Khởi chạy cleaner task
    asyncio.create_task(_cleaner())


# Helper: lưu file tạm và return path string
def save_tmp_file(video_bytes: bytes, suffix: str = '.mp4') -> str:
    tmp_path = Path(tempfile.gettempdir()) / f"{int(time.time()*1000)}{suffix}"
    tmp_path.write_bytes(video_bytes)
    return str(tmp_path)


# Helper: lấy Redis lock (per clientId)
def get_redis_lock(redis_client, buffer_key):
    return redis_client.lock(f"lock:{buffer_key}", timeout=5, blocking_timeout=1)


# Endpoint /upload (giữ nguyên)
@app.post("/upload")
async def upload(
    label: str = Form(..., description="Từ/nhãn của video, ví dụ: 'Đi bộ'"),
    video_file: UploadFile = File(...),
):
    ext = Path(video_file.filename).suffix
    if ext.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    content = await video_file.read()
    saved_path = save_video(label.strip(), content, ext)
    return ORJSONResponse(content={"message": "Upload thành công", "saved_to": str(saved_path)})


# Endpoint /spoter (full-video inference)
@app.post("/spoter")
async def spoter(
    video_file: UploadFile = File(...),
    angle_threshold: int = 130,
    top_k: int = 3,
    rotate: bool = Form(False),
):
    if not video_file.filename:
        raise HTTPException(400, "No video file")

    content = await video_file.read()
    tmp = save_tmp_file(content, suffix=Path(video_file.filename).suffix)

    try:
        model_cfg = ModelConfig(
            arch="spoter",
            hidden_dim=108,
            onnx_path="models/spoter_v3.0.onnx",
            gloss_csv_path="gloss.csv",
        )
        infer_cfg = InferenceConfig(
            source=str(tmp),
            visibility=0.5,
            angle_threshold=angle_threshold,
            top_k=top_k,
        )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            thread_local.pool,
            run_inference,
            model_cfg,
            infer_cfg,
            app.state.spoter_inferer,
            rotate,
        )
        return ORJSONResponse(content={"results_merged": result.get("results_merged", []), "message": "Inference completed successfully."})
    finally:
        Path(tmp).unlink(missing_ok=True)


# Endpoint /spoter_segmented (chunked inference with Redis)
@app.post("/spoter_segmented")
async def spoter_segmented(
    video_file: UploadFile = File(...),
    clientId: str = Form(...),
    angle_threshold: int = 130,
    rotate: bool = Form(False),
):
    if not video_file.filename:
        raise HTTPException(400, "No video file")

    # 1. Đọc bytes từ video file
    video_bytes = await video_file.read()

    # 2. Lưu file tạm
    tmp_path = save_tmp_file(video_bytes, suffix='.mp4')

    # 3. Chuẩn bị Redis lock
    redis_client = app.state.redis_client
    buffer_key = f"buffer:{clientId}"
    lock = get_redis_lock(redis_client, buffer_key)
    have_lock = lock.acquire(blocking=True)
    if not have_lock:
        raise HTTPException(503, "Resource busy, try again shortly")

    cap = cv2.VideoCapture(tmp_path)
    mp_holo = None
    try:
        # 4. Đọc từng frame thô & (nếu cần) rotate
        frames = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if rotate:
                    frame = rotate_frame(frame)
                frames.append(frame)
        finally:
            cap.release()

        if not frames:
            return ORJSONResponse(content={"message": "Empty chunk"})

        # 5. Lấy dữ liệu cũ từ Redis (nếu có)
        data = redis_client.hgetall(buffer_key)
        if data:
            buffer_list = orjson.loads(data[b'data'])
            buffer = [dict_to_mediapipe(d) for d in buffer_list]
            left_arm_data = orjson.loads(data[b'left_arm'])
            right_arm_data = orjson.loads(data[b'right_arm'])
            left_arm = Arm.from_dict(left_arm_data, "left", 0.5)
            right_arm = Arm.from_dict(right_arm_data, "right", 0.5)
        else:
            buffer = []
            left_arm = Arm("left", 0.5)
            right_arm = Arm("right", 0.5)

        # 6. Lấy 3 frame mẫu để ước tính tay (chỉ cần 3 frame)
        length = len(frames)
        if length >= 3:
            sidx = [length // 3, 2 * length // 3, length - 1]
        else:
            sidx = list(range(length))

        # 7. Tái sử dụng 1 instance Mediapipe Holistic
        mp_holo = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.6
        )
        hand_flags = []
        for i in sidx:
            r = mp_holo.process(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
            left_arm.set_pose(r.pose_landmarks.landmark if r.pose_landmarks else None)
            right_arm.set_pose(r.pose_landmarks.landmark if r.pose_landmarks else None)
            flag = (
                (left_arm.angle and left_arm.angle < angle_threshold)
                or (right_arm.angle and right_arm.angle < angle_threshold)
            )
            hand_flags.append(flag)

        # 8. Xác định start, end cho chunk
        start, end = 0, length - 1
        infer_now = False
        if length >= 3:
            if hand_flags[0] and not hand_flags[1] and not hand_flags[2]:
                end = sidx[0]
                infer_now = True
            elif hand_flags[0] and hand_flags[1] and not hand_flags[2]:
                end = sidx[1]
                infer_now = True
            elif not hand_flags[0] and hand_flags[1] and hand_flags[2]:
                start = sidx[0] + 1
            elif not hand_flags[0] and not hand_flags[1] and hand_flags[2]:
                start = sidx[1] + 1
            elif not any(hand_flags):
                if buffer:
                    preds, t = await _infer_async(buffer)
                    left_arm.reset_state()
                    right_arm.reset_state()
                    redis_client.delete(buffer_key)
                    # Chỉ close Mediapipe nếu mp_holo không None
                    if mp_holo is not None:
                        mp_holo.close()
                        mp_holo = None
                    return ORJSONResponse(content={"message": "Hand down – run inference", "predictions": preds, "inference_time": t})
                # Chỉ close Mediapipe nếu mp_holo không None
                if mp_holo is not None:
                    mp_holo.close()
                    mp_holo = None
                return ORJSONResponse(content={"message": "No hand"})

        # 9. Build chunk_res (list of MediapipeResult)
        chunk_res = []
        for i in range(start, end + 1):
            r = mp_holo.process(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
            chunk_res.append(r)
        # Sau khi xong, close Mediapipe
        if mp_holo is not None:
            mp_holo.close()
            mp_holo = None

        # 10. Ghép chunk_res vào buffer
        buffer.extend(chunk_res)

        # 11. Nếu infer_now, chạy inference
        if infer_now:
            preds, t = await _infer_async(buffer)
            left_arm.reset_state()
            right_arm.reset_state()
            redis_client.delete(buffer_key)
            return ORJSONResponse(content={"message": "Hand action ended – inference now", "predictions": preds, "inference_time": t})

        # 12. Nếu chưa infer, lưu buffer mới vào Redis
        buffer_list = [mediapipe_to_dict(res) for res in buffer]
        left_arm_json = orjson.dumps(left_arm.to_dict())
        right_arm_json = orjson.dumps(right_arm.to_dict())
        serialized_data = orjson.dumps(buffer_list)

        pipe = redis_client.pipeline()
        pipe.hset(buffer_key, "data", serialized_data)
        pipe.hset(buffer_key, "left_arm", left_arm_json)
        pipe.hset(buffer_key, "right_arm", right_arm_json)
        pipe.hset(buffer_key, "last_update", str(time.time()))
        pipe.expire(buffer_key, 30)
        pipe.execute()
        return ORJSONResponse(content={"message": "Buffered", "buffer_size": len(buffer)})

    finally:
        # 13. Giải phóng Mediapipe nếu còn
        if mp_holo is not None:
            mp_holo.close()
        # 14. Xóa file tạm
        os.remove(tmp_path)
        # 15. Thả Redis lock
        lock.release()


# Healthcheck
@app.get("/healthcheck")
def _health():
    return ORJSONResponse(content={"status": "ok"})


# Internal async infer helper (gọi inferer trong thread pool)
async def _infer_async(buffer):
    loop = asyncio.get_event_loop()
    t0 = time.time()
    preds = await loop.run_in_executor(thread_local.pool, app.state.spoter_inferer.infer, buffer)
    return preds, time.time() - t0

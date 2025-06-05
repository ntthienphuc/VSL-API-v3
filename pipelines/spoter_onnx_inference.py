import csv
import numpy as np
import onnxruntime as ort
from typing import List, Dict
from torchvision.transforms import Compose

from utils.constants import LANDMARKS, BODY_LANDMARKS, HAND_LANDMARKS


# --------------------------------------------------------------------------- #
# utils                                                                       #
# --------------------------------------------------------------------------- #
def load_id2label(csv_path: str) -> Dict[str, str]:
    """
    Đọc file CSV (id, gloss) → dict {id: gloss}.
    """
    id2label = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                id2label[row[0].strip()] = row[1].strip()
    return id2label


# --------------------------------------------------------------------------- #
# Inference wrapper                                                           #
# --------------------------------------------------------------------------- #
class SPOTERONNXInferer:
    """
    Bao wrapper ONNX Runtime – thread-safe.
    """

    def __init__(
        self,
        onnx_path: str,
        id2label: Dict[str, str],
        num_frames: int = 100,
        top_k: int = 3,
        device: str = "auto",
        num_cpu_threads: int = 4
    ):
        self.top_k = top_k
        self.num_frames = num_frames
        self.id2label = id2label

        # ------------- SessionOptions
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_mem_pattern = True
        # Nếu muốn tận dụng đa luồng trong chính inference graph:
        so.execution_mode = ort.ExecutionMode.ORT_PARALLEL

        # Mỗi process FastAPI ≈ 1 worker → tránh over-subscription
        if device == "gpu":
            so.intra_op_num_threads = 1       # GPU sẽ tự lo phần tính toán, CPU chỉ copy dữ liệu
            so.inter_op_num_threads = 1
        else:                                 # fallback CPU
            so.intra_op_num_threads = num_cpu_threads   # 4-8 hợp lý nếu CPU
            so.inter_op_num_threads = 1

        # ------------- Providers
        providers = ["CPUExecutionProvider"]
        if device != "cpu" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": 8 * 1024 * 1024 * 1024,  # 8 GB VRAM
                        "do_copy_in_default_stream": 1,
                    },
                ),
                "CPUExecutionProvider",
            ]

        self.session = ort.InferenceSession(
            onnx_path, sess_options=so, providers=providers
        )

        # ------------- Transforms
        self.transforms = Compose(
            [
                JointSelect(),
                Pad(self.num_frames),
                TensorToDict(),
                SingleBodyDictNormalize(),
                SingleHandDictNormalize(),
                DictToTensor(),
                Shift(),
            ]
        )

    # --------------------------------------------------------------------- #
    # Public                                                                 #
    # --------------------------------------------------------------------- #
    def infer(self, mediapipe_list: list) -> List[Dict[str, float]]:
        """
        Trả về top-k list dict: {"gloss": str, "score": float}
        - Dùng hoàn toàn NumPy để tính softmax & top-k, tránh chuyển qua PyTorch.
        """
        # 1. Áp transform (ra torch.Tensor), sau đó chuyển sang NumPy
        tensor = self.transforms(mediapipe_list).unsqueeze(0).float()  # shape: (1, T, P, 2)
        input_numpy = tensor.numpy()

        # 2. Chạy ONNX
        logits = self.session.run(None, {"poses": input_numpy})[0]  # shape: (1, C)

        # 3. Tính softmax NumPy
        #     logits: (1, C)
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)  # shape: (1, C)

        # 4. Chọn top-k indices
        #    argsort giảm dần (dấu "-")
        idxs = np.argsort(-probs, axis=1)[:, : self.top_k].squeeze().tolist()
        if self.top_k == 1:
            idxs = [idxs]  # ensure list-of-list

        # 5. Lấy scores tương ứng
        if self.top_k > 1:
            scores = [float(probs[0, i]) for i in idxs]
        else:
            scores = [float(probs[0, idxs[0]])]

        # 6. Build output
        results = []
        for i, s in zip(idxs, scores):
            gloss = self.id2label.get(str(i), f"unk_{i}")
            results.append({"gloss": gloss, "score": s})

        return results


# --------------------------------------------------------------------------- #
# -------------  Các transform bên dưới giữ nguyên logic gốc  -------------- #
# --------------------------------------------------------------------------- #
class JointSelect:
    def __init__(self):
        self.pose_idxs = (0, -1, 5, 2, 8, 7, 12, 11, 14, 13, 16, 15)
        self.hand_idxs = (
            0,
            8, 7, 6, 5,
            12, 11, 10, 9,
            16, 15, 14, 13,
            20, 19, 18, 17,
            4, 3, 2, 1,
        )

    def __parse(self, idxs, lm):
        if lm is None:
            return np.zeros((len(idxs), 2))
        out = []
        for idx in idxs:
            if idx == -1:
                out.append([0, 0])
            else:
                l = lm.landmark[idx]
                out.append([l.x, l.y])
        return np.asarray(out)

    def __call__(self, poses):
        import torch

        total = len(self.pose_idxs) + len(self.hand_idxs) * 2
        arr = np.zeros((len(poses), total, 2))
        for i, p in enumerate(poses):
            arr[i] = np.vstack(
                [
                    self.__parse(self.pose_idxs, p.pose_landmarks),
                    self.__parse(self.hand_idxs, p.left_hand_landmarks),
                    self.__parse(self.hand_idxs, p.right_hand_landmarks),
                ]
            )
        return torch.from_numpy(arr)


class Pad:
    def __init__(self, num_frames: int = 150):
        self.num_frames = num_frames

    def __call__(self, data):
        """
        Parameters
        ----------
        data : torch.Tensor
            Tensor (T, num_points, 2).  T có thể <, = hoặc > self.num_frames.

        Returns
        -------
        torch.Tensor
            Tensor đã được pad/truncate về đúng self.num_frames.
        """
        import torch

        T, P, _ = data.shape

        # --- Trường hợp khớp chiều ---
        if T == self.num_frames:
            return data

        # --- Trường hợp VIDEO NGẮN hơn ---
        if T < self.num_frames:
            out = torch.zeros((self.num_frames, P, 2), dtype=data.dtype)
            out[:T] = data
            rest = self.num_frames - T

            # lặp lại toàn bộ chuỗi gốc cho đủ độ dài
            loops = int(np.ceil(rest / T))
            pad_block = torch.cat([data] * loops, dim=0)[:rest]
            out[T:] = pad_block
            return out

        # --- Trường hợp VIDEO DÀI hơn ---
        indices = np.linspace(0, T - 1, self.num_frames, dtype=int)
        return data[indices]


class TensorToDict:
    def __call__(self, data):
        arr = data.numpy()
        return {lm: arr[:, i] for i, lm in enumerate(LANDMARKS)}


class SingleBodyDictNormalize:
    def __call__(self, row):
        sequence_size = len(row["leftEar"])
        valid = True
        last_start, last_end = None, None

        for t in range(sequence_size):
            ls, rs = row["leftShoulder"][t], row["rightShoulder"][t]
            neck, nose = row["neck"][t], row["nose"][t]

            if (ls[0] == 0 or rs[0] == 0) and (neck[0] == 0 or nose[0] == 0):
                if not last_start:
                    valid = False
                    continue
                start, end = last_start, last_end
            else:
                if ls[0] != 0 and rs[0] != 0:
                    dist = np.linalg.norm(ls - rs)
                else:
                    dist = np.linalg.norm(neck - nose)
                start = [neck[0] - 3 * dist, row["leftEye"][t][1] + dist]
                end = [neck[0] + 3 * dist, start[1] - 6 * dist]
                last_start, last_end = start, end

            start[0] = max(start[0], 0)
            start[1] = max(start[1], 0)
            end[0] = max(end[0], 0)
            end[1] = max(end[1], 0)

            for id_ in BODY_LANDMARKS:
                if row[id_][t][0] == 0:
                    continue
                denom_x = end[0] - start[0]
                denom_y = start[1] - end[1]
                if denom_x == 0 or denom_y == 0:
                    valid = False
                    break
                row[id_][t][0] = (row[id_][t][0] - start[0]) / denom_x
                row[id_][t][1] = (row[id_][t][1] - end[1]) / denom_y

        return row if valid else row


class SingleHandDictNormalize:
    def __call__(self, row):
        range_size = 2 if "wrist_1" in row else 1
        hand_map = {h: [f"{l}_{h}" for l in HAND_LANDMARKS] for h in range(range_size)}

        for h in range(range_size):
            seq = len(row[f"wrist_{h}"])
            for t in range(seq):
                xs = [row[k][t][0] for k in hand_map[h] if row[k][t][0] != 0]
                ys = [row[k][t][1] for k in hand_map[h] if row[k][t][1] != 0]
                if not xs or not ys:
                    continue
                w, hgt = max(xs) - min(xs), max(ys) - min(ys)
                if w > hgt:
                    dx = 0.1 * w
                    dy = dx + (w - hgt) / 2
                else:
                    dy = 0.1 * hgt
                    dx = dy + (hgt - w) / 2
                start = (min(xs) - dx, min(ys) - dy)
                end = (max(xs) + dx, max(ys) + dy)
                for k in hand_map[h]:
                    px, py = row[k][t]
                    if px == 0:
                        continue
                    row[k][t][0] = (px - start[0]) / (end[0] - start[0])
                    row[k][t][1] = (py - start[1]) / (end[1] - start[1])
        return row


class DictToTensor:
    def __call__(self, data):
        import torch

        T = len(data["leftEar"])
        P = len(LANDMARKS)
        out = np.empty((T, P, 2))
        for i, lm in enumerate(LANDMARKS):
            out[:, i] = data[lm]
        return torch.from_numpy(out)


class Shift:
    def __call__(self, data):
        return data - 0.5

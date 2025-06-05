import cv2
import numpy as np
from time import time

import mediapipe as mp

from utils.loggers import logging, config_logger
from configs.arguments import ModelConfig, InferenceConfig
from pipelines.spoter_onnx_inference import load_id2label, SPOTERONNXInferer
from data.utils import Arm, ok_to_get_frame


def rotate_frame(frame: np.ndarray) -> np.ndarray:
    """
    Rotate một frame 90 độ theo chiều kim đồng hồ.

    Args:
        frame (np.ndarray): Input frame in BGR format.

    Returns:
        np.ndarray: Rotated frame.
    """
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)


def get_one_arm_timestamp(left_arm: Arm, right_arm: Arm):
    """
    Determine the start_time & end_time của sign dựa vào left/right arms.

    Args:
        left_arm (Arm): The left arm instance.
        right_arm (Arm): The right arm instance.

    Returns:
        (int, int): (start_ms, end_ms) in milliseconds.
    """
    left_has_time = left_arm.start_time > 0 and left_arm.end_time > 0
    right_has_time = right_arm.start_time > 0 and right_arm.end_time > 0

    start_ms = 0
    end_ms = 0
    if left_has_time and not right_has_time:
        start_ms = left_arm.start_time
        end_ms = left_arm.end_time
    elif right_has_time and not left_has_time:
        start_ms = right_arm.start_time
        end_ms = right_arm.end_time
    elif left_has_time and right_has_time:
        start_ms = min(left_arm.start_time, right_arm.start_time)
        end_ms = max(left_arm.end_time, right_arm.end_time)
    return start_ms, end_ms


def end_inference_if_arm_up(
    left_arm: Arm,
    right_arm: Arm,
    data: list,
    last_time_ms: float,
    spoter_inferer: SPOTERONNXInferer,
    results_merged: list,
):
    """
    Nếu video kết thúc nhưng arms vẫn "up", compute final inference.

    Args:
        left_arm (Arm): The left arm instance.
        right_arm (Arm): The right arm instance.
        data (list): Collected mediapipe frames so far.
        last_time_ms (float): Last time (ms) của video.
        spoter_inferer (SPOTERONNXInferer): Model để infer.
        results_merged (list): List kết quả để append thêm.
    """
    if left_arm.is_up and left_arm.start_time > 0 and left_arm.end_time == 0:
        left_arm.end_time = last_time_ms

    if right_arm.is_up and right_arm.start_time > 0 and right_arm.end_time == 0:
        right_arm.end_time = last_time_ms

    start_ms, end_ms = get_one_arm_timestamp(left_arm, right_arm)
    if start_ms == 0 or end_ms == 0 or not data:
        return

    start_sec = start_ms / 1000
    end_sec = end_ms / 1000
    logging.info(f"Final sign from {start_sec:.2f}s to {end_sec:.2f}s (video ended).")

    t0 = time()
    pred_output = spoter_inferer.infer(data)
    t_infer = time() - t0

    final_result = {
        "predictions": pred_output,
        "inference_time": t_infer,
        "start_time": start_sec,
        "end_time": end_sec,
    }
    logging.info(f"Final Inference results: {final_result}")
    results_merged.append(final_result)


def run_inference(
    model_config: ModelConfig,
    inference_config: InferenceConfig,
    spoter_inferer: SPOTERONNXInferer,
    rotate: bool = False
):
    """
    Run core SPOTER ONNX inference trên 1 video hoàn chỉnh.
    Chỉ trả về dữ liệu JSON-friendly (no CSV, no video).

    Args:
        model_config (ModelConfig): Config cho model.
        inference_config (InferenceConfig): Config cho inference parameters.
        spoter_inferer (SPOTERONNXInferer): Inference object.
        rotate (bool): Nếu True, rotate frames 90° CW.

    Returns:
        dict: {
            "results_merged": [...],
            "message": "Inference completed successfully."
        }
    """
    config_logger(level=logging.INFO)
    logging.info("Starting SPOTER ONNX inference...")

    id2label = load_id2label(model_config.gloss_csv_path)
    logging.info(f"id2label loaded with {len(id2label)} entries.")

    source_path = (
        str(inference_config.source) if inference_config.source.is_file() else 0
    )
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video source: {source_path}")
        return {}

    # ---------- Tái sử dụng một instance Mediapipe Holistic cho toàn bộ video ----------
    mp_holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=0.8,
        min_tracking_confidence=0.5
    )

    right_arm = Arm("right", inference_config.visibility)
    left_arm = Arm("left", inference_config.visibility)

    data = []
    results_merged = []

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                logging.info("End of video or cannot read frame.")
                break

            if rotate:
                frame = rotate_frame(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_results = mp_holistic.process(frame_rgb)

            if detection_results.pose_landmarks:
                left_arm.set_pose(detection_results.pose_landmarks.landmark)
                right_arm.set_pose(detection_results.pose_landmarks.landmark)

            current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            left_up = ok_to_get_frame(
                arm=left_arm,
                angle_threshold=inference_config.angle_threshold,
                min_num_up_frames=inference_config.min_num_up_frames,
                min_num_down_frames=inference_config.min_num_down_frames,
                current_time=current_time_ms,
                delay=inference_config.delay,
            )
            right_up = ok_to_get_frame(
                arm=right_arm,
                angle_threshold=inference_config.angle_threshold,
                min_num_up_frames=inference_config.min_num_up_frames,
                min_num_down_frames=inference_config.min_num_down_frames,
                current_time=current_time_ms,
                delay=inference_config.delay,
            )

            if left_up or right_up:
                logging.info("At least one arm is up → capturing frame.")
                data.append(detection_results)

            start_ms, end_ms = get_one_arm_timestamp(left_arm, right_arm)
            start_sec = start_ms / 1000
            end_sec = end_ms / 1000

            # Nếu đã có start và end, chạy inference
            if start_sec != 0 and end_sec != 0:
                logging.info(f"Detected sign from {start_sec:.2f}s to {end_sec:.2f}s.")
                t0 = time()
                pred_output = spoter_inferer.infer(data)
                t_infer = time() - t0

                result = {
                    "predictions": pred_output,
                    "inference_time": t_infer,
                    "start_time": start_sec,
                    "end_time": end_sec,
                }
                logging.info(f"Inference results: {result}")
                results_merged.append(result)

                # Reset arms state & data
                left_arm.reset_state()
                right_arm.reset_state()
                data = []
    finally:
        cap.release()
        mp_holistic.close()

    # Nếu video kết thúc mà tay vẫn up
    last_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    end_inference_if_arm_up(
        left_arm,
        right_arm,
        data,
        last_time_ms,
        spoter_inferer,
        results_merged,
    )

    return {
        "results_merged": results_merged,
        "message": "Inference completed successfully.",
    }

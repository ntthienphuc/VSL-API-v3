from argparse import Namespace
from pathlib import Path
from dataclasses import dataclass

from simple_parsing import ArgumentParser

from utils.constants import MODELS, VIDEO_EXTENSIONS


@dataclass
class ModelConfig:
    """
    Configuration class for the SPOTER model.
    Contains information about model architecture, ONNX path, and gloss CSV path.
    """
    arch: str = "spoter"
    hidden_dim: int = 108
    onnx_path: str = "models/spoter_v3.0.onnx"
    gloss_csv_path: str = "gloss.csv"

    def __post_init__(self) -> None:
        """
        Validate the model configuration and ensure required files exist.
        """
        if self.arch not in MODELS:
            raise ValueError(f"Model '{self.arch}' is not supported.")
        if not Path(self.onnx_path).exists():
            raise FileNotFoundError(f"ONNX file not found: {self.onnx_path}")
        if not Path(self.gloss_csv_path).exists():
            raise FileNotFoundError(f"Gloss CSV file not found: {self.gloss_csv_path}")


@dataclass
class InferenceConfig:
    """
    Configuration class for the inference process.
    Contains information about the source video, device, threshold, etc.
    """
    source: str = "1.mp4"
    output_dir: str = "demo"
    use_onnx: bool = True
    device: str = "cpu"
    cache_dir: str = "models"

    visualize: bool = False
    show_skeleton: bool = False

    visibility: float = 0.5
    angle_threshold: int = 140
    min_num_up_frames: int = 10
    min_num_down_frames: int = 10
    delay: int = 400
    top_k: int = 3

    def __post_init__(self) -> None:
        """
        Validate that the source is 'webcam' or a valid video path.
        """
        self.source = Path(self.source)
        if not (
            str(self.source) == "webcam"
            or (self.source.exists() and str(self.source).endswith(VIDEO_EXTENSIONS))
        ):
            raise ValueError(
                f"Only 'webcam' or a supported video file is allowed (got {self.source})."
            )


def get_args() -> Namespace:
    """
    Retrieve command line arguments using simple_parsing.
    """
    parser = ArgumentParser(description="Offline Inference with SPOTER ONNX (no HF).")
    parser.add_arguments(ModelConfig, "model")
    parser.add_arguments(InferenceConfig, "inference")
    return parser.parse_args()

import numpy as np
from mediapipe.python.solutions import pose

def calculate_angle(a: tuple, b: tuple, c: tuple) -> float:
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = (
        np.arctan2(c[1] - b[1], c[0] - b[0])
        - np.arctan2(a[1] - b[1], a[0] - b[0])
    )
    angle = abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

class Arm:
    """
    Class representing a single arm (left or right) for pose detection.
    Provides methods to set pose, track angle, and manage up/down state.
    """

    def __init__(self, side: str, visibility: float = 0.5) -> None:
        self.side = side
        if side == "left":
            self.shoulde_idx = pose.PoseLandmark.LEFT_SHOULDER.value
            self.elbow_idx = pose.PoseLandmark.LEFT_ELBOW.value
            self.wrist_idx = pose.PoseLandmark.LEFT_WRIST.value
        elif side == "right":
            self.shoulde_idx = pose.PoseLandmark.RIGHT_SHOULDER.value
            self.elbow_idx = pose.PoseLandmark.RIGHT_ELBOW.value
            self.wrist_idx = pose.PoseLandmark.RIGHT_WRIST.value
        else:
            raise ValueError("Side must be either 'left' or 'right'.")

        self.visibility = visibility
        self.is_up = False
        self.num_up_frames = 0
        self.num_down_frames = 0
        self.start_time = 0
        self.end_time = 0
        self.shoulder = None
        self.elbow = None
        self.wrist = None
        self.angle = 0

    def reset_state(self) -> None:
        self.is_up = False
        self.num_up_frames = 0
        self.num_down_frames = 0
        self.start_time = 0
        self.end_time = 0
        self.shoulder = None
        self.elbow = None
        self.wrist = None
        self.angle = 0

    def set_pose(self, landmarks) -> bool:
        if not landmarks:
            self.angle = 180
            return False

        if (
            self.shoulde_idx >= len(landmarks)
            or self.elbow_idx >= len(landmarks)
            or self.wrist_idx >= len(landmarks)
        ):
            return False

        if (
            landmarks[self.shoulde_idx].visibility < self.visibility
            or landmarks[self.elbow_idx].visibility < self.visibility
            or landmarks[self.wrist_idx].visibility < self.visibility
        ):
            return False

        self.shoulder = (
            landmarks[self.shoulde_idx].x,
            landmarks[self.shoulde_idx].y,
        )
        self.elbow = (
            landmarks[self.elbow_idx].x,
            landmarks[self.elbow_idx].y,
        )
        self.wrist = (
            landmarks[self.wrist_idx].x,
            landmarks[self.wrist_idx].y,
        )
        self.angle = calculate_angle(self.shoulder, self.elbow, self.wrist)
        return True

    def to_dict(self):
        return {
            'is_up': bool(self.is_up),
            'num_up_frames': int(self.num_up_frames),
            'num_down_frames': int(self.num_down_frames),
            'start_time': int(self.start_time),
            'end_time': int(self.end_time),
            'shoulder': [float(x) for x in self.shoulder] if self.shoulder else None,
            'elbow': [float(x) for x in self.elbow] if self.elbow else None,
            'wrist': [float(x) for x in self.wrist] if self.wrist else None,
            'angle': float(self.angle),
        }

    @classmethod
    def from_dict(cls, data, side, visibility):
        arm = cls(side, visibility)
        arm.is_up = data['is_up']
        arm.num_up_frames = data['num_up_frames']
        arm.num_down_frames = data['num_down_frames']
        arm.start_time = data['start_time']
        arm.end_time = data['end_time']
        arm.shoulder = tuple(data['shoulder']) if data['shoulder'] else None
        arm.elbow = tuple(data['elbow']) if data['elbow'] else None
        arm.wrist = tuple(data['wrist']) if data['wrist'] else None
        arm.angle = data['angle']
        return arm

def ok_to_get_frame(
    arm: Arm,
    angle_threshold: int,
    min_num_up_frames: int,
    min_num_down_frames: int,
    current_time: int,
    delay: int,
) -> bool:
    if 0 < arm.angle < angle_threshold:
        # "Up" logic
        if arm.is_up:
            arm.num_down_frames = 0
            arm.end_time = 0
        else:
            if arm.num_up_frames == min_num_up_frames:
                arm.is_up = True
                arm.num_up_frames = 0
            else:
                if arm.num_up_frames == 0:
                    arm.start_time = current_time - delay
                arm.num_up_frames += 1
                return False
    else:
        # "Down" logic
        if arm.is_up:
            if arm.num_down_frames == min_num_down_frames:
                arm.is_up = False
                arm.num_down_frames = 0
            else:
                if arm.num_down_frames == 0:
                    arm.end_time = current_time + delay
                arm.num_down_frames += 1
                return True
        else:
            arm.num_up_frames = 0
            arm.start_time = 0

    return arm.is_up

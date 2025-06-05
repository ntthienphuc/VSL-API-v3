VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
MODELS = ("spoter",)

HAND_LANDMARKS = [
    "wrist",
    "indexTip",
    "indexDIP",
    "indexPIP",
    "indexMCP",
    "middleTip",
    "middleDIP",
    "middlePIP",
    "middleMCP",
    "ringTip",
    "ringDIP",
    "ringPIP",
    "ringMCP",
    "littleTip",
    "littleDIP",
    "littlePIP",
    "littleMCP",
    "thumbTip",
    "thumbIP",
    "thumbMP",
    "thumbCMC",
]

BODY_LANDMARKS = [
    "nose",
    "neck",
    "rightEye",
    "leftEye",
    "rightEar",
    "leftEar",
    "rightShoulder",
    "leftShoulder",
    "rightElbow",
    "leftElbow",
    "rightWrist",
    "leftWrist",
]

HANDS_LANDMARKS = [
    f"{id_}{suffix}"
    for id_ in HAND_LANDMARKS
    for suffix in ["_0", "_1"]
]

LANDMARKS = BODY_LANDMARKS + HANDS_LANDMARKS

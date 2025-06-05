from pathlib import Path
from typing import Union

# Chấp nhận các định dạng video thường gặp
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def save_video(
    label: str,
    data: Union[bytes, bytearray],
    ext: str,
    root_dir: Union[str, Path] = r"D:\VSL-Data",
) -> Path:
    """
    Lưu một video vào thư mục ``<root_dir>/<label>/<n>.<ext>`` (n = 1, 2, 3…).

    Parameters
    ----------
    label : str
        Từ/nhãn của clip (ví dụ: ``"Đi bộ"``).
    data : bytes | bytearray
        Nội dung video.
    ext : str
        Phần mở rộng gốc (``.mp4``, ``.mov`` …).
    root_dir : str | Path, default ``D:\\VSL-Data``
        Thư mục gốc chứa dataset.

    Returns
    -------
    Path
        Đường dẫn tuyệt đối tới file vừa lưu.
    """
    ext = ext.lower()
    if ext not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {ext}")

    label_dir = Path(root_dir) / label
    label_dir.mkdir(parents=True, exist_ok=True)

    # Xác định chỉ số kế tiếp (1.mp4, 2.mp4, ...)
    nums = [
        int(p.stem) for p in label_dir.iterdir()
        if p.is_file() and p.stem.isdigit()
    ]
    next_idx = max(nums) + 1 if nums else 1

    dst = label_dir / f"{next_idx}{ext}"
    dst.write_bytes(data)
    return dst

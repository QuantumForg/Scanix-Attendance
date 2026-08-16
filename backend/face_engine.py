import os
import cv2
import numpy as np
from PIL import Image

# Optional DeepFace
try:
    from deepface import DeepFace
    DEEPFACE_OK = True
except Exception as e:
    DEEPFACE_OK = False
    print("DeepFace not available:", e)

from backend.database import KNOWN_FACES_DIR, DATA_DIR


def save_student_photo(name, roll_no, image):
    """Save photo to known_faces folder and return path"""
    safe_name = "".join(c for c in name if c.isalnum() or c in " _-").strip()
    filename = f"{safe_name}_{roll_no}.jpg"
    photo_path = os.path.join(KNOWN_FACES_DIR, filename)

    if isinstance(image, np.ndarray):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
        Image.fromarray(img_rgb).save(photo_path)
    else:
        # PIL Image
        image.save(photo_path)

    return photo_path


def recognize_face(image_array):
    """
    Match face against known_faces folder using DeepFace.
    Returns (name, roll_no, confidence) or None
    """
    if not DEEPFACE_OK:
        print("DeepFace not ready")
        return None

    # Save temp query image
    temp_path = os.path.join(DATA_DIR, "temp_query.jpg")
    if isinstance(image_array, np.ndarray):
        img = Image.fromarray(image_array)
        img.save(temp_path)
    else:
        image_array.save(temp_path)

    try:
        dfs = DeepFace.find(
            img_path=temp_path,
            db_path=KNOWN_FACES_DIR,
            model_name="VGG-Face",
            distance_metric="cosine",
            enforce_detection=False,
            silent=True
        )

        if dfs and len(dfs) > 0 and not dfs[0].empty:
            best = dfs[0].iloc[0]
            identity = best.get("identity", "")
            distance = best.get("distance", 1.0)

            # Confidence = (1 - distance) * 100 for cosine
            confidence = max(0, min(100, (1 - distance) * 100))

            if confidence < 45:
                return None

            # Parse Name_Roll.jpg
            basename = os.path.basename(identity)
            name_part = basename.rsplit(".", 1)[0]
            if "_" in name_part:
                parts = name_part.rsplit("_", 1)
                name = parts[0].replace("_", " ")
                roll = parts[1]
            else:
                name = name_part
                roll = "Unknown"

            return name, roll, confidence

    except Exception as e:
        print("Recognition error:", e)

    return None


def capture_from_rtsp(rtsp_url, max_frames=5, timeout_sec=8):
    """
    Basic RTSP support.
    CCTV / IP camera se frames capture karta hai.
    Returns list of RGB frames or empty list on failure.
    """
    frames = []
    cap = None
    try:
        # OpenCV se RTSP open
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_sec * 1000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_sec * 1000)

        if not cap.isOpened():
            print("RTSP connection failed")
            return []

        count = 0
        while count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            count += 1

    except Exception as e:
        print("RTSP error:", e)
    finally:
        if cap is not None:
            cap.release()

    return frames

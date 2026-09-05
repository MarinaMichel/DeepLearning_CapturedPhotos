"""
Turns a video into a (num_frames, 1662) array of MediaPipe Holistic
keypoints per frame: pose (33*4) + face (468*3) + left hand (21*3) +
right hand (21*3) = 1662, matching the Encoder's input_dim in model.py.

Used both to pre-extract features for every training video (see
extract_all_features.py) and, at inference time, by the backend on the
video a user uploads.
"""
import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic


def _landmarks_to_array(landmarks, n_points, n_dims):
    if landmarks is None:
        return np.zeros(n_points * n_dims, dtype=np.float32)
    if n_dims == 4:
        vals = [[p.x, p.y, p.z, p.visibility] for p in landmarks.landmark]
    else:
        vals = [[p.x, p.y, p.z] for p in landmarks.landmark]
    return np.array(vals, dtype=np.float32).flatten()


def extract_keypoints(results) -> np.ndarray:
    pose = _landmarks_to_array(results.pose_landmarks, 33, 4)
    face = _landmarks_to_array(results.face_landmarks, 468, 3)
    lh = _landmarks_to_array(results.left_hand_landmarks, 21, 3)
    rh = _landmarks_to_array(results.right_hand_landmarks, 21, 3)
    return np.concatenate([pose, face, lh, rh])


def extract_features_from_video(video_path: str, max_frames: int = None) -> np.ndarray:
    """Returns an (T, 1662) float32 array, one row of keypoints per frame."""
    cap = cv2.VideoCapture(video_path)
    frame_features = []

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)

            frame_features.append(extract_keypoints(results))

            if max_frames and len(frame_features) >= max_frames:
                break

    cap.release()

    if len(frame_features) == 0:
        # No frames could be read (corrupt/empty video) -> return one
        # all-zero frame so downstream code doesn't crash on an empty array.
        return np.zeros((1, 1662), dtype=np.float32)

    return np.stack(frame_features).astype(np.float32)

"""
Frame to label
input: a facial expression image (e.g., from the video frames)
output: a label (e.g., emotion category)

Uses DeepFace to analyze facial expressions frame-by-frame.
The clip-level label is determined by majority vote across all frames.
"""

from collections import Counter
from deepface import DeepFace


# Mapping from DeepFace emotion labels to codebook categories
DEEPFACE_TO_CODEBOOK = {
    "happy": "positive",
    "surprise": "positive",
    "neutral": "neutral",
    "sad": "negative",
    "angry": "negative",
    "disgust": "negative",
    "fear": "negative",
}


def analyze_frame(frame_path: str) -> str | None:
    """
    Run DeepFace emotion analysis on a single frame image.

    Args:
        frame_path: Path to the image file.

    Returns:
        Dominant emotion string (e.g. 'happy'), or None if analysis fails.
    """
    try:
        results = DeepFace.analyze(
            img_path=frame_path,
            actions=["emotion"],
            enforce_detection=False,  # Don't crash if no face detected
            silent=True,
        )
        # results is a list; take the first face found
        if isinstance(results, list) and len(results) > 0:
            return results[0]["dominant_emotion"]
        return None
    except Exception as e:
        print(f"[DeepFace] Error analyzing {frame_path}: {e}")
        return None


def majority_vote(labels: list[str]) -> str | None:
    """
    Return the most common label from a list. Returns None if the list is empty.
    """
    filtered = [l for l in labels if l is not None]
    if not filtered:
        return None
    return Counter(filtered).most_common(1)[0][0]


def predict_clip_emotion(frame_paths: list[str]) -> dict:
    """
    Analyze all frames for a TCU clip and return a clip-level emotion label
    via majority vote.

    Args:
        frame_paths: Ordered list of file paths for the clip's extracted frames.

    Returns:
        dict with keys:
            auto_facial_mapped   – codebook-mapped label (positive/neutral/negative)
            auto_facial_raw      – raw DeepFace dominant_emotion label
            auto_facial_frame_count – number of frames successfully analyzed
    """
    raw_labels = [analyze_frame(fp) for fp in frame_paths]
    successful = [l for l in raw_labels if l is not None]

    raw_winner = majority_vote(successful)
    mapped = DEEPFACE_TO_CODEBOOK.get(raw_winner, "unknown") if raw_winner else "unknown"

    return {
        "auto_facial_mapped": mapped,
        "auto_facial_raw": raw_winner,
        "auto_facial_frame_count": len(successful),
    }
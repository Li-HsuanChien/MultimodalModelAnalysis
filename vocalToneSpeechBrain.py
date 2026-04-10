"""
wav to label
input: a speech audio file (e.g., from the video)
output: a label (e.g., emotion category)

Uses SpeechBrain's pre-trained wav2vec2 emotion recognition model
(speechbrain/emotion-recognition-wav2vec2-IEMOCAP).

Falls back to openSMILE + a simple classifier if SpeechBrain is unavailable
or too resource-intensive.
"""

import os
import numpy as np

# Mapping from SpeechBrain / IEMOCAP emotion labels to codebook categories
SPEECHBRAIN_TO_CODEBOOK = {
    "hap": "positive",   # happy
    "exc": "positive",   # excited
    "neu": "neutral",    # neutral
    "sad": "negative",   # sad
    "ang": "negative",   # angry
    "fru": "negative",   # frustrated
    "fea": "negative",   # fearful
    "dis": "negative",   # disgusted
}


# ---------------------------------------------------------------------------
# Primary backend: SpeechBrain
# ---------------------------------------------------------------------------

def _load_speechbrain_classifier():
    """Lazy-load the SpeechBrain classifier (downloads weights once)."""
    from speechbrain.pretrained import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        savedir="pretrained_models/emotion-recognition-wav2vec2-IEMOCAP",
        run_opts={"device": "cpu"},
    )


def predict_with_speechbrain(wav_path: str, classifier=None) -> dict:
    """
    Run SpeechBrain emotion recognition on a .wav file.

    Args:
        wav_path:   Path to the .wav audio clip.
        classifier: Optional pre-loaded SpeechBrain classifier (avoids
                    reloading weights on every call).

    Returns:
        dict with keys: raw_label, confidence, backend
    """
    import torch

    if classifier is None:
        classifier = _load_speechbrain_classifier()

    out_prob, score, index, text_lab = classifier.classify_file(wav_path)

    # text_lab is a list of string labels, e.g. ['ang']
    raw_label = text_lab[0] if text_lab else "neu"
    confidence = float(score.squeeze()) if score is not None else 0.0

    return {"raw_label": raw_label, "confidence": confidence, "backend": "speechbrain"}


# ---------------------------------------------------------------------------
# Fallback backend: openSMILE + heuristic classifier
# ---------------------------------------------------------------------------

def _extract_opensmile_features(wav_path: str) -> np.ndarray | None:
    """
    Extract eGeMAPSv02 features via openSMILE Python bindings.
    Returns a 1-D numpy array, or None on failure.
    """
    try:
        import opensmile
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        df = smile.process_file(wav_path)
        return df.values.flatten()
    except Exception as e:
        print(f"[openSMILE] Feature extraction failed for {wav_path}: {e}")
        return None


def _heuristic_classify(features: np.ndarray) -> tuple[str, float]:
    """
    Very simple heuristic using loudness (F0 mean) and speech rate
    as a proxy classifier when no trained model is available.

    In a real project, replace with a pickled sklearn model:
        import joblib; clf = joblib.load("opensmile_clf.pkl")
        label = clf.predict([features])[0]
    """
    # eGeMAPSv02 index 0 = F0semitoneFrom27.5Hz_sma3nz_amean
    # index 22 = loudness_sma3_amean (approximate; depends on feature version)
    try:
        loudness = float(features[22])
    except IndexError:
        loudness = 0.0

    if loudness > 0.6:
        return "ang", 0.5   # High energy → negative proxy
    elif loudness < 0.2:
        return "sad", 0.5
    else:
        return "neu", 0.5


def predict_with_opensmile(wav_path: str) -> dict:
    """
    Fallback emotion prediction using openSMILE features + heuristic.
    """
    features = _extract_opensmile_features(wav_path)
    if features is None:
        return {"raw_label": "neu", "confidence": 0.0, "backend": "opensmile_failed"}

    raw_label, confidence = _heuristic_classify(features)
    return {"raw_label": raw_label, "confidence": confidence, "backend": "opensmile"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_clip_tone(wav_path: str, classifier=None) -> dict:
    """
    Predict the emotional tone of a .wav clip. Tries SpeechBrain first;
    falls back to openSMILE on ImportError or runtime failure.

    Args:
        wav_path:   Path to the audio clip (.wav).
        classifier: Optional pre-loaded SpeechBrain classifier instance,
                    passed through to avoid reloading weights on each TCU.

    Returns:
        dict with keys:
            auto_tone_label      - raw model label (e.g. 'ang', 'hap')
            auto_tone_mapped     - codebook-mapped label (positive/neutral/negative)
            auto_tone_confidence - model confidence score [0, 1]
            auto_tone_backend    - which backend was used
    """
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    # Try SpeechBrain first
    try:
        result = predict_with_speechbrain(wav_path, classifier=classifier)
    except (ImportError, Exception) as e:
        print(f"[SpeechBrain] Unavailable or failed ({e}), falling back to openSMILE.")
        result = predict_with_opensmile(wav_path)

    raw_label = result["raw_label"]
    mapped = SPEECHBRAIN_TO_CODEBOOK.get(raw_label, "unknown")

    return {
        "auto_tone_label": raw_label,
        "auto_tone_mapped": mapped,
        "auto_tone_confidence": round(result["confidence"], 4),
        "auto_tone_backend": result["backend"],
    }
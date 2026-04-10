"""
text to label
input: a string of text (e.g., tcu_transcript)
output: a label (e.g., sentiment or stance)

Uses cardiffnlp/twitter-roberta-base-sentiment-latest via HuggingFace
Transformers to classify text sentiment, then maps to codebook stance
categories.

Note: The model was trained on Twitter data, not political speech.
The sentiment→stance mapping is approximate and serves as a baseline only.
Store results in auto_stance_* columns; never overwrite human annotations.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_TOKENS = 512  # RoBERTa hard limit


# Mapping from model output labels to codebook stance categories.
# This is intentionally approximate — see README section 3.3.
SENTIMENT_TO_STANCE = {
    "positive": "support",
    "negative": "oppose",
    "neutral": "neutral",
}


# ---------------------------------------------------------------------------
# Lazy-loaded pipeline (singleton to avoid reloading weights each call)
# ---------------------------------------------------------------------------

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=-1,          # CPU; set to 0 for GPU
            truncation=True,
            max_length=MAX_TOKENS,
            top_k=None,         # Return scores for all labels
        )
    return _pipeline


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_stance(tcu_transcript: str) -> dict:
    """
    Run sentiment / stance classification on a single TCU transcript.

    Args:
        tcu_transcript: Raw transcript string for one TCU.

    Returns:
        dict with keys:
            auto_stance_label      - raw model label (positive/negative/neutral)
            auto_stance_mapped     - codebook-mapped stance (support/oppose/neutral)
            auto_stance_confidence - confidence for the winning label [0, 1]
    """
    if not tcu_transcript or not tcu_transcript.strip():
        return {
            "auto_stance_label": "neutral",
            "auto_stance_mapped": "neutral",
            "auto_stance_confidence": 0.0,
        }

    clf = _get_pipeline()

    # pipeline returns a list of lists when top_k=None: [[{label, score}, ...]]
    results = clf(tcu_transcript.strip())
    scores = results[0] if isinstance(results[0], list) else results

    # Normalise label keys to lower-case
    scores = [{"label": s["label"].lower(), "score": s["score"]} for s in scores]

    # Pick the highest-scoring label
    best = max(scores, key=lambda x: x["score"])
    raw_label = best["label"]
    confidence = round(best["score"], 4)
    mapped = SENTIMENT_TO_STANCE.get(raw_label, "unknown")

    return {
        "auto_stance_label": raw_label,
        "auto_stance_mapped": mapped,
        "auto_stance_confidence": confidence,
    }
"""
Controller — multimodal TCU annotation pipeline

Source: SQLite database (TCU table from schema.sql), extended with three
processing-state columns that are added automatically on first run:

    tone_processed        BOOLEAN DEFAULT 0
    expression_processed  BOOLEAN DEFAULT 0
    text_processed        BOOLEAN DEFAULT 0

For each unprocessed TCU the pipeline runs only the modalities that have
not yet been completed, writes predictions to the DB immediately after each
TCU, and flips the corresponding flag — so a crash mid-run is safely
resumable with no repeated work.

Modality outputs written to the TCU table:
    auto_facial_mapped, auto_facial_raw, auto_facial_frame_count
    auto_tone_label, auto_tone_mapped, auto_tone_confidence, auto_tone_backend
    auto_stance_label, auto_stance_mapped, auto_stance_confidence

Human annotation columns (in the Annotation table) are never touched.
"""

import os
import glob
import sqlite3
import argparse
from tqdm import tqdm

from facialExpressionDeepFaceModel import predict_clip_emotion
from vocalToneSpeechBrain import predict_clip_tone
from bertTextModel import predict_stance


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

# Columns added to the TCU table (beyond the original schema.sql definition).
# Each tuple: (column_name, sqlite_type, default_value)
EXTRA_COLUMNS: list[tuple[str, str, str]] = [
    # Processing-state flags
    ("tone_processed",        "BOOLEAN", "0"),
    ("expression_processed",  "BOOLEAN", "0"),
    ("text_processed",        "BOOLEAN", "0"),
    # Facial expression outputs
    ("auto_facial_mapped",     "TEXT",    "NULL"),
    ("auto_facial_raw",        "TEXT",    "NULL"),
    ("auto_facial_frame_count","INTEGER", "NULL"),
    # Vocal tone outputs
    ("auto_tone_label",        "TEXT",    "NULL"),
    ("auto_tone_mapped",       "TEXT",    "NULL"),
    ("auto_tone_confidence",   "REAL",    "NULL"),
    ("auto_tone_backend",      "TEXT",    "NULL"),
    # Text / stance outputs
    ("auto_stance_label",      "TEXT",    "NULL"),
    ("auto_stance_mapped",     "TEXT",    "NULL"),
    ("auto_stance_confidence", "REAL",    "NULL"),
]


def ensure_extra_columns(conn: sqlite3.Connection) -> None:
    """
    Add any missing extra columns to the TCU table.
    Uses ALTER TABLE … ADD COLUMN which is a no-op-safe pattern (catches the
    duplicate-column error so re-runs are safe).
    """
    for col_name, col_type, default in EXTRA_COLUMNS:
        try:
            conn.execute(
                f'ALTER TABLE TCU ADD COLUMN "{col_name}" {col_type} DEFAULT {default}'
            )
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e).lower():
                pass  # Already exists — safe to continue
            else:
                raise
    conn.commit()


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

FRAME_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")


def get_frame_paths(frames_dir: str) -> list[str]:
    paths = []
    for ext in FRAME_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(frames_dir, ext)))
    return sorted(paths)


# ---------------------------------------------------------------------------
# Per-TCU processing
# ---------------------------------------------------------------------------

def process_expression(row: dict) -> dict:
    """Run DeepFace on the TCU's frames directory."""
    frames_dir = row.get("frames_dir") or ""
    if not frames_dir or not os.path.isdir(frames_dir):
        print(f"  [facial] Missing/invalid frames_dir for {row['TCUID']}")
        return {
            "auto_facial_mapped": "unknown",
            "auto_facial_raw": None,
            "auto_facial_frame_count": 0,
        }
    frame_paths = get_frame_paths(frames_dir)
    return predict_clip_emotion(frame_paths)


def process_tone(row: dict, sb_classifier=None) -> dict:
    """Run SpeechBrain (or openSMILE fallback) on the TCU's .wav file."""
    # Derive wav path from the TCU's audio_saved flag + TCUID convention.
    # Adjust the path pattern below to match your file layout.
    wav_path = row.get("wav_path") or os.path.join("audio", f"{row['TCUID']}.wav")
    if not os.path.isfile(wav_path):
        print(f"  [tone]   Missing wav for {row['TCUID']} (tried: {wav_path})")
        return {
            "auto_tone_label": "unknown",
            "auto_tone_mapped": "unknown",
            "auto_tone_confidence": 0.0,
            "auto_tone_backend": "skipped",
        }
    return predict_clip_tone(wav_path, classifier=sb_classifier)


def process_text(row: dict) -> dict:
    """Run RoBERTa stance classifier on the TCU transcript."""
    return predict_stance(row.get("tcu_transcript") or "")


# ---------------------------------------------------------------------------
# Per-TCU DB write-back
# ---------------------------------------------------------------------------

def write_expression(conn: sqlite3.Connection, tcuid: str, preds: dict) -> None:
    conn.execute(
        """UPDATE TCU SET
               auto_facial_mapped      = :auto_facial_mapped,
               auto_facial_raw         = :auto_facial_raw,
               auto_facial_frame_count = :auto_facial_frame_count,
               expression_processed    = 1
           WHERE TCUID = :tcuid""",
        {**preds, "tcuid": tcuid},
    )
    conn.commit()


def write_tone(conn: sqlite3.Connection, tcuid: str, preds: dict) -> None:
    conn.execute(
        """UPDATE TCU SET
               auto_tone_label      = :auto_tone_label,
               auto_tone_mapped     = :auto_tone_mapped,
               auto_tone_confidence = :auto_tone_confidence,
               auto_tone_backend    = :auto_tone_backend,
               tone_processed       = 1
           WHERE TCUID = :tcuid""",
        {**preds, "tcuid": tcuid},
    )
    conn.commit()


def write_text(conn: sqlite3.Connection, tcuid: str, preds: dict) -> None:
    conn.execute(
        """UPDATE TCU SET
               auto_stance_label      = :auto_stance_label,
               auto_stance_mapped     = :auto_stance_mapped,
               auto_stance_confidence = :auto_stance_confidence,
               text_processed         = 1
           WHERE TCUID = :tcuid""",
        {**preds, "tcuid": tcuid},
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    db_path: str,
    frames_base_dir: str = "frames",
    wav_base_dir: str = "audio",
) -> None:
    """
    Load unprocessed TCUs from the SQLite DB, run all three models per modality,
    and write results back immediately — resumable if interrupted.

    Args:
        db_path:         Path to the SQLite database file.
        frames_base_dir: Root directory for frame folders (used to resolve
                         frames_dir when not stored in the DB).
        wav_base_dir:    Root directory for .wav files.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row   # access columns by name

    print("Ensuring extra columns exist in TCU table …")
    ensure_extra_columns(conn)

    # Fetch all TCUs that still have at least one unprocessed modality
    rows = conn.execute(
        """SELECT
               TCUID,
               tcu_transcript,
               audio_saved,
               frames_saved,
               tone_processed,
               expression_processed,
               text_processed
           FROM TCU
           WHERE tone_processed = 0
              OR expression_processed = 0
              OR text_processed = 0
           ORDER BY TCUID"""
    ).fetchall()

    total = len(rows)
    print(f"{total} TCU(s) have at least one unprocessed modality.")

    if total == 0:
        print("Nothing to do.")
        conn.close()
        return

    # Pre-load SpeechBrain weights once — only if any tone work remains
    sb_classifier = None
    if any(not r["tone_processed"] for r in rows):
        try:
            from speechbrain.pretrained import EncoderClassifier
            print("Pre-loading SpeechBrain classifier …")
            sb_classifier = EncoderClassifier.from_hparams(
                source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                savedir="pretrained_models/emotion-recognition-wav2vec2-IEMOCAP",
                run_opts={"device": "cpu"},
            )
            print("  SpeechBrain ready.")
        except ImportError:
            print("  SpeechBrain not installed; will fall back to openSMILE.")

    for row in tqdm(rows, desc="Processing TCUs"):
        tcuid = row["TCUID"]
        row_dict = dict(row)

        # Resolve file paths from base dirs when not stored as DB columns
        row_dict.setdefault("frames_dir", os.path.join(frames_base_dir, tcuid))
        row_dict.setdefault("wav_path",   os.path.join(wav_base_dir, f"{tcuid}.wav"))

        # ── 1. Facial expression ──────────────────────────────────────────
        if not row["expression_processed"]:
            try:
                preds = process_expression(row_dict)
                write_expression(conn, tcuid, preds)
            except Exception as e:
                print(f"\n[ERROR] expression {tcuid}: {e}")

        # ── 2. Vocal tone ─────────────────────────────────────────────────
        if not row["tone_processed"]:
            try:
                preds = process_tone(row_dict, sb_classifier=sb_classifier)
                write_tone(conn, tcuid, preds)
            except Exception as e:
                print(f"\n[ERROR] tone {tcuid}: {e}")

        # ── 3. Text / stance ──────────────────────────────────────────────
        if not row["text_processed"]:
            try:
                preds = process_text(row_dict)
                write_text(conn, tcuid, preds)
            except Exception as e:
                print(f"\n[ERROR] text {tcuid}: {e}")

    conn.close()
    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multimodal automated annotation; source and sink are SQLite."
    )
    parser.add_argument("db_path", help="Path to the SQLite database file")
    parser.add_argument(
        "--frames-dir",
        default="frames",
        help="Root directory containing per-TCU frame folders (default: frames/)",
    )
    parser.add_argument(
        "--wav-dir",
        default="audio",
        help="Root directory containing per-TCU .wav files (default: audio/)",
    )
    args = parser.parse_args()

    run_pipeline(
        db_path=args.db_path,
        frames_base_dir=args.frames_dir,
        wav_base_dir=args.wav_dir,
    )
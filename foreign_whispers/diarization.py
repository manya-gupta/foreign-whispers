"""Speaker diarization using pyannote.audio.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M2-align).

Optional dependency: pyannote.audio
    pip install pyannote.audio
Requires accepting the pyannote/speaker-diarization-3.1 licence on HuggingFace
and providing an HF token.  Returns empty list with a warning if the dep is
absent or the token is missing.
"""
import copy
import logging

logger = logging.getLogger(__name__)


def diarize_audio(audio_path: str, hf_token: str | None = None) -> list[dict]:
    """Return speaker-labeled intervals for *audio_path*.

    Returns:
        List of ``{start_s: float, end_s: float, speaker: str}``.
        Empty list when pyannote.audio is absent, token is missing, or diarization fails.
    """
    if not hf_token:
        logger.warning("No HF token provided — diarization skipped.")
        return []

    try:
        from pyannote.audio import Pipeline
    except (ImportError, TypeError):
        logger.warning("pyannote.audio not installed — returning empty diarization.")
        return []

    try:
        pipeline    = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        diarization = pipeline(audio_path)
        return [
            {"start_s": turn.start, "end_s": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
    except Exception as exc:
        logger.warning("Diarization failed for %s: %s", audio_path, exc)
        return []

# Stub — copy this into foreign_whispers/diarization.py (at the bottom)

def assign_speakers(
    segments: list[dict],
    diarization: list[dict],
) -> list[dict]:
    """Assign a speaker label to each transcription segment.

    For each segment, finds the diarization interval with the greatest
    temporal overlap and copies its speaker label. If diarization is
    empty, all segments default to ``SPEAKER_00``.

    Args:
        segments: Whisper-style ``[{id, start, end, text, ...}]``.
        diarization: pyannote-style ``[{start_s, end_s, speaker}]``.

    Returns:
        New list of segment dicts, each with an added ``speaker`` key.
        Original list is not mutated.
    """
    
    new_dict = copy.deepcopy(segments)

    # iterate through list of segments and assign using my_dict["new_key"] = "new_value"
    # dicts are tuple key-value pairs. we are adding a speaker category for each item of the list
    # both times are in seconds

    for segment, new_item in zip(segments, new_dict):
        max_overlap = 0
        best_speaker = "SPEAKER_00"
        start_time = segment['start']
        end_time = segment['end']
        for diar in diarization:
            # diary takes up entire segment
            if start_time >= diar['start_s'] and end_time <= diar['end_s']:
                best_speaker = diar['speaker'] 
                break
            # diary starts in middle of segment or
            # segment starts in middle of diary
            elif start_time <= diar['start_s'] < end_time or diar['start_s'] <= start_time < diar['end_s']:
                overlap = min(end_time, diar['end_s']) - start_time
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = diar['speaker']           
            # diary is only part of the segment
            elif start_time <= diar['start_s'] and end_time >= diar['end_s']:
                overlap = diar['end_s'] - diar['start_s']
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = diar['speaker']
        new_item['speaker'] = best_speaker
    
    return new_dict

        

    
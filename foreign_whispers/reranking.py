"""Deterministic failure analysis and translation re-ranking stubs.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function is a **student assignment**
— see the docstring for inputs, outputs, and implementation guidance.
"""

import dataclasses
import logging
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


# Lookup tables for rule-based translation shortening
SPANISH_SHORTENING_LUT = {
    # Filler words & discourse markers
    "fillers": {
        "pues": "",
        "bueno": "",
        "bien": "",
        "vaya": "",
        "mira": "",
        "oye": "",
        "ya sabes": "",
        "verdad": "",
    },
    # Common phrases → shorter alternatives
    "phrases": {
        "en este momento": "ahora",
        "en la actualidad": "hoy",
        "por lo tanto": "luego",
        "sin embargo": "pero",
        "de todas formas": "igual",
        "a pesar de": "aunque",
        "con respecto a": "sobre",
        "en cuanto a": "sobre",
        "más o menos": "aprox.",
        "de alguna manera": "algo",
        "de cierta forma": "así",
        "por favor": "plis",
        "gracias de antemano": "gracias",
        "muchas gracias": "gracias",
        "muy importante": "importante",
        "donde estás": "ontas",
        "hasta luego": "talogo",
        "buenos días": "nosidas",
        "buenas noches": "nosnoches",
        "buenas tardes": "nostardes",
        "fin de semana": "finde",
    },
    # Auxiliary verbs that can sometimes be dropped
    "auxiliaries": {
        "estar": "",
        "ser": "",
        "haber": "",
    },
    # Shorter synonyms 
    "synonyms": {
        "pequeño": "chico",
        "rápido": "veloz",
        "lento": "tardo",
        "importante": "clave",
        "interesante": "curioso",
    },
    # Contractions 
    "contractions": {
        "de el": "del",
        "a el": "al",
    },
    # Apocopes (omission of sounds or letters at the end of a word)
    "apocopes": {
        "grande": "gran",
        "bueno": "buen",
        "malo": "mal",
        "primero": "primer",
        "tercero": "tercer",
        "universidad": "uni",
        "televisión": "tele",
        "fotografía": "foto",
        "profesor": "profe",
        "profesora": "profe",
        "absolutamente": "absolut",
        "abuelo": "abu",
        "abuela": "abu",
        "colegio": "cole",
        "instituto": "insti",
        "hospital": "hospi",
        "restaurante": "resto",
        "biblioteca": "biblio",
        "automóvil": "auto",
        "bicicleta": "bici",
        "computadora": "compu",
        "metropolitano": "metro",
        "facultad": "facu",
        "bolígrafo": "boli",
        "matemáticas": "mate",
        "motocicleta": "moto",
        "oficina": "ofi",
    },
    # Linguistic abbreviations    
    "abbreviations": {
        "autobus": "bus",
        "por si acaso": "porsiaca",
        "escuela preparatoria": "prepa",
        "información": "info",
        "aproximadamente": "aprox.",
        "administración": "admin.",
        "medicina": "med",
        "máximo": "máx",
        "mínimo": "mín",
    }
}


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.

    .. admonition:: Student Assignment — Duration-Aware Translation Re-ranking

       This function is intentionally a **stub that returns an empty list**.
       Your task is to implement a strategy that produces shorter
       target-language translations when the baseline translation is too long
       for the time budget.

       **Inputs**

       ============== ======== ==================================================
       Parameter      Type     Description
       ============== ======== ==================================================
       source_text    str      Original source-language segment text
       baseline_es    str      Baseline target-language translation (from argostranslate)
       target_duration_s float Time budget in seconds for this segment
       context_prev   str      Text of the preceding segment (for coherence)
       context_next   str      Text of the following segment (for coherence)
       ============== ======== ==================================================

       **Outputs**

       A list of ``TranslationCandidate`` objects, sorted shortest first.
       Each candidate has:

       - ``text``: the shortened target-language translation
       - ``char_count``: ``len(text)``
       - ``brevity_rationale``: short note on what was changed

       **Duration heuristic**: target-language TTS produces ~15 characters/second
       (or ~4.5 syllables/second for Romance languages).  So a 3-second budget
       ≈ 45 characters.

       **Approaches to consider** (pick one or combine):

       1. **Rule-based shortening** — strip filler words, use shorter synonyms
          from a lookup table, contract common phrases
          (e.g. "en este momento" → "ahora").
       2. **Multiple translation backends** — call argostranslate with
          paraphrased input, or use a second translation model, then pick
          the shortest output that preserves meaning.
       3. **LLM re-ranking** — use an LLM (e.g. via an API) to generate
          condensed alternatives.  This was the previous approach but adds
          latency, cost, and a runtime dependency.
       4. **Hybrid** — rule-based first, fall back to LLM only for segments
          that still exceed the budget.

       **Evaluation criteria**: the caller selects the candidate whose
       ``len(text) / 15.0`` is closest to ``target_duration_s``.

    Returns:
        Empty list (stub).  Implement to return ``TranslationCandidate`` items.
    """
    ### Hybrid approach ###
    ### use a Spanish LUT first on the translation and then call an LLM if needed ###

    logger.info(
        "get_shorter_translations called for %.1fs budget (%d chars baseline) — ",
        target_duration_s,
        len(baseline_es),
    )

    # check if text meets target duration
    benchmark = True if (len(baseline_es) / 15.0) <= target_duration_s else False

    lut = SPANISH_SHORTENING_LUT
    candidates = []

    if not benchmark:  
        for category, replacements in lut.items():
            for long_form, short_form in replacements.items():
                shortened_text = baseline_es.replace(long_form, short_form)
                # track this as a candidate
                candidates.append(TranslationCandidate(
                    text=shortened_text,
                    char_count=len(shortened_text),
                    brevity_rationale=f"Applied {category} replacement"
                ))
        
        # Sort by char_count (shortest first)
        candidates.sort(key=lambda c: c.char_count)          
        new_shortened_text = candidates[0].text if candidates else baseline_es
        benchmark = True if (len(new_shortened_text) / 15.0) <= target_duration_s else False

        if len(candidates) > 1 and not benchmark:
            cat_list = []
            for category, replacements in lut.items():
                for long_form, short_form in replacements.items():
                    new_shortened_text = new_shortened_text.replace(long_form, short_form)
                    cat_list.append(category)
            # add to beginning of list as shortest candidate
            candidates.insert(0, TranslationCandidate(
                text=new_shortened_text,
                char_count=len(new_shortened_text),
                brevity_rationale=f"Applied all potential replacements: {cat_list}"
            ))
        
        # try again with all abbreviations applied
        benchmark = True if (len(new_shortened_text) / 15.0) <= target_duration_s else False

        # if its still too long, invoke LLM
        # LLM info: https://huggingface.co/BSC-LT/salamandra-7b-instruct
        if not benchmark:
            
            logger.info("Invoking LLM for further condensation of translation.")
            
            try:
                from huggingface_hub import InferenceClient
            except (ImportError, TypeError):
                logger.warning("huggingface_hub not installed — skipping LLM re-ranking.")
                return candidates
            
            # load SALAMANDRA_TOKEN from .env
            try:
                load_dotenv()
            except Exception as exc:
                logger.warning("Failed to load .env file: %s", exc)

            api_token = os.getenv("SALAMANDRA_TOKEN")

            reranking_client = InferenceClient(api_key=api_token)
            response = reranking_client.text_generation(
                model="BSC-LT/salamandra-7b-instruct",
                inputs=(
                    f"Condense the following Spanish text to fit within approximately {int(target_duration_s * 15)} characters"
                    f"Preserve the original meaning as much as possible.\n\n"
                    f"Original text: {baseline_es}\n\n"
                    f"Context (previous segment): {context_prev}\n"
                    f"Context (next segment): {context_next}\n\n"
                    "Reply with only the shortened version:"
                )
            )
            shortened_text = response.generated_text.strip()
            candidates.append(TranslationCandidate(
                text=shortened_text,
                char_count=len(shortened_text),
                brevity_rationale="LLM-generated condensation"
            ))
            # sort again
            candidates.sort(key=lambda c: c.char_count)    
                
    return candidates 

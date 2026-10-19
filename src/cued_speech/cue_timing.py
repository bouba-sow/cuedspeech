"""Cued Speech keys and hand-transition timing.

Turns a phone alignment (e.g. from MFA) into Cued Speech keys and predicts:

- [M1, M2]: when the hand moves from the previous vowel position to the next;
- [D1, D2]: when the fingers change from the previous handshape to the next.

The timing models follow Attina (2005) and the rule-based models of Auto-CS
(B. Bigi, CNRS-LPL, https://auto-cuedspeech.org). This is an independent
implementation that reuses only the published rules and their parameters.

References:
    V. Attina (2005). La Langue française Parlée Complétée (LPC) : production
    et perception. PhD thesis, INPG Grenoble.
    B. Bigi (2023). https://hal.science/hal-04081282
"""

from __future__ import annotations

import bisect
import logging
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Position code of the rest position, away from the face.
NEUTRAL = "n"

# French Cued Speech (LPC): consonant -> handshape 1..8.
CONSONANT_SHAPES: Dict[str, int] = {
    "p": 1, "d": 1, "ʒ": 1,
    "k": 2, "v": 2, "z": 2, "c": 2,
    "s": 3, "ʁ": 3, "r": 3, "ʀ": 3,
    "b": 4, "n": 4, "ɥ": 4,
    "t": 5, "m": 5, "f": 5,
    "l": 6, "ʃ": 6, "ɲ": 6, "w": 6,
    "g": 7, "ɡ": 7, "ɟ": 7,
    "j": 8, "ŋ": 8, "ʎ": 8,
}

# French Cued Speech: vowel -> position code.
# s = side, b = cheekbone, m = mouth, c = chin, t = throat.
VOWEL_POSITIONS: Dict[str, str] = {
    "a": "s", "ɑ": "s", "o": "s", "œ": "s", "ə": "s",
    "ɛ̃": "b", "ø": "b",
    "i": "m", "ɔ̃": "m", "ɑ̃": "m",
    "u": "c", "ɛ": "c", "ɔ": "c",
    "œ̃": "t", "y": "t", "e": "t",
}

VOWEL_ONLY_SHAPE = 5          # handshape of a vowel without consonant (-V)
CONSONANT_ONLY_POSITION = "s"  # position of a consonant without vowel (C-)

# Multi-symbol phones that are cued as a sequence of consonants.
PHONE_SPLITS: Dict[str, Tuple[str, ...]] = {
    "ts": ("t", "s"), "tʃ": ("t", "ʃ"), "dʒ": ("d", "ʒ"), "mʲ": ("m",),
}

# Labels that are not speech sounds (silences, unknown words).
IGNORED_PHONES = {"", "sil", "sp", "spn", "<eps>", "<unk>"}


@dataclass
class Phone:
    label: str
    start: float
    end: float


@dataclass
class CueKey:
    """One Cued Speech key: a handshape at a position, over [a1, a3]."""

    consonant: Optional[str]
    vowel: Optional[str]
    a1: float
    a3: float
    rank: int = 1  # 1 = first key after a silence
    m1: float = 0.0
    m2: float = 0.0
    d1: float = 0.0
    d2: float = 0.0

    @property
    def shape(self) -> int:
        if self.consonant is None:
            return VOWEL_ONLY_SHAPE
        return CONSONANT_SHAPES[self.consonant]

    @property
    def position(self) -> str:
        if self.vowel is None:
            return CONSONANT_ONLY_POSITION
        return VOWEL_POSITIONS[self.vowel]

    @property
    def structure(self) -> str:
        if self.consonant and self.vowel:
            return "CV"
        return "C-" if self.consonant else "-V"

    @property
    def label(self) -> str:
        return (self.consonant or "") + (self.vowel or "")


def normalize_phone(label: str) -> str:
    return unicodedata.normalize("NFC", label.strip()).replace("ː", "")


def _expand(phones: Sequence[Phone]) -> List[Phone]:
    """Normalize labels and split affricates into their consonants."""
    out = []
    for ph in phones:
        label = normalize_phone(ph.label)
        parts = PHONE_SPLITS.get(label)
        if parts is None:
            out.append(Phone(label, ph.start, ph.end))
            continue
        step = (ph.end - ph.start) / len(parts)
        for i, part in enumerate(parts):
            out.append(Phone(part, ph.start + i * step, ph.start + (i + 1) * step))
    return out


def build_keys(phones: Sequence[Phone], pause_threshold: float = 0.25) -> List[List[CueKey]]:
    """Group phones into Cued Speech keys, split into speech segments.

    A consonant directly followed by a vowel forms a CV key; any other
    consonant is a C- key and any other vowel a -V key. A new segment starts
    after a silence or unknown word, or a gap longer than ``pause_threshold``.
    """
    segments: List[List[CueKey]] = []
    current: List[CueKey] = []
    pending: Optional[Phone] = None  # consonant waiting for a vowel
    last_end: Optional[float] = None

    def close_segment():
        nonlocal current
        if current:
            segments.append(current)
        current = []

    def flush_pending():
        nonlocal pending
        if pending is not None:
            current.append(CueKey(pending.label, None, pending.start, pending.end))
            pending = None

    for ph in _expand(phones):
        if ph.label in IGNORED_PHONES:
            flush_pending()
            continue
        is_cons = ph.label in CONSONANT_SHAPES
        is_vowel = ph.label in VOWEL_POSITIONS
        if not (is_cons or is_vowel):
            logger.warning(f"Phone '{ph.label}' has no Cued Speech mapping; skipped")
            continue
        if last_end is not None and ph.start - last_end > pause_threshold:
            flush_pending()
            close_segment()
        last_end = ph.end

        if is_vowel:
            if pending is not None:
                current.append(CueKey(pending.label, ph.label, pending.start, ph.end))
                pending = None
            else:
                current.append(CueKey(None, ph.label, ph.start, ph.end))
        else:
            flush_pending()
            pending = ph
    flush_pending()
    close_segment()

    for segment in segments:
        for rank, key in enumerate(segment, start=1):
            key.rank = rank
    return segments


# ---------------------------------------------------------------------------
# Transition timing models
# ---------------------------------------------------------------------------


class TransitionModel:
    """Base timing model: transitions happen at A1 (Bratakos et al., 1998)."""

    STATIC_DURATION = 0.3  # default key duration, from the CLeLfPC corpus

    def __init__(self):
        self._durations: List[float] = []

    def observe(self, key: CueKey) -> None:
        self._durations.append(key.a3 - key.a1)

    def avg_duration(self) -> float:
        """Running average of observed key durations (static value until enough keys)."""
        n = len(self._durations)
        if n < 2:
            return self.STATIC_DURATION
        if n < 3:
            return (sum(self._durations) + self.STATIC_DURATION) / (n + 1)
        return sum(self._durations) / n

    def position(self, key: CueKey) -> Tuple[float, float]:
        return key.a1, key.a1

    def shape(self, key: CueKey) -> Tuple[float, float]:
        return key.a1, key.a1

    def position_to_neutral(self, a1: float) -> Tuple[float, float]:
        """[M1, M2] of the move to the rest position after a segment ending at ``a1``."""
        d = self.STATIC_DURATION
        m1 = a1 + d * 0.2
        return m1, m1 + d * 0.8


class AttinaModel(TransitionModel):
    """Proportional timings from Attina (2005), p. 117 and p. 136."""

    STATIC_DURATION = 0.4

    def position(self, key):
        d = self.avg_duration()
        if key.consonant is None:  # -V
            return max(0.0, key.a1 - d * 0.46), key.a1 + d * 0.21
        return max(0.0, key.a1 - d * 0.62), key.a1 + d * 0.10

    def shape(self, key):
        d = self.avg_duration()
        span = d * (0.36 if key.consonant is None else 0.54)
        return max(0.0, key.a1 - span), key.a1


class RulesModel(TransitionModel):
    """Auto-CS "custom rules": anticipation shrinks with the rank in the segment."""

    def position(self, key):
        d = self.avg_duration()
        if key.rank == 1:
            m1a1, m2a1 = max(0.500, d * 1.40), min(0.150, d * 0.20)
        elif key.rank == 2:
            m1a1, m2a1 = max(0.350, d * 1.15), min(0.150, d * 0.15)
        elif key.rank == 3:
            m1a1, m2a1 = max(0.250, d), min(0.100, d * 0.10)
        else:
            m1a1, m2a1 = d * 0.9, d * 0.05
        m1 = max(0.0, key.a1 - m1a1)
        m2 = max(m1, key.a1 - m2a1)
        m2 = min(m2, key.a3)
        return min(m1, m2), m2

    def shape(self, key):
        d = self.avg_duration()
        if key.rank == 1:
            d1a1, d2a1 = max(0.450, d * 1.1), d * 0.8
        elif key.rank == 2:
            d1a1, d2a1 = max(0.300, d * 0.95), d * 0.65
        elif key.rank == 3:
            d1a1, d2a1 = max(0.200, d * 0.8), d * 0.5
        else:
            d1a1, d2a1 = d * 0.7, d * 0.4
        d1 = max(0.0, key.a1 - d1a1)
        return d1, max(d1, key.a1 - d2a1)


class RevisedRulesModel(TransitionModel):
    """Auto-CS "revised rules", fitted on the CLeLfPC corpus; depends on key structure."""

    def position(self, key):
        d = self.avg_duration()
        if key.rank == 1:  # from the rest position: strongly anticipated
            m1 = max(0.0, key.a1 - max(0.600, d * 1.60))
            return m1, max(m1, key.a1 - min(0.150, d * 0.10))
        if key.structure == "C-":
            m1a1, a1m2 = max(0.500, d * 1.60), d * -0.30
        elif key.structure == "-V":
            m1a1, a1m2 = max(0.600, d * 2.40), d * -0.60
        else:
            m1a1 = max(0.350, d * 0.80)
            a1m2 = 0.0 if key.rank == 2 else d * 0.11
        m1 = max(0.0, key.a1 - m1a1)
        m2 = min(max(m1, key.a1 + a1m2), key.a3)
        return min(m1, m2), m2

    def shape(self, key):
        d = self.avg_duration()
        if key.rank == 1:
            d1a1, a1d2 = d * 1.30, d * -0.90
        elif key.structure == "C-":
            d1a1, a1d2 = d * 1.15, d * -0.55
        elif key.structure == "-V":
            d1a1, a1d2 = max(0.400, d * 1.80), d * -1.20
        elif key.rank == 2:
            d1a1, a1d2 = d * 0.8, d * -0.6
        else:
            d1a1, a1d2 = d * 0.65, d * -0.4
        d1 = max(0.0, key.a1 - d1a1)
        return d1, max(d1, key.a1 + a1d2)

    def position_to_neutral(self, a1):
        d = self.STATIC_DURATION
        m1 = max(0.0, a1 - d * 0.1)
        return m1, max(m1, a1 + d * 1.25)


TIMING_MODELS: Dict[str, Callable[[], TransitionModel]] = {
    "base": TransitionModel,
    "attina": AttinaModel,
    "rules": RulesModel,
    "revised": RevisedRulesModel,
}


def predict_transitions(segments: List[List[CueKey]], model: str = "rules"):
    """Fill M1/M2/D1/D2 of every key and return the position and shape transitions.

    Returns ``(position_transitions, shape_transitions)``, lists of
    ``(start, end, target)``. Position targets are position codes (including
    NEUTRAL at the end of each segment); shape targets are handshape numbers.
    """
    if model not in TIMING_MODELS:
        raise ValueError(f"Unknown timing model '{model}'. Choose from {sorted(TIMING_MODELS)}")
    predictor = TIMING_MODELS[model]()
    positions: List[Tuple[float, float, Hashable]] = []
    shapes: List[Tuple[float, float, Hashable]] = []
    for segment in segments:
        for key in segment:
            predictor.observe(key)
            key.m1, key.m2 = predictor.position(key)
            key.d1, key.d2 = predictor.shape(key)
            positions.append((key.m1, key.m2, key.position))
            shapes.append((key.d1, key.d2, key.shape))
        m1, m2 = predictor.position_to_neutral(segment[-1].a3)
        positions.append((m1, m2, NEUTRAL))
    return positions, shapes


# ---------------------------------------------------------------------------
# Interpolation between targets
# ---------------------------------------------------------------------------


def _blend(state: Dict[Hashable, float], target: Hashable, p: float) -> Dict[Hashable, float]:
    out = {k: w * (1.0 - p) for k, w in state.items()}
    out[target] = out.get(target, 0.0) + p
    return {k: w for k, w in out.items() if abs(w) > 1e-4}


class Track:
    """Piecewise interpolation between discrete targets over time.

    The state at time t is a set of weights over targets (summing to 1). A
    transition that starts before the previous one has finished starts from
    wherever the previous one had got to, so overlapping predictions never
    make the hand jump.
    """

    def __init__(self, initial: Hashable, transitions: Sequence[Tuple[float, float, Hashable]]):
        self.initial = initial
        self.transitions: List[Tuple[float, float, Hashable]] = []
        last_start = float("-inf")
        for start, end, target in sorted(transitions, key=lambda tr: tr[0]):
            start = max(start, last_start)
            self.transitions.append((start, max(end, start), target))
            last_start = start
        self._starts = [tr[0] for tr in self.transitions]
        self._from: List[Dict[Hashable, float]] = []
        state = {initial: 1.0}
        for k, (start, _, _) in enumerate(self.transitions):
            if k > 0:
                prev = self.transitions[k - 1]
                state = _blend(self._from[k - 1], prev[2], self._progress(prev, start))
            self._from.append(state)

    def _progress(self, transition, t: float) -> float:
        start, end, _ = transition
        if end <= start:
            return 1.0 if t >= start else 0.0
        return min(1.0, max(0.0, (t - start) / (end - start)))

    def _index(self, t: float) -> int:
        return bisect.bisect_right(self._starts, t) - 1

    def weights(self, t: float) -> Dict[Hashable, float]:
        k = self._index(t)
        if k < 0:
            return {self.initial: 1.0}
        return _blend(self._from[k], self.transitions[k][2], self._progress(self.transitions[k], t))

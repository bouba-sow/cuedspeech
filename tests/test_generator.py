"""Tests for the cued speech generator: keys, timing, placement and hand rendering."""

import numpy as np
import pytest

from cued_speech.cue_placement import FaceGeometry
from cued_speech.cue_timing import (
    NEUTRAL,
    Phone,
    Track,
    build_keys,
    predict_transitions,
)
from cued_speech.generator import CuedSpeechGenerator
from cued_speech.hand_assets import HandShape, HandShapeSet


def phones(*items, start=0.0, dur=0.1):
    out, t = [], start
    for label in items:
        out.append(Phone(label, t, t + dur))
        t += dur
    return out


class TestKeys:
    def test_cv_c_and_v_keys(self):
        # "merci" -> m ɛ ʁ s i : mɛ (CV), ʁ (C-), si (CV)
        (segment,) = build_keys(phones("m", "ɛ", "ʁ", "s", "i"))
        assert [k.label for k in segment] == ["mɛ", "ʁ", "si"]
        assert [k.structure for k in segment] == ["CV", "C-", "CV"]
        assert [(k.shape, k.position) for k in segment] == [(5, "c"), (3, "s"), (3, "m")]
        assert [k.rank for k in segment] == [1, 2, 3]

    def test_vowel_alone_uses_shape_5(self):
        (segment,) = build_keys(phones("a"))
        assert (segment[0].shape, segment[0].position) == (5, "s")

    def test_gn_is_shape_6_and_g_is_shape_7(self):
        (segment,) = build_keys(phones("ɲ", "a", "ɡ", "o"))
        assert [k.shape for k in segment] == [6, 7]

    def test_nasal_vowels_and_affricates(self):
        (segment,) = build_keys(phones("b", "ɔ̃", "tʃ", "a"))
        assert [k.label for k in segment] == ["bɔ̃", "t", "ʃa"]
        assert segment[0].position == "m"

    def test_unknown_words_are_not_cued(self):
        (segment,) = build_keys(phones("spn", "a"))
        assert [k.label for k in segment] == ["a"]

    def test_long_pause_splits_segments(self):
        segments = build_keys(phones("p", "a") + phones("t", "u", start=1.0))
        assert [[k.label for k in s] for s in segments] == [["pa"], ["tu"]]
        assert [s[0].rank for s in segments] == [1, 1]


class TestTiming:
    @pytest.mark.parametrize("model", ["base", "attina", "rules", "revised"])
    def test_transitions_are_ordered(self, model):
        segments = build_keys(phones("m", "ɛ", "ʁ", "s", "i", "a", "t", "u"))
        positions, shapes = predict_transitions(segments, model)
        assert positions[-1][2] == NEUTRAL
        for m1, m2, _ in positions + shapes:
            assert 0.0 <= m1 <= m2

    def test_hand_moves_before_the_sound(self):
        (segment,) = build_keys(phones("m", "ɛ", start=1.0))
        predict_transitions([segment], "rules")
        key = segment[0]
        assert key.m1 < key.a1 and key.m2 <= key.a1
        assert key.d1 < key.a1

    def test_unknown_model(self):
        with pytest.raises(ValueError):
            predict_transitions(build_keys(phones("a")), "nope")


class TestTrack:
    def test_interpolates_between_targets(self):
        track = Track("n", [(1.0, 2.0, "m"), (3.0, 4.0, "s")])
        assert track.weights(0.5) == {"n": 1.0}
        assert track.weights(1.5) == pytest.approx({"n": 0.5, "m": 0.5})
        assert track.weights(2.5) == {"m": 1.0}
        assert track.weights(5.0) == {"s": 1.0}

    def test_overlapping_transition_starts_where_the_hand_is(self):
        track = Track("n", [(0.0, 1.0, "m"), (0.5, 1.5, "s")])
        w = track.weights(0.5)
        assert w == pytest.approx({"n": 0.5, "m": 0.5})
        assert sum(track.weights(1.0).values()) == pytest.approx(1.0)
        assert track.weights(2.0) == pytest.approx({"s": 1.0})

    def test_moves_at_constant_speed(self):
        track = Track("a", [(0.0, 1.0, "b")])
        values = [track.weights(t / 10).get("b", 0.0) for t in range(11)]
        assert values == pytest.approx([t / 10 for t in range(11)])


class TestPlacement:
    def face(self):
        # An upright face, 200 px from brows to chin.
        pts = {
            0: (300, 250), 1: (302, 280), 2: (306, 310), 3: (312, 340), 4: (322, 368),
            6: (355, 410), 7: (378, 425), 8: (400, 430), 19: (350, 220), 24: (450, 220),
            27: (400, 240), 33: (400, 330), 36: (335, 260), 48: (370, 380), 57: (400, 395),
            60: (375, 380),
        }
        return FaceGeometry({k: np.array(v, float) for k, v in pts.items()}, 200.0)

    def test_positions_layout(self):
        pos = self.face().vowel_positions()
        assert pos["s"][0] < pos["m"][0]                  # side is outside the face
        assert pos["t"][1] > pos["c"][1] > pos["m"][1]    # throat below chin below mouth
        assert pos["b"][1] < pos["m"][1]                  # cheekbone above mouth
        assert pos["n"][1] > pos["t"][1]                  # rest is lowest

    def test_upright_face_angle(self):
        assert self.face().angle == pytest.approx(90.0)


class TestHandComposite:
    def hand_set(self):
        img = np.zeros((100, 40, 4), np.uint8)
        img[10:95, 10:30] = (0, 200, 255, 255)
        shape = HandShape(img, target=(20, 10), wrist=(20, 95))
        return HandShapeSet({n: shape for n in range(1, 9)})

    def test_fingertip_lands_on_target(self):
        hands = self.hand_set()
        frame = np.zeros((300, 300, 3), np.uint8)
        hands.composite(frame, {1: 1.0}, (150, 50), angle=90.0, scale=1.0)
        ys, xs = np.nonzero(frame[:, :, 2])
        assert ys.min() == pytest.approx(50, abs=1)
        assert xs.mean() == pytest.approx(150, abs=1)

    def test_rotation_tilts_towards_the_face(self):
        hands = self.hand_set()
        frame = np.zeros((300, 300, 3), np.uint8)
        hands.composite(frame, {1: 1.0}, (150, 50), angle=60.0, scale=1.0)
        ys, xs = np.nonzero(frame[:, :, 2])
        # Fingers lean right, so the wrist ends up left of the fingertip.
        assert xs[ys.argmax()] < 150

    def test_hand_outside_frame_is_skipped(self):
        frame = np.zeros((50, 50, 3), np.uint8)
        self.hand_set().composite(frame, {1: 1.0}, (-500, -500), angle=90.0, scale=1.0)
        assert not frame.any()


def test_unknown_config_key_is_ignored():
    gen = CuedSpeechGenerator({"not_a_key": 1, "hand_scale": 1.2})
    assert "not_a_key" not in gen.config
    assert gen.config["hand_scale"] == 1.2

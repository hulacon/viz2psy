"""Tests for viz2psy.viz.merge — multi-CSV feature merging for the dashboard."""

import numpy as np
import pandas as pd
import pytest

from viz2psy.viz.merge import (
    collect_feature_csvs,
    load_and_merge,
    merge_feature_frames,
)


def _static(model, n=4):
    return pd.DataFrame({
        "stimulus_id": [f"stim{i:02d}" for i in range(n)],
        "filename": [f"stim{i:02d}.png" for i in range(n)],
        f"{model}_score": np.arange(n, dtype=float) + hash(model) % 7,
    })


def _timed(model, times, stim="film-a"):
    return pd.DataFrame({
        "stimulus_id": [stim] * len(times),
        "time": list(times),
        f"{model}_score": np.arange(len(times), dtype=float),
    })


class TestStaticMerge:
    def test_merges_on_stimulus_id_and_dedupes_carriers(self):
        merged = merge_feature_frames(
            {"a.csv": _static("alpha"), "b.csv": _static("beta")})
        assert len(merged) == 4
        assert list(merged.columns).count("filename") == 1
        assert {"alpha_score", "beta_score"} <= set(merged.columns)

    def test_outer_merge_keeps_unmatched_rows(self):
        merged = merge_feature_frames(
            {"a.csv": _static("alpha", n=3), "b.csv": _static("beta", n=5)})
        assert len(merged) == 5
        assert merged["alpha_score"].isna().sum() == 2

    def test_duplicate_stimulus_id_is_an_error(self):
        dup = _static("alpha")
        dup.loc[1, "stimulus_id"] = dup.loc[0, "stimulus_id"]
        with pytest.raises(ValueError, match="duplicate stimulus_id"):
            merge_feature_frames({"a.csv": dup, "b.csv": _static("beta")})


class TestGuards:
    def test_colliding_columns_get_filename_suffixes(self):
        # Same model run on two sources: both copies survive, suffixed by
        # what distinguishes the filenames.
        merged = merge_feature_frames(
            {"caption_alpha_chunks.csv": _static("alpha"),
             "humancap_alpha_chunks.csv": _static("alpha")})
        assert "alpha_score__caption" in merged.columns
        assert "alpha_score__humancap" in merged.columns

    def test_indistinguishable_collision_is_an_error(self):
        with pytest.raises(ValueError, match="even after"):
            merge_feature_frames(
                {"x/alpha.csv": _static("alpha"),
                 "y/alpha.csv": _static("alpha")})

    def test_mixed_timed_and_static_is_an_error(self):
        with pytest.raises(ValueError, match="time-indexed and static"):
            merge_feature_frames(
                {"a.csv": _static("alpha"),
                 "b.csv": _timed("beta", [0.0, 0.5])})

    def test_missing_stimulus_id_is_an_error(self):
        with pytest.raises(ValueError, match="stimulus_id"):
            merge_feature_frames(
                {"a.csv": pd.DataFrame({"x": [1]})})


class TestTimedMerge:
    def test_identical_grids_merge_exactly(self):
        times = [0.0, 0.5, 1.0, 1.5]
        merged = merge_feature_frames(
            {"a.csv": _timed("alpha", times), "b.csv": _timed("beta", times)})
        assert len(merged) == 4
        assert merged["beta_score"].notna().all()

    def test_offset_grid_merges_to_nearest_spine_row(self):
        # Video frames at 0.0+0.5k, audio frames centered at 0.25+0.5k —
        # the real layout of viz vs. audio features for the same film.
        video = _timed("video", [0.0, 0.5, 1.0, 1.5, 2.0])
        audio = _timed("audio", [0.25, 0.75, 1.25, 1.75])
        merged = merge_feature_frames({"v.csv": video, "a.csv": audio})
        assert len(merged) == 5
        # Every spine row adopts its nearest audio sample (an audio row may
        # serve two adjacent video frames — no NaN gaps on an offset grid).
        assert merged["audio_score"].notna().all()
        assert merged.loc[merged["time"] == 0.0, "audio_score"].iloc[0] == 0.0
        assert merged.loc[merged["time"] == 2.0, "audio_score"].iloc[0] == 3.0

    def test_multiple_stimuli_align_within_stimulus(self):
        a = pd.concat([_timed("alpha", [0.0, 0.5], stim="f1"),
                       _timed("alpha", [0.0, 0.5], stim="f2")])
        b = pd.concat([_timed("beta", [0.0, 0.5], stim="f1"),
                       _timed("beta", [0.0, 0.5], stim="f2")])
        merged = merge_feature_frames({"a.csv": a, "b.csv": b})
        assert len(merged) == 4
        assert merged["beta_score"].notna().all()

    def test_far_off_rows_stay_nan(self):
        video = _timed("video", [0.0, 0.5, 1.0])
        sparse = _timed("sparse", [10.0])
        merged = merge_feature_frames({"v.csv": video, "s.csv": sparse})
        assert merged["sparse_score"].isna().all()


class TestFilesystemEntry:
    def test_collect_excludes_word_scaffolds(self, tmp_path):
        for name in ("clip.csv", "emonet.csv", "caption_clip_words.csv",
                     "transcribe_transcript.csv"):
            (tmp_path / name).write_text("stimulus_id\n")
        found = [p.name for p in collect_feature_csvs(tmp_path)]
        assert found == ["clip.csv", "emonet.csv"]

    def test_collect_empty_dir_errors_with_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=str(tmp_path)):
            collect_feature_csvs(tmp_path)

    def test_load_and_merge_expands_directories(self, tmp_path):
        _static("alpha").to_csv(tmp_path / "alpha.csv", index=False)
        _static("beta").to_csv(tmp_path / "beta.csv", index=False)
        merged = load_and_merge([tmp_path])
        assert {"alpha_score", "beta_score"} <= set(merged.columns)
        assert len(merged) == 4

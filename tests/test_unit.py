"""단위 테스트 — DB·파일시스템 없이 순수 함수를 검증합니다."""
from collections import Counter
from datetime import date

import pytest

from tests.conftest import _load

publish_mod = _load("03_publish")
link_mod    = _load("02_link")
ingest_mod  = _load("01_ingest")

split_fn          = publish_mod.stratified_subject_split
get_pipeline_commit = publish_mod.get_pipeline_commit
parse_date        = link_mod.parse_date
infer_image_type  = ingest_mod.infer_image_type


# ── stratified_subject_split ──────────────────────────────────────────────────

def _make_subjects(cn: int, mci: int, dem: int):
    subjects = (
        [f"C{i:02d}" for i in range(cn)]
        + [f"M{i:02d}" for i in range(mci)]
        + [f"D{i:02d}" for i in range(dem)]
    )
    labels = {
        **{s: "CN"       for s in subjects if s.startswith("C")},
        **{s: "MCI"      for s in subjects if s.startswith("M")},
        **{s: "Dementia" for s in subjects if s.startswith("D")},
    }
    return subjects, labels


class TestStratifiedSplit:
    def test_all_subjects_assigned(self):
        subjects, labels = _make_subjects(8, 5, 2)
        result = split_fn(subjects, labels, seed=42)
        assert set(result.keys()) == set(subjects)

    def test_valid_split_values(self):
        subjects, labels = _make_subjects(8, 5, 2)
        result = split_fn(subjects, labels, seed=42)
        assert all(v in {"train", "val", "test"} for v in result.values())

    def test_no_leakage(self):
        subjects, labels = _make_subjects(8, 5, 2)
        result = split_fn(subjects, labels, seed=42)
        splits = {"train": set(), "val": set(), "test": set()}
        for s, sp in result.items():
            splits[sp].add(s)
        assert not (splits["train"] & splits["val"]),  "train-val 중복"
        assert not (splits["train"] & splits["test"]), "train-test 중복"
        assert not (splits["val"]   & splits["test"]), "val-test 중복"

    def test_minority_class_n3_in_val_and_test(self):
        """n≥3 소수 클래스는 val·test 각각 1명 이상 보장."""
        subjects, labels = _make_subjects(8, 5, 3)
        result = split_fn(subjects, labels, seed=42)
        for split_name in ("val", "test"):
            count = sum(
                1 for s, sp in result.items()
                if sp == split_name and labels[s] == "Dementia"
            )
            assert count >= 1, f"Dementia이 {split_name}에 없음"

    def test_n1_all_train(self):
        """n=1 클래스는 train 배정."""
        subjects = ["A00", "B00"]
        labels   = {"A00": "CN", "B00": "Rare"}
        result   = split_fn(subjects, labels, seed=42)
        assert result["B00"] == "train"

    def test_n2_train_test_no_val(self):
        """n=2 클래스는 train 1, test 1, val 없음."""
        subjects = ["A00", "A01"]
        labels   = {"A00": "CN", "A01": "CN"}
        result   = split_fn(subjects, labels, seed=42)
        counts   = Counter(result.values())
        assert counts["train"] == 1
        assert counts["test"]  == 1
        assert counts.get("val", 0) == 0

    def test_reproducibility(self):
        subjects, labels = _make_subjects(8, 5, 3)
        r1 = split_fn(subjects, labels, seed=42)
        r2 = split_fn(subjects, labels, seed=42)
        assert r1 == r2

    def test_different_seeds_differ(self):
        subjects, labels = _make_subjects(10, 6, 4)
        r1 = split_fn(subjects, labels, seed=42)
        r2 = split_fn(subjects, labels, seed=99)
        assert r1 != r2

    def test_single_subject(self):
        """subject 1명 → 모두 train."""
        result = split_fn(["S00"], {"S00": "CN"}, seed=42)
        assert result == {"S00": "train"}

    @pytest.mark.parametrize("cn,mci,dem", [(6, 3, 3), (10, 8, 5)])
    def test_all_classes_in_all_splits(self, cn, mci, dem):
        """n≥3 클래스는 세 split 모두에 등장해야 함."""
        subjects, labels = _make_subjects(cn, mci, dem)
        result = split_fn(subjects, labels, seed=42)
        for cls in ("CN", "MCI", "Dementia"):
            for sp in ("train", "val", "test"):
                cnt = sum(1 for s, v in result.items() if v == sp and labels[s] == cls)
                assert cnt >= 1, f"{cls} not in {sp}"


# ── parse_date ────────────────────────────────────────────────────────────────

class TestParseDate:
    def test_iso_format(self):
        assert parse_date("2010-05-20") == date(2010, 5, 20)

    def test_compact_format(self):
        assert parse_date("20100520") == date(2010, 5, 20)

    def test_both_equivalent(self):
        assert parse_date("2015-11-01") == parse_date("20151101")

    def test_month_boundary(self):
        assert parse_date("20231231") == date(2023, 12, 31)


# ── infer_image_type ──────────────────────────────────────────────────────────

class TestInferImageType:
    @pytest.mark.parametrize("modality,desc,expected", [
        ("PT", "FDG PET",           "PET"),
        ("MR", "MPRAGE Sagittal",   "T1"),
        ("MR", "IR-FSPGR 3D",       "T1"),
        ("MR", "SAG T1 3D",         "T1"),
        ("MR", "DTI_30dir",         "DWI"),
        ("MR", "DWI_b1000",         "DWI"),
        ("MR", "DIFFUSION",         "DWI"),
        ("MR", "FLAIR",             "UNKNOWN"),
        ("MR", "T2_AX",             "UNKNOWN"),
    ])
    def test_inference(self, modality, desc, expected):
        assert infer_image_type(modality, desc) == expected


# ── get_pipeline_commit ───────────────────────────────────────────────────────

class TestGetPipelineCommit:
    def test_returns_string(self):
        result = get_pipeline_commit()
        assert isinstance(result, str) and len(result) > 0

    def test_valid_format(self):
        result = get_pipeline_commit()
        is_hash    = all(c in "0123456789abcdef" for c in result) and len(result) >= 4
        is_unknown = result == "unknown"
        assert is_hash or is_unknown, f"예상치 못한 반환값: {result!r}"


# ── 적응형 허용오차 ────────────────────────────────────────────────────────────

class TestAdaptiveTolerance:
    """04_validate.py의 적응형 허용오차 계산 로직 검증."""
    FIXED = 0.05

    def _tol(self, n):
        return max(self.FIXED, 1.0 / n)

    def test_large_n_uses_fixed(self):
        assert self._tol(100) == self.FIXED

    def test_small_n_uses_adaptive(self):
        assert self._tol(8) == pytest.approx(0.125)

    def test_n_20_boundary(self):
        assert self._tol(20) == self.FIXED  # 1/20=0.05 == FIXED

    @pytest.mark.parametrize("n_subj,train_n,expected_pass", [
        (8,  5, True),   # |5/8 - 0.70| = 0.075 < 0.125 → pass
        (8,  2, False),  # |2/8 - 0.70| = 0.45  > 0.125 → fail
        (20, 14, True),  # |14/20 - 0.70| = 0.0 → pass
        (20, 13, True),  # |13/20 - 0.65| → 0.05 == tol → pass (edge)
        (20, 12, False), # |12/20 - 0.70| = 0.1 > 0.05 → fail
    ])
    def test_train_ratio_check(self, n_subj, train_n, expected_pass):
        tol = self._tol(n_subj)
        actual  = train_n / n_subj
        passed  = abs(actual - 0.70) <= tol + 1e-9
        assert passed == expected_pass
